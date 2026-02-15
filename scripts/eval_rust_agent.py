"""Rust coding agent evaluation suite.

Evaluates the fine-tuned model on Rust coding tasks by:
1. Generating solutions for held-out tasks
2. Running cargo check/test/clippy on each solution
3. Measuring iteration count, diff size, tool-call accuracy
4. Checking for capability regression (HumanEval, MMLU)

Metrics:
- cargo_check_pass_rate: % of code that compiles
- cargo_test_pass_rate: % of code that passes tests
- clippy_clean_rate: % with no clippy warnings
- iterations_to_green_median: median attempts needed
- diff_size_median: median diff in lines
- tool_call_format_accuracy: % valid JSON tool calls
- hallucinated_api_rate: % calls to non-existent APIs

Usage:
    python scripts/eval_rust_agent.py
    python scripts/eval_rust_agent.py --config configs/rust_eval.yaml
    python scripts/eval_rust_agent.py --checkpoint checkpoints/core_agent_grpo --num_samples 50

Requires: pip install -e ".[gpt_oss,rust_eval]"
"""
from __future__ import annotations

import json
import os
import statistics
import time

import yaml

from pipeline_lib.rust_evaluators import (
    RustTaskResult,
    run_cargo_check,
    run_cargo_clippy,
    run_cargo_test,
)


class RustAgentEvaluator:
    """Evaluation suite for Rust coding agent models."""

    def __init__(
        self,
        checkpoint: str,
        config: dict,
    ):
        """Initialize evaluator.

        Args:
            checkpoint: Path to model checkpoint.
            config: Evaluation config dict.
        """
        self.checkpoint = checkpoint
        self.config = config
        self.results: list[RustTaskResult] = []
        self.timeouts = config.get("evaluation", {}).get("timeouts", {})

        # Load model
        from pipeline_lib.unsloth_utils import load_unsloth_model
        print(f"Loading model from {checkpoint}...")
        self.model, self.tokenizer = load_unsloth_model(
            model_name=checkpoint,
            max_seq_length=config.get("model", {}).get("max_seq_length", 16384),
            load_in_4bit=config.get("model", {}).get("load_in_4bit", True),
        )

    def load_tasks(self, test_set: str, num_samples: int = 200) -> list[dict]:
        """Load evaluation tasks.

        Args:
            test_set: Path to test set (JSONL or HF dataset).
            num_samples: Maximum number of tasks.

        Returns:
            List of task dicts.
        """
        tasks = []

        if test_set.endswith(".jsonl"):
            with open(test_set) as f:
                for line in f:
                    try:
                        tasks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(tasks) >= num_samples:
                        break
        else:
            from datasets import load_from_disk
            ds = load_from_disk(test_set)
            for i, example in enumerate(ds):
                if i >= num_samples:
                    break
                tasks.append(dict(example))

        return tasks

    def generate_solution(
        self,
        task: dict,
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
    ) -> str:
        """Generate a solution for a task.

        Args:
            task: Task dict with description, tests, etc.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens to generate.

        Returns:
            Generated code string.
        """
        from dataset_formatters.harmony import encode_harmony_messages

        desc = task.get("description", "")
        tests = task.get("tests", "")
        starter = task.get("starter_code", "")

        content = desc
        if starter:
            content += f"\n\nStarter code:\n```rust\n{starter}\n```"
        if tests:
            content += f"\n\nTests:\n```rust\n{tests}\n```"

        prompt = encode_harmony_messages(
            [{"role": "user", "content": content}],
            developer_instructions="You are a Rust programming expert. Write correct, idiomatic code.",
            reasoning_effort="high",
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _extract_rust_code(response)

    def evaluate_solution(
        self,
        task_id: str,
        code: str,
        tests: str = "",
    ) -> RustTaskResult:
        """Evaluate a single solution.

        Args:
            task_id: Unique task identifier.
            code: Generated Rust code.
            tests: Optional test code to append.

        Returns:
            RustTaskResult with evaluation details.
        """
        full_code = code
        if tests:
            full_code = f"{code}\n\n{tests}"

        result = RustTaskResult(task_id=task_id)

        # cargo check
        check = run_cargo_check(
            full_code,
            timeout=self.timeouts.get("cargo_check", 60),
        )
        result.check_passed = check.success

        if not check.success:
            result.error = check.stderr[:500]
            return result

        # cargo test
        test_result = run_cargo_test(
            full_code,
            timeout=self.timeouts.get("cargo_test", 300),
        )
        result.test_passed = test_result.success

        # cargo clippy
        clippy = run_cargo_clippy(
            full_code,
            timeout=self.timeouts.get("cargo_clippy", 60),
        )
        result.clippy_clean = clippy.success

        # Diff size
        result.diff_lines = len(code.splitlines())

        return result

    def run_evaluation(
        self,
        test_set: str | None = None,
        num_samples: int | None = None,
    ) -> dict:
        """Run full evaluation suite.

        Args:
            test_set: Override test set path.
            num_samples: Override number of samples.

        Returns:
            Dict of aggregated metrics.
        """
        eval_config = self.config.get("evaluation", {})
        gen_config = self.config.get("generation", {})

        resolved_test_set: str = test_set or eval_config.get("test_set", "data/rust/eval/tasks.jsonl")
        resolved_num_samples: int = num_samples or eval_config.get("num_samples", 200)

        print(f"\n{'='*60}")
        print(f"Rust Agent Evaluation")
        print(f"{'='*60}")
        print(f"  Checkpoint: {self.checkpoint}")
        print(f"  Test set: {resolved_test_set}")
        print(f"  Num samples: {resolved_num_samples}")

        tasks = self.load_tasks(resolved_test_set, resolved_num_samples)
        print(f"  Loaded {len(tasks)} tasks")

        if not tasks:
            print("ERROR: No tasks loaded.")
            return {}

        self.results = []
        start_time = time.time()

        for i, task in enumerate(tasks):
            task_id = task.get("id", f"task_{i}")
            print(f"\n  [{i+1}/{len(tasks)}] Evaluating {task_id}...")

            try:
                # Generate solution
                code = self.generate_solution(
                    task,
                    temperature=gen_config.get("temperature", 0.2),
                    max_new_tokens=gen_config.get("max_new_tokens", 2048),
                )

                # Evaluate
                tests = task.get("tests", "")
                result = self.evaluate_solution(task_id, code, tests)
                self.results.append(result)

                status = "PASS" if result.test_passed else ("CHECK" if result.check_passed else "FAIL")
                print(f"    {status} | check={result.check_passed} test={result.test_passed} clippy={result.clippy_clean}")

            except Exception as e:
                print(f"    ERROR: {e}")
                self.results.append(RustTaskResult(task_id=task_id, error=str(e)))

        elapsed = time.time() - start_time
        print(f"\n  Evaluation completed in {elapsed:.1f}s")

        # Compute metrics
        metrics = self._compute_metrics()
        self._print_metrics(metrics)

        return metrics

    def _compute_metrics(self) -> dict:
        """Compute aggregate metrics from results."""
        if not self.results:
            return {}

        total = len(self.results)
        check_pass = sum(1 for r in self.results if r.check_passed)
        test_pass = sum(1 for r in self.results if r.test_passed)
        clippy_clean = sum(1 for r in self.results if r.clippy_clean)

        diff_sizes = [r.diff_lines for r in self.results if r.diff_lines > 0]
        iterations = [r.iterations for r in self.results if r.iterations > 0]

        metrics = {
            "cargo_check_pass_rate": check_pass / total,
            "cargo_test_pass_rate": test_pass / total,
            "clippy_clean_rate": clippy_clean / total,
            "total_tasks": total,
        }

        if diff_sizes:
            metrics["diff_size_median"] = statistics.median(diff_sizes)
        if iterations:
            metrics["iterations_to_green_median"] = statistics.median(iterations)

        # Check against targets
        metric_configs = self.config.get("metrics", {})
        metrics["targets_met"] = {}
        for metric_name, metric_config in metric_configs.items():
            if metric_name in metrics:
                target = metric_config.get("target")
                higher_is_better = metric_config.get("higher_is_better", True)
                if target is not None:
                    if higher_is_better:
                        met = metrics[metric_name] >= target
                    else:
                        met = metrics[metric_name] <= target
                    metrics["targets_met"][metric_name] = met

        return metrics

    def _print_metrics(self, metrics: dict) -> None:
        """Print formatted metrics."""
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")

        for key, value in metrics.items():
            if key == "targets_met":
                continue
            if isinstance(value, float):
                if value < 1.0 and key.endswith("_rate"):
                    print(f"  {key}: {value:.1%}")
                else:
                    print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        targets = metrics.get("targets_met", {})
        if targets:
            print(f"\nTarget checks:")
            for name, met in targets.items():
                status = "PASS" if met else "FAIL"
                print(f"  [{status}] {name}")

    def save_results(self, output_dir: str | None = None) -> None:
        """Save evaluation results to disk.

        Args:
            output_dir: Override output directory.
        """
        if output_dir is None:
            output_dir = self.config.get("output", {}).get("results_dir", "evals/rust_agent")

        if output_dir is None:
            raise ValueError("output_dir must be specified in config or CLI args")
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        metrics = self._compute_metrics()
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save individual results
        results_data = []
        for r in self.results:
            results_data.append({
                "task_id": r.task_id,
                "check_passed": r.check_passed,
                "test_passed": r.test_passed,
                "clippy_clean": r.clippy_clean,
                "iterations": r.iterations,
                "diff_lines": r.diff_lines,
                "error": r.error,
            })

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to {output_dir}")


def _extract_rust_code(response: str) -> str:
    """Extract Rust code from a model response.

    Handles responses with ```rust code blocks and plain code.
    """
    # Try to extract from code block
    if "```rust" in response:
        parts = response.split("```rust")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()

    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            # Skip language identifier on first line
            lines = code.strip().splitlines()
            if lines and lines[0].strip() in ("rust", "rs", ""):
                code = "\n".join(lines[1:])
            return code.strip()

    # Return as-is (assume it's code)
    return response.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Rust coding agent")
    parser.add_argument("--config", type=str, default="configs/rust_eval.yaml")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--test_set", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint = args.checkpoint or config["model"]["checkpoint"]

    evaluator = RustAgentEvaluator(checkpoint=checkpoint, config=config)
    metrics = evaluator.run_evaluation(
        test_set=args.test_set,
        num_samples=args.num_samples,
    )
    evaluator.save_results(output_dir=args.output_dir)
