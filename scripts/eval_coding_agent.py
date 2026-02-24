"""Evaluate coding agent across languages.

Runs evaluation tasks through the model and scores completions using
language-specific evaluators (Rust, Python, TypeScript, Go).

Usage:
    python scripts/eval_coding_agent.py --config configs/rust_eval.yaml
    python scripts/eval_coding_agent.py --config configs/python_eval.yaml
    python scripts/eval_coding_agent.py --config configs/typescript_eval.yaml --language typescript
    python scripts/eval_coding_agent.py --config configs/go_eval.yaml --language go

Requires: pip install -e ".[gpt_oss]"
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Disable wandb and parallelism before other imports
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pipeline_lib.evaluator_dispatch import compute_execution_reward
from pipeline_lib.unsloth_utils import load_unsloth_model, print_trainable_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported languages
SUPPORTED_LANGUAGES = {"rust", "python", "typescript", "go"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_tasks(test_set: str, num_samples: int) -> list[dict[str, Any]]:
    """Load evaluation tasks from a JSONL file or a HuggingFace dataset path.

    Args:
        test_set: Path to a .jsonl file or a HuggingFace dataset identifier.
        num_samples: Maximum number of tasks to load.  Pass 0 or a negative
            value to load all available tasks.

    Returns:
        List of task dicts, each expected to contain at minimum a ``task_id``
        and ``prompt`` key.

    Raises:
        FileNotFoundError: If ``test_set`` looks like a local path but does
            not exist on disk.
        ValueError: If the loaded dataset is empty.
    """
    tasks: list[dict[str, Any]] = []

    if test_set.endswith(".jsonl") or Path(test_set).exists():
        path = Path(test_set)
        if not path.exists():
            raise FileNotFoundError(f"Test set not found: {test_set}")
        logger.info("Loading eval tasks from JSONL: %s", path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    else:
        # Attempt to load as a HuggingFace dataset
        try:
            from datasets import load_dataset  # type: ignore[import]

            logger.info("Loading eval tasks from HuggingFace dataset: %s", test_set)
            ds = load_dataset(test_set, split="test")
            tasks = list(ds)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load test set '{test_set}' as JSONL or HF dataset: {exc}"
            ) from exc

    if not tasks:
        raise ValueError(f"No tasks found in test set: {test_set}")

    if num_samples and num_samples > 0:
        tasks = tasks[:num_samples]

    logger.info("Loaded %d evaluation tasks", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------


def _extract_code_from_completion(completion: str) -> str:
    """Extract the first fenced code block from a model completion.

    Falls back to returning the entire completion string when no fence is
    found so that callers always receive a non-empty string to evaluate.
    """
    lines = completion.splitlines()
    inside = False
    code_lines: list[str] = []

    for line in lines:
        if not inside and line.strip().startswith("```"):
            inside = True
            continue
        if inside:
            if line.strip().startswith("```"):
                break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)
    return completion


def _count_valid_tool_calls(tool_calls: list[dict[str, Any]] | None) -> tuple[int, int]:
    """Return (valid_count, total_count) for a list of tool call dicts."""
    if not tool_calls:
        return 0, 0

    total = len(tool_calls)
    valid = 0
    for call in tool_calls:
        try:
            # A call is considered valid when it has a ``name`` and its
            # ``arguments`` field is parseable JSON (or already a dict).
            if "name" not in call:
                continue
            args = call.get("arguments", "{}")
            if isinstance(args, str):
                json.loads(args)
            valid += 1
        except (json.JSONDecodeError, TypeError):
            pass

    return valid, total


def evaluate_task(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    language: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate a completion for a single task and evaluate it.

    Args:
        model: Loaded (Unsloth) model.
        tokenizer: Corresponding tokenizer.
        task: Task dict with at least ``task_id`` and ``prompt`` keys.
        language: One of ``SUPPORTED_LANGUAGES``.
        config: Full evaluation config dict (used for generation settings and
            per-task timeout).

    Returns:
        Result dict with keys:
            task_id, reward, check_passed, test_passed, lint_clean,
            iterations, tool_calls_valid, tool_calls_total,
            completion_tokens, elapsed_seconds, error (str or None).
    """
    import torch  # imported lazily to avoid slowing startup for --help

    task_id = task.get("task_id", "unknown")
    prompt = task.get("prompt", "")
    result: dict[str, Any] = {
        "task_id": task_id,
        "reward": 0.0,
        "check_passed": False,
        "test_passed": False,
        "lint_clean": False,
        "iterations": 0,
        "tool_calls_valid": 0,
        "tool_calls_total": 0,
        "completion_tokens": 0,
        "elapsed_seconds": 0.0,
        "error": None,
    }

    gen_cfg = config.get("generation", {})
    temperature: float = float(gen_cfg.get("temperature", 0.2))
    max_new_tokens: int = int(gen_cfg.get("max_new_tokens", 2048))
    per_task_timeout: int = int(
        config.get("evaluation", {}).get("timeouts", {}).get("per_task", 600)
    )
    max_iterations: int = int(
        config.get("evaluation", {}).get("max_iterations_per_task", 10)
    )

    t_start = time.monotonic()

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        best_reward = 0.0
        best_code = ""
        tool_calls_valid_total = 0
        tool_calls_grand_total = 0

        for iteration in range(1, max_iterations + 1):
            elapsed = time.monotonic() - t_start
            if elapsed >= per_task_timeout:
                logger.debug(
                    "Task %s: per-task timeout reached after %d iterations",
                    task_id,
                    iteration - 1,
                )
                break

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            code = _extract_code_from_completion(completion)

            # Gather tool calls if the task carries a reference list
            tool_calls: list[dict[str, Any]] | None = task.get("tool_calls")
            valid, total = _count_valid_tool_calls(tool_calls)
            tool_calls_valid_total += valid
            tool_calls_grand_total += total

            reward_result = compute_execution_reward(
                code,
                language=language,
                tool_calls=tool_calls,
                reward_config=config.get("reward_config"),
            )

            # ``compute_execution_reward`` may return a float or a dict
            if isinstance(reward_result, dict):
                reward: float = float(reward_result.get("reward", 0.0))
                check_passed: bool = bool(reward_result.get("check_passed", False))
                test_passed: bool = bool(reward_result.get("test_passed", False))
                lint_clean: bool = bool(reward_result.get("lint_clean", False))
            else:
                reward = float(reward_result)
                check_passed = reward >= 0.5
                test_passed = reward >= 0.9
                lint_clean = False

            if reward > best_reward:
                best_reward = reward
                best_code = code  # noqa: F841 — stored for future trajectory saving

            result["iterations"] = iteration

            if test_passed:
                break

        result["reward"] = best_reward
        result["check_passed"] = check_passed
        result["test_passed"] = test_passed
        result["lint_clean"] = lint_clean
        result["tool_calls_valid"] = tool_calls_valid_total
        result["tool_calls_total"] = tool_calls_grand_total
        result["completion_tokens"] = int(new_tokens.shape[-1]) if "new_tokens" in dir() else 0

    except Exception as exc:
        logger.warning("Task %s failed with error: %s", task_id, exc, exc_info=True)
        result["error"] = str(exc)

    result["elapsed_seconds"] = round(time.monotonic() - t_start, 3)
    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    results: list[dict[str, Any]],
    metrics_config: dict[str, Any],
) -> dict[str, float]:
    """Compute aggregate metrics from a list of per-task result dicts.

    Metrics are derived dynamically from the keys present in each result dict
    and from what the ``metrics_config`` defines.  Unrecognised metric names
    in the config are silently skipped.

    Args:
        results: List of dicts returned by :func:`evaluate_task`.
        metrics_config: The ``metrics`` section of the YAML eval config.

    Returns:
        Dict mapping metric name to its computed float value.
    """
    if not results:
        return {}

    n = len(results)

    # ---- boolean pass-rate metrics ----------------------------------------
    bool_field_to_metric: dict[str, str] = {
        # Generic keys present in all language results
        "check_passed": "check_pass_rate",
        "test_passed": "test_pass_rate",
        "lint_clean": "lint_clean_rate",
    }

    # Language-specific aliases recognised from config keys
    lang_metric_aliases: dict[str, str] = {
        # Rust
        "cargo_check_pass_rate": "check_pass_rate",
        "cargo_test_pass_rate": "test_pass_rate",
        "clippy_clean_rate": "lint_clean_rate",
        # Python
        "syntax_check_pass_rate": "check_pass_rate",
        "pytest_pass_rate": "test_pass_rate",
        "mypy_clean_rate": "lint_clean_rate",
        "ruff_clean_rate": "lint_clean_rate",
        # TypeScript
        "tsc_pass_rate": "check_pass_rate",
        "jest_pass_rate": "test_pass_rate",
        "eslint_clean_rate": "lint_clean_rate",
        # Go
        "go_build_pass_rate": "check_pass_rate",
        "go_test_pass_rate": "test_pass_rate",
        "go_vet_clean_rate": "lint_clean_rate",
        "golangci_lint_clean_rate": "lint_clean_rate",
    }

    # Compute base rates from result fields
    base: dict[str, float] = {
        "check_pass_rate": sum(1 for r in results if r.get("check_passed")) / n,
        "test_pass_rate": sum(1 for r in results if r.get("test_passed")) / n,
        "lint_clean_rate": sum(1 for r in results if r.get("lint_clean")) / n,
    }

    # Tool-call format accuracy
    total_tc = sum(r.get("tool_calls_total", 0) for r in results)
    valid_tc = sum(r.get("tool_calls_valid", 0) for r in results)
    base["tool_call_format_accuracy"] = (valid_tc / total_tc) if total_tc > 0 else 1.0

    # Hallucinated API rate: stored directly in result if evaluator provides it
    hallucinated_values = [
        r["hallucinated_api_rate"]
        for r in results
        if "hallucinated_api_rate" in r
    ]
    if hallucinated_values:
        base["hallucinated_api_rate"] = statistics.mean(hallucinated_values)
    else:
        base["hallucinated_api_rate"] = 0.0

    # Median-based metrics
    iterations_list = [r.get("iterations", 0) for r in results if r.get("iterations")]
    base["iterations_to_green_median"] = float(
        statistics.median(iterations_list) if iterations_list else 0
    )

    diff_sizes = [r["diff_size"] for r in results if "diff_size" in r]
    base["diff_size_median"] = float(
        statistics.median(diff_sizes) if diff_sizes else 0.0
    )

    # ---- Map to the names used in the YAML config --------------------------
    computed: dict[str, float] = {}
    for metric_name in metrics_config:
        if metric_name in base:
            computed[metric_name] = base[metric_name]
        elif metric_name in lang_metric_aliases:
            computed[metric_name] = base[lang_metric_aliases[metric_name]]
        else:
            logger.debug("Metric '%s' not computed — no matching result field", metric_name)

    return computed


# ---------------------------------------------------------------------------
# Target checking
# ---------------------------------------------------------------------------


def check_targets(
    metrics: dict[str, float],
    metrics_config: dict[str, Any],
) -> bool:
    """Print a pass/fail report for each metric against its target.

    Args:
        metrics: Dict of computed metric values (from :func:`compute_metrics`).
        metrics_config: The ``metrics`` section of the YAML eval config.

    Returns:
        ``True`` when ALL defined targets are met, ``False`` otherwise.
    """
    all_pass = True
    col_w = max((len(k) for k in metrics_config), default=30) + 2

    print("\n" + "=" * 72)
    print("  EVALUATION RESULTS")
    print("=" * 72)
    print(f"  {'Metric':<{col_w}} {'Value':>8}  {'Target':>8}  {'Status':>6}")
    print("-" * 72)

    for metric_name, cfg in metrics_config.items():
        if not isinstance(cfg, dict):
            continue

        target = cfg.get("target")
        higher_is_better: bool = cfg.get("higher_is_better", True)
        value = metrics.get(metric_name)

        if value is None:
            print(f"  {metric_name:<{col_w}} {'N/A':>8}  {str(target):>8}  {'SKIP':>6}")
            continue

        if target is None:
            print(f"  {metric_name:<{col_w}} {value:>8.4f}  {'N/A':>8}  {'INFO':>6}")
            continue

        if higher_is_better:
            passed = value >= target
        else:
            passed = value <= target

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"  {metric_name:<{col_w}} {value:>8.4f}  {target:>8}  {status:>6}")

    print("=" * 72)
    overall = "ALL TARGETS MET" if all_pass else "SOME TARGETS MISSED"
    print(f"  Overall: {overall}")
    print("=" * 72 + "\n")

    return all_pass


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    config_path: str,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config, run model evaluation, save results, and check targets.

    Args:
        config_path: Path to a YAML evaluation config file.
        cli_overrides: Optional dict of CLI-level overrides that take
            precedence over the YAML config.  Recognised keys:
            ``language``, ``checkpoint``, ``num_samples``, ``output_dir``.

    Returns:
        Dict with keys ``metrics`` (computed metrics), ``all_targets_met``
        (bool), and ``results_path`` (str path to saved JSON).
    """
    if cli_overrides is None:
        cli_overrides = {}

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_path = str(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # 2. Determine language (CLI > config top-level > evaluation sub-key > default)
    # ------------------------------------------------------------------
    language: str = (
        cli_overrides.get("language")
        or config.get("language")
        or config.get("evaluation", {}).get("language")
        or "rust"
    )
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Must be one of: {sorted(SUPPORTED_LANGUAGES)}"
        )

    logger.info("Language: %s", language)

    # ------------------------------------------------------------------
    # 3. Resolve checkpoint and output dir (CLI overrides win)
    # ------------------------------------------------------------------
    model_cfg: dict[str, Any] = config.get("model", {})
    if cli_overrides.get("checkpoint"):
        model_cfg["checkpoint"] = cli_overrides["checkpoint"]

    checkpoint: str = model_cfg.get("checkpoint", "checkpoints/core_agent_grpo")
    max_seq_length: int = int(model_cfg.get("max_seq_length", 16384))
    load_in_4bit: bool = bool(model_cfg.get("load_in_4bit", True))
    dtype = model_cfg.get("dtype")  # None → auto

    output_cfg: dict[str, Any] = config.get("output", {})
    results_dir: str = cli_overrides.get("output_dir") or output_cfg.get(
        "results_dir", f"evals/{language}_agent"
    )
    save_trajectories: bool = bool(output_cfg.get("save_trajectories", True))

    eval_cfg: dict[str, Any] = config.get("evaluation", {})
    num_samples: int = int(
        cli_overrides.get("num_samples") or eval_cfg.get("num_samples", 200)
    )
    test_set: str = eval_cfg.get("test_set", f"data/{language}/eval/tasks.jsonl")

    run_name: str = config.get("run_name", f"eval-{language}-agent")
    logger.info("Run name: %s", run_name)

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    logger.info("Loading model from checkpoint: %s", checkpoint)
    model, tokenizer = load_unsloth_model(
        model_name=checkpoint,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )
    print_trainable_params(model)

    # ------------------------------------------------------------------
    # 5. Load eval tasks
    # ------------------------------------------------------------------
    tasks = load_eval_tasks(test_set, num_samples)

    # ------------------------------------------------------------------
    # 6. Evaluation loop
    # ------------------------------------------------------------------
    logger.info("Starting evaluation loop: %d tasks, language=%s", len(tasks), language)
    results: list[dict[str, Any]] = []
    n_tasks = len(tasks)

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("task_id", f"task_{idx}")
        logger.info("[%d/%d] Evaluating task: %s", idx, n_tasks, task_id)

        result = evaluate_task(model, tokenizer, task, language, config)
        results.append(result)

        # Incremental progress log
        if idx % 10 == 0 or idx == n_tasks:
            n_passed = sum(1 for r in results if r.get("test_passed"))
            logger.info(
                "Progress %d/%d — test pass rate so far: %.1f%%",
                idx,
                n_tasks,
                100.0 * n_passed / idx,
            )

    # ------------------------------------------------------------------
    # 7. Compute and display metrics
    # ------------------------------------------------------------------
    metrics_config: dict[str, Any] = config.get("metrics", {})
    metrics = compute_metrics(results, metrics_config)
    all_targets_met = check_targets(metrics, metrics_config)

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    results_path_obj = Path(results_dir)
    results_path_obj.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_file = results_path_obj / f"{run_name}_{timestamp}.json"

    output_payload: dict[str, Any] = {
        "run_name": run_name,
        "language": language,
        "checkpoint": checkpoint,
        "timestamp": timestamp,
        "num_tasks": len(results),
        "metrics": metrics,
        "all_targets_met": all_targets_met,
        "config": config,
    }
    if save_trajectories:
        output_payload["results"] = results

    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)

    logger.info("Results saved to: %s", results_file)

    return {
        "metrics": metrics,
        "all_targets_met": all_targets_met,
        "results_path": str(results_file),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate coding agent across languages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML evaluation config (e.g. configs/rust_eval.yaml).",
    )
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default=None,
        help=(
            "Override the language from config. "
            "Priority: --language > config.language > config.evaluation.language > 'rust'."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override model checkpoint path from config.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Override number of evaluation samples from config.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override results output directory from config.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.language is not None:
        overrides["language"] = args.language
    if args.checkpoint is not None:
        overrides["checkpoint"] = args.checkpoint
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir

    outcome = run_evaluation(args.config, cli_overrides=overrides)

    exit_code = 0 if outcome["all_targets_met"] else 1
    sys.exit(exit_code)
