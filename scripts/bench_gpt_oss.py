#!/usr/bin/env python3
"""
GPT-OSS Coding Benchmark Suite for DGX Spark.
Runs HumanEval, coding generation, and tool-use evaluations.

Fixes from v1:
  - Uses Harmony chat template (required by GPT-OSS models)
  - BF16 precision for 20B (fits in 128GB unified memory), 4-bit for 120B
  - Full 164-problem HumanEval by default (--quick for 20)
  - Proper generation params: temp=0.1, top_p=0.95, max_tokens=2048
  - Integrated HumanEval evaluation (runs tests after generation)

Usage:
    docker run --gpus all --ipc=host -v /home/rmarnold:/home/rmarnold \
        unsloth-dgx-spark:latest python3 /home/rmarnold/Projects/llm-training-pipeline/scripts/bench_gpt_oss.py \
        --model openai/gpt-oss-20b
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# Generation with Harmony chat template
# ---------------------------------------------------------------------------

def generate_chat(model, tokenizer, user_message, max_new_tokens=2048,
                  system_message="Reasoning: low", temperature=0.1, top_p=0.95):
    """Generate using Harmony chat template (required for GPT-OSS models)."""
    import torch

    messages = [{"role": "system", "content": system_message}]
    if isinstance(user_message, list):
        messages.extend(user_message)
    else:
        messages.append({"role": "user", "content": user_message})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    n = outputs.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    # Strip Harmony reasoning prefix — extract final response only
    text = _strip_harmony_reasoning(text)
    return text, n


def _strip_harmony_reasoning(text):
    """Strip Harmony format reasoning/analysis prefix from model output.

    Harmony outputs: analysis<thinking>assistantfinal<answer>
    or: analysis<thinking>assistant<answer>
    We want only the <answer> part.
    """
    # Try various Harmony markers — model may use assistant or commentary format
    for marker in ["assistantfinal", "commentaryfinal", "assistant final",
                    "commentary final", "assistant", "final"]:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                result = parts[1].strip()
                if result:
                    return result
    # If no marker found, strip leading "analysis..." or "commentary..." block
    for prefix in ("analysis", "commentary"):
        if text.startswith(prefix):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                return lines[1].strip()
    return text


def generate_raw(model, tokenizer, prompt, max_new_tokens=512):
    """Raw text completion (legacy, for reference only)."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    n = outputs.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text, n


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, output_dir, use_bf16=False):
    """Load model in 4-bit quantization (default) or BF16.

    4-bit is the default — proven to work well for GPT-OSS MoE models.
    BF16 available via --bf16 flag if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}\nLoading: {model_name}\n{'='*60}")
    from unsloth import FastLanguageModel
    import torch

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    use_4bit = not use_bf16
    dtype_label = "bfloat16" if use_bf16 else "4bit"
    print(f"Precision: {dtype_label} (GPU: {gpu_mem:.0f}GB)")

    # Unified memory patch for DGX Spark
    _orig_mem_get_info = None
    if gpu_mem > 100:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        _orig_mem_get_info = torch.cuda.mem_get_info
        def _patched_mem_get_info(device=None):
            free, total = _orig_mem_get_info(device)
            return (int(total * 0.95), total)
        torch.cuda.mem_get_info = _patched_mem_get_info
        print(f"Unified memory detected ({gpu_mem:.0f}GB), patched mem_get_info")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=use_4bit,
        dtype=torch.bfloat16 if use_bf16 else None,
    )

    if _orig_mem_get_info is not None:
        torch.cuda.mem_get_info = _orig_mem_get_info

    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    # Verify chat template works
    print("Verifying Harmony chat template...")
    test_text, _ = generate_chat(model, tokenizer,
                                 "Write a Python function that returns 42.",
                                 max_new_tokens=128)
    print(f"Chat template test: {test_text[:200]}")

    info = {
        "model": model_name,
        "dtype": dtype_label,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(gpu_mem, 1),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1),
    }
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"GPU mem allocated: {info['allocated_gb']} GB / {info['gpu_mem_gb']} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# HumanEval benchmark + evaluation
# ---------------------------------------------------------------------------

def bench_humaneval(model, tokenizer, output_dir, limit=None):
    """Generate HumanEval completions using Harmony chat template.

    Each HumanEval prompt is wrapped in a chat message asking the model to
    complete the function. The completion is then extracted (code only).
    """
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    t0 = time.time()
    print(f"\nHumanEval ({len(ds)} problems)...")

    for i, ex in enumerate(ds):
        # Wrap HumanEval prompt in chat template
        user_msg = (
            "Complete the following Python function. "
            "Return ONLY the function body (the code that goes after the signature). "
            "Do not repeat the function signature or add any explanation.\n\n"
            f"{ex['prompt']}"
        )
        comp, ntok = generate_chat(model, tokenizer, user_msg, max_new_tokens=2048)

        # Extract just the function body from the completion
        clean = _extract_function_body(comp, ex["entry_point"])

        results.append({
            "task_id": ex["task_id"],
            "completion": clean,
            "raw_completion": comp,
            "tokens": ntok,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(ds)}] {(i+1)/elapsed:.1f} prob/s  "
                  f"ETA: {elapsed / (i+1) * (len(ds) - i - 1):.0f}s")

    f = os.path.join(output_dir, "humaneval_completions.jsonl")
    with open(f, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"Generated: {len(results)} in {time.time()-t0:.0f}s -> {f}")
    return results


def _extract_function_body(completion, entry_point):
    """Extract clean function body from a chat completion.

    Handles cases where the model wraps code in markdown blocks, repeats
    the signature, or adds explanations.
    """
    text = completion.strip()

    # Strip markdown code blocks
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            text = parts[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            text = parts[1]

    # If the model repeated the full function, extract just the body
    lines = text.split("\n")
    body_lines = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        # Skip function signature if model repeated it
        if stripped.startswith(f"def {entry_point}") or stripped.startswith(f"def {entry_point}("):
            in_body = True
            continue
        if in_body or not stripped.startswith("def "):
            # Stop at next top-level function/class definition
            if stripped.startswith(("def ", "class ", "if __name__")) and body_lines:
                break
            body_lines.append(line)
            in_body = True

    if body_lines:
        return "\n".join(body_lines)
    return text


def eval_humaneval(output_dir):
    """Evaluate HumanEval completions against test cases."""
    completions_file = os.path.join(output_dir, "humaneval_completions.jsonl")
    if not os.path.exists(completions_file):
        print("  No completions file found, skipping evaluation")
        return {}

    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = {ex["task_id"]: ex for ex in ds}

    completions = []
    with open(completions_file) as f:
        for line in f:
            completions.append(json.loads(line.strip()))

    correct = 0
    details = []
    print(f"\nEvaluating {len(completions)} HumanEval completions...")

    for comp in completions:
        task_id = comp["task_id"]
        completion = comp["completion"]

        if task_id not in problems:
            details.append({"task_id": task_id, "passed": False, "error": "task not found"})
            continue

        problem = problems[task_id]
        full_code = problem["prompt"] + completion + "\n" + problem["test"]
        full_code += f"\ncheck({problem['entry_point']})\n"

        passed = _run_code_safe(full_code, timeout=10)
        if passed:
            correct += 1
        details.append({"task_id": task_id, "passed": passed})

    total = len(completions)
    pass_at_1 = correct / total if total > 0 else 0

    results = {
        "total": total,
        "correct": correct,
        "pass_at_1": round(pass_at_1, 4),
        "details": details,
    }

    output_path = os.path.join(output_dir, "humaneval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Pass@1: {pass_at_1:.1%} ({correct}/{total})")

    # Show per-task results
    for d in details:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"    {status}  {d['task_id']}")

    return results


def _run_code_safe(code, timeout=10):
    """Execute code in a subprocess with timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Coding generation benchmark
# ---------------------------------------------------------------------------

def bench_coding(model, tokenizer, output_dir):
    """Run coding generation benchmark using Harmony chat template."""
    tasks = [
        ("fibonacci", "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"),
        ("binary_search", "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Binary search. Return index or -1.\"\"\"\n"),
        ("merge_sort", "def merge_sort(arr: list) -> list:\n    \"\"\"Merge sort implementation.\"\"\"\n"),
        ("safe_json", "def safe_parse_json(s: str):\n    \"\"\"Parse JSON string, return None on failure.\"\"\"\n"),
        ("argparse_cli", "def create_parser():\n    \"\"\"Create argparse parser with --input, --output, --verbose.\"\"\"\n"),
        ("find_files", "def find_files(directory: str, pattern: str) -> list:\n    \"\"\"Recursively find files matching glob pattern.\"\"\"\n"),
        ("parse_diff", "def parse_git_diff(diff_text: str) -> list:\n    \"\"\"Parse unified git diff, return list of changed files.\"\"\"\n"),
        ("retry_deco", "def retry(max_retries=3, backoff=2):\n    \"\"\"Decorator: retry with exponential backoff.\"\"\"\n"),
        ("tool_call", "def format_tool_call(name: str, args: dict, call_id: str = None) -> dict:\n    \"\"\"Format a tool call as JSON with name, arguments, id.\"\"\"\n"),
        ("conv_state", "class ConversationState:\n    \"\"\"Multi-turn conversation state: message history, tool results, context management.\"\"\"\n"),
    ]
    results = []
    t0 = time.time()
    print(f"\nCoding bench ({len(tasks)} tasks)...")
    for tid, prompt in tasks:
        user_msg = f"Complete this Python code. Return ONLY the implementation.\n\n{prompt}"
        comp, ntok = generate_chat(model, tokenizer, user_msg, max_new_tokens=1024)
        results.append({"id": tid, "prompt": prompt, "completion": comp, "tokens": ntok})
        print(f"  {tid}: {ntok} tokens")
    f = os.path.join(output_dir, "coding_bench.json")
    with open(f, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Done: {time.time()-t0:.0f}s -> {f}")
    return results


# ---------------------------------------------------------------------------
# Tool-use benchmark
# ---------------------------------------------------------------------------

def bench_tool_use(model, tokenizer, output_dir):
    """Run tool-use benchmark using Harmony chat template."""
    TOOL_SYSTEM = (
        "You are a coding assistant with access to these tools:\n"
        "- read_file(path: str) -> str: Read file contents\n"
        "- write_file(path: str, content: str) -> bool: Write to a file\n"
        "- run_command(cmd: str) -> str: Execute a shell command\n"
        "- search_files(pattern: str) -> list: Search for files\n\n"
        "When you need to use a tool, format it as: tool_name(args)\n"
        "Reasoning: low"
    )

    tasks = [
        {
            "id": "single_tool_call",
            "messages": [
                {"role": "user", "content": "Read the file at /tmp/config.json"},
            ],
        },
        {
            "id": "multi_step_debug",
            "messages": [
                {"role": "user", "content":
                    "The tests in tests/test_auth.py are failing with "
                    "'AttributeError: module has no attribute validate_token'. "
                    "Debug and fix the issue."},
            ],
        },
        {
            "id": "multi_turn_plan",
            "messages": [
                {"role": "user", "content":
                    "Add a --dry-run flag to the deploy script at scripts/deploy.sh "
                    "that prints what would happen without executing."},
            ],
        },
        {
            "id": "error_recovery",
            "messages": [
                {"role": "user", "content": "Fix the build error."},
                {"role": "assistant", "content": "Let me check the build output first.\n\n"
                    "run_command(\"cargo build 2>&1\")"},
                {"role": "user", "content":
                    "Tool result:\nerror[E0308]: mismatched types\n"
                    "  --> src/parser.rs:42:12\n"
                    "   | expected `Vec<Token>`, found `Option<Vec<Token>>`"},
            ],
        },
    ]

    results = []
    t0 = time.time()
    print(f"\nTool-use bench ({len(tasks)} tasks)...")
    for task in tasks:
        comp, ntok = generate_chat(model, tokenizer, task["messages"],
                                   system_message=TOOL_SYSTEM,
                                   max_new_tokens=1024)
        results.append({
            "id": task["id"],
            "messages": task["messages"],
            "completion": comp,
            "tokens": ntok,
        })
        print(f"  {task['id']}: {ntok} tokens")
    f = os.path.join(output_dir, "tool_use_bench.json")
    with open(f, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Done: {time.time()-t0:.0f}s -> {f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS Coding Benchmark Suite (v2 — Harmony chat template + BF16)")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", default="/home/rmarnold/benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Limit HumanEval to 20 samples (default: full 164)")
    parser.add_argument("--humaneval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip HumanEval test execution (only generate completions)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 precision instead of 4-bit quantization")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    out = os.path.join(args.output_dir, model_short)
    limit = 20 if args.quick else None

    model, tokenizer = load_model(args.model, out, use_bf16=args.bf16)

    if not args.humaneval_only:
        bench_coding(model, tokenizer, out)
        bench_tool_use(model, tokenizer, out)

    bench_humaneval(model, tokenizer, out, limit=limit)

    if not args.skip_eval:
        eval_humaneval(out)

    print(f"\n{'='*60}\nAll benchmarks complete -> {out}\n{'='*60}")


if __name__ == "__main__":
    main()
