#!/usr/bin/env python3
"""
GPT-OSS Benchmark Suite — vLLM API version.

Uses the vLLM OpenAI-compatible API for fast inference with NVFP4/MXFP4 kernels.
Sends concurrent requests for HumanEval to maximize throughput.

Usage:
    # Start vLLM server first, then:
    python3 bench_gpt_oss_vllm.py --base-url http://localhost:8888
    python3 bench_gpt_oss_vllm.py --base-url http://localhost:8888 --quick
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def chat_complete(base_url, model, messages, max_tokens=2048,
                  temperature=0.1, top_p=0.95, tools=None):
    """Send a chat completion request to vLLM.

    Returns (content, token_count) for normal requests.
    When tools are provided, returns (message_dict, token_count) where
    message_dict contains 'content', 'tool_calls', and 'reasoning' fields.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    ntok = usage.get("completion_tokens", 0)
    if tools:
        return msg, ntok
    return msg.get("content", ""), ntok


def get_model_name(base_url):
    """Get the model name from the vLLM server."""
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


# ---------------------------------------------------------------------------
# HumanEval benchmark + evaluation
# ---------------------------------------------------------------------------

def bench_humaneval(base_url, model, output_dir, limit=None, concurrency=8):
    """Generate HumanEval completions via vLLM API with concurrent requests."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    print(f"\nHumanEval ({len(ds)} problems, concurrency={concurrency})...")
    t0 = time.time()

    def _generate_one(ex):
        user_msg = (
            "Complete the following Python function. "
            "Return ONLY the function body (the code that goes after the signature). "
            "Do not repeat the function signature or add any explanation.\n\n"
            f"{ex['prompt']}"
        )
        messages = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": user_msg},
        ]
        comp, ntok = chat_complete(base_url, model, messages)
        clean = _extract_function_body(comp, ex["entry_point"])
        return {
            "task_id": ex["task_id"],
            "completion": clean,
            "raw_completion": comp,
            "tokens": ntok,
        }

    results = [None] * len(ds)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_generate_one, ds[i]): i for i in range(len(ds))}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "task_id": ds[idx]["task_id"],
                    "completion": "",
                    "raw_completion": f"ERROR: {e}",
                    "tokens": 0,
                }
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(ds) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(ds)}] {rate:.1f} prob/s  ETA: {eta:.0f}s")

    f = os.path.join(output_dir, "humaneval_completions.jsonl")
    with open(f, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    elapsed = time.time() - t0
    print(f"Generated: {len(results)} in {elapsed:.0f}s ({len(results)/elapsed:.1f} prob/s) -> {f}")
    return results


def _extract_function_body(completion, entry_point):
    """Extract clean function body from a chat completion.

    IMPORTANT: Preserves indentation — HumanEval expects indented body lines
    that are concatenated directly after the function signature.
    """
    text = completion

    # Strip markdown code blocks
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            text = parts[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            text = parts[1]

    # Check if the model repeated the full function signature
    lines = text.split("\n")
    sig_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"def {entry_point}("):
            sig_idx = i
            break

    if sig_idx >= 0:
        # Model repeated the signature — take everything after it
        body_lines = []
        for line in lines[sig_idx + 1:]:
            stripped = line.strip()
            # Stop at next top-level definition
            if stripped.startswith(("def ", "class ", "if __name__")) and body_lines:
                break
            body_lines.append(line)
        if body_lines:
            return "\n".join(body_lines)

    # Model returned just the body — return as-is (preserve indentation)
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
    total = len(completions)
    print(f"\nEvaluating {total} HumanEval completions...")

    results = []
    for i, comp in enumerate(completions):
        task_id = comp["task_id"]
        if task_id not in problems:
            results.append({"task_id": task_id, "passed": False, "error": "not found"})
            continue

        problem = problems[task_id]
        full_code = (
            problem["prompt"] + comp["completion"] + "\n"
            + problem["test"] + f"\ncheck({problem['entry_point']})\n"
        )
        passed = _run_code_safe(full_code)
        if passed:
            correct += 1
        results.append({"task_id": task_id, "passed": passed})

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{total}] {correct} correct so far")

    pass_at_1 = correct / total if total > 0 else 0
    summary = {
        "total": total,
        "correct": correct,
        "pass_at_1": round(pass_at_1, 4),
        "details": results,
    }

    out = os.path.join(output_dir, "humaneval_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Pass@1: {pass_at_1:.1%} ({correct}/{total})")
    print(f"  Results: {out}")

    # Print failures
    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\n  Failed ({len(fails)}):")
        for r in fails[:10]:
            status = "ERROR" if r.get("error") else "FAIL"
            print(f"    {status}  {r['task_id']}")
        if len(fails) > 10:
            print(f"    ... and {len(fails)-10} more")

    return summary


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

def bench_coding(base_url, model, output_dir):
    """Run coding generation benchmark via vLLM API."""
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
        messages = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": user_msg},
        ]
        comp, ntok = chat_complete(base_url, model, messages, max_tokens=1024)
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

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def _format_tool_calls(tool_calls):
    """Format tool_calls array into a readable string for scoring."""
    if not tool_calls:
        return ""
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "unknown")
        args = fn.get("arguments", "{}")
        parts.append(f"{name}({args})")
    return "\n".join(parts)


def bench_tool_use(base_url, model, output_dir):
    """Run tool-use benchmark via vLLM API with proper OpenAI tool definitions.

    Sends structured tools via the OpenAI tools parameter so vLLM can use
    --tool-call-parser openai to return Harmony-format tool calls as
    structured tool_calls in the response.
    """
    TOOL_SYSTEM = "You are a coding assistant. Use the provided tools to help the user."

    tasks = [
        {
            "id": "single_tool_call",
            "messages": [
                {"role": "user", "content": "Read the file at /tmp/config.json"},
            ],
            # Direct file read — expect read_file specifically
            "expect_tools": ["read_file"],
        },
        {
            "id": "multi_step_debug",
            "messages": [
                {"role": "user", "content":
                    "The tests in tests/test_auth.py are failing with "
                    "'AttributeError: module has no attribute validate_token'. "
                    "Debug and fix the issue."},
            ],
            # Multi-step: any tool call is a valid first step
            "expect_tools": ["read_file", "run_command", "search_files"],
        },
        {
            "id": "multi_turn_plan",
            "messages": [
                {"role": "user", "content":
                    "Add a --dry-run flag to the deploy script at scripts/deploy.sh "
                    "that prints what would happen without executing."},
            ],
            # May search for file first, then read — both valid
            "expect_tools": ["read_file", "search_files"],
        },
        {
            "id": "error_recovery",
            "messages": [
                {"role": "user", "content": "Fix the build error."},
                {"role": "assistant", "content": None,
                 "tool_calls": [{
                     "id": "call_1", "type": "function",
                     "function": {"name": "run_command",
                                  "arguments": json.dumps({"cmd": "cargo build 2>&1"})},
                 }]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": "error[E0308]: mismatched types\n"
                    "  --> src/parser.rs:42:12\n"
                    "   | expected `Vec<Token>`, found `Option<Vec<Token>>`"},
            ],
            # Given error context, should read the file or search for it
            "expect_tools": ["read_file", "write_file", "search_files"],
        },
    ]

    results = []
    t0 = time.time()
    print(f"\nTool-use bench ({len(tasks)} tasks)...")
    for task in tasks:
        messages = [{"role": "system", "content": TOOL_SYSTEM}] + task["messages"]
        try:
            msg, ntok = chat_complete(base_url, model, messages,
                                      max_tokens=1024, tools=OPENAI_TOOLS)
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", "")

            # Build a readable completion string for dashboard scoring
            if tool_calls:
                completion = _format_tool_calls(tool_calls)
            else:
                completion = content or ""

            # Check if expected tools were called
            called_tools = [tc["function"]["name"] for tc in tool_calls] if tool_calls else []
            expected = task.get("expect_tools", [])
            matched = any(t in called_tools for t in expected) if expected else True

            results.append({
                "id": task["id"],
                "messages": task["messages"],
                "completion": completion,
                "tool_calls": tool_calls,
                "reasoning": reasoning,
                "content": content,
                "tokens": ntok,
                "called_tools": called_tools,
                "expected_tools": expected,
                "tool_match": matched,
            })
            tc_str = ", ".join(called_tools) if called_tools else "(none)"
            match_str = "PASS" if matched else "MISS"
            print(f"  {task['id']}: {ntok} tok  tools=[{tc_str}]  {match_str}")
        except Exception as e:
            results.append({
                "id": task["id"],
                "messages": task["messages"],
                "completion": f"ERROR: {e}",
                "tool_calls": [],
                "tokens": 0,
                "called_tools": [],
                "expected_tools": task.get("expect_tools", []),
                "tool_match": False,
            })
            print(f"  {task['id']}: ERROR {e}")

    # Summary
    matched = sum(1 for r in results if r.get("tool_match"))
    print(f"\nTool-use: {matched}/{len(results)} tasks called expected tools")

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
        description="GPT-OSS Benchmark Suite (vLLM API version)")
    parser.add_argument("--base-url", default="http://localhost:8888",
                        help="vLLM server URL")
    parser.add_argument("--output-dir", default="/home/rmarnold/benchmarks-v2")
    parser.add_argument("--quick", action="store_true",
                        help="Limit HumanEval to 20 samples")
    parser.add_argument("--humaneval-only", action="store_true")
    parser.add_argument("--tool-use-only", action="store_true",
                        help="Run only the tool-use benchmark")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent HumanEval requests (default: 8)")
    args = parser.parse_args()

    # Get model info from server
    model = get_model_name(args.base_url)
    model_short = model.split("/")[-1]
    out = os.path.join(args.output_dir, model_short)
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GPT-OSS Benchmark — vLLM API")
    print(f"Model: {model}")
    print(f"Server: {args.base_url}")
    print(f"Output: {out}")
    print(f"{'='*60}")

    # Save model info
    try:
        health = requests.get(f"{args.base_url}/health", timeout=5)
        server_ok = health.status_code == 200
    except Exception:
        server_ok = False

    info = {
        "model": model,
        "dtype": "mxfp4",
        "server": args.base_url,
        "server_healthy": server_ok,
    }
    with open(os.path.join(out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    limit = 20 if args.quick else None

    if args.tool_use_only:
        bench_tool_use(args.base_url, model, out)
    elif args.humaneval_only:
        bench_humaneval(args.base_url, model, out, limit=limit,
                        concurrency=args.concurrency)
        if not args.skip_eval:
            eval_humaneval(out)
    else:
        bench_coding(args.base_url, model, out)
        bench_tool_use(args.base_url, model, out)
        bench_humaneval(args.base_url, model, out, limit=limit,
                        concurrency=args.concurrency)
        if not args.skip_eval:
            eval_humaneval(out)

    print(f"\n{'='*60}\nAll benchmarks complete -> {out}\n{'='*60}")


if __name__ == "__main__":
    main()
