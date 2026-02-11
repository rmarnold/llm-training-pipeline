"""Generate agent trajectory training data for core_agent SFT.

Creates synthetic multi-turn agent trajectories by:
1. Loading Rust task descriptions (from Strandset, manual, or generated)
2. Simulating an agent session that reads, patches, and tests code
3. Formatting each trajectory in Harmony format for SFT

The generated trajectories teach the model:
- Tool-call format (run_command, read_file, apply_patch, etc.)
- Debugging workflow: read error → hypothesize → patch → test → iterate
- When to stop (tests pass) vs. when to retry

Usage:
    python scripts/15_generate_trajectories.py
    python scripts/15_generate_trajectories.py --config configs/data_sources_rust.yaml
    python scripts/15_generate_trajectories.py --num_tasks 500 --output_dir data/rust/core_agent/train

Requires: pip install -e ".[gpt_oss]"
"""
from __future__ import annotations

import json
import os
import random
import uuid
from typing import Any

from dataset_formatters.harmony import format_harmony_agent


# ==========================================================================
# Trajectory Templates
# ==========================================================================

TRAJECTORY_TEMPLATES = {
    "fix_compilation_error": {
        "developer_prompt": (
            "You are a Rust coding agent. Fix the compilation error. "
            "Use tools to read files, understand the error, and apply a patch."
        ),
        "task_template": (
            "The following Rust code fails to compile:\n\n"
            "```rust\n{buggy_code}\n```\n\n"
            "Error:\n```\n{error_message}\n```\n\n"
            "Fix the compilation error."
        ),
    },
    "fix_failing_tests": {
        "developer_prompt": (
            "You are a Rust coding agent. Fix the failing test. "
            "Use tools to read the test, understand the failure, "
            "and patch the implementation."
        ),
        "task_template": (
            "The test suite is failing:\n\n"
            "```\n{error_message}\n```\n\n"
            "The relevant code is in `{file_path}`.\n"
            "Fix the code so all tests pass."
        ),
    },
    "fix_clippy_warnings": {
        "developer_prompt": (
            "You are a Rust coding agent. Fix clippy warnings to make "
            "the code more idiomatic. Apply minimal changes."
        ),
        "task_template": (
            "Clippy reports the following warnings:\n\n"
            "```\n{error_message}\n```\n\n"
            "Fix the warnings in `{file_path}`."
        ),
    },
    "add_feature_with_tdd": {
        "developer_prompt": (
            "You are a Rust coding agent. Implement the requested feature "
            "using test-driven development: write tests first, then implement."
        ),
        "task_template": (
            "Add the following feature to the codebase:\n\n"
            "{description}\n\n"
            "Write tests first, then implement."
        ),
    },
}


def _make_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create a tool call dict with a unique ID."""
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "arguments": arguments,
    }


def _make_tool_result(call_id: str, output: str) -> dict[str, Any]:
    """Create a tool result dict."""
    return {"id": call_id, "output": output}


# ==========================================================================
# Trajectory Generators
# ==========================================================================

def generate_fix_trajectory(
    buggy_code: str,
    fixed_code: str,
    error_message: str,
    file_path: str = "src/lib.rs",
    trajectory_type: str = "fix_compilation_error",
) -> dict[str, Any]:
    """Generate a trajectory for fixing a code error.

    Simulates the agent:
    1. Reading the file
    2. Analyzing the error
    3. Applying a patch
    4. Running tests to verify

    Args:
        buggy_code: The broken code.
        fixed_code: The corrected code.
        error_message: Compiler/test error message.
        file_path: Path to the file being fixed.
        trajectory_type: Type of trajectory template to use.

    Returns:
        Dict with task, trajectory, developer_prompt keys.
    """
    template = TRAJECTORY_TEMPLATES.get(trajectory_type, TRAJECTORY_TEMPLATES["fix_compilation_error"])

    task = template["task_template"].format(
        buggy_code=buggy_code[:3000],
        error_message=error_message[:2000],
        file_path=file_path,
        description="",
    )

    # Step 1: Read the file
    read_call = _make_tool_call("read_file", {"path": file_path})
    read_result = _make_tool_result(read_call["id"], buggy_code[:4000])

    # Step 2: Apply the fix
    # Generate a simple unified diff
    diff = _generate_simple_diff(buggy_code, fixed_code, file_path)
    patch_call = _make_tool_call("apply_patch", {"diff": diff})
    patch_result = _make_tool_result(patch_call["id"], "Patch applied successfully.")

    # Step 3: Run tests
    test_call = _make_tool_call("run_tests", {"cmd": "cargo test"})
    test_result = _make_tool_result(test_call["id"], "test result: ok. 0 passed; 0 failed; 0 ignored")

    trajectory = [
        {
            "thinking": (
                f"The error is in `{file_path}`. Let me read the file to understand the context."
            ),
            "tool_calls": [read_call],
            "tool_results": [read_result],
        },
        {
            "thinking": (
                f"I can see the issue. The error message says:\n{error_message[:500]}\n\n"
                "I need to apply a fix."
            ),
            "tool_calls": [patch_call],
            "tool_results": [patch_result],
        },
        {
            "thinking": "Patch applied. Let me run the tests to verify the fix.",
            "tool_calls": [test_call],
            "tool_results": [test_result],
            "response": "The fix has been applied and all tests pass.",
        },
    ]

    return {
        "task": task,
        "trajectory": trajectory,
        "developer_prompt": template["developer_prompt"],
    }


def generate_multi_step_trajectory(
    buggy_code: str,
    fixed_code: str,
    error_message: str,
    file_path: str = "src/lib.rs",
) -> dict[str, Any]:
    """Generate a multi-step trajectory with an initial failed attempt.

    Teaches the model to handle partial failures and iterate.
    """
    template = TRAJECTORY_TEMPLATES["fix_compilation_error"]

    task = template["task_template"].format(
        buggy_code=buggy_code[:3000],
        error_message=error_message[:2000],
        file_path=file_path,
        description="",
    )

    # Step 1: Read file
    read_call = _make_tool_call("read_file", {"path": file_path})
    read_result = _make_tool_result(read_call["id"], buggy_code[:4000])

    # Step 2: First (wrong) attempt — teach retry behavior
    wrong_patch = _generate_simple_diff(buggy_code, buggy_code, file_path)
    wrong_call = _make_tool_call("apply_patch", {"diff": wrong_patch})
    wrong_result = _make_tool_result(wrong_call["id"], "Patch applied successfully.")

    # Step 3: Tests still fail
    test_fail_call = _make_tool_call("run_tests", {"cmd": "cargo test"})
    test_fail_result = _make_tool_result(
        test_fail_call["id"],
        f"test result: FAILED.\n{error_message[:1000]}",
    )

    # Step 4: Re-read and apply correct fix
    read2_call = _make_tool_call("read_file", {"path": file_path})
    read2_result = _make_tool_result(read2_call["id"], buggy_code[:4000])

    diff = _generate_simple_diff(buggy_code, fixed_code, file_path)
    fix_call = _make_tool_call("apply_patch", {"diff": diff})
    fix_result = _make_tool_result(fix_call["id"], "Patch applied successfully.")

    # Step 5: Tests pass
    test_pass_call = _make_tool_call("run_tests", {"cmd": "cargo test"})
    test_pass_result = _make_tool_result(
        test_pass_call["id"],
        "test result: ok. 0 passed; 0 failed; 0 ignored",
    )

    trajectory = [
        {
            "thinking": f"Let me read `{file_path}` to understand the issue.",
            "tool_calls": [read_call],
            "tool_results": [read_result],
        },
        {
            "thinking": "I think I see the problem. Let me try a fix.",
            "tool_calls": [wrong_call],
            "tool_results": [wrong_result],
        },
        {
            "thinking": "Let me verify by running tests.",
            "tool_calls": [test_fail_call],
            "tool_results": [test_fail_result],
        },
        {
            "thinking": (
                "The tests still fail. My first attempt was wrong. "
                "Let me re-read the code and try a different approach."
            ),
            "tool_calls": [read2_call],
            "tool_results": [read2_result],
        },
        {
            "thinking": (
                "I see — the real issue is different from what I initially thought. "
                "Let me apply the correct fix."
            ),
            "tool_calls": [fix_call],
            "tool_results": [fix_result],
        },
        {
            "thinking": "Let me verify the fix passes all tests.",
            "tool_calls": [test_pass_call],
            "tool_results": [test_pass_result],
            "response": "Fixed! The correct approach was to address the root cause. All tests pass now.",
        },
    ]

    return {
        "task": task,
        "trajectory": trajectory,
        "developer_prompt": template["developer_prompt"],
    }


def _generate_simple_diff(old_code: str, new_code: str, file_path: str) -> str:
    """Generate a simplified unified diff between old and new code."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    try:
        import difflib
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}")
        return "".join(diff)
    except Exception:
        return f"--- a/{file_path}\n+++ b/{file_path}\n"


# ==========================================================================
# Dataset Generation
# ==========================================================================

def generate_trajectories_from_mutations(
    mutations_path: str,
    output_dir: str,
    max_samples: int = 5000,
    multi_step_ratio: float = 0.3,
) -> int:
    """Generate trajectory data from cargo-mutants output.

    Args:
        mutations_path: Path to mutations JSONL file (from 16_generate_mutations.py).
        output_dir: Output directory for HF dataset.
        max_samples: Maximum number of trajectories to generate.
        multi_step_ratio: Fraction of trajectories with multi-step retries.

    Returns:
        Number of trajectories generated.
    """
    if not os.path.exists(mutations_path):
        print(f"  Mutations file not found: {mutations_path}")
        return 0

    mutations = []
    with open(mutations_path) as f:
        for line in f:
            try:
                mutations.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not mutations:
        print("  No mutations found.")
        return 0

    print(f"  Loaded {len(mutations)} mutations")

    trajectories = []
    random.shuffle(mutations)

    for i, m in enumerate(mutations[:max_samples]):
        buggy = m.get("buggy_code", "")
        fixed = m.get("fixed_code", "")
        error = m.get("error_message", "")
        file_path = m.get("file_path", "src/lib.rs")

        if not all([buggy, fixed, error]):
            continue

        # Determine trajectory type
        if "compiler error" in error.lower() or "error[E" in error:
            ttype = "fix_compilation_error"
        elif "test" in error.lower() or "FAILED" in error:
            ttype = "fix_failing_tests"
        elif "clippy" in error.lower() or "warning" in error.lower():
            ttype = "fix_clippy_warnings"
        else:
            ttype = "fix_compilation_error"

        # Generate multi-step or single-step
        if random.random() < multi_step_ratio:
            traj = generate_multi_step_trajectory(buggy, fixed, error, file_path)
        else:
            traj = generate_fix_trajectory(buggy, fixed, error, file_path, ttype)

        # Format as Harmony
        formatted = format_harmony_agent(traj)
        if formatted.get("text"):
            trajectories.append({"text": formatted["text"]})

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{min(len(mutations), max_samples)} trajectories")

    if not trajectories:
        print("  No valid trajectories generated.")
        return 0

    # Save as HF dataset
    from datasets import Dataset
    dataset = Dataset.from_list(trajectories)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"  Saved {len(trajectories)} trajectories to {output_dir}")

    return len(trajectories)


def generate_trajectories_from_strandset(
    dataset_name: str = "Fortytwo-Network/Strandset-Rust-v1",
    output_dir: str = "data/rust/core_agent/strandset_trajectories",
    max_samples: int = 5000,
) -> int:
    """Convert Strandset Rust examples into agent trajectories.

    Takes code completion examples and wraps them as agent sessions.

    Args:
        dataset_name: HuggingFace dataset name.
        output_dir: Output directory.
        max_samples: Maximum samples.

    Returns:
        Number of trajectories generated.
    """
    from datasets import load_dataset, Dataset

    print(f"  Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")

    trajectories = []
    for i, example in enumerate(ds):
        if i >= max_samples:
            break

        prompt = example.get("prompt", "") or example.get("instruction", "")
        solution = example.get("completion", "") or example.get("solution", "") or example.get("output", "")

        if not prompt or not solution:
            continue

        # Wrap as a simple agent trajectory: user asks → agent writes code
        write_call = _make_tool_call("apply_patch", {"diff": f"+{solution[:2000]}"})
        write_result = _make_tool_result(write_call["id"], "Patch applied successfully.")

        test_call = _make_tool_call("run_tests", {"cmd": "cargo test"})
        test_result = _make_tool_result(test_call["id"], "test result: ok.")

        traj = {
            "task": prompt,
            "trajectory": [
                {
                    "thinking": "I'll implement this step by step.",
                    "tool_calls": [write_call],
                    "tool_results": [write_result],
                },
                {
                    "thinking": "Let me verify the implementation passes tests.",
                    "tool_calls": [test_call],
                    "tool_results": [test_result],
                    "response": "Implementation complete and tests pass.",
                },
            ],
            "developer_prompt": (
                "You are a Rust coding agent. Implement the requested feature."
            ),
        }

        formatted = format_harmony_agent(traj)
        if formatted.get("text"):
            trajectories.append({"text": formatted["text"]})

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} examples")

    if not trajectories:
        print("  No valid trajectories generated.")
        return 0

    dataset = Dataset.from_list(trajectories)
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"  Saved {len(trajectories)} trajectories to {output_dir}")

    return len(trajectories)


# ==========================================================================
# CLI
# ==========================================================================

def main(
    output_dir: str = "data/rust/core_agent/train",
    mutations_path: str | None = None,
    max_samples: int = 5000,
    include_strandset: bool = True,
    multi_step_ratio: float = 0.3,
) -> None:
    """Generate all trajectory training data.

    Args:
        output_dir: Base output directory.
        mutations_path: Path to mutations JSONL (from 16_generate_mutations.py).
        max_samples: Max trajectories per source.
        include_strandset: Whether to include Strandset-derived trajectories.
        multi_step_ratio: Fraction with multi-step retry behavior.
    """
    print(f"\n{'='*60}")
    print("Generating Agent Trajectory Training Data")
    print(f"{'='*60}")

    total = 0

    # Generate from mutations (if available)
    if mutations_path is None:
        mutations_path = "data/rust/mutations/mutations.jsonl"

    if os.path.exists(mutations_path):
        print(f"\n[1] Generating from mutations: {mutations_path}")
        mutation_output = os.path.join(output_dir, "from_mutations")
        count = generate_trajectories_from_mutations(
            mutations_path, mutation_output, max_samples, multi_step_ratio,
        )
        total += count
    else:
        print(f"\n[1] Mutations file not found: {mutations_path} (skipping)")
        print("    Run 16_generate_mutations.py first to generate mutation data.")

    # Generate from Strandset
    if include_strandset:
        print(f"\n[2] Generating from Strandset...")
        strandset_output = os.path.join(output_dir, "from_strandset")
        count = generate_trajectories_from_strandset(
            output_dir=strandset_output,
            max_samples=max_samples,
        )
        total += count
    else:
        print("\n[2] Strandset generation skipped.")

    print(f"\n{'='*60}")
    print(f"Trajectory generation complete!")
    print(f"  Total trajectories: {total:,}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate agent trajectory training data")
    parser.add_argument("--config", type=str, default="configs/data_sources_rust.yaml")
    parser.add_argument("--output_dir", type=str, default="data/rust/core_agent/train")
    parser.add_argument("--mutations_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--no-strandset", action="store_true", help="Skip Strandset generation")
    parser.add_argument("--multi_step_ratio", type=float, default=0.3)
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        mutations_path=args.mutations_path,
        max_samples=args.max_samples,
        include_strandset=not args.no_strandset,
        multi_step_ratio=args.multi_step_ratio,
    )
