#!/usr/bin/env python3
"""Generate GRPO task prompts from TUI training data.

Creates a JSONL file of task prompts for GRPO training by extracting
user messages from the tool-calling and agent-trajectory datasets.

The TUI pipeline trains GPT-OSS 20B for tool-calling and agent behavior.
These tasks become the prompts that GRPO generates completions for and
scores using format-based rewards (no execution sandbox required).

Data sources (HF datasets on disk):
  data/coding_tui/tool_calling/train   — tool-calling SFT examples
  data/coding_tui/agent_traj/train     — multi-turn agent trajectory examples

Extraction logic:
  Each example's `text` field is Harmony-formatted. We extract the first
  <|user|> message as the GRPO task description. Messages without a clear
  user turn are skipped.

Mix:
  60% tool-calling tasks, 40% agent-trajectory tasks (interleaved).

Output format (JSONL):
  {"description": "<user message>", "task_type": "tool_calling"|"agent_traj"}

Usage:
    python scripts/generate_tui_grpo_tasks.py
    python scripts/generate_tui_grpo_tasks.py --quick-test
    python scripts/generate_tui_grpo_tasks.py --output data/coding_tui/grpo_tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys


# ---------------------------------------------------------------------------
# Harmony text parsing
# ---------------------------------------------------------------------------

def extract_first_user_message(harmony_text: str) -> str | None:
    """Extract the first user message from a Harmony-formatted text string.

    Searches for the content that appears between a <|user|> token and the
    next role token (or end of string). Returns None if no user turn is found
    or if the extracted content is empty after stripping.

    Args:
        harmony_text: A full Harmony-encoded conversation string.

    Returns:
        Stripped user message string, or None if not found.
    """
    if not harmony_text or "<|user|>" not in harmony_text:
        return None

    # Split on any role-boundary token so we can grab just the user segment.
    # Role boundaries: <|user|>, <|assistant|>, <|developer|>, <|system|>,
    #                  <|tool_call|>, <|tool_result|>, <|thinking|>, <|endoftext|>
    role_token_pattern = re.compile(
        r"<\|(?:user|assistant|developer|system|tool_call|tool_result|thinking|endoftext|end)\|>"
    )

    # Find all token positions
    token_matches = list(role_token_pattern.finditer(harmony_text))

    for i, match in enumerate(token_matches):
        token = match.group()
        if token != "<|user|>":
            continue

        # Content starts after the token (and its optional trailing newline)
        content_start = match.end()
        if content_start < len(harmony_text) and harmony_text[content_start] == "\n":
            content_start += 1

        # Content ends at the start of the next token, or end of string
        if i + 1 < len(token_matches):
            content_end = token_matches[i + 1].start()
        else:
            content_end = len(harmony_text)

        user_content = harmony_text[content_start:content_end].strip()
        if user_content:
            return user_content

    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hf_dataset_texts(dataset_path: str, limit: int | None = None) -> list[str]:
    """Load `text` field from an HF dataset saved to disk.

    Falls back gracefully if the path does not exist or the datasets library
    is unavailable.

    Args:
        dataset_path: Absolute or relative path to the HF dataset directory.
        limit: Maximum number of records to load. None means load all.

    Returns:
        List of text strings.
    """
    if not os.path.exists(dataset_path):
        print(f"  [warn] Dataset path not found, skipping: {dataset_path}")
        return []

    try:
        from datasets import load_from_disk
    except ImportError:
        print("  [warn] `datasets` library not installed. Cannot load HF datasets.")
        return []

    try:
        ds = load_from_disk(dataset_path)
    except Exception as exc:
        print(f"  [warn] Failed to load dataset at {dataset_path}: {exc}")
        return []

    texts: list[str] = []
    for i, example in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = example.get("text", "")
        if text:
            texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_tasks(
    tool_calling_texts: list[str],
    agent_traj_texts: list[str],
    total_target: int,
    tool_calling_fraction: float = 0.60,
) -> list[dict]:
    """Build a mixed task list from tool-calling and agent-trajectory texts.

    The two sources are sampled to hit the target count with the requested
    fraction split. Each entry contains the first user message extracted from
    the Harmony-formatted text.

    Args:
        tool_calling_texts: List of Harmony texts from the tool-calling dataset.
        agent_traj_texts: List of Harmony texts from the agent-traj dataset.
        total_target: Desired number of output tasks.
        tool_calling_fraction: Fraction of tasks from tool-calling data (default 0.60).

    Returns:
        List of task dicts: {"description": str, "task_type": str}.
    """
    n_tool = min(math.ceil(total_target * tool_calling_fraction), len(tool_calling_texts))
    n_traj = min(total_target - n_tool, len(agent_traj_texts))

    # If one source is short, backfill from the other
    if n_tool + n_traj < total_target:
        shortfall = total_target - (n_tool + n_traj)
        extra_tool = min(shortfall, len(tool_calling_texts) - n_tool)
        n_tool += extra_tool
        shortfall -= extra_tool
        extra_traj = min(shortfall, len(agent_traj_texts) - n_traj)
        n_traj += extra_traj

    tasks: list[dict] = []

    for text in tool_calling_texts[:n_tool]:
        desc = extract_first_user_message(text)
        if desc:
            tasks.append({"description": desc, "task_type": "tool_calling"})

    for text in agent_traj_texts[:n_traj]:
        desc = extract_first_user_message(text)
        if desc:
            tasks.append({"description": desc, "task_type": "agent_traj"})

    # Interleave: tool_calling and agent_traj alternate so batches are mixed
    tool_tasks = [t for t in tasks if t["task_type"] == "tool_calling"]
    traj_tasks = [t for t in tasks if t["task_type"] == "agent_traj"]
    interleaved: list[dict] = []
    for i in range(max(len(tool_tasks), len(traj_tasks))):
        if i < len(tool_tasks):
            interleaved.append(tool_tasks[i])
        if i < len(traj_tasks):
            interleaved.append(traj_tasks[i])

    return interleaved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GRPO task prompts from TUI training data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/coding_tui/grpo_tasks.jsonl",
        help="Output JSONL path (default: data/coding_tui/grpo_tasks.jsonl)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=500,
        help="Target number of tasks to generate (default: 500)",
    )
    parser.add_argument(
        "--tool-calling-dir",
        type=str,
        default="data/coding_tui/tool_calling/train",
        help="HF dataset directory for tool-calling data",
    )
    parser.add_argument(
        "--agent-traj-dir",
        type=str,
        default="data/coding_tui/agent_traj/train",
        help="HF dataset directory for agent-trajectory data",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick-test mode: limit output to 50 tasks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_target = 50 if args.quick_test else args.num_tasks
    mode_label = "quick-test" if args.quick_test else "full"

    print(f"\nGenerating TUI GRPO tasks ({mode_label} mode, target={total_target})")
    print(f"  Tool-calling source : {args.tool_calling_dir}")
    print(f"  Agent-traj source   : {args.agent_traj_dir}")
    print(f"  Output              : {args.output}")

    # Determine how many raw records we need to load.
    # We load more than total_target because some records may fail extraction.
    load_limit = min(total_target * 3, 5000)

    print("\nLoading datasets...")
    tool_texts = load_hf_dataset_texts(args.tool_calling_dir, limit=load_limit)
    traj_texts = load_hf_dataset_texts(args.agent_traj_dir, limit=load_limit)

    print(f"  tool_calling records loaded : {len(tool_texts)}")
    print(f"  agent_traj records loaded   : {len(traj_texts)}")

    if not tool_texts and not traj_texts:
        print(
            "\n[error] No data loaded from either source. "
            "Ensure the HF datasets exist at the specified paths and that the "
            "`datasets` library is installed.\n"
            "You can still create placeholder tasks for testing:\n"
            "  echo '{\"description\": \"Test task\", \"task_type\": \"tool_calling\"}' "
            f"> {args.output}"
        )
        sys.exit(1)

    print("\nExtracting user messages...")
    tasks = generate_tasks(
        tool_calling_texts=tool_texts,
        agent_traj_texts=traj_texts,
        total_target=total_target,
    )

    if not tasks:
        print(
            "\n[error] No tasks generated. "
            "Check that the dataset `text` fields contain <|user|> tokens."
        )
        sys.exit(1)

    # Trim to target in case interleaving produced extras
    tasks = tasks[:total_target]

    # Write output
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1

    # Summary
    tool_count = sum(1 for t in tasks if t["task_type"] == "tool_calling")
    traj_count = sum(1 for t in tasks if t["task_type"] == "agent_traj")

    print(f"\nWrote {written} tasks to {output_path}")
    print(f"  tool_calling : {tool_count} ({tool_count / written * 100:.0f}%)")
    print(f"  agent_traj   : {traj_count} ({traj_count / written * 100:.0f}%)")
    print("\nDone.")


if __name__ == "__main__":
    main()
