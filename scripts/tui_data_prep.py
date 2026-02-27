#!/usr/bin/env python3
"""Download and format TUI training datasets from HuggingFace.

Mirrors the data preparation from train_gpt_oss_coding_tui.ipynb.
Skips datasets that already exist at their output paths.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets import load_dataset
from dataset_formatters.harmony import encode_harmony_messages


def prepare_tool_calling(output_dir="data/coding_tui/tool_calling/train", quick_test=False):
    """Download and format tool calling datasets: Glaive v2, xLAM-60k, Hermes v1."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Tool calling data already exists at {output_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    from dataset_formatters.function_calling import (
        format_glaive_function_calling,
        format_hermes_function_calling,
    )

    all_examples = []

    # 1. Glaive v2 (113K)
    print("Downloading glaiveai/glaive-function-calling-v2...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    limit = 500 if quick_test else len(ds)
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = format_glaive_function_calling(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  Glaive: {len(all_examples)} examples")

    # 2. xLAM-60K
    print("Downloading Salesforce/xlam-function-calling-60k...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    limit = 200 if quick_test else len(ds)
    count_before = len(all_examples)
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = _format_xlam(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  xLAM: {len(all_examples) - count_before} examples")

    # 3. Hermes v1
    print("Downloading NousResearch/hermes-function-calling-v1...")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    limit = 300 if quick_test else len(ds)
    count_before = len(all_examples)
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = format_hermes_function_calling(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  Hermes: {len(all_examples) - count_before} examples")

    _save_dataset(all_examples, output_dir)
    print(f"Total tool calling examples: {len(all_examples)}")


def prepare_agent_trajectories(output_dir="data/coding_tui/agent_traj/train", quick_test=False):
    """Download and format agent trajectory datasets: code-act, commitpackft, EditPackFT."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Agent trajectory data already exists at {output_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_examples = []

    CODING_AGENT_DEV_PROMPT = (
        "You are a coding agent. Use tools to read files, write code, run tests, and "
        "complete programming tasks. Do not just analyze - always take action and produce "
        "working code. After making changes, verify they work by running the relevant tests. "
        "If a tool call fails, diagnose and retry with corrected parameters."
    )

    # 1. code-act
    print("Downloading xingyaoww/code-act...")
    for split_name in ("codeact", "general"):
        try:
            ds = load_dataset("xingyaoww/code-act", split=split_name)
            limit = 100 if quick_test else len(ds)
            for ex in ds.select(range(min(limit, len(ds)))):
                formatted = _format_code_act(ex, CODING_AGENT_DEV_PROMPT)
                if formatted:
                    all_examples.append(formatted)
        except Exception as e:
            print(f"  Warning: code-act split '{split_name}' failed: {e}")
    print(f"  code-act: {len(all_examples)} examples")

    # 2. commitpackft (multi-language)
    print("Downloading bigcode/commitpackft...")
    languages = ["python", "javascript", "go", "rust", "java", "typescript"]
    count_before = len(all_examples)
    for lang in languages:
        try:
            ds = load_dataset("bigcode/commitpackft", lang, split="train")
            limit = 50 if quick_test else min(5000, len(ds))
            for ex in ds.select(range(min(limit, len(ds)))):
                formatted = _format_commitpackft(ex)
                if formatted:
                    all_examples.append(formatted)
        except Exception as e:
            print(f"  Warning: commitpackft/{lang} failed: {e}")
    print(f"  commitpackft: {len(all_examples) - count_before} examples")

    # 3. EditPackFT
    print("Downloading nuprl/EditPackFT...")
    count_before = len(all_examples)
    try:
        ds = load_dataset("nuprl/EditPackFT", split="train")
        limit = 300 if quick_test else min(10000, len(ds))
        for ex in ds.select(range(min(limit, len(ds)))):
            formatted = _format_editpackft(ex)
            if formatted:
                all_examples.append(formatted)
    except Exception as e:
        print(f"  Warning: EditPackFT failed: {e}")
    print(f"  EditPackFT: {len(all_examples) - count_before} examples")

    _save_dataset(all_examples, output_dir)
    print(f"Total agent trajectory examples: {len(all_examples)}")


def prepare_preferences(output_dir="data/coding_tui/preference/train", quick_test=False):
    """Download and format preference datasets: hh-rlhf, CodeFeedback."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Preference data already exists at {output_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_examples = []

    # 1. hh-rlhf
    print("Downloading Anthropic/hh-rlhf...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    limit = 300 if quick_test else min(20000, len(ds))
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = _format_hh_rlhf(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  hh-rlhf: {len(all_examples)} examples")

    # 2. CodeFeedback
    print("Downloading m-a-p/CodeFeedback-Filtered-Instruction...")
    count_before = len(all_examples)
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    limit = 200 if quick_test else min(10000, len(ds))
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = _format_code_feedback(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  CodeFeedback: {len(all_examples) - count_before} examples")

    _save_dataset(all_examples, output_dir, columns=["text", "pref_prompt", "pref_chosen", "pref_rejected"])
    print(f"Total preference examples: {len(all_examples)}")


# --- Internal formatters (mirror notebook logic exactly) ---

def _format_xlam(ex):
    """Format xLAM function calling example to Harmony."""
    import json
    try:
        tools = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
        answers = json.loads(ex["answers"]) if isinstance(ex["answers"], str) else ex["answers"]
    except (json.JSONDecodeError, KeyError):
        return None
    if not tools or not answers:
        return None

    tool_desc = "\n".join(
        f"- {t.get('name', 'unknown')}: {t.get('description', '')}"
        for t in tools
    )
    tool_calls = []
    for ans in answers:
        tool_calls.append({
            "name": ans.get("name", "unknown"),
            "arguments": json.dumps(ans.get("arguments", {})),
        })

    messages = [
        {"role": "developer", "content": f"Available tools:\n{tool_desc}"},
        {"role": "user", "content": ex["query"]},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]
    return {"text": encode_harmony_messages(messages)}


def _format_code_act(ex, dev_prompt):
    """Format code-act example to Harmony."""
    convos = ex.get("conversations", [])
    if not convos:
        return None
    messages = [{"role": "developer", "content": dev_prompt}]
    for turn in convos:
        role_raw = turn.get("role") or turn.get("from", "")
        content = turn.get("content") or turn.get("value", "")
        if role_raw in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role_raw in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
        elif role_raw in ("tool", "function", "observation"):
            messages.append({"role": "tool", "content": content})
    if len(messages) < 3:
        return None
    return {"text": encode_harmony_messages(messages, reasoning_effort="high")}


def _format_commitpackft(ex):
    """Format commitpackft example to Harmony."""
    old_code = ex.get("old_contents", "")
    new_code = ex.get("new_contents", "")
    msg = ex.get("message", "") or ex.get("subject", "")
    if not old_code or not new_code or old_code == new_code:
        return None
    diff_len = len(new_code) - len(old_code)
    if abs(diff_len) > 8000:
        return None
    messages = [
        {"role": "user", "content": f"Apply this change: {msg}\n\nCurrent code:\n```\n{old_code}\n```"},
        {"role": "assistant", "content": f"```\n{new_code}\n```"},
    ]
    return {"text": encode_harmony_messages(messages, reasoning_effort="medium")}


def _format_editpackft(ex):
    """Format EditPackFT example to Harmony."""
    instruction = ex.get("instruction", "")
    old_code = ex.get("old_contents") or ex.get("input", "")
    new_code = ex.get("new_contents") or ex.get("output", "")
    if not instruction or not old_code or not new_code or old_code == new_code:
        return None
    if abs(len(new_code) - len(old_code)) > 8000:
        return None
    messages = [
        {"role": "user", "content": f"{instruction}\n\nCurrent code:\n```\n{old_code}\n```"},
        {"role": "assistant", "content": f"```\n{new_code}\n```"},
    ]
    return {"text": encode_harmony_messages(messages, reasoning_effort="medium")}


def _format_hh_rlhf(ex):
    """Format hh-rlhf example to Harmony preference pair."""
    chosen_raw = ex.get("chosen", "")
    rejected_raw = ex.get("rejected", "")
    if not chosen_raw or not rejected_raw:
        return None

    def _parse_hh(text):
        messages = []
        parts = text.split("\n\nHuman: ")
        for part in parts:
            if not part.strip():
                continue
            if "\n\nAssistant: " in part:
                human_part, assistant_part = part.split("\n\nAssistant: ", 1)
                if human_part.strip():
                    messages.append({"role": "user", "content": human_part.strip()})
                if assistant_part.strip():
                    messages.append({"role": "assistant", "content": assistant_part.strip()})
            else:
                messages.append({"role": "user", "content": part.strip()})
        return messages

    chosen_msgs = _parse_hh(chosen_raw)
    rejected_msgs = _parse_hh(rejected_raw)
    if len(chosen_msgs) < 2 or len(rejected_msgs) < 2:
        return None

    prompt_msgs = chosen_msgs[:-1]
    prompt_text = encode_harmony_messages(prompt_msgs)
    chosen_text = encode_harmony_messages(chosen_msgs)
    rejected_text = encode_harmony_messages(rejected_msgs)

    return {
        "text": chosen_text,
        "pref_prompt": prompt_text,
        "pref_chosen": chosen_text,
        "pref_rejected": rejected_text,
    }


def _format_code_feedback(ex):
    """Format CodeFeedback example to Harmony preference pair (synthetic rejected)."""
    query = ex.get("query", "")
    answer = ex.get("answer", "")
    if not query or not answer or len(answer) < 100:
        return None

    messages_chosen = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    # Synthetic rejected: truncate at 30% + stalling response
    truncated = answer[:int(len(answer) * 0.3)]
    messages_rejected = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": truncated + " I would need to analyze this further before proceeding."},
    ]
    prompt_msgs = [{"role": "user", "content": query}]

    prompt_text = encode_harmony_messages(prompt_msgs)
    chosen_text = encode_harmony_messages(messages_chosen)
    rejected_text = encode_harmony_messages(messages_rejected)

    return {
        "text": chosen_text,
        "pref_prompt": prompt_text,
        "pref_chosen": chosen_text,
        "pref_rejected": rejected_text,
    }


def _save_dataset(examples, output_dir, columns=None):
    """Save formatted examples as HuggingFace dataset."""
    import random
    from datasets import Dataset

    random.seed(42)
    random.shuffle(examples)

    if columns:
        # Filter to only include specified columns
        examples = [{k: ex[k] for k in columns if k in ex} for ex in examples]

    ds = Dataset.from_list(examples)
    ds.save_to_disk(output_dir)
    print(f"Saved {len(ds)} examples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and format TUI training data")
    parser.add_argument("--quick-test", action="store_true", help="Download small subset for testing")
    parser.add_argument("--phase", choices=["all", "tool_calling", "agent_traj", "preference"],
                        default="all", help="Which phase data to prepare")
    args = parser.parse_args()

    print("=" * 60)
    print("TUI Training Data Preparation")
    print(f"Mode: {'quick test' if args.quick_test else 'full'}")
    print(f"Phase: {args.phase}")
    print("=" * 60)

    if args.phase in ("all", "tool_calling"):
        prepare_tool_calling(quick_test=args.quick_test)
    if args.phase in ("all", "agent_traj"):
        prepare_agent_trajectories(quick_test=args.quick_test)
    if args.phase in ("all", "preference"):
        prepare_preferences(quick_test=args.quick_test)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
