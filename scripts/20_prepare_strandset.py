"""Download, parse, and format Strandset-Rust-v1 for Harmony training.

Downloads the Strandset-Rust-v1 dataset from HuggingFace, parses JSON fields,
maps task categories to Harmony formatters, and saves HF Datasets for each
training stage.

Output structure:
    data/rust/strandset/
      lang_rust/train/    # ~130K completion examples (HF Dataset with "text")
      core_agent/train/   # ~60K agent/debug examples (HF Dataset with "text")
      ipo/train/          # ~20K preference pairs (prompt, chosen, rejected)
      eval/test/          # 225 examples from Strandset test split
      stats.json          # Conversion statistics

Task category → target mapping:
    Lang adapter (format_harmony_completion):
        code_generation, code_completion, docstring_generation,
        comment_generation, code_summarization, variable_naming,
        function_naming

    Core agent — debug (format_harmony_debug):
        bug_detection

    Core agent — review (format_harmony_completion):
        code_review, code_refactoring, optimization, test_generation

    IPO preferences (format_harmony_preference):
        bug_detection subset (fixed=chosen, buggy=rejected)

Usage:
    python scripts/20_prepare_strandset.py
    python scripts/20_prepare_strandset.py --output_dir data/rust/strandset
    python scripts/20_prepare_strandset.py --max_samples 100

Requires: pip install -e ".[gpt_oss]"
"""
from __future__ import annotations

import json
import os
from typing import Any

from dataset_formatters.harmony import (
    format_harmony_completion,
    format_harmony_debug,
    format_harmony_preference,
)


# ======================================================================
# Category → target mapping
# ======================================================================

LANG_CATEGORIES = {
    "code_generation",
    "code_completion",
    "docstring_generation",
    "comment_generation",
    "code_summarization",
    "variable_naming",
    "function_naming",
}

DEBUG_CATEGORIES = {
    "bug_detection",
}

REVIEW_CATEGORIES = {
    "code_review",
    "code_refactoring",
    "optimization",
    "test_generation",
}


# ======================================================================
# JSON field parsing
# ======================================================================

def _parse_json_field(raw: Any) -> dict[str, Any]:
    """Parse input_data / output_data which may be a JSON string or dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except (json.JSONDecodeError, TypeError):
            return {"value": raw}
    return {}


def _extract_code(data: dict[str, Any]) -> str:
    """Extract the primary code field from parsed JSON data."""
    for key in ("code", "source_code", "original_code", "buggy_code",
                "function", "snippet", "content"):
        if key in data and data[key]:
            return str(data[key])
    return ""


def _extract_context(data: dict[str, Any]) -> str:
    """Extract context/instructions from parsed JSON data."""
    for key in ("code_context", "context", "imports", "dependencies",
                "instructions", "task_description", "description"):
        if key in data and data[key]:
            return str(data[key])
    return ""


def _extract_output(data: dict[str, Any]) -> str:
    """Extract the primary output field from parsed JSON data."""
    for key in ("generated_code", "completed_code", "output", "solution",
                "commented_code", "documented_code", "docstring",
                "summary", "name", "suggested_name", "refactored_code",
                "optimized_code", "review", "test_code", "tests",
                "fixed_code", "fix", "value"):
        if key in data and data[key]:
            return str(data[key])
    # Fall back to first non-empty string value
    for v in data.values():
        if isinstance(v, str) and len(v.strip()) > 10:
            return v
    return ""


def _extract_error(data: dict[str, Any]) -> str:
    """Extract error/bug description from parsed JSON data."""
    for key in ("error_message", "compiler_error", "error", "bug_description",
                "bug_type", "issue", "warning"):
        if key in data and data[key]:
            return str(data[key])
    return ""


# ======================================================================
# Per-category converters
# ======================================================================

def _convert_lang_example(
    task_category: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    crate_name: str,
) -> dict[str, str]:
    """Convert a lang-adapter category example to format_harmony_completion input."""
    code = _extract_code(input_data)
    context = _extract_context(input_data)
    output = _extract_output(output_data)

    if not output:
        return {"text": ""}

    # Build a natural prompt based on the task category
    prompts = {
        "code_generation": f"Generate Rust code for the following:\n\n{context or code}" if (context or code) else "",
        "code_completion": f"Complete the following Rust code:\n\n```rust\n{code}\n```" if code else "",
        "docstring_generation": f"Generate documentation for this Rust code:\n\n```rust\n{code}\n```" if code else "",
        "comment_generation": f"Add comments to this Rust code:\n\n```rust\n{code}\n```" if code else "",
        "code_summarization": f"Summarize what this Rust code does:\n\n```rust\n{code}\n```" if code else "",
        "variable_naming": f"Suggest a good variable name for this Rust code:\n\n```rust\n{code}\n```" if code else "",
        "function_naming": f"Suggest a good function name for this Rust code:\n\n```rust\n{code}\n```" if code else "",
    }

    prompt = prompts.get(task_category, "")
    if not prompt:
        # Generic fallback
        prompt = code or context
    if not prompt:
        return {"text": ""}

    if crate_name:
        prompt = f"[crate: {crate_name}] {prompt}"

    return format_harmony_completion({"prompt": prompt, "completion": output})


def _convert_agent_debug_example(
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    crate_name: str,
) -> dict[str, str]:
    """Convert a bug_detection example to format_harmony_debug input."""
    buggy_code = _extract_code(input_data)
    error_msg = _extract_error(input_data) or _extract_error(output_data)
    fixed_code = _extract_output(output_data)

    if not buggy_code or not fixed_code:
        return {"text": ""}

    # If no explicit error, synthesize one from the output description
    if not error_msg:
        error_msg = "Bug detected in this code. See the fix below."

    explanation = f"[crate: {crate_name}]" if crate_name else ""

    return format_harmony_debug({
        "buggy_code": buggy_code,
        "error_message": error_msg,
        "fixed_code": fixed_code,
        "explanation": explanation,
    })


def _convert_agent_review_example(
    task_category: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    crate_name: str,
) -> dict[str, str]:
    """Convert review/refactor/optimization/test examples to completion format."""
    code = _extract_code(input_data)
    context = _extract_context(input_data)
    output = _extract_output(output_data)

    if not output:
        return {"text": ""}

    prompts = {
        "code_review": f"Review this Rust code and suggest improvements:\n\n```rust\n{code}\n```",
        "code_refactoring": f"Refactor this Rust code for better readability and idioms:\n\n```rust\n{code}\n```",
        "optimization": f"Optimize this Rust code for performance:\n\n```rust\n{code}\n```",
        "test_generation": f"Write tests for this Rust code:\n\n```rust\n{code}\n```",
    }

    prompt = prompts.get(task_category, f"Improve this Rust code:\n\n```rust\n{code}\n```")
    if not code:
        prompt = context or ""
    if not prompt:
        return {"text": ""}

    if crate_name:
        prompt = f"[crate: {crate_name}] {prompt}"

    return format_harmony_completion({"prompt": prompt, "completion": output})


def _convert_to_preference_pair(
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    crate_name: str,
) -> dict[str, str]:
    """Convert bug_detection example to IPO preference pair.

    The fixed code is the chosen response; the buggy code is rejected.
    """
    buggy_code = _extract_code(input_data)
    fixed_code = _extract_output(output_data)
    error_msg = _extract_error(input_data) or _extract_error(output_data)

    if not buggy_code or not fixed_code:
        return {"text": ""}

    prompt_text = "Fix the bug in this Rust code"
    if error_msg:
        prompt_text += f":\n\nError: {error_msg}"
    if crate_name:
        prompt_text = f"[crate: {crate_name}] {prompt_text}"
    prompt_text += f"\n\n```rust\n{buggy_code}\n```"

    return format_harmony_preference({
        "prompt": prompt_text,
        "chosen": f"```rust\n{fixed_code}\n```",
        "rejected": f"```rust\n{buggy_code}\n```",
    })


# ======================================================================
# Main pipeline
# ======================================================================

def prepare_strandset(
    dataset_name: str = "Fortytwo-Network/Strandset-Rust-v1",
    output_dir: str = "data/rust/strandset",
    max_samples: int = 0,
    include_preferences: bool = True,
) -> str:
    """Download, parse, and format Strandset-Rust-v1.

    Args:
        dataset_name: HuggingFace dataset identifier.
        output_dir: Root output directory.
        max_samples: Max examples to process (0 = all).
        include_preferences: Whether to generate IPO preference pairs.

    Returns:
        Path to output directory.
    """
    from datasets import Dataset, load_dataset

    print(f"\n{'='*60}")
    print("Preparing Strandset-Rust-v1")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Output: {output_dir}")
    if max_samples:
        print(f"  Max samples: {max_samples}")

    # Load dataset
    print(f"\nLoading dataset...")
    ds_train = load_dataset(dataset_name, split="train")
    print(f"  Train split: {len(ds_train):,} examples")

    try:
        ds_test = load_dataset(dataset_name, split="test")
        print(f"  Test split: {len(ds_test):,} examples")
    except Exception:
        ds_test = None
        print("  Test split: not available")

    # Process train split
    lang_examples: list[dict[str, str]] = []
    agent_examples: list[dict[str, str]] = []
    ipo_examples: list[dict[str, str]] = []

    stats: dict[str, Any] = {
        "total_processed": 0,
        "lang_rust": 0,
        "core_agent_debug": 0,
        "core_agent_review": 0,
        "ipo": 0,
        "skipped_empty": 0,
        "skipped_parse_error": 0,
    }
    category_counts: dict[str, int] = {}

    n = len(ds_train) if not max_samples else min(max_samples, len(ds_train))

    print(f"\nProcessing {n:,} examples...")
    for i in range(n):
        example = ds_train[i]
        stats["total_processed"] += 1

        task_category = example.get("task_category", "")
        crate_name = example.get("crate_name", "")
        category_counts[task_category] = category_counts.get(task_category, 0) + 1

        # Parse JSON fields
        input_data = _parse_json_field(example.get("input_data", ""))
        output_data = _parse_json_field(example.get("output_data", ""))

        if not input_data and not output_data:
            stats["skipped_parse_error"] += 1
            continue

        # Route to appropriate converter
        if task_category in LANG_CATEGORIES:
            result = _convert_lang_example(task_category, input_data, output_data, crate_name)
            if result.get("text"):
                lang_examples.append(result)
                stats["lang_rust"] += 1
            else:
                stats["skipped_empty"] += 1

        elif task_category in DEBUG_CATEGORIES:
            # Debug examples go to core_agent
            result = _convert_agent_debug_example(input_data, output_data, crate_name)
            if result.get("text"):
                agent_examples.append(result)
                stats["core_agent_debug"] += 1
            else:
                stats["skipped_empty"] += 1

            # Also generate preference pairs for IPO
            if include_preferences:
                pref = _convert_to_preference_pair(input_data, output_data, crate_name)
                if pref.get("text"):
                    ipo_examples.append(pref)
                    stats["ipo"] += 1

        elif task_category in REVIEW_CATEGORIES:
            result = _convert_agent_review_example(task_category, input_data, output_data, crate_name)
            if result.get("text"):
                agent_examples.append(result)
                stats["core_agent_review"] += 1
            else:
                stats["skipped_empty"] += 1

        else:
            # Unknown category — try as lang completion
            result = _convert_lang_example(task_category, input_data, output_data, crate_name)
            if result.get("text"):
                lang_examples.append(result)
                stats["lang_rust"] += 1
            else:
                stats["skipped_empty"] += 1

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,}/{n:,} "
                  f"(lang={stats['lang_rust']:,}, agent={stats['core_agent_debug'] + stats['core_agent_review']:,}, "
                  f"ipo={stats['ipo']:,})")

    # Save datasets
    print(f"\nSaving datasets...")

    # Lang adapter data
    lang_path = os.path.join(output_dir, "lang_rust", "train")
    if lang_examples:
        os.makedirs(lang_path, exist_ok=True)
        lang_ds = Dataset.from_list(lang_examples)
        lang_ds.save_to_disk(lang_path)
        print(f"  lang_rust/train: {len(lang_ds):,} examples -> {lang_path}")

    # Core agent data
    agent_path = os.path.join(output_dir, "core_agent", "train")
    if agent_examples:
        os.makedirs(agent_path, exist_ok=True)
        agent_ds = Dataset.from_list(agent_examples)
        agent_ds.save_to_disk(agent_path)
        print(f"  core_agent/train: {len(agent_ds):,} examples -> {agent_path}")

    # IPO preference pairs
    ipo_path = os.path.join(output_dir, "ipo", "train")
    if ipo_examples:
        os.makedirs(ipo_path, exist_ok=True)
        ipo_ds = Dataset.from_list(ipo_examples)
        ipo_ds.save_to_disk(ipo_path)
        print(f"  ipo/train: {len(ipo_ds):,} preference pairs -> {ipo_path}")

    # Eval data (from test split)
    eval_path = os.path.join(output_dir, "eval", "test")
    if ds_test is not None and len(ds_test) > 0:
        eval_examples = []
        for i in range(len(ds_test)):
            example = ds_test[i]
            input_data = _parse_json_field(example.get("input_data", ""))  # type: ignore[union-attr]
            output_data = _parse_json_field(example.get("output_data", ""))  # type: ignore[union-attr]
            task_category = example.get("task_category", "")  # type: ignore[union-attr]
            crate_name = example.get("crate_name", "")  # type: ignore[union-attr]

            # Format all test examples as completions for eval
            result = _convert_lang_example(task_category, input_data, output_data, crate_name)
            if result.get("text"):
                eval_examples.append(result)

        if eval_examples:
            os.makedirs(eval_path, exist_ok=True)
            eval_ds = Dataset.from_list(eval_examples)
            eval_ds.save_to_disk(eval_path)
            print(f"  eval/test: {len(eval_ds):,} examples -> {eval_path}")

    # Save stats
    stats["category_counts"] = category_counts
    stats_path = os.path.join(output_dir, "stats.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("Strandset preparation complete!")
    print(f"{'='*60}")
    print(f"  Total processed: {stats['total_processed']:,}")
    print(f"  Lang adapter:    {stats['lang_rust']:,}")
    print(f"  Core agent:      {stats['core_agent_debug'] + stats['core_agent_review']:,} "
          f"(debug={stats['core_agent_debug']:,}, review={stats['core_agent_review']:,})")
    print(f"  IPO pairs:       {stats['ipo']:,}")
    print(f"  Skipped (empty): {stats['skipped_empty']:,}")
    print(f"  Skipped (parse): {stats['skipped_parse_error']:,}")
    print(f"\n  Category breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count:,}")
    print(f"\n  Stats: {stats_path}")
    print(f"{'='*60}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and format Strandset-Rust-v1 for Harmony training"
    )
    parser.add_argument("--dataset", type=str, default="Fortytwo-Network/Strandset-Rust-v1")
    parser.add_argument("--output_dir", type=str, default="data/rust/strandset")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max examples to process (0 = all)")
    parser.add_argument("--no-preferences", dest="include_preferences", action="store_false",
                        help="Skip generating IPO preference pairs")
    args = parser.parse_args()

    prepare_strandset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        include_preferences=args.include_preferences,
    )
