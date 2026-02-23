"""Profile dataset token length distributions for optimal seq_len selection.

Samples a dataset, tokenizes, and reports percentile statistics so you can
set max_seq_length to cover the actual data rather than guessing.

Usage:
    from pipeline_lib.data_profiler import profile_seq_lengths

    recommended = profile_seq_lengths(
        dataset_path="data/coding_tui/agent_traj/train",
        tokenizer_name="openai/gpt-oss-20b",
        sample_size=5000,
    )
    # Returns recommended seq_len (P99 rounded up to nearest power of 2)
"""
from __future__ import annotations

import math


def profile_seq_lengths(
    dataset_path: str,
    tokenizer_name: str | None = None,
    tokenizer: object | None = None,
    text_column: str = "text",
    sample_size: int = 5000,
    target_percentile: float = 99.0,
) -> dict:
    """Profile token length distribution and recommend optimal seq_len.

    Args:
        dataset_path: Path to HF dataset on disk.
        tokenizer_name: HuggingFace tokenizer name (loaded if tokenizer not provided).
        tokenizer: Pre-loaded tokenizer (avoids reloading).
        text_column: Column name containing text data.
        sample_size: Number of examples to sample (0 = all).
        target_percentile: Percentile to cover (default 99).

    Returns:
        Dict with distribution stats and recommended seq_len.
    """
    import numpy as np
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    total = len(dataset)

    if sample_size > 0 and total > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        sampled = sample_size
    else:
        sampled = total

    # Load tokenizer if not provided
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Tokenize and measure lengths
    lengths = []
    for example in dataset:
        text = example.get(text_column, "")
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    if not lengths:
        print(f"  WARNING: No valid text found in column '{text_column}'")
        return {"recommended_seq_len": 8192, "error": "no_data"}

    lengths = np.array(lengths)

    # Compute percentiles
    p50 = int(np.percentile(lengths, 50))
    p75 = int(np.percentile(lengths, 75))
    p90 = int(np.percentile(lengths, 90))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    p100 = int(np.max(lengths))
    mean = int(np.mean(lengths))

    # Recommend: P99 rounded up to nearest power of 2 (standard seq_len values)
    raw_target = int(np.percentile(lengths, target_percentile))
    recommended = _next_power_of_2(raw_target)
    # Cap at 65536 (practical GPU limit)
    recommended = min(recommended, 65536)

    # Count how many examples would be truncated at various seq_lens
    truncation = {}
    for sl in [2048, 4096, 8192, 16384, 32768, 65536]:
        truncated = int(np.sum(lengths > sl))
        truncation[sl] = truncated

    result = {
        "total_examples": total,
        "sampled": sampled,
        "mean": mean,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": p100,
        "recommended_seq_len": recommended,
        "target_percentile": target_percentile,
        "truncation_at": truncation,
    }

    # Print report
    print(f"\n  {'─' * 50}")
    print(f"  TOKEN LENGTH PROFILE ({sampled:,} of {total:,} examples)")
    print(f"  {'─' * 50}")
    print(f"  Mean:    {mean:>7,} tokens")
    print(f"  P50:     {p50:>7,}")
    print(f"  P75:     {p75:>7,}")
    print(f"  P90:     {p90:>7,}")
    print(f"  P95:     {p95:>7,}")
    print(f"  P99:     {p99:>7,}")
    print(f"  Max:     {p100:>7,}")
    print(f"  {'─' * 50}")
    print(f"  Truncation at common seq_lens:")
    for sl, count in sorted(truncation.items()):
        pct = 100 * count / sampled
        bar = "#" * int(pct / 2)
        print(f"    {sl:>6}: {count:>5} truncated ({pct:>5.1f}%) {bar}")
    print(f"  {'─' * 50}")
    print(f"  Recommended seq_len: {recommended} (covers P{target_percentile:.0f}={raw_target})")
    print(f"  {'─' * 50}\n")

    return result


def _next_power_of_2(n: int) -> int:
    """Round up to the nearest power of 2 (min 1024)."""
    if n <= 1024:
        return 1024
    return 2 ** math.ceil(math.log2(n))
