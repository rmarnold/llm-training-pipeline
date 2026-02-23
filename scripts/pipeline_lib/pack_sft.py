"""Pre-pack SFT dataset by tokenizing, concatenating, and chunking into fixed-length sequences.

Eliminates padding waste for datasets where mean token length << seq_len.
Follows the same greedy concatenation approach as scripts/03_tokenize_and_pack.py.

Examples already end with <|endoftext|> from Harmony format, so add_special_tokens=False.

Usage (standalone):
    python -m pipeline_lib.pack_sft \
        --dataset_path data/coding_tui/agent_traj/train \
        --tokenizer_name openai/gpt-oss-20b \
        --seq_len 8192

Usage (from notebook/script):
    from pipeline_lib.pack_sft import pack_sft_dataset

    result = pack_sft_dataset(
        dataset_path="data/coding_tui/agent_traj/train",
        tokenizer=tokenizer,
        seq_len=8192,
    )
"""
from __future__ import annotations

from pathlib import Path


def pack_sft_dataset(
    dataset_path: str,
    tokenizer: object,
    seq_len: int = 8192,
    output_path: str | None = None,
    text_column: str = "text",
    val_fraction: float = 0.01,
    val_max: int = 500,
    seed: int = 42,
) -> dict:
    """Tokenize, concatenate, and chunk an SFT dataset into fixed-length sequences.

    Args:
        dataset_path: Path to HuggingFace dataset on disk (must have ``text_column``).
        tokenizer: A HuggingFace tokenizer (already loaded).
        seq_len: Target sequence length for packed chunks.
        output_path: Where to save the packed HF Dataset.  Defaults to
            ``<dataset_path>_packed_<seq_len>``.
        text_column: Column containing text data.
        val_fraction: Fraction of packed sequences to hold out for validation.
        val_max: Maximum number of validation sequences.
        seed: Random seed for train/val split.

    Returns:
        Dict with packing statistics: paths, counts, compression ratio, etc.
    """
    import numpy as np
    from datasets import Dataset, load_from_disk

    # ── Resolve output path ──────────────────────────────────────────
    if output_path is None:
        output_path = f"{dataset_path}_packed_{seq_len}"

    train_path = f"{output_path}/train"
    val_path = f"{output_path}/val"

    # ── Load source dataset ──────────────────────────────────────────
    print(f"\n  Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    total_examples = len(dataset)
    print(f"  Source examples: {total_examples:,}")

    # ── Tokenize ─────────────────────────────────────────────────────
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    all_token_ids: list[list[int]] = []
    total_tokens = 0
    skipped = 0
    truncated = 0

    print(f"  Tokenizing (add_special_tokens=False, seq_len={seq_len})...")
    for example in dataset:
        text = example.get(text_column, "")
        if not text:
            skipped += 1
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            skipped += 1
            continue

        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
            truncated += 1

        all_token_ids.append(tokens)
        total_tokens += len(tokens)

    if not all_token_ids:
        print("  ERROR: No valid examples found after tokenization.")
        return {"error": "no_data"}

    # ── Greedy concatenation ─────────────────────────────────────────
    print(f"  Packing {len(all_token_ids):,} tokenized examples into {seq_len}-token chunks...")

    packed_input_ids: list[list[int]] = []
    packed_attention_mask: list[list[int]] = []
    packed_labels: list[list[int]] = []

    current_chunk: list[int] = []

    for tokens in all_token_ids:
        if len(current_chunk) + len(tokens) <= seq_len:
            # Fits in current chunk
            current_chunk.extend(tokens)
        else:
            # Flush current chunk (pad to seq_len)
            if current_chunk:
                _flush_chunk(current_chunk, seq_len, pad_token_id,
                             packed_input_ids, packed_attention_mask, packed_labels)
            # Start new chunk with this example
            current_chunk = list(tokens)

    # Flush final chunk
    if current_chunk:
        _flush_chunk(current_chunk, seq_len, pad_token_id,
                     packed_input_ids, packed_attention_mask, packed_labels)

    num_packed = len(packed_input_ids)

    # ── Create HF Dataset ────────────────────────────────────────────
    full_dataset = Dataset.from_dict({
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
        "labels": packed_labels,
    })

    # ── Train/val split ──────────────────────────────────────────────
    val_size = max(1, min(val_max, int(num_packed * val_fraction)))
    if num_packed <= val_size + 1:
        val_size = 0

    if val_size > 0:
        splits = full_dataset.train_test_split(test_size=val_size, seed=seed)
        splits["train"].save_to_disk(train_path)
        splits["test"].save_to_disk(val_path)
        train_count = len(splits["train"])
        val_count = len(splits["test"])
    else:
        full_dataset.save_to_disk(train_path)
        train_count = num_packed
        val_count = 0

    # ── Compute stats ────────────────────────────────────────────────
    mean_tokens = total_tokens / len(all_token_ids) if all_token_ids else 0
    ideal_packed = total_tokens / seq_len if seq_len else 0
    compression_ratio = total_examples / num_packed if num_packed else 0
    utilization = total_tokens / (num_packed * seq_len) * 100 if num_packed else 0

    result = {
        "output_path": output_path,
        "train_path": train_path,
        "val_path": val_path,
        "source_examples": total_examples,
        "packed_sequences": num_packed,
        "train_count": train_count,
        "val_count": val_count,
        "compression_ratio": round(compression_ratio, 1),
        "utilization_pct": round(utilization, 1),
        "mean_tokens": int(mean_tokens),
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "skipped": skipped,
        "truncated": truncated,
    }

    # ── Print report ─────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  PACKING REPORT")
    print(f"  {'='*55}")
    print(f"  Source examples:    {total_examples:>10,}")
    print(f"  Packed sequences:   {num_packed:>10,}")
    print(f"  Compression ratio:  {compression_ratio:>10.1f}x")
    print(f"  {'─'*55}")
    print(f"  Mean tokens/example:{mean_tokens:>10,.0f}")
    print(f"  Target seq_len:     {seq_len:>10,}")
    print(f"  Utilization:        {utilization:>9.1f}%")
    print(f"  Ideal packed seqs:  {ideal_packed:>10,.0f}")
    print(f"  {'─'*55}")
    print(f"  Skipped (empty):    {skipped:>10,}")
    print(f"  Truncated (>seq_len):{truncated:>9,}")
    print(f"  {'─'*55}")
    print(f"  Train sequences:    {train_count:>10,}")
    print(f"  Val sequences:      {val_count:>10,}")
    print(f"  Saved to:           {output_path}")
    print(f"  {'='*55}\n")

    return result


def _flush_chunk(
    current_chunk: list[int],
    seq_len: int,
    pad_token_id: int,
    packed_input_ids: list[list[int]],
    packed_attention_mask: list[list[int]],
    packed_labels: list[list[int]],
) -> None:
    """Pad a chunk to seq_len and append to output lists."""
    pad_len = seq_len - len(current_chunk)
    input_ids = current_chunk + [pad_token_id] * pad_len
    attention_mask = [1] * len(current_chunk) + [0] * pad_len
    labels = list(current_chunk) + [-100] * pad_len

    packed_input_ids.append(input_ids)
    packed_attention_mask.append(attention_mask)
    packed_labels.append(labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-pack an SFT dataset into fixed-length tokenized sequences",
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to HF dataset on disk")
    parser.add_argument("--tokenizer_name", type=str, default="openai/gpt-oss-20b",
                        help="Tokenizer to use for encoding")
    parser.add_argument("--seq_len", type=int, default=8192,
                        help="Target sequence length (default: 8192)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path (default: <dataset_path>_packed_<seq_len>)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing text (default: text)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    pack_sft_dataset(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        output_path=args.output_path,
        text_column=args.text_column,
    )
