"""GPU deduplication implementation - fast three-phase pipeline."""
from __future__ import annotations

import gc
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from dedup.common import CUDA_AVAILABLE
from dedup.minhash import fast_shingle_hash, compute_minhash_signatures_gpu
from dedup.lsh import build_lsh_index, find_duplicates_lsh


def gpu_dedup_fast(
    input_path: str,
    output_path: str,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    char_ngrams: int,
    num_perm: int,
    num_bands: int,
    memory_info: dict,
    show_progress: bool,
) -> str:
    """Fast GPU deduplication without datasketch overhead.

    Three phases:
    1. Extract shingles (CPU, parallelized with xxhash)
    2. Compute MinHash signatures (GPU, batched)
    3. LSH deduplication (CPU, vectorized numpy)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    # Get files
    if input_p.is_file():
        parquet_files = [input_p]
    else:
        parquet_files = sorted(input_p.glob("*.parquet"))

    # Count docs
    n_docs = sum(pq.read_metadata(pf).num_rows for pf in parquet_files)

    if show_progress:
        print(f"  Input: {input_p}")
        print(f"  Output: {output_p}")
        print(f"  Files: {len(parquet_files):,}, Documents: {n_docs:,}")
        print(f"  Threshold: {similarity_threshold}, N-grams: {char_ngrams}")
        print(f"  MinHash: {num_perm} permutations, {num_bands} bands")

    # Determine batch sizes based on memory
    # Signatures: n_docs * num_perm * 8 bytes
    sig_memory_gb = n_docs * num_perm * 8 / (1024**3)
    available_ram = memory_info['ram_free_gb'] * 0.6

    if sig_memory_gb > available_ram:
        # Need to process in chunks
        chunk_size = int(available_ram * (1024**3) / (num_perm * 8))
        if show_progress:
            print(f"  WARNING: Large dataset, processing in chunks of {chunk_size:,}")

    # Adaptive num_perm based on RAM
    if memory_info['ram_free_gb'] < 32:
        num_perm = min(num_perm, 128)
        num_bands = min(num_bands, 16)
    if memory_info['ram_free_gb'] < 16:
        num_perm = min(num_perm, 64)
        num_bands = min(num_bands, 8)

    if show_progress:
        print(f"  Adjusted: {num_perm} permutations, {num_bands} bands")

    # Determine CPU workers
    n_workers = min(os.cpu_count() or 4, 16)

    device = 'cuda' if CUDA_AVAILABLE else 'cpu'

    # PHASE 1: Load all documents and extract shingles
    if show_progress:
        print(f"\n  Phase 1: Extracting shingles ({n_workers} CPU workers)...")

    all_texts = []
    all_ids = []

    for pf in tqdm(parquet_files, desc="  Loading files", disable=not show_progress):
        try:
            df = pd.read_parquet(pf)

            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            texts = df[text_column].fillna('').tolist()
            ids = df[id_column].tolist()

            all_texts.extend(texts)
            all_ids.extend(ids)

            del df
        except Exception as e:
            if show_progress:
                print(f"    Warning: {pf.name}: {e}")

    if show_progress:
        print(f"  Loaded {len(all_texts):,} documents")

    # Extract shingles in parallel
    def extract_shingles(text):
        return fast_shingle_hash(text, char_ngrams)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        all_shingles = list(tqdm(
            executor.map(extract_shingles, all_texts),
            total=len(all_texts),
            desc="  Extracting shingles",
            disable=not show_progress
        ))

    # Free texts to save memory
    del all_texts
    gc.collect()

    if show_progress:
        print(f"  Extracted shingles for {len(all_shingles):,} documents")

    # PHASE 2: Compute MinHash signatures on GPU
    if show_progress:
        print(f"\n  Phase 2: Computing MinHash signatures on {device}...")

    signatures = compute_minhash_signatures_gpu(
        all_shingles,
        num_perm=num_perm,
        device=device,
        batch_size=50000,  # Process 50K docs per GPU kernel
    )

    # Free shingles
    del all_shingles
    gc.collect()

    if show_progress:
        print(f"  Computed signatures: {signatures.shape}")

    # PHASE 3: LSH deduplication
    if show_progress:
        print(f"\n  Phase 3: LSH deduplication ({num_bands} bands)...")

    # Build LSH index
    hash_tables = build_lsh_index(signatures, num_bands)

    if show_progress:
        total_buckets = sum(len(t) for t in hash_tables)
        print(f"  Built LSH index: {total_buckets:,} buckets across {num_bands} bands")

    # Find duplicates
    duplicate_indices = find_duplicates_lsh(
        hash_tables,
        signatures,
        similarity_threshold,
    )

    if show_progress:
        print(f"  Found {len(duplicate_indices):,} duplicates to remove")

    # Free signatures and LSH
    del signatures, hash_tables
    gc.collect()

    # PHASE 4: Write deduplicated output
    if show_progress:
        print(f"\n  Phase 4: Writing deduplicated output...")

    # Create set of IDs to keep
    keep_indices = set(range(len(all_ids))) - duplicate_indices
    keep_ids = {all_ids[i] for i in keep_indices}

    # Stream through files and write filtered output
    writer = None
    output_file = output_p / "deduplicated.parquet"
    n_kept = 0

    for pf in tqdm(parquet_files, desc="  Writing output", disable=not show_progress):
        try:
            df = pd.read_parquet(pf)

            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            # Filter to kept IDs
            mask = df[id_column].isin(keep_ids)
            filtered_df = df[mask]

            if len(filtered_df) > 0:
                table = pa.Table.from_pandas(filtered_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_file), table.schema)
                writer.write_table(table)
                n_kept += len(filtered_df)

            del df, filtered_df
        except Exception as e:
            if show_progress:
                print(f"    Warning: {pf.name}: {e}")

    if writer:
        writer.close()

    if show_progress:
        n_removed = len(all_ids) - n_kept
        pct_removed = 100 * n_removed / max(1, len(all_ids))
        print(f"\n  Results:")
        print(f"    Input: {len(all_ids):,} documents")
        print(f"    Removed: {n_removed:,} duplicates ({pct_removed:.1f}%)")
        print(f"    Output: {n_kept:,} unique documents")

    return str(output_p)
