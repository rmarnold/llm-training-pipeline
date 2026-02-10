"""CPU fallback deduplication using datasketch with streaming."""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dedup.common import DATASKETCH_AVAILABLE


def cpu_dedup_streaming(
    input_path: str,
    output_path: str,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    memory_info: dict,
    show_progress: bool,
) -> str:
    """CPU fallback using datasketch with streaming.

    For when GPU is not available.
    """
    if not DATASKETCH_AVAILABLE:
        raise ImportError("datasketch not installed. Run: pip install datasketch")

    from datasketch import MinHash, MinHashLSH
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"CPU Deduplication (streaming mode)")
        print(f"  Input: {input_p}")
        print(f"  Output: {output_p}")

    # Get files
    if input_p.is_file():
        parquet_files = [input_p]
    else:
        parquet_files = sorted(input_p.glob("*.parquet"))

    n_docs = sum(pq.read_metadata(pf).num_rows for pf in parquet_files)

    if show_progress:
        print(f"  Files: {len(parquet_files):,}, Documents: {n_docs:,}")

    # Adaptive num_perm
    num_perm = 64 if n_docs > 10_000_000 else 128
    if memory_info['ram_free_gb'] < 16:
        num_perm = 64

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)

    if show_progress:
        print(f"  MinHash permutations: {num_perm}")
        print("  Pass 1: Building LSH index...")

    # Build index
    duplicate_ids = set()

    for pf in tqdm(parquet_files, desc="  Building index", disable=not show_progress):
        try:
            df = pd.read_parquet(pf)

            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            for _, row in df.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                doc_id = str(row[id_column])

                m = MinHash(num_perm=num_perm)
                for word in text.split()[:1000]:  # Limit for speed
                    m.update(word.encode('utf8'))

                if lsh.query(m):
                    duplicate_ids.add(doc_id)
                else:
                    lsh.insert(doc_id, m)

            del df
            gc.collect()
        except Exception as e:
            if show_progress:
                print(f"    Warning: {pf.name}: {e}")

    if show_progress:
        print(f"  Found {len(duplicate_ids):,} duplicates")
        print("  Pass 2: Writing output...")

    del lsh
    gc.collect()

    # Write filtered output
    writer = None
    output_file = output_p / "deduplicated.parquet"
    n_kept = 0

    for pf in tqdm(parquet_files, desc="  Writing output", disable=not show_progress):
        try:
            df = pd.read_parquet(pf)

            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            mask = ~df[id_column].isin(duplicate_ids)
            filtered_df = df[mask]

            if len(filtered_df) > 0:
                table = pa.Table.from_pandas(filtered_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_file), table.schema)
                writer.write_table(table)
                n_kept += len(filtered_df)

            del df, filtered_df
            gc.collect()
        except Exception:
            pass

    if writer:
        writer.close()

    if show_progress:
        print(f"  Kept: {n_kept:,} documents")

    return str(output_p)
