"""GPU-accelerated deduplication using pure PyTorch/CUDA.

Auto-detects GPU memory (VRAM) and system RAM, then processes data in
appropriately-sized chunks to avoid OOM while maximizing GPU utilization.

Falls back to NeMo Curator if available, or CPU datasketch as last resort.
"""
from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Check for torch/CUDA
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Check for NeMo Curator availability (as fallback)
NEMO_AVAILABLE = False
NEMO_API_VERSION = None
NEMO_IMPORT_ERROR = None

def _try_nemo_import():
    """Try to import NeMo Curator with multiple API paths."""
    global NEMO_AVAILABLE, NEMO_API_VERSION, NEMO_IMPORT_ERROR

    # Try 1: NeMo Curator 1.0.0 / 25.x workflow API
    try:
        from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
        import dask.dataframe as dd
        NEMO_AVAILABLE = True
        NEMO_API_VERSION = "workflow"
        return
    except ImportError as e:
        NEMO_IMPORT_ERROR = str(e)

    # Try 2: modules API (some older versions)
    try:
        from nemo_curator.modules import FuzzyDuplicates
        from nemo_curator.datasets import DocumentDataset
        import dask.dataframe as dd
        NEMO_AVAILABLE = True
        NEMO_API_VERSION = "modules"
        return
    except ImportError as e:
        NEMO_IMPORT_ERROR = str(e)

    # Try 3: Legacy direct import (NeMo Curator < 25.x)
    try:
        from nemo_curator import FuzzyDuplicates
        from nemo_curator.datasets import DocumentDataset
        import dask.dataframe as dd
        NEMO_AVAILABLE = True
        NEMO_API_VERSION = "legacy"
        return
    except ImportError as e:
        NEMO_IMPORT_ERROR = str(e)

# Run import check
_try_nemo_import()

# Check for dask-cuda
try:
    from dask_cuda import LocalCUDACluster
    DASK_CUDA_AVAILABLE = True
except ImportError:
    DASK_CUDA_AVAILABLE = False

# Fallback to datasketch
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False


def get_memory_info() -> dict:
    """Get available GPU and system memory.

    Returns:
        Dict with 'gpu_total_gb', 'gpu_free_gb', 'ram_total_gb', 'ram_free_gb'
    """
    info = {
        'gpu_total_gb': 0,
        'gpu_free_gb': 0,
        'ram_total_gb': 0,
        'ram_free_gb': 0,
        'gpu_name': None,
    }

    # GPU memory
    if CUDA_AVAILABLE:
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_name'] = gpu_props.name
            info['gpu_total_gb'] = gpu_props.total_memory / (1024**3)

            # Get free memory
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info['gpu_free_gb'] = free_mem / (1024**3)
        except Exception:
            pass

    # System RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        info['ram_total_gb'] = vm.total / (1024**3)
        info['ram_free_gb'] = vm.available / (1024**3)
    except ImportError:
        # Fallback: try to read from /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        info['ram_total_gb'] = int(line.split()[1]) / (1024**2)
                    elif 'MemAvailable' in line:
                        info['ram_free_gb'] = int(line.split()[1]) / (1024**2)
        except Exception:
            info['ram_total_gb'] = 16  # Assume 16GB if unknown
            info['ram_free_gb'] = 8

    return info


def compute_optimal_batch_size(
    n_docs: int,
    avg_doc_len: int = 500,
    num_perm: int = 128,
    memory_info: Optional[dict] = None,
) -> Tuple[int, int, int, str]:
    """Compute optimal batch sizes for GPU and CPU processing.

    Args:
        n_docs: Total number of documents
        avg_doc_len: Average document length in characters
        num_perm: Number of MinHash permutations
        memory_info: Pre-computed memory info (optional)

    Returns:
        Tuple of (gpu_batch_size, cpu_batch_size, n_cpu_workers, device)
    """
    if memory_info is None:
        memory_info = get_memory_info()

    # Memory estimates per document (approximate)
    # - Text storage: ~avg_doc_len bytes
    # - Shingles (5-grams): ~avg_doc_len * 4 bytes (as int32 hashes)
    # - MinHash signature: num_perm * 4 bytes (int32)
    # - Working memory: ~2x buffer
    bytes_per_doc_text = avg_doc_len
    bytes_per_doc_shingles = avg_doc_len * 4  # Rough estimate
    bytes_per_doc_minhash = num_perm * 4
    bytes_per_doc_total = (bytes_per_doc_text + bytes_per_doc_shingles + bytes_per_doc_minhash) * 2

    # GPU batch size (use 70% of free VRAM to leave headroom)
    if memory_info['gpu_free_gb'] > 2:
        gpu_mem_bytes = memory_info['gpu_free_gb'] * 0.7 * (1024**3)
        gpu_batch_size = int(gpu_mem_bytes / bytes_per_doc_total)
        gpu_batch_size = max(10_000, min(gpu_batch_size, 500_000))
        device = 'cuda'
    else:
        gpu_batch_size = 0
        device = 'cpu'

    # CPU batch size (use 50% of free RAM for LSH index building)
    ram_mem_bytes = memory_info['ram_free_gb'] * 0.5 * (1024**3)
    cpu_batch_size = int(ram_mem_bytes / bytes_per_doc_total)
    cpu_batch_size = max(10_000, min(cpu_batch_size, 1_000_000))

    # CPU workers for parallel shingle extraction
    # Use more workers if RAM is abundant, fewer if limited
    import os
    n_cpus = os.cpu_count() or 4
    if memory_info['ram_free_gb'] > 32:
        n_cpu_workers = min(n_cpus, 8)
    elif memory_info['ram_free_gb'] > 16:
        n_cpu_workers = min(n_cpus, 4)
    else:
        n_cpu_workers = min(n_cpus, 2)

    return gpu_batch_size, cpu_batch_size, n_cpu_workers, device


def is_gpu_dedup_available() -> bool:
    """Check if GPU deduplication is available."""
    return CUDA_AVAILABLE and TORCH_AVAILABLE


def gpu_fuzzy_dedup(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    id_column: str = "id",
    similarity_threshold: float = 0.8,
    char_ngrams: int = 5,
    num_buckets: int = 20,
    hashes_per_bucket: int = 13,
    cache_path: Optional[str] = None,
    use_gpu: Optional[bool] = None,
    n_workers: int = 1,
    show_progress: bool = True,
) -> str:
    """GPU-accelerated fuzzy deduplication with automatic memory management.

    Automatically detects available GPU memory and system RAM, then processes
    data in appropriately-sized chunks to avoid OOM.

    Args:
        input_path: Path to input parquet file(s) or directory
        output_path: Output path for deduplicated data
        text_column: Name of text column
        id_column: Name of ID column (will be created if missing)
        similarity_threshold: Jaccard similarity threshold (0.8 = 80% similar = duplicate)
        char_ngrams: Character n-gram size for shingling
        num_buckets: Number of LSH buckets (more = higher recall, slower)
        hashes_per_bucket: MinHash signatures per bucket
        cache_path: Cache directory for intermediate results
        use_gpu: Force GPU (True) or CPU (False). None = auto-detect.
        n_workers: Number of GPU workers (for multi-GPU)
        show_progress: Show progress information

    Returns:
        Path to deduplicated output
    """
    if use_gpu is None:
        use_gpu = is_gpu_dedup_available()

    # Get memory info for adaptive sizing
    memory_info = get_memory_info()

    if show_progress:
        print(f"GPU Deduplication - Memory-Adaptive Mode")
        print(f"  GPU: {memory_info['gpu_name'] or 'Not available'}")
        if memory_info['gpu_free_gb'] > 0:
            print(f"  VRAM: {memory_info['gpu_free_gb']:.1f} GB free / {memory_info['gpu_total_gb']:.1f} GB total")
        print(f"  RAM: {memory_info['ram_free_gb']:.1f} GB free / {memory_info['ram_total_gb']:.1f} GB total")

    # Use our PyTorch-based GPU implementation
    if use_gpu and CUDA_AVAILABLE:
        return _gpu_dedup_pytorch(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            char_ngrams=char_ngrams,
            num_perm=num_buckets * hashes_per_bucket,
            memory_info=memory_info,
            show_progress=show_progress,
        )
    else:
        # Fall back to CPU datasketch with streaming
        return _cpu_dedup_datasketch(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            memory_info=memory_info,
            show_progress=show_progress,
        )


def _extract_shingles_single(args):
    """Extract shingles from a single document (for multiprocessing)."""
    idx, text, char_ngrams = args
    if not text or len(text) < char_ngrams:
        return (idx, None)

    # Extract character n-grams and hash them
    shingle_hashes = np.array([
        hash(text[j:j + char_ngrams]) & 0xFFFFFFFF
        for j in range(len(text) - char_ngrams + 1)
    ], dtype=np.int64)

    if len(shingle_hashes) > 0:
        return (idx, shingle_hashes)
    return (idx, None)


def _gpu_minhash_batch(
    texts: list[str],
    char_ngrams: int = 5,
    num_perm: int = 128,
    device: str = 'cuda',
    micro_batch_size: int = 1000,
    n_cpu_workers: int = 4,
) -> np.ndarray:
    """Compute MinHash signatures for a batch of texts using GPU.

    Optimized for throughput by:
    1. Parallel CPU preprocessing for shingle extraction
    2. Batching GPU operations for multiple documents
    3. Using vectorized operations where possible

    Args:
        texts: List of text documents
        char_ngrams: Size of character n-grams
        num_perm: Number of MinHash permutations
        device: 'cuda' or 'cpu'
        micro_batch_size: Number of documents to process per GPU kernel
        n_cpu_workers: Number of CPU workers for parallel shingle extraction

    Returns:
        numpy array of shape (len(texts), num_perm) with MinHash signatures
    """
    if not texts:
        return np.array([], dtype=np.uint32).reshape(0, num_perm)

    n_texts = len(texts)
    PRIME = 2**61 - 1
    MAX_HASH = np.uint32(2**32 - 1)

    # Generate random hash coefficients (same across all docs for consistency)
    torch.manual_seed(42)
    a = torch.randint(1, PRIME, (num_perm,), dtype=torch.int64, device=device)
    b = torch.randint(0, PRIME, (num_perm,), dtype=torch.int64, device=device)

    # Pre-allocate output array
    all_signatures = np.full((n_texts, num_perm), MAX_HASH, dtype=np.uint32)

    # STEP 1: Pre-hash all shingles on CPU (parallelized)
    doc_shingle_hashes = [None] * n_texts
    doc_indices_with_shingles = []

    # Use multiprocessing for large batches
    if n_texts > 5000 and n_cpu_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        # Prepare arguments
        args_list = [(i, texts[i], char_ngrams) for i in range(n_texts)]

        # Use ThreadPoolExecutor for smaller overhead (GIL released during hash())
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_cpu_workers) as executor:
            results = list(executor.map(_extract_shingles_single, args_list))

        for idx, shingles in results:
            if shingles is not None:
                doc_shingle_hashes[idx] = shingles
                doc_indices_with_shingles.append(idx)
    else:
        # Sequential processing for small batches
        for i, text in enumerate(texts):
            if not text or len(text) < char_ngrams:
                continue

            shingle_hashes = np.array([
                hash(text[j:j + char_ngrams]) & 0xFFFFFFFF
                for j in range(len(text) - char_ngrams + 1)
            ], dtype=np.int64)

            if len(shingle_hashes) > 0:
                doc_shingle_hashes[i] = shingle_hashes
                doc_indices_with_shingles.append(i)

    # STEP 2: Process documents in micro-batches on GPU
    # This maximizes GPU utilization by processing multiple docs at once
    for batch_start in range(0, len(doc_indices_with_shingles), micro_batch_size):
        batch_end = min(batch_start + micro_batch_size, len(doc_indices_with_shingles))
        batch_indices = doc_indices_with_shingles[batch_start:batch_end]

        # Collect shingles for this micro-batch
        batch_shingles = []
        batch_doc_boundaries = [0]  # Start index of each doc's shingles

        for idx in batch_indices:
            shingles = doc_shingle_hashes[idx]
            if shingles is not None:
                batch_shingles.extend(shingles.tolist())
            batch_doc_boundaries.append(len(batch_shingles))

        if not batch_shingles:
            continue

        # Move all shingles to GPU at once
        shingles_tensor = torch.tensor(batch_shingles, dtype=torch.int64, device=device)

        # Compute all hash values at once: (num_perm, num_all_shingles)
        hash_values = (a.unsqueeze(1) * shingles_tensor.unsqueeze(0) + b.unsqueeze(1)) % PRIME % MAX_HASH

        # Extract min for each document using the boundaries
        hash_values_cpu = hash_values.cpu().numpy().astype(np.uint32)

        for j, idx in enumerate(batch_indices):
            start = batch_doc_boundaries[j]
            end = batch_doc_boundaries[j + 1]
            if end > start:
                # Min over shingles for this document
                all_signatures[idx] = hash_values_cpu[:, start:end].min(axis=1)

        # Free GPU memory
        del shingles_tensor, hash_values, hash_values_cpu

    return all_signatures


def _gpu_dedup_pytorch(
    input_path: str,
    output_path: str,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    char_ngrams: int,
    num_perm: int,
    memory_info: dict,
    show_progress: bool,
) -> str:
    """GPU deduplication using pure PyTorch with adaptive memory management.

    Two-pass approach:
    1. Compute MinHash signatures on GPU, build LSH index on CPU
    2. Query LSH to find duplicates, filter and write output
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"  Input: {input_p}")
        print(f"  Output: {output_p}")
        print(f"  Threshold: {similarity_threshold}")
        print(f"  N-grams: {char_ngrams}, Permutations: {num_perm}")

    # Get list of files to process
    if input_p.is_file():
        parquet_files = [input_p]
    else:
        parquet_files = sorted(input_p.glob("*.parquet"))

    # Count total docs
    n_docs = 0
    for pf in parquet_files:
        try:
            n_docs += pq.read_metadata(pf).num_rows
        except Exception:
            pass

    if show_progress:
        print(f"  Files: {len(parquet_files):,}")
        print(f"  Documents: {n_docs:,}")

    # Compute optimal batch sizes
    gpu_batch, cpu_batch, n_cpu_workers, device = compute_optimal_batch_size(
        n_docs=n_docs,
        num_perm=num_perm,
        memory_info=memory_info,
    )

    if show_progress:
        print(f"  Device: {device}")
        print(f"  GPU batch size: {gpu_batch:,}")
        print(f"  CPU batch size: {cpu_batch:,}")
        print(f"  CPU workers: {n_cpu_workers}")

    # Initialize LSH index
    # Adjust number of permutations based on memory
    actual_num_perm = min(num_perm, 128 if memory_info['ram_free_gb'] > 16 else 64)

    if not DATASKETCH_AVAILABLE:
        raise ImportError("datasketch not installed. Run: pip install datasketch")

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=actual_num_perm)

    if show_progress:
        print(f"  LSH permutations: {actual_num_perm}")
        print("  Pass 1: Computing MinHash signatures and building LSH index...")

    # PASS 1: Compute MinHash on GPU, build LSH index
    n_processed = 0
    n_unique = 0
    duplicate_ids = set()

    # Process files in batches
    current_batch_texts = []
    current_batch_ids = []

    file_iterator = parquet_files
    if show_progress:
        file_iterator = tqdm(parquet_files, desc="  Building index", unit="file")

    for pf in file_iterator:
        try:
            df = pd.read_parquet(pf)

            # Add ID column if missing
            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            texts = df[text_column].fillna('').tolist()
            ids = df[id_column].tolist()

            del df
            gc.collect()

            # Add to current batch
            current_batch_texts.extend(texts)
            current_batch_ids.extend(ids)

            # Process batch when it reaches gpu_batch size
            while len(current_batch_texts) >= gpu_batch:
                batch_texts = current_batch_texts[:gpu_batch]
                batch_ids = current_batch_ids[:gpu_batch]
                current_batch_texts = current_batch_texts[gpu_batch:]
                current_batch_ids = current_batch_ids[gpu_batch:]

                # Compute MinHash on GPU
                signatures = _gpu_minhash_batch(
                    batch_texts,
                    char_ngrams=char_ngrams,
                    num_perm=actual_num_perm,
                    device=device,
                    n_cpu_workers=n_cpu_workers,
                )

                # Add to LSH index (on CPU)
                for i, (doc_id, sig) in enumerate(zip(batch_ids, signatures)):
                    m = MinHash(num_perm=actual_num_perm, hashfunc=lambda x: x)
                    m.hashvalues = sig

                    if lsh.query(m):
                        duplicate_ids.add(doc_id)
                    else:
                        lsh.insert(doc_id, m)
                        n_unique += 1

                    n_processed += 1

                del signatures, batch_texts, batch_ids
                gc.collect()
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()

            del texts, ids
            gc.collect()

        except Exception as e:
            if show_progress:
                print(f"    Warning: Error processing {pf.name}: {e}")

    # Process remaining batch
    if current_batch_texts:
        signatures = _gpu_minhash_batch(
            current_batch_texts,
            char_ngrams=char_ngrams,
            num_perm=actual_num_perm,
            device=device,
            n_cpu_workers=n_cpu_workers,
        )

        for i, (doc_id, sig) in enumerate(zip(current_batch_ids, signatures)):
            m = MinHash(num_perm=actual_num_perm, hashfunc=lambda x: x)
            m.hashvalues = sig

            if lsh.query(m):
                duplicate_ids.add(doc_id)
            else:
                lsh.insert(doc_id, m)
                n_unique += 1

            n_processed += 1

        del signatures, current_batch_texts, current_batch_ids
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    if show_progress:
        print(f"  Index built: {n_unique:,} unique, {len(duplicate_ids):,} duplicates")
        print("  Pass 2: Filtering and writing results...")

    # Clear LSH to free memory
    del lsh
    gc.collect()

    # PASS 2: Filter duplicates and write output
    writer = None
    output_file = output_p / "deduplicated.parquet"
    n_kept = 0

    file_iterator = parquet_files
    if show_progress:
        file_iterator = tqdm(parquet_files, desc="  Writing output", unit="file")

    for pf in file_iterator:
        try:
            df = pd.read_parquet(pf)

            # Add ID column if missing (same logic as pass 1)
            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            # Filter out duplicates
            keep_mask = ~df[id_column].isin(duplicate_ids)
            filtered_df = df[keep_mask]

            if len(filtered_df) > 0:
                table = pa.Table.from_pandas(filtered_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_file), table.schema)
                writer.write_table(table)
                n_kept += len(filtered_df)
                del table

            del df, filtered_df, keep_mask
            gc.collect()

        except Exception as e:
            if show_progress:
                print(f"    Warning: Error writing {pf.name}: {e}")

    if writer:
        writer.close()

    if show_progress:
        n_removed = n_docs - n_kept
        pct_removed = 100 * n_removed / max(1, n_docs)
        print(f"  Removed: {n_removed:,} duplicates ({pct_removed:.1f}%)")
        print(f"  Kept: {n_kept:,} unique documents")

    return str(output_p)


def _cpu_dedup_datasketch(
    input_path: str,
    output_path: str,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    memory_info: Optional[dict] = None,
    show_progress: bool = True,
) -> str:
    """CPU fallback using datasketch MinHashLSH with streaming mode.

    Uses adaptive batch sizes based on available RAM.
    """
    if not DATASKETCH_AVAILABLE:
        raise ImportError("datasketch not installed. Run: pip install datasketch")

    import pyarrow as pa
    import pyarrow.parquet as pq

    if memory_info is None:
        memory_info = get_memory_info()

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"CPU Deduplication (datasketch) - Streaming Mode")
        print(f"  RAM: {memory_info['ram_free_gb']:.1f} GB free")
        print(f"  Input: {input_p}")
        print(f"  Output: {output_p}")
        print(f"  Threshold: {similarity_threshold}")

    # Get list of files to process
    if input_p.is_file():
        parquet_files = [input_p]
    else:
        parquet_files = sorted(input_p.glob("*.parquet"))

    # Count total docs
    n_docs = 0
    for pf in parquet_files:
        try:
            n_docs += pq.read_metadata(pf).num_rows
        except Exception:
            pass

    if show_progress:
        print(f"  Files: {len(parquet_files):,}")
        print(f"  Documents: {n_docs:,}")

    # Adaptive settings based on RAM
    # Use fewer permutations for very large datasets to reduce memory
    if n_docs > 20_000_000:
        num_perm = 64
    elif n_docs > 10_000_000:
        num_perm = 96
    else:
        num_perm = 128

    # Reduce further if RAM is limited
    if memory_info['ram_free_gb'] < 16:
        num_perm = min(num_perm, 64)

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)

    if show_progress:
        print(f"  MinHash permutations: {num_perm}")
        print("  Pass 1: Building LSH index (streaming)...")

    # PASS 1: Build LSH index
    n_processed = 0
    n_unique = 0
    duplicate_ids = set()

    file_iterator = parquet_files
    if show_progress:
        file_iterator = tqdm(parquet_files, desc="  Building index", unit="file")

    for pf in file_iterator:
        try:
            df = pd.read_parquet(pf)

            # Add ID column if missing
            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            # Process each document
            for idx, row in df.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                doc_id = str(row[id_column])

                # Compute MinHash
                m = MinHash(num_perm=num_perm)
                words = text.split()

                # Use character 3-grams for short texts, words for longer
                if len(words) < 20:
                    for i in range(len(text) - 2):
                        m.update(text[i:i+3].encode('utf8'))
                else:
                    for word in words:
                        m.update(word.encode('utf8'))

                # Check for duplicates
                if lsh.query(m):
                    duplicate_ids.add(doc_id)
                else:
                    lsh.insert(doc_id, m)
                    n_unique += 1

                n_processed += 1

            del df
            gc.collect()

        except Exception as e:
            if show_progress:
                print(f"    Warning: Error processing {pf.name}: {e}")

    if show_progress:
        print(f"  Index built: {n_unique:,} unique, {len(duplicate_ids):,} duplicates")
        print("  Pass 2: Filtering and writing results (streaming)...")

    # Clear LSH to free memory
    del lsh
    gc.collect()

    # PASS 2: Filter and write
    writer = None
    output_file = output_p / "deduplicated.parquet"
    n_kept = 0

    file_iterator = parquet_files
    if show_progress:
        file_iterator = tqdm(parquet_files, desc="  Writing output", unit="file")

    for pf in file_iterator:
        try:
            df = pd.read_parquet(pf)

            if id_column not in df.columns:
                df[id_column] = [f"{pf.stem}_{i}" for i in range(len(df))]

            keep_mask = ~df[id_column].isin(duplicate_ids)
            filtered_df = df[keep_mask]

            if len(filtered_df) > 0:
                table = pa.Table.from_pandas(filtered_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_file), table.schema)
                writer.write_table(table)
                n_kept += len(filtered_df)
                del table

            del df, filtered_df, keep_mask
            gc.collect()

        except Exception as e:
            if show_progress:
                print(f"    Warning: Error writing {pf.name}: {e}")

    if writer:
        writer.close()

    if show_progress:
        n_removed = n_docs - n_kept
        pct_removed = 100 * n_removed / max(1, n_docs)
        print(f"  Removed: {n_removed:,} duplicates ({pct_removed:.1f}%)")
        print(f"  Kept: {n_kept:,} unique documents")

    return str(output_p)


def benchmark_dedup(n_samples: int = 10_000) -> dict:
    """Benchmark GPU vs CPU deduplication.

    Args:
        n_samples: Number of test samples

    Returns:
        Dict with timing results
    """
    import time
    import tempfile

    # Generate test data with some duplicates
    texts = []
    for i in range(n_samples):
        if i % 10 == 0 and i > 0:
            # 10% duplicates with minor variations
            texts.append(texts[i - 10] + " extra")
        else:
            texts.append(f"This is unique document number {i} with some content.")

    df = pd.DataFrame({
        'id': [str(i) for i in range(n_samples)],
        'text': texts
    })

    results = {'n_samples': n_samples}
    memory_info = get_memory_info()
    results['memory_info'] = memory_info

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.parquet"
        df.to_parquet(input_path)

        # CPU benchmark
        cpu_output = Path(tmpdir) / "cpu_output"
        start = time.time()
        _cpu_dedup_datasketch(
            str(input_path), str(cpu_output),
            text_column='text', id_column='id',
            similarity_threshold=0.8,
            memory_info=memory_info,
            show_progress=False
        )
        cpu_time = time.time() - start
        results['cpu_time'] = cpu_time
        results['cpu_docs_per_sec'] = n_samples / cpu_time

        # GPU benchmark (if available)
        if is_gpu_dedup_available():
            gpu_output = Path(tmpdir) / "gpu_output"
            start = time.time()
            _gpu_dedup_pytorch(
                str(input_path), str(gpu_output),
                text_column='text', id_column='id',
                similarity_threshold=0.8,
                char_ngrams=5,
                num_perm=128,
                memory_info=memory_info,
                show_progress=False
            )
            gpu_time = time.time() - start
            results['gpu_time'] = gpu_time
            results['gpu_docs_per_sec'] = n_samples / gpu_time
            results['speedup'] = cpu_time / gpu_time
        else:
            results['gpu_time'] = None
            results['gpu_docs_per_sec'] = None
            results['speedup'] = None

    return results


if __name__ == '__main__':
    print("GPU Dedup Utils - Testing")
    print("=" * 50)

    # Show memory info
    memory_info = get_memory_info()
    print(f"GPU: {memory_info['gpu_name'] or 'Not available'}")
    if memory_info['gpu_free_gb'] > 0:
        print(f"VRAM: {memory_info['gpu_free_gb']:.1f} GB free / {memory_info['gpu_total_gb']:.1f} GB total")
    print(f"RAM: {memory_info['ram_free_gb']:.1f} GB free / {memory_info['ram_total_gb']:.1f} GB total")
    print()

    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"GPU dedup available: {is_gpu_dedup_available()}")
    print(f"NeMo Curator available: {NEMO_AVAILABLE} ({NEMO_API_VERSION})")
    print(f"Datasketch available: {DATASKETCH_AVAILABLE}")

    if DATASKETCH_AVAILABLE:
        print("\nRunning small benchmark...")
        results = benchmark_dedup(n_samples=5_000)
        print(f"\nResults for {results['n_samples']:,} samples:")
        print(f"  CPU: {results['cpu_time']:.2f}s ({results['cpu_docs_per_sec']:,.0f} docs/sec)")
        if results['gpu_time']:
            print(f"  GPU: {results['gpu_time']:.2f}s ({results['gpu_docs_per_sec']:,.0f} docs/sec)")
            print(f"  Speedup: {results['speedup']:.1f}x")
