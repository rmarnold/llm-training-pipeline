"""GPU-accelerated deduplication using NeMo Curator.

Provides 16-107x speedup over CPU MinHash on A100/H100 GPUs.
Falls back to CPU datasketch if NeMo Curator is not available.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Check for NeMo Curator availability
try:
    from nemo_curator import FuzzyDuplicates
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import get_client
    import dask.dataframe as dd
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

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


def is_gpu_dedup_available() -> bool:
    """Check if GPU deduplication is available."""
    return NEMO_AVAILABLE and DASK_CUDA_AVAILABLE


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
    """GPU-accelerated fuzzy deduplication.

    16-107x faster than CPU MinHash on A100.

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

    if use_gpu and NEMO_AVAILABLE:
        return _gpu_dedup_nemo(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            char_ngrams=char_ngrams,
            num_buckets=num_buckets,
            hashes_per_bucket=hashes_per_bucket,
            cache_path=cache_path,
            n_workers=n_workers,
            show_progress=show_progress,
        )
    else:
        return _cpu_dedup_datasketch(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            show_progress=show_progress,
        )


def _gpu_dedup_nemo(
    input_path: str,
    output_path: str,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    char_ngrams: int,
    num_buckets: int,
    hashes_per_bucket: int,
    cache_path: Optional[str],
    n_workers: int,
    show_progress: bool,
) -> str:
    """GPU deduplication using NeMo Curator."""
    from nemo_curator import FuzzyDuplicates
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import get_client
    from dask_cuda import LocalCUDACluster
    import dask.dataframe as dd

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if cache_path is None:
        cache_path = output_path / ".nemo_cache"
    else:
        cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"GPU Deduplication (NeMo Curator)")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Threshold: {similarity_threshold}")
        print(f"  N-grams: {char_ngrams}, Buckets: {num_buckets}")

    # Start Dask CUDA cluster
    cluster = LocalCUDACluster(n_workers=n_workers)
    client = get_client(cluster)

    if show_progress:
        print(f"  Dask cluster: {n_workers} GPU workers")

    try:
        # Load data
        if input_path.is_file():
            ddf = dd.read_parquet(str(input_path))
        else:
            ddf = dd.read_parquet(str(input_path / "*.parquet"))

        # Add ID column if missing
        if id_column not in ddf.columns:
            ddf[id_column] = ddf.index.astype(str)

        # Create NeMo dataset
        dataset = DocumentDataset(ddf)

        if show_progress:
            n_docs = len(ddf)
            print(f"  Documents: {n_docs:,}")

        # Configure fuzzy deduplication
        fuzzy_dedup = FuzzyDuplicates(
            id_field=id_column,
            text_field=text_column,
            seed=42,
            char_ngrams=char_ngrams,
            num_buckets=num_buckets,
            hashes_per_bucket=hashes_per_bucket,
            use_64_bit_hash=False,
            buckets_per_shuffle=1,
            false_positive_check=True,
            num_anchors=2,
            jaccard_threshold=similarity_threshold,
            cache_dir=str(cache_path),
        )

        # Find duplicates
        if show_progress:
            print("  Computing MinHash signatures...")

        duplicates = fuzzy_dedup(dataset)

        # Remove duplicates
        if show_progress:
            print("  Filtering duplicates...")

        # Get duplicate IDs
        dup_ids = duplicates.df[id_column].compute().tolist()
        dup_set = set(dup_ids)

        # Filter original dataset
        result_ddf = ddf[~ddf[id_column].isin(dup_set)]

        # Save results
        result_ddf.to_parquet(str(output_path), write_index=False)

        if show_progress:
            n_kept = len(result_ddf)
            n_removed = n_docs - n_kept
            print(f"  Removed: {n_removed:,} duplicates ({100*n_removed/n_docs:.1f}%)")
            print(f"  Kept: {n_kept:,} unique documents")

    finally:
        client.close()
        cluster.close()

    return str(output_path)


def _cpu_dedup_datasketch(
    input_path: str,
    output_path: str,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    show_progress: bool,
) -> str:
    """CPU fallback using datasketch MinHashLSH."""
    if not DATASKETCH_AVAILABLE:
        raise ImportError("datasketch not installed. Run: pip install datasketch")

    from datasketch import MinHash, MinHashLSH

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"CPU Deduplication (datasketch)")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Threshold: {similarity_threshold}")

    # Load data
    if input_path.is_file():
        df = pd.read_parquet(input_path)
    else:
        parquet_files = list(input_path.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

    # Add ID column if missing
    if id_column not in df.columns:
        df[id_column] = df.index.astype(str)

    n_docs = len(df)
    if show_progress:
        print(f"  Documents: {n_docs:,}")

    # Initialize LSH
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
    keep_mask = []

    # Process documents
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=n_docs, desc="  Deduplicating", unit="doc")

    for idx, row in iterator:
        text = str(row[text_column])
        doc_id = str(row[id_column])

        # Compute MinHash
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))

        # Check for duplicates
        if lsh.query(m):
            keep_mask.append(False)
        else:
            lsh.insert(doc_id, m)
            keep_mask.append(True)

    # Filter and save
    result_df = df[keep_mask]
    output_file = output_path / "deduplicated.parquet"
    result_df.to_parquet(output_file, index=False)

    if show_progress:
        n_kept = len(result_df)
        n_removed = n_docs - n_kept
        print(f"  Removed: {n_removed:,} duplicates ({100*n_removed/n_docs:.1f}%)")
        print(f"  Kept: {n_kept:,} unique documents")

    return str(output_path)


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

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.parquet"
        df.to_parquet(input_path)

        # CPU benchmark
        cpu_output = Path(tmpdir) / "cpu_output"
        start = time.time()
        _cpu_dedup_datasketch(
            str(input_path), str(cpu_output),
            text_column='text', id_column='id',
            similarity_threshold=0.8, show_progress=False
        )
        cpu_time = time.time() - start
        results['cpu_time'] = cpu_time
        results['cpu_docs_per_sec'] = n_samples / cpu_time

        # GPU benchmark (if available)
        if is_gpu_dedup_available():
            gpu_output = Path(tmpdir) / "gpu_output"
            start = time.time()
            _gpu_dedup_nemo(
                str(input_path), str(gpu_output),
                text_column='text', id_column='id',
                similarity_threshold=0.8, char_ngrams=5,
                num_buckets=20, hashes_per_bucket=13,
                cache_path=str(Path(tmpdir) / "cache"),
                n_workers=1, show_progress=False
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
    print(f"NeMo Curator available: {NEMO_AVAILABLE}")
    print(f"Dask CUDA available: {DASK_CUDA_AVAILABLE}")
    print(f"GPU dedup available: {is_gpu_dedup_available()}")
    print(f"Datasketch available: {DATASKETCH_AVAILABLE}")

    if DATASKETCH_AVAILABLE:
        print("\nRunning small benchmark...")
        results = benchmark_dedup(n_samples=5_000)
        print(f"\nResults for {results['n_samples']:,} samples:")
        print(f"  CPU: {results['cpu_time']:.2f}s ({results['cpu_docs_per_sec']:,.0f} docs/sec)")
        if results['gpu_time']:
            print(f"  GPU: {results['gpu_time']:.2f}s ({results['gpu_docs_per_sec']:,.0f} docs/sec)")
            print(f"  Speedup: {results['speedup']:.1f}x")
