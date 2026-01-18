"""GPU-accelerated deduplication using NeMo Curator.

Provides 16-107x speedup over CPU MinHash on A100/H100 GPUs.
Falls back to CPU datasketch if NeMo Curator is not available.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Check for NeMo Curator availability
# NeMo Curator API has changed across versions - try multiple import paths
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

    # Try 2: NeMo Curator 1.0.0 stages API (alternative path)
    try:
        import nemo_curator.stages as stages
        if hasattr(stages, 'deduplication'):
            # Check what's in the deduplication module
            dedup_mod = stages.deduplication
            if hasattr(dedup_mod, 'fuzzy'):
                fuzzy_mod = dedup_mod.fuzzy
                if hasattr(fuzzy_mod, 'FuzzyDeduplicationWorkflow'):
                    import dask.dataframe as dd
                    NEMO_AVAILABLE = True
                    NEMO_API_VERSION = "workflow_v2"
                    return
    except ImportError as e:
        NEMO_IMPORT_ERROR = str(e)

    # Try 3: modules API (some older versions)
    try:
        from nemo_curator.modules import FuzzyDuplicates
        from nemo_curator.datasets import DocumentDataset
        import dask.dataframe as dd
        NEMO_AVAILABLE = True
        NEMO_API_VERSION = "modules"
        return
    except ImportError as e:
        NEMO_IMPORT_ERROR = str(e)

    # Try 4: Legacy direct import (NeMo Curator < 25.x)
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
    """GPU deduplication using NeMo Curator.

    Supports multiple API versions:
    - workflow: NeMo Curator 25.x (FuzzyDeduplicationWorkflow)
    - modules: Some versions (from nemo_curator.modules import FuzzyDuplicates)
    - legacy: NeMo Curator < 25.x (from nemo_curator import FuzzyDuplicates)
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    if cache_path is None:
        cache_p = output_p / ".nemo_cache"
    else:
        cache_p = Path(cache_path)
    cache_p.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"GPU Deduplication (NeMo Curator - {NEMO_API_VERSION} API)")
        print(f"  Input: {input_p}")
        print(f"  Output: {output_p}")
        print(f"  Threshold: {similarity_threshold}")
        print(f"  N-grams: {char_ngrams}, Buckets: {num_buckets}")

    # Use workflow API for NeMo Curator 25.x
    if NEMO_API_VERSION == "workflow":
        return _gpu_dedup_workflow_api(
            input_path=input_p,
            output_path=output_p,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            char_ngrams=char_ngrams,
            num_buckets=num_buckets,
            hashes_per_bucket=hashes_per_bucket,
            cache_path=cache_p,
            show_progress=show_progress,
        )

    # Use legacy/modules API for older versions
    return _gpu_dedup_legacy_api(
        input_path=input_p,
        output_path=output_p,
        text_column=text_column,
        id_column=id_column,
        similarity_threshold=similarity_threshold,
        char_ngrams=char_ngrams,
        num_buckets=num_buckets,
        hashes_per_bucket=hashes_per_bucket,
        cache_path=cache_p,
        n_workers=n_workers,
        show_progress=show_progress,
    )


def _gpu_dedup_workflow_api(
    input_path: Path,
    output_path: Path,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    char_ngrams: int,
    num_buckets: int,
    hashes_per_bucket: int,
    cache_path: Path,
    show_progress: bool,
) -> str:
    """GPU deduplication using NeMo Curator 1.0.0+ workflow API.

    Uses FuzzyDeduplicationWorkflow with proper RayClient initialization.
    The workflow requires:
    1. RayClient started with GPU support
    2. ID generator actor created after RayClient starts
    3. Then workflow can be run
    """
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
    from nemo_curator.core.client import RayClient
    from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor

    # Prepare input directory (workflow expects directory, not single file)
    if input_path.is_file():
        # Copy single file to temp directory
        input_dir = cache_path / "input_temp"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_path, input_dir / input_path.name)
        input_path_str = str(input_dir)
    else:
        input_path_str = str(input_path)

    # Workflow output paths - use output_path directly for deduplicated data
    workflow_cache = cache_path / "workflow_cache"
    workflow_cache.mkdir(parents=True, exist_ok=True)

    # Output goes directly to our output path
    dedup_output = output_path / "deduplicated"
    dedup_output.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print("  Running FuzzyDeduplicationWorkflow (identify duplicates)...")
        print(f"    char_ngrams: {char_ngrams} (recommended: 20+ for lower false positives)")
        print(f"    num_bands: {num_buckets}, minhashes_per_band: {hashes_per_bucket}")

    # Step 1: Shutdown any existing Ray instances to ensure clean state
    try:
        import ray
        if ray.is_initialized():
            if show_progress:
                print("  Shutting down existing Ray instance...")
            ray.shutdown()
            import time
            time.sleep(2)  # Give Ray time to fully shutdown
    except Exception:
        pass

    # Step 2: Initialize RayClient with GPU support
    # This is REQUIRED before creating the ID generator actor
    if show_progress:
        print("  Initializing RayClient with GPU support...")

    # Detect available GPUs
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 1  # Assume 1 GPU if torch not available

    # Use fewer CPUs to avoid resource contention in Colab
    import os
    num_cpus = min(os.cpu_count() or 4, 8)

    ray_client = RayClient(num_cpus=num_cpus, num_gpus=num_gpus)
    ray_client.start()

    try:
        # Step 3: Create ID generator actor (MUST be done after RayClient starts)
        if show_progress:
            print("  Creating ID generator actor...")
        create_id_generator_actor()

        # Step 4: Create and run workflow
        # NeMo Curator 1.0.0: perform_removal is not implemented yet
        # So we use perform_removal=False to identify duplicates, then remove manually
        id_output = workflow_cache / "fuzzy_ids"
        id_output.mkdir(parents=True, exist_ok=True)

        if show_progress:
            print("  Starting FuzzyDeduplicationWorkflow...")

        workflow = FuzzyDeduplicationWorkflow(
            input_path=input_path_str,
            cache_path=str(workflow_cache / "minhash_cache"),
            output_path=str(id_output),
            text_field=text_column,
            perform_removal=False,  # Identify only - removal not implemented in 1.0.0
            input_filetype="parquet",
            seed=42,
            char_ngrams=char_ngrams,
            num_bands=num_buckets,
            minhashes_per_band=hashes_per_bucket,
        )
        workflow.run()

        if show_progress:
            print("  MinHash + LSH complete, loading duplicate IDs...")

        # Load duplicate IDs from workflow output
        dup_ids_path = id_output / "FuzzyDuplicateIds"
        dup_ids = set()

        if dup_ids_path.exists():
            dup_files = list(dup_ids_path.glob("*.parquet"))
            if show_progress:
                print(f"  Found {len(dup_files)} duplicate ID files")

            for dup_file in dup_files:
                try:
                    dup_df = pd.read_parquet(dup_file)
                    # NeMo Curator uses various ID column names
                    for col in ['_curator_dedup_id', 'duplicate_id', 'id', id_column]:
                        if col in dup_df.columns:
                            dup_ids.update(dup_df[col].astype(str).tolist())
                            break
                except Exception as e:
                    if show_progress:
                        print(f"    Warning: Could not read {dup_file.name}: {e}")

        if show_progress:
            print(f"  Found {len(dup_ids):,} duplicate document IDs")

        # Load original data and remove duplicates
        if show_progress:
            print("  Removing duplicates from original data...")

        if input_path.is_file():
            df = pd.read_parquet(input_path)
        else:
            parquet_files = list(input_path.glob("*.parquet"))
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)

        n_docs = len(df)

        # The workflow adds _curator_dedup_id to processed data
        # Check for it in the output, or use original ID column
        id_col_to_use = None
        for col in ['_curator_dedup_id', id_column, 'id']:
            if col in df.columns:
                id_col_to_use = col
                break

        if id_col_to_use:
            # Convert to string for matching
            df[id_col_to_use] = df[id_col_to_use].astype(str)
            result_df = df[~df[id_col_to_use].isin(dup_ids)]
        else:
            # No ID column - use index-based approach
            # This happens if the input didn't have IDs and workflow didn't add them
            df['_temp_idx'] = df.index.astype(str)
            result_df = df[~df['_temp_idx'].isin(dup_ids)]
            result_df = result_df.drop('_temp_idx', axis=1)

        # Save deduplicated results
        result_df.to_parquet(dedup_output / "deduplicated.parquet", index=False)

        if show_progress:
            n_kept = len(result_df)
            n_removed = n_docs - n_kept
            print(f"  Removed: {n_removed:,} duplicates ({100*n_removed/n_docs:.1f}%)")
            print(f"  Kept: {n_kept:,} unique documents")

        return str(output_path)

    except Exception as e:
        error_msg = str(e)
        if show_progress:
            print(f"  Warning: GPU workflow failed: {error_msg}")
            print("  Falling back to CPU datasketch...")

        # Fall back to CPU deduplication
        return _cpu_dedup_datasketch(
            input_path=str(input_path),
            output_path=str(output_path),
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            show_progress=show_progress,
        )

    finally:
        # Step 5: Always stop RayClient when done (success or failure)
        try:
            if show_progress:
                print("  Stopping RayClient...")
            ray_client.stop()
        except Exception:
            pass  # Ignore cleanup errors


def _gpu_dedup_legacy_api(
    input_path: Path,
    output_path: Path,
    text_column: str,
    id_column: str,
    similarity_threshold: float,
    char_ngrams: int,
    num_buckets: int,
    hashes_per_bucket: int,
    cache_path: Path,
    n_workers: int,
    show_progress: bool,
) -> str:
    """GPU deduplication using legacy NeMo Curator API (< 25.x)."""
    from dask_cuda import LocalCUDACluster
    import dask.dataframe as dd

    # Import based on API version
    if NEMO_API_VERSION == "modules":
        from nemo_curator.modules import FuzzyDuplicates
        from nemo_curator.datasets import DocumentDataset
        from nemo_curator.utils.distributed_utils import get_client
    else:
        from nemo_curator import FuzzyDuplicates
        from nemo_curator.datasets import DocumentDataset
        from nemo_curator.utils.distributed_utils import get_client

    # Start Dask CUDA cluster
    cluster = LocalCUDACluster(n_workers=n_workers)
    client = get_client(cluster)

    if show_progress:
        print(f"  Dask cluster: {n_workers} GPU workers")

    n_docs = 0
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

        n_docs = len(ddf)
        if show_progress:
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

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"CPU Deduplication (datasketch)")
        print(f"  Input: {input_p}")
        print(f"  Output: {output_p}")
        print(f"  Threshold: {similarity_threshold}")

    # Load data
    if input_p.is_file():
        df = pd.read_parquet(input_p)
    else:
        parquet_files = list(input_p.glob("*.parquet"))
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
    output_file = output_p / "deduplicated.parquet"
    result_df.to_parquet(output_file, index=False)

    if show_progress:
        n_kept = len(result_df)
        n_removed = n_docs - n_kept
        print(f"  Removed: {n_removed:,} duplicates ({100*n_removed/n_docs:.1f}%)")
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
    print(f"NeMo API version: {NEMO_API_VERSION}")
    if NEMO_IMPORT_ERROR and not NEMO_AVAILABLE:
        print(f"NeMo import error: {NEMO_IMPORT_ERROR}")
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
