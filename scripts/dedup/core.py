"""Core deduplication entry point - dispatches to GPU or CPU implementation."""
from __future__ import annotations

from dedup.common import (
    get_memory_info,
    is_gpu_dedup_available,
    CUDA_AVAILABLE,
    XXHASH_AVAILABLE,
)
from dedup.gpu_impl import gpu_dedup_fast
from dedup.cpu_fallback import cpu_dedup_streaming


def gpu_fuzzy_dedup(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    id_column: str = "id",
    similarity_threshold: float = 0.8,
    char_ngrams: int = 5,
    num_buckets: int = 20,
    hashes_per_bucket: int = 13,
    cache_path: str | None = None,
    use_gpu: bool | None = None,
    n_workers: int = 1,
    show_progress: bool = True,
) -> str:
    """GPU-accelerated fuzzy deduplication - FAST implementation.

    Optimized for 20M+ documents with:
    - xxhash for fast shingle hashing
    - Vectorized GPU MinHash computation
    - Band-based LSH without datasketch overhead

    Args:
        input_path: Path to input parquet file(s) or directory
        output_path: Output path for deduplicated data
        text_column: Name of text column
        id_column: Name of ID column (will be created if missing)
        similarity_threshold: Jaccard similarity threshold (0.8 = 80% similar)
        char_ngrams: Character n-gram size for shingling
        num_buckets: Number of LSH bands
        hashes_per_bucket: Rows per band (num_perm = num_buckets * hashes_per_bucket)
        cache_path: Cache directory for intermediate results
        use_gpu: Force GPU (True) or CPU (False). None = auto-detect.
        n_workers: Number of workers (unused, for API compatibility)
        show_progress: Show progress information

    Returns:
        Path to deduplicated output
    """
    if use_gpu is None:
        use_gpu = is_gpu_dedup_available()

    memory_info = get_memory_info()

    if show_progress:
        print(f"GPU Deduplication - FAST Mode")
        print(f"  GPU: {memory_info['gpu_name'] or 'Not available'}")
        if memory_info['gpu_free_gb'] > 0:
            print(f"  VRAM: {memory_info['gpu_free_gb']:.1f} GB free / {memory_info['gpu_total_gb']:.1f} GB total")
        print(f"  RAM: {memory_info['ram_free_gb']:.1f} GB free / {memory_info['ram_total_gb']:.1f} GB total")
        print(f"  xxhash: {'Available (10-100x faster)' if XXHASH_AVAILABLE else 'Not available'}")

    num_perm = num_buckets * hashes_per_bucket

    if use_gpu and CUDA_AVAILABLE:
        return gpu_dedup_fast(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            char_ngrams=char_ngrams,
            num_perm=num_perm,
            num_bands=num_buckets,
            memory_info=memory_info,
            show_progress=show_progress,
        )
    else:
        return cpu_dedup_streaming(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            id_column=id_column,
            similarity_threshold=similarity_threshold,
            memory_info=memory_info,
            show_progress=show_progress,
        )
