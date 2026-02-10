"""Deduplication package - GPU-accelerated and CPU fallback implementations.

Import modules directly where needed:
    from dedup.core import gpu_fuzzy_dedup
    from dedup.common import get_memory_info, is_gpu_dedup_available
    from dedup.gpu_impl import gpu_dedup_fast
    from dedup.cpu_fallback import cpu_dedup_streaming
    from dedup.minhash import fast_shingle_hash, compute_minhash_signatures_gpu
    from dedup.lsh import build_lsh_index, find_duplicates_lsh
"""
