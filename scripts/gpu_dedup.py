"""GPU-accelerated deduplication using pure PyTorch/CUDA.

This module re-exports from dedup/ package for backward compatibility.
"""
from dedup.common import (
    get_memory_info,
    is_gpu_dedup_available,
    CUDA_AVAILABLE,
    TORCH_AVAILABLE,
    XXHASH_AVAILABLE,
    DATASKETCH_AVAILABLE,
)
from dedup.core import gpu_fuzzy_dedup

__all__ = [
    "get_memory_info",
    "is_gpu_dedup_available",
    "gpu_fuzzy_dedup",
    "CUDA_AVAILABLE",
    "TORCH_AVAILABLE",
    "XXHASH_AVAILABLE",
    "DATASKETCH_AVAILABLE",
]


if __name__ == '__main__':
    print("GPU Dedup - FAST Mode")
    print("=" * 50)

    memory_info = get_memory_info()
    print(f"GPU: {memory_info['gpu_name'] or 'Not available'}")
    if memory_info['gpu_free_gb'] > 0:
        print(f"VRAM: {memory_info['gpu_free_gb']:.1f} GB free")
    print(f"RAM: {memory_info['ram_free_gb']:.1f} GB free")
    print(f"xxhash: {XXHASH_AVAILABLE}")
    print(f"CUDA: {CUDA_AVAILABLE}")
    print(f"datasketch: {DATASKETCH_AVAILABLE}")
