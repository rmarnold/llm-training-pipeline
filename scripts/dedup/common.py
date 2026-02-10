"""Common utilities and feature detection for deduplication."""
from __future__ import annotations

# Check for xxhash (10-100x faster than Python hash)
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

# Check for torch/CUDA
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Fallback to datasketch for CPU mode
try:
    from datasketch import MinHash, MinHashLSH  # noqa: F401
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False


def get_memory_info() -> dict:
    """Get available GPU and system memory."""
    info = {
        'gpu_total_gb': 0,
        'gpu_free_gb': 0,
        'ram_total_gb': 0,
        'ram_free_gb': 0,
        'gpu_name': None,
    }

    if CUDA_AVAILABLE:
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_name'] = gpu_props.name
            info['gpu_total_gb'] = gpu_props.total_memory / (1024**3)
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info['gpu_free_gb'] = free_mem / (1024**3)
        except Exception:
            pass

    try:
        import psutil
        vm = psutil.virtual_memory()
        info['ram_total_gb'] = vm.total / (1024**3)
        info['ram_free_gb'] = vm.available / (1024**3)
    except ImportError:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        info['ram_total_gb'] = int(line.split()[1]) / (1024**2)
                    elif 'MemAvailable' in line:
                        info['ram_free_gb'] = int(line.split()[1]) / (1024**2)
        except Exception:
            info['ram_total_gb'] = 16
            info['ram_free_gb'] = 8

    return info


def is_gpu_dedup_available() -> bool:
    """Check if GPU deduplication is available."""
    return CUDA_AVAILABLE and TORCH_AVAILABLE
