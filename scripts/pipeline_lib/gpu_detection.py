"""GPU detection and backend configuration utilities."""
from __future__ import annotations

from typing import TypedDict

import torch


class GPUInfo(TypedDict):
    """Type definition for GPU information dictionary."""
    gpu_name: str
    compute_capability: str
    is_h100: bool
    is_a100: bool
    fp8_available: bool
    compile_mode: str
    batch_size: int


def check_fp8_available() -> bool:
    """Check if FP8 training is available (H100 + transformer-engine)."""
    try:
        import transformer_engine  # noqa: F401
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 9:
                return True
    except ImportError:
        pass
    return False


def detect_gpu_type() -> GPUInfo:
    """Detect GPU type and return optimized settings."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability()
        is_h100 = "H100" in gpu_name or capability[0] >= 9
        is_a100 = "A100" in gpu_name or (capability[0] == 8 and capability[1] == 0)
        fp8_available = check_fp8_available()

        return {
            "gpu_name": gpu_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "is_h100": is_h100,
            "is_a100": is_a100,
            "fp8_available": fp8_available and is_h100,
            "compile_mode": "max-autotune" if is_h100 else "default",
            "batch_size": 8,
        }
    return {
        "gpu_name": "CPU",
        "compute_capability": "N/A",
        "is_h100": False,
        "is_a100": False,
        "fp8_available": False,
        "compile_mode": "default",
        "batch_size": 4
    }


def print_gpu_info(gpu_info: GPUInfo) -> None:
    """Print GPU information."""
    print(f"GPU detected: {gpu_info['gpu_name']}")
    print(f"  Compute capability: {gpu_info['compute_capability']}")
    if gpu_info['is_h100']:
        if gpu_info['fp8_available']:
            print("  H100 with FP8 support - maximum performance available")
        else:
            print("  H100 detected (install transformer-engine for FP8)")
    elif gpu_info['is_a100']:
        print("  A100 detected - BF16 precision")


def setup_torch_backends() -> None:
    """Configure PyTorch backends for optimal performance."""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+ GPUs
        # Note: Use legacy API only - torch.compile's inductor backend expects it
        # and mixing with new API causes RuntimeError
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

    # Configure torch.compile/dynamo for better compatibility with custom kernels
    # torch._dynamo requires PyTorch >= 2.1
    if hasattr(torch, '_dynamo'):
        _dynamo = torch._dynamo
        # Capture scalar outputs like .item() to avoid graph breaks from Liger kernels
        _dynamo.config.capture_scalar_outputs = True
        # Treat layer_idx as dynamic to avoid recompilation per layer (32 layers = 32 recompiles)
        _dynamo.config.assume_static_by_default = False
        # Suppress excessive recompilation warnings
        _dynamo.config.suppress_errors = False
        # Increase cache size limit to handle more graph variations
        _dynamo.config.cache_size_limit = 64
