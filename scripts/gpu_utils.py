"""GPU detection and optimization utilities for LLM training"""
import torch

def check_fp8_available():
    """Check if FP8 training is available (H100 + transformer-engine)"""
    try:
        import transformer_engine
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # H100 has compute capability 9.0+
            if capability[0] >= 9:
                return True
    except ImportError:
        pass
    return False

def detect_gpu_type():
    """Detect GPU type and return optimized settings"""
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

def print_gpu_info(gpu_info):
    """Print GPU information"""
    print(f"GPU detected: {gpu_info['gpu_name']}")
    print(f"  Compute capability: {gpu_info['compute_capability']}")
    if gpu_info['is_h100']:
        if gpu_info['fp8_available']:
            print("  H100 with FP8 support - maximum performance available")
        else:
            print("  H100 detected (install transformer-engine for FP8)")
    elif gpu_info['is_a100']:
        print("  A100 detected - BF16 precision")

def get_fp8_accelerator(gradient_accumulation_steps=4):
    """Create an Accelerator configured for FP8 training"""
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs

    fp8_kwargs = FP8RecipeKwargs(
        backend="te",  # Transformer Engine backend
        fp8_format="HYBRID",  # E4M3 for forward, E5M2 for backward
        amax_history_len=1024,
        amax_compute_algo="max",
    )

    accelerator = Accelerator(
        mixed_precision="fp8",
        kwargs_handlers=[fp8_kwargs],
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    return accelerator

def setup_torch_backends():
    """Configure PyTorch backends for optimal performance"""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
