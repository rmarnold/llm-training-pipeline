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


def check_tokenizer_exists(tokenizer_path="configs/tokenizer"):
    """Check if tokenizer exists and provide helpful message if not.

    Args:
        tokenizer_path: Path to tokenizer directory

    Returns:
        True if tokenizer exists, False otherwise
    """
    import os

    required_files = ["tokenizer_config.json", "tokenizer.json"]
    alt_files = ["vocab.json", "merges.txt"]  # For some tokenizer types

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("")
        print("To create the tokenizer, run:")
        print("  python scripts/demo_tokenize.py")
        print("")
        print("Or for production, run the data preparation pipeline:")
        print("  python scripts/03_tokenize_and_pack.py")
        return False

    # Check for tokenizer files
    has_required = any(
        os.path.exists(os.path.join(tokenizer_path, f))
        for f in required_files + alt_files
    )

    if not has_required:
        print(f"Error: Tokenizer directory exists but appears incomplete: {tokenizer_path}")
        print(f"Missing expected files: {required_files}")
        return False

    return True


def check_checkpoint_exists(checkpoint_path, checkpoint_type="model"):
    """Check if a checkpoint/model exists at the given path.

    Args:
        checkpoint_path: Path to checkpoint directory
        checkpoint_type: Type of checkpoint ('model', 'tokenizer', 'data')

    Returns:
        True if checkpoint exists and is valid, False otherwise
    """
    import os

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_type.title()} not found at {checkpoint_path}")

        # Provide helpful suggestions
        if checkpoint_type == "model":
            if "init" in checkpoint_path:
                print("\nTo initialize the model, run:")
                print("  python scripts/04_init_model.py")
            elif "pretrain" in checkpoint_path:
                print("\nTo create this checkpoint, run pretraining:")
                print("  python scripts/05_pretrain.py")
            elif "sft" in checkpoint_path:
                print("\nTo create this checkpoint, run SFT:")
                print("  python scripts/07_sft.py")
            elif "dpo" in checkpoint_path:
                print("\nTo create this checkpoint, run DPO:")
                print("  python scripts/09_dpo.py")
        return False

    # Check for model files
    model_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    has_model = any(
        os.path.exists(os.path.join(checkpoint_path, f))
        for f in model_files
    )

    if not has_model:
        print(f"Error: Directory exists but no model files found: {checkpoint_path}")
        print(f"Expected one of: {model_files}")
        return False

    return True


def validate_training_prerequisites(
    model_path=None,
    tokenizer_path="configs/tokenizer",
    data_path=None
):
    """Validate all prerequisites before starting training.

    Args:
        model_path: Path to model checkpoint (optional)
        tokenizer_path: Path to tokenizer
        data_path: Path to training data (optional)

    Returns:
        True if all prerequisites are met, False otherwise
    """
    import os

    all_valid = True

    # Check tokenizer
    if not check_tokenizer_exists(tokenizer_path):
        all_valid = False

    # Check model if specified
    if model_path and not check_checkpoint_exists(model_path, "model"):
        all_valid = False

    # Check data if specified
    if data_path:
        if not os.path.exists(data_path):
            print(f"Error: Training data not found at {data_path}")
            print("\nTo prepare training data, run:")
            print("  python scripts/01_download_data.py")
            print("  python scripts/02_clean_deduplicate_optimized.py")
            print("  python scripts/03_tokenize_and_pack.py")
            all_valid = False

    return all_valid
