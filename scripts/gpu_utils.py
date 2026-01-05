"""GPU detection and optimization utilities for LLM training"""
from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypedDict, TypeVar

import torch

F = TypeVar('F', bound=Callable[..., Any])


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

def get_fp8_accelerator(gradient_accumulation_steps: int = 4):
    """Create an Accelerator configured for FP8 training.

    Args:
        gradient_accumulation_steps: Number of gradient accumulation steps.

    Returns:
        Configured Accelerator instance with FP8 settings.
    """
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

def setup_torch_backends() -> None:
    """Configure PyTorch backends for optimal performance."""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True


def check_tokenizer_exists(tokenizer_path: str = "configs/tokenizer") -> bool:
    """Check if tokenizer exists and provide helpful message if not.

    Args:
        tokenizer_path: Path to tokenizer directory.

    Returns:
        True if tokenizer exists, False otherwise.
    """
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


def check_checkpoint_exists(checkpoint_path: str, checkpoint_type: str = "model") -> bool:
    """Check if a checkpoint/model exists at the given path.

    Args:
        checkpoint_path: Path to checkpoint directory.
        checkpoint_type: Type of checkpoint ('model', 'tokenizer', 'data').

    Returns:
        True if checkpoint exists and is valid, False otherwise.
    """
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
    model_path: Optional[str] = None,
    tokenizer_path: str = "configs/tokenizer",
    data_path: Optional[str] = None
) -> bool:
    """Validate all prerequisites before starting training.

    Args:
        model_path: Path to model checkpoint (optional).
        tokenizer_path: Path to tokenizer.
        data_path: Path to training data (optional).

    Returns:
        True if all prerequisites are met, False otherwise.
    """
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


class OOMHandler:
    """Handler for GPU out-of-memory errors with automatic batch size reduction.

    This class provides utilities for gracefully handling CUDA OOM errors by:
    1. Catching OOM exceptions
    2. Clearing GPU memory
    3. Reducing batch size automatically
    4. Retrying the operation

    Usage:
        handler = OOMHandler(initial_batch_size=8, min_batch_size=1)

        while handler.batch_size >= handler.min_batch_size:
            try:
                train_step(batch_size=handler.batch_size)
                break
            except RuntimeError as e:
                if not handler.handle_oom(e):
                    raise
    """

    def __init__(
        self,
        initial_batch_size: int = 8,
        min_batch_size: int = 1,
        reduction_factor: float = 0.5,
        max_retries: int = 3,
        cooldown_seconds: float = 2.0,
    ) -> None:
        """Initialize OOM handler.

        Args:
            initial_batch_size: Starting batch size.
            min_batch_size: Minimum batch size before giving up.
            reduction_factor: Factor to reduce batch size by (0.5 = halve).
            max_retries: Maximum number of OOM retries.
            cooldown_seconds: Seconds to wait after clearing memory.
        """
        self.initial_batch_size = initial_batch_size
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.retry_count = 0
        self.oom_count = 0

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def is_oom_error(self, error: Exception) -> bool:
        """Check if an exception is a CUDA OOM error."""
        error_msg = str(error).lower()
        return (
            isinstance(error, RuntimeError) and
            ("cuda" in error_msg or "gpu" in error_msg) and
            ("out of memory" in error_msg or "oom" in error_msg or
             "memory" in error_msg and "allocat" in error_msg)
        )

    def reduce_batch_size(self) -> int:
        """Reduce batch size by the reduction factor.

        Returns:
            New batch size (may be less than min if no more reductions possible).
        """
        new_size = max(self.min_batch_size, int(self.batch_size * self.reduction_factor))
        if new_size >= self.batch_size:
            # Can't reduce further
            new_size = self.batch_size - 1
        self.batch_size = max(1, new_size)
        return self.batch_size

    def handle_oom(self, error: Exception) -> bool:
        """Handle an OOM error.

        Args:
            error: The exception that occurred.

        Returns:
            True if OOM was handled and should retry, False if should raise.
        """
        if not self.is_oom_error(error):
            return False

        self.oom_count += 1
        self.retry_count += 1

        if self.retry_count > self.max_retries:
            print(f"\n[OOM] Max retries ({self.max_retries}) exceeded. Giving up.")
            return False

        if self.batch_size <= self.min_batch_size:
            print(f"\n[OOM] Already at minimum batch size ({self.min_batch_size}). Cannot reduce further.")
            return False

        old_size = self.batch_size
        new_size = self.reduce_batch_size()

        print(f"\n{'='*60}")
        print(f"[OOM RECOVERY] GPU out of memory detected!")
        print(f"{'='*60}")
        print(f"  Clearing GPU memory...")
        self.clear_gpu_memory()
        print(f"  Reducing batch size: {old_size} -> {new_size}")
        print(f"  Retry count: {self.retry_count}/{self.max_retries}")
        print(f"  Waiting {self.cooldown_seconds}s before retry...")
        time.sleep(self.cooldown_seconds)
        print(f"  Resuming training...")
        print(f"{'='*60}\n")

        return True

    def reset(self) -> None:
        """Reset handler state (call after successful training step)."""
        self.retry_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get OOM handling statistics."""
        return {
            "initial_batch_size": self.initial_batch_size,
            "current_batch_size": self.batch_size,
            "total_oom_events": self.oom_count,
            "batch_size_reduced": self.initial_batch_size != self.batch_size,
        }


@contextmanager
def oom_recovery_context(
    initial_batch_size: int = 8,
    min_batch_size: int = 1,
    reduction_factor: float = 0.5,
) -> Generator[OOMHandler, None, None]:
    """Context manager for OOM recovery.

    Yields an OOMHandler that can be used to get the current batch size
    and handle OOM errors.

    Usage:
        with oom_recovery_context(initial_batch_size=8) as handler:
            for batch in dataloader:
                try:
                    # Use handler.batch_size for dynamic sizing
                    output = model(batch[:handler.batch_size])
                except RuntimeError as e:
                    if not handler.handle_oom(e):
                        raise
                else:
                    handler.reset()  # Reset retry count on success
    """
    handler = OOMHandler(
        initial_batch_size=initial_batch_size,
        min_batch_size=min_batch_size,
        reduction_factor=reduction_factor,
    )
    try:
        yield handler
    finally:
        if handler.oom_count > 0:
            stats = handler.get_stats()
            print(f"\n[OOM Summary] Total OOM events: {stats['total_oom_events']}")
            if stats['batch_size_reduced']:
                print(f"  Batch size reduced: {stats['initial_batch_size']} -> {stats['current_batch_size']}")


def with_oom_retry(
    max_retries: int = 3,
    reduction_factor: float = 0.5,
    min_batch_size: int = 1,
) -> Callable[[F], F]:
    """Decorator for functions that may encounter OOM errors.

    The decorated function must accept a `batch_size` keyword argument.
    On OOM, it will be retried with a reduced batch size.

    Usage:
        @with_oom_retry(max_retries=3)
        def train_step(batch_size: int = 8):
            # Training code here
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            batch_size = kwargs.get('batch_size', 8)
            handler = OOMHandler(
                initial_batch_size=batch_size,
                min_batch_size=min_batch_size,
                reduction_factor=reduction_factor,
                max_retries=max_retries,
            )

            while True:
                try:
                    kwargs['batch_size'] = handler.batch_size
                    result = func(*args, **kwargs)
                    handler.reset()
                    return result
                except RuntimeError as e:
                    if not handler.handle_oom(e):
                        raise
        return wrapper  # type: ignore
    return decorator


def get_safe_batch_size(
    model_params_billions: float = 7.0,
    gpu_memory_gb: Optional[float] = None,
    sequence_length: int = 2048,
    dtype_bytes: int = 2,  # BF16
    safety_factor: float = 0.8,
) -> int:
    """Estimate a safe batch size based on GPU memory.

    This is a rough heuristic - actual memory usage depends on many factors.

    Args:
        model_params_billions: Model size in billions of parameters.
        gpu_memory_gb: GPU memory in GB (auto-detected if None).
        sequence_length: Maximum sequence length.
        dtype_bytes: Bytes per parameter (2 for BF16/FP16, 4 for FP32).
        safety_factor: Fraction of memory to use (0.8 = 80%).

    Returns:
        Recommended batch size.
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            return 1  # CPU fallback

    # Rough memory estimation (very approximate):
    # - Model weights: params * dtype_bytes
    # - Activations: ~4x model size per batch element
    # - Gradients: ~2x model size
    # - Optimizer states: ~4x model size
    model_memory_gb = model_params_billions * dtype_bytes
    overhead_gb = model_memory_gb * 6  # Gradients + optimizer + activations base

    available_gb = (gpu_memory_gb * safety_factor) - overhead_gb
    if available_gb <= 0:
        return 1

    # Per-sample activation memory (rough estimate)
    per_sample_gb = (sequence_length * 4096 * dtype_bytes * 2) / (1024**3)

    batch_size = max(1, int(available_gb / per_sample_gb))

    # Clamp to reasonable range
    return min(batch_size, 32)
