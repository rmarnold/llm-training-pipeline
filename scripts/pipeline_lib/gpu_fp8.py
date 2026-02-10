"""FP8 training configuration using NVIDIA Transformer Engine."""
from __future__ import annotations


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
