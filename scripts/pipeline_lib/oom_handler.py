"""GPU out-of-memory error handling with automatic batch size reduction."""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

import torch

F = TypeVar('F', bound=Callable[..., Any])


class OOMHandler:
    """Handler for GPU out-of-memory errors with automatic batch size reduction.

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
        """Reduce batch size by the reduction factor."""
        new_size = max(self.min_batch_size, int(self.batch_size * self.reduction_factor))
        if new_size >= self.batch_size:
            new_size = self.batch_size - 1
        self.batch_size = max(1, new_size)
        return self.batch_size

    def handle_oom(self, error: Exception) -> bool:
        """Handle an OOM error. Returns True if should retry, False if should raise."""
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

    Usage:
        with oom_recovery_context(initial_batch_size=8) as handler:
            for batch in dataloader:
                try:
                    output = model(batch[:handler.batch_size])
                except RuntimeError as e:
                    if not handler.handle_oom(e):
                        raise
                else:
                    handler.reset()
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
    gpu_memory_gb: float | None = None,
    sequence_length: int = 2048,
    dtype_bytes: int = 2,  # BF16
    safety_factor: float = 0.8,
) -> int:
    """Estimate a safe batch size based on GPU memory.

    This is a rough heuristic - actual memory usage depends on many factors.
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            return 1  # CPU fallback

    model_memory_gb = model_params_billions * dtype_bytes
    overhead_gb = model_memory_gb * 6  # Gradients + optimizer + activations base

    available_gb = (gpu_memory_gb * safety_factor) - overhead_gb
    if available_gb <= 0:
        return 1

    per_sample_gb = (sequence_length * 4096 * dtype_bytes * 2) / (1024**3)
    batch_size = max(1, int(available_gb / per_sample_gb))

    return min(batch_size, 32)
