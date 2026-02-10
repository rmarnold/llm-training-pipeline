"""GPU detection and optimization utilities for LLM training.

This module re-exports from pipeline_lib for backward compatibility.
"""
from pipeline_lib.gpu_detection import (
    GPUInfo,
    check_fp8_available,
    detect_gpu_type,
    print_gpu_info,
    setup_torch_backends,
)
from pipeline_lib.gpu_fp8 import get_fp8_accelerator
from pipeline_lib.training_validation import (
    check_tokenizer_exists,
    check_checkpoint_exists,
    validate_training_prerequisites,
)
from pipeline_lib.oom_handler import (
    OOMHandler,
    oom_recovery_context,
    with_oom_retry,
    get_safe_batch_size,
)

__all__ = [
    "GPUInfo",
    "check_fp8_available",
    "detect_gpu_type",
    "print_gpu_info",
    "setup_torch_backends",
    "get_fp8_accelerator",
    "check_tokenizer_exists",
    "check_checkpoint_exists",
    "validate_training_prerequisites",
    "OOMHandler",
    "oom_recovery_context",
    "with_oom_retry",
    "get_safe_batch_size",
]
