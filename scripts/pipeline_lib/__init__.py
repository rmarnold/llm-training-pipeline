"""Shared utilities for the LLM training pipeline.

Submodule imports are lazy to avoid pulling in transformers before unsloth.
Unsloth must be imported before transformers to apply its optimizations.
Import directly from submodules: e.g. ``from pipeline_lib.model_utils import ...``
"""


def __getattr__(name: str):
    if name == "unwrap_compiled_model" or name == "load_compiled_checkpoint":
        from .model_utils import unwrap_compiled_model, load_compiled_checkpoint
        return unwrap_compiled_model if name == "unwrap_compiled_model" else load_compiled_checkpoint
    if name == "OOMRecoveryCallback":
        from .training_callbacks import OOMRecoveryCallback
        return OOMRecoveryCallback
    raise AttributeError(f"module 'pipeline_lib' has no attribute {name!r}")


__all__ = [
    "unwrap_compiled_model",
    "load_compiled_checkpoint",
    "OOMRecoveryCallback",
]
