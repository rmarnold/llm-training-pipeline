"""Flexible model configuration for different model sizes.

This module provides pre-configured model architectures from 125M to 7B parameters,
optimized for LLaMA-style training with GQA (Grouped Query Attention).

Usage:
    from configs.model_configs import get_model_config, list_available_sizes

    # Get a specific size
    config = get_model_config("3b")

    # List all available sizes
    sizes = list_available_sizes()

    # Custom configuration
    config = get_model_config("1b", max_position_embeddings=8192)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

# Type for model sizes
ModelSize = Literal["125m", "350m", "1b", "3b", "7b"]


@dataclass
class ModelConfig:
    """LLaMA-style model configuration with computed properties."""

    # Core architecture
    vocab_size: int = 50304  # Padded for efficiency (divisible by 64)
    hidden_size: int = 4096
    intermediate_size: int = 11008  # ~2.7x hidden for SwiGLU
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA ratio

    # Context & positional encoding
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0

    # Regularization
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # Activations
    hidden_act: str = "silu"

    # Training efficiency
    use_cache: bool = False  # Disabled for training
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Precision
    torch_dtype: str = "bfloat16"

    # Metadata
    size_name: str = "7b"

    @property
    def total_params(self) -> float:
        """Calculate approximate total parameters in billions."""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size

        # Attention parameters per layer (Q, K, V, O projections)
        # With GQA: Q has full heads, K/V have fewer heads
        q_params = self.hidden_size * self.hidden_size
        kv_params = 2 * self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads)
        o_params = self.hidden_size * self.hidden_size
        attention_params = self.num_hidden_layers * (q_params + kv_params + o_params)

        # FFN parameters (SwiGLU has gate, up, down projections)
        ffn_params = self.num_hidden_layers * (
            3 * self.hidden_size * self.intermediate_size  # gate + up + down
        )

        # LayerNorm parameters
        norm_params = self.num_hidden_layers * 2 * self.hidden_size  # attention + FFN norms
        norm_params += self.hidden_size  # Final norm

        # LM head (usually tied with embeddings, but count separately)
        lm_head_params = self.hidden_size * self.vocab_size

        total = embed_params + attention_params + ffn_params + norm_params + lm_head_params
        return total / 1e9

    @property
    def total_params_str(self) -> str:
        """Human-readable parameter count."""
        params = self.total_params
        if params >= 1.0:
            return f"{params:.2f}B"
        else:
            return f"{params * 1000:.0f}M"

    @property
    def gqa_ratio(self) -> int:
        """GQA ratio (heads per KV head)."""
        return self.num_attention_heads // self.num_key_value_heads

    def summary(self) -> str:
        """Print configuration summary."""
        return f"""
Model Configuration: {self.size_name.upper()}
{'='*50}
Parameters:     {self.total_params_str} ({self.total_params:.3f}B)
Hidden size:    {self.hidden_size}
Layers:         {self.num_hidden_layers}
Attention:      {self.num_attention_heads} heads, {self.num_key_value_heads} KV heads (GQA {self.gqa_ratio}:1)
FFN size:       {self.intermediate_size}
Context:        {self.max_position_embeddings} tokens
Vocab:          {self.vocab_size}
Precision:      {self.torch_dtype}
{'='*50}
"""


# Pre-defined model configurations
MODEL_CONFIGS: Dict[str, Dict] = {
    # ~125M parameters (GPT-2 small equivalent)
    "125m": {
        "hidden_size": 768,
        "intermediate_size": 2048,  # ~2.7x
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,  # No GQA for small models
        "max_position_embeddings": 2048,
        "gradient_checkpointing": False,  # Not needed for small model
        "size_name": "125m",
    },

    # ~350M parameters (GPT-2 medium equivalent)
    "350m": {
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # No GQA
        "max_position_embeddings": 2048,
        "gradient_checkpointing": False,
        "size_name": "350m",
    },

    # ~1B parameters
    "1b": {
        "hidden_size": 2048,
        "intermediate_size": 5504,  # ~2.7x
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,  # 4:1 GQA
        "max_position_embeddings": 4096,
        "gradient_checkpointing": True,
        "size_name": "1b",
    },

    # ~3B parameters
    "3b": {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 26,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,  # 3:1 GQA
        "max_position_embeddings": 4096,
        "gradient_checkpointing": True,
        "size_name": "3b",
    },

    # ~7B parameters (LLaMA 7B equivalent)
    "7b": {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # 4:1 GQA
        "max_position_embeddings": 4096,
        "gradient_checkpointing": True,
        "size_name": "7b",
    },
}


def get_model_config(
    size: str = "7b",
    vocab_size: int = 50304,
    max_position_embeddings: Optional[int] = None,
    use_flash_attention: bool = True,
    torch_dtype: str = "bfloat16",
    **overrides
) -> ModelConfig:
    """Get a model configuration for a specific size.

    Args:
        size: Model size preset ("125m", "350m", "1b", "3b", "7b")
        vocab_size: Vocabulary size (should match tokenizer)
        max_position_embeddings: Override context length
        use_flash_attention: Enable Flash Attention 2
        torch_dtype: Data type ("bfloat16", "float16", "float32")
        **overrides: Additional overrides for any config field

    Returns:
        ModelConfig instance

    Example:
        >>> config = get_model_config("3b", max_position_embeddings=8192)
        >>> print(config.total_params_str)
        "3.21B"
    """
    size = size.lower().strip()

    if size not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model size '{size}'. Available: {available}")

    # Start with preset config
    config_dict = MODEL_CONFIGS[size].copy()

    # Apply common settings
    config_dict["vocab_size"] = vocab_size
    config_dict["use_flash_attention"] = use_flash_attention
    config_dict["torch_dtype"] = torch_dtype

    # Apply max_position_embeddings override if provided
    if max_position_embeddings is not None:
        config_dict["max_position_embeddings"] = max_position_embeddings

    # Apply any additional overrides
    config_dict.update(overrides)

    return ModelConfig(**config_dict)


def list_available_sizes() -> Dict[str, str]:
    """List available model sizes with descriptions.

    Returns:
        Dict mapping size name to parameter count
    """
    result = {}
    for size_name in MODEL_CONFIGS:
        config = get_model_config(size_name)
        result[size_name] = config.total_params_str
    return result


def get_optimal_batch_size(size: str, gpu_memory_gb: float = 80) -> Dict[str, int]:
    """Get recommended batch sizes for a model size and GPU memory.

    Args:
        size: Model size ("125m", "350m", "1b", "3b", "7b")
        gpu_memory_gb: GPU memory in GB

    Returns:
        Dict with batch_size and gradient_accumulation_steps
    """
    # Rough estimates for A100 80GB with gradient checkpointing
    # Format: (batch_size, grad_accum) for effective batch of ~32
    recommendations = {
        "125m": {"batch_size": 32, "gradient_accumulation_steps": 1},
        "350m": {"batch_size": 16, "gradient_accumulation_steps": 2},
        "1b": {"batch_size": 16, "gradient_accumulation_steps": 2},
        "3b": {"batch_size": 8, "gradient_accumulation_steps": 4},
        "7b": {"batch_size": 8, "gradient_accumulation_steps": 4},
    }

    size = size.lower().strip()
    if size not in recommendations:
        return {"batch_size": 4, "gradient_accumulation_steps": 8}

    rec = recommendations[size].copy()

    # Adjust for smaller GPUs
    if gpu_memory_gb < 40:
        rec["batch_size"] = max(1, rec["batch_size"] // 2)
        rec["gradient_accumulation_steps"] *= 2
    elif gpu_memory_gb < 24:
        rec["batch_size"] = max(1, rec["batch_size"] // 4)
        rec["gradient_accumulation_steps"] *= 4

    return rec


# Convenience function for backward compatibility
def get_7b_config(**overrides) -> ModelConfig:
    """Get the 7B model configuration (backward compatible)."""
    return get_model_config("7b", **overrides)


if __name__ == "__main__":
    # Print all available configurations
    print("Available Model Configurations")
    print("=" * 60)

    for size_name in MODEL_CONFIGS:
        config = get_model_config(size_name)
        print(config.summary())
