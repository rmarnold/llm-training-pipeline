from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Tiny model for demo/testing - ~124M parameters (GPT-2 small size)"""
    # Architecture
    vocab_size: int = 50304  # Match GPT-2 tokenizer + special tokens (padded to multiple of 64)
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12  # No GQA for demo

    # Context & embeddings
    max_position_embeddings: int = 512  # Shorter context
    rope_theta: float = 10000.0

    # Regularization
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # Activations
    hidden_act: str = "silu"

    # Training efficiency
    use_cache: bool = False
    use_flash_attention: bool = False  # Disable for CPU demo
    gradient_checkpointing: bool = False  # Disable for speed

    # Precision
    torch_dtype: str = "float32"  # Use float32 for CPU

    @property
    def total_params(self):
        """Approximate parameter count"""
        embed_params = self.vocab_size * self.hidden_size
        attention_params = self.num_hidden_layers * (
            4 * self.hidden_size * self.hidden_size
        )
        ffn_params = self.num_hidden_layers * (
            2 * self.hidden_size * self.intermediate_size
        )
        norm_params = 2 * self.num_hidden_layers * self.hidden_size

        total = embed_params + attention_params + ffn_params + norm_params
        return total / 1e6  # Convert to millions
