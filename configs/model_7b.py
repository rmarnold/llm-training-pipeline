from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 50304  # Match tokenizer (padded for efficiency)
    hidden_size: int = 4096
    intermediate_size: int = 11008  # SwiGLU FFN
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA (Grouped Query Attention)

    # Context & embeddings
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
    use_cache: bool = False
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Precision
    torch_dtype: str = "bfloat16"

    @property
    def total_params(self):
        """Approximate parameter count"""
        embed_params = self.vocab_size * self.hidden_size
        attention_params = self.num_hidden_layers * (
            4 * self.hidden_size * self.hidden_size  # Q,K,V,O projections
        )
        ffn_params = self.num_hidden_layers * (
            2 * self.hidden_size * self.intermediate_size  # up/down projections
        )
        norm_params = 2 * self.num_hidden_layers * self.hidden_size  # LayerNorm

        total = embed_params + attention_params + ffn_params + norm_params
        return total / 1e9  # Convert to billions
