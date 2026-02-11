"""GPT-OSS 20B MoE model configuration.

GPT-OSS 20B is OpenAI's first open-weight MoE model (Apache 2.0):
- Total parameters: 20.9B
- Active parameters per token: 3.6B (top-4 of 32 experts)
- Context: 128K native
- Weight format: MXFP4 (MoE layers)
- Attention: Grouped multi-query (group size 8)
- Positional encoding: RoPE
- Tokenizer: o200k_harmony (201K vocab)
- Chat format: Harmony (mandatory)

Usage:
    from configs.gpt_oss_20b import GPT_OSS_20B_CONFIG, MoEModelConfig
    print(GPT_OSS_20B_CONFIG.summary())
"""

from dataclasses import dataclass


@dataclass
class MoEModelConfig:
    """Mixture-of-Experts model configuration for GPT-OSS family."""

    # Core architecture
    vocab_size: int = 200064  # o200k_harmony tokenizer (padded for efficiency)
    hidden_size: int = 2880  # Residual stream dimension
    intermediate_size: int = 7680  # Expert FFN dimension
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 3  # Grouped multi-query (group size 8)

    # MoE specific
    num_experts: int = 32  # Total experts per MoE layer
    num_experts_per_tok: int = 4  # Active experts per token (top-4 routing)
    router_aux_loss_coef: float = 0.02  # Load balancing loss
    moe_layer_frequency: int = 1  # Every layer is MoE (alternating dense+sparse attention)

    # Context & positional encoding
    max_position_embeddings: int = 131072  # 128K native context
    rope_theta: float = 500000.0  # RoPE base for long context

    # Attention pattern
    attention_pattern: str = "alternating"  # Dense + sparse banded (bandwidth 128)
    banded_attention_bandwidth: int = 128

    # Regularization
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # Activations
    hidden_act: str = "silu"  # Gated SwiGLU

    # Training efficiency
    use_cache: bool = False
    gradient_checkpointing: bool = True

    # Precision
    torch_dtype: str = "bfloat16"
    native_quantization: str = "mxfp4"  # Base weights in MXFP4

    # Tokenizer
    tokenizer_name: str = "o200k_harmony"

    # Metadata
    model_name: str = "gpt-oss-20b"
    hf_model_id: str = "openai/gpt-oss-20b"
    license: str = "Apache-2.0"

    @property
    def total_params(self) -> float:
        """Approximate total parameter count in billions."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size

        # Attention per layer (shared across all tokens)
        head_dim = self.hidden_size // self.num_attention_heads
        q_params = self.hidden_size * self.hidden_size
        kv_params = 2 * self.hidden_size * (head_dim * self.num_key_value_heads)
        o_params = self.hidden_size * self.hidden_size
        attention_params = self.num_hidden_layers * (q_params + kv_params + o_params)

        # MoE FFN: each expert has gate + up + down projections (SwiGLU)
        expert_params = self.num_hidden_layers * self.num_experts * (
            3 * self.hidden_size * self.intermediate_size
        )

        # Router: projects hidden_size → num_experts
        router_params = self.num_hidden_layers * (
            self.hidden_size * self.num_experts
        )

        # LayerNorm (RMSNorm)
        norm_params = self.num_hidden_layers * 2 * self.hidden_size + self.hidden_size

        # LM head (usually tied with embeddings)
        lm_head_params = self.hidden_size * self.vocab_size

        total = embed_params + attention_params + expert_params + router_params + norm_params + lm_head_params
        return total / 1e9

    @property
    def active_params(self) -> float:
        """Parameters active per forward pass (with top-k routing) in billions."""
        embed_params = self.vocab_size * self.hidden_size

        head_dim = self.hidden_size // self.num_attention_heads
        q_params = self.hidden_size * self.hidden_size
        kv_params = 2 * self.hidden_size * (head_dim * self.num_key_value_heads)
        o_params = self.hidden_size * self.hidden_size
        attention_params = self.num_hidden_layers * (q_params + kv_params + o_params)

        # Only num_experts_per_tok experts active
        active_expert_params = self.num_hidden_layers * self.num_experts_per_tok * (
            3 * self.hidden_size * self.intermediate_size
        )

        router_params = self.num_hidden_layers * self.hidden_size * self.num_experts
        norm_params = self.num_hidden_layers * 2 * self.hidden_size + self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        total = embed_params + attention_params + active_expert_params + router_params + norm_params + lm_head_params
        return total / 1e9

    @property
    def total_params_str(self) -> str:
        """Human-readable total parameter count."""
        return f"{self.total_params:.1f}B"

    @property
    def active_params_str(self) -> str:
        """Human-readable active parameter count."""
        return f"{self.active_params:.1f}B"

    def summary(self) -> str:
        return f"""
Model Configuration: {self.model_name.upper()}
{'='*60}
Total Parameters:   {self.total_params_str} ({self.total_params:.2f}B)
Active Parameters:  {self.active_params_str} ({self.active_params:.2f}B)
Hidden size:        {self.hidden_size}
Layers:             {self.num_hidden_layers}
Attention:          {self.num_attention_heads} heads, {self.num_key_value_heads} KV heads
MoE:                {self.num_experts} experts, top-{self.num_experts_per_tok} routing
Expert FFN size:    {self.intermediate_size}
Context:            {self.max_position_embeddings // 1024}K tokens
Vocab:              {self.vocab_size} ({self.tokenizer_name})
Precision:          {self.torch_dtype} (base: {self.native_quantization})
License:            {self.license}
{'='*60}
"""

    def get_lora_target_modules(self) -> list[str]:
        """Return recommended LoRA target modules for this architecture.

        Targets attention + expert FFN layers. Does NOT target the
        router/gate layer (frozen during fine-tuning).
        """
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # Expert FFN (SwiGLU)
        ]


# Default configuration matching published GPT-OSS 20B specs
GPT_OSS_20B_CONFIG = MoEModelConfig()


if __name__ == "__main__":
    config = GPT_OSS_20B_CONFIG
    print(config.summary())
    print(f"LoRA target modules: {config.get_lora_target_modules()}")
