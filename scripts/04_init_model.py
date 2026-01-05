import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sys
sys.path.append('configs')
from model_7b import ModelConfig

def initialize_model():
    config = ModelConfig()

    # Create HuggingFace config
    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_dropout=config.attention_dropout,
        hidden_act=config.hidden_act,
        torch_dtype=torch.bfloat16,
    )

    # Initialize model
    model = LlamaForCausalLM(hf_config)

    # Save initial checkpoint
    model.save_pretrained("checkpoints/init", safe_serialization=True)

    print(f"✓ Model initialized: {config.total_params:.2f}B parameters")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Max context: {config.max_position_embeddings}")

    return model

if __name__ == "__main__":
    initialize_model()
