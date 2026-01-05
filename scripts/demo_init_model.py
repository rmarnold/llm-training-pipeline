"""Initialize tiny model for demo/testing"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sys
import os

# Add configs to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))
from model_tiny_demo import ModelConfig

def initialize_model():
    config = ModelConfig()

    print("Initializing tiny demo model...")
    print(f"  Parameters: ~{config.total_params:.1f}M")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Context length: {config.max_position_embeddings}")

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
        torch_dtype=torch.float32,
    )

    # Initialize model
    print("  Creating model...")
    model = LlamaForCausalLM(hf_config)

    # Create checkpoint directory
    os.makedirs("checkpoints/demo_init", exist_ok=True)

    # Save initial checkpoint
    print("  Saving checkpoint...")
    model.save_pretrained("checkpoints/demo_init", safe_serialization=True)

    print(f"\n✓ Model initialized: {config.total_params:.1f}M parameters")
    print(f"  Checkpoint saved to: checkpoints/demo_init/")

    # Test forward pass
    print("\n  Testing forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        outputs = model(input_ids)
    print(f"  ✓ Forward pass successful! Output shape: {outputs.logits.shape}")

    return model

if __name__ == "__main__":
    initialize_model()
