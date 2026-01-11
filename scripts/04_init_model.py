#!/usr/bin/env python3
"""Initialize a LLaMA-style model with configurable size.

This script creates and saves an initial model checkpoint that can be used
for pretraining. Supports multiple model sizes from 125M to 7B parameters.

Usage:
    # Initialize default 7B model
    python scripts/04_init_model.py

    # Initialize a 1B model
    python scripts/04_init_model.py --size 1b

    # Initialize with custom context length
    python scripts/04_init_model.py --size 3b --context-length 8192

    # List available sizes
    python scripts/04_init_model.py --list-sizes
"""

import argparse
import os
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
from model_configs import get_model_config, list_available_sizes, get_optimal_batch_size


def initialize_model(
    size: str = "7b",
    output_dir: str = "checkpoints/init",
    context_length: int = None,
    vocab_size: int = 50304,
    use_flash_attention: bool = True,
    dtype: str = "bfloat16",
) -> None:
    """Initialize and save a model checkpoint.

    Args:
        size: Model size preset ("125m", "350m", "1b", "3b", "7b")
        output_dir: Directory to save the checkpoint
        context_length: Override max context length
        vocab_size: Vocabulary size (should match tokenizer)
        use_flash_attention: Enable Flash Attention 2
        dtype: Data type ("bfloat16", "float16", "float32")
    """
    # Get model configuration
    config = get_model_config(
        size=size,
        vocab_size=vocab_size,
        max_position_embeddings=context_length,
        use_flash_attention=use_flash_attention,
        torch_dtype=dtype,
    )

    print(config.summary())

    # Determine torch dtype
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

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
        torch_dtype=torch_dtype,
    )

    print(f"Initializing {config.total_params_str} model...")

    # Initialize model with random weights
    model = LlamaForCausalLM(hf_config)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save checkpoint
    print(f"Saving checkpoint to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save model size info for later use
    size_info_path = os.path.join(output_dir, "model_size.txt")
    with open(size_info_path, "w") as f:
        f.write(f"size={size}\n")
        f.write(f"params={config.total_params:.3f}B\n")
        f.write(f"hidden_size={config.hidden_size}\n")
        f.write(f"num_layers={config.num_hidden_layers}\n")
        f.write(f"num_heads={config.num_attention_heads}\n")
        f.write(f"num_kv_heads={config.num_key_value_heads}\n")
        f.write(f"context_length={config.max_position_embeddings}\n")

    # Get batch size recommendations
    batch_rec = get_optimal_batch_size(size)

    print(f"\nModel initialized successfully!")
    print(f"  Checkpoint: {output_dir}")
    print(f"  Parameters: {config.total_params_str}")
    print(f"  Architecture: {config.num_hidden_layers}L / {config.hidden_size}H / {config.num_attention_heads}A")
    print(f"  GQA ratio: {config.gqa_ratio}:1 ({config.num_key_value_heads} KV heads)")
    print(f"  Context: {config.max_position_embeddings} tokens")
    print(f"\nRecommended training settings (A100 80GB):")
    print(f"  batch_size: {batch_rec['batch_size']}")
    print(f"  gradient_accumulation_steps: {batch_rec['gradient_accumulation_steps']}")
    print(f"  effective_batch_size: {batch_rec['batch_size'] * batch_rec['gradient_accumulation_steps']}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a LLaMA-style model with configurable size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/04_init_model.py --size 7b          # Full 7B model
  python scripts/04_init_model.py --size 1b          # Smaller 1B model
  python scripts/04_init_model.py --size 3b --context-length 8192
  python scripts/04_init_model.py --list-sizes       # Show all sizes
        """
    )

    parser.add_argument(
        "--size", "-s",
        type=str,
        default="7b",
        choices=["125m", "350m", "1b", "3b", "7b"],
        help="Model size preset (default: 7b)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="checkpoints/init",
        help="Output directory for checkpoint (default: checkpoints/init)"
    )
    parser.add_argument(
        "--context-length", "-c",
        type=int,
        default=None,
        help="Maximum context length (default: size-dependent)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50304,
        help="Vocabulary size (default: 50304, should match tokenizer)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights (default: bfloat16)"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2"
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="List available model sizes and exit"
    )

    args = parser.parse_args()

    # List sizes and exit
    if args.list_sizes:
        print("\nAvailable Model Sizes")
        print("=" * 40)
        sizes = list_available_sizes()
        for name, params in sizes.items():
            config = get_model_config(name)
            print(f"  {name:>5}: {params:>6} - {config.num_hidden_layers}L/{config.hidden_size}H/{config.num_attention_heads}A")
        print("\nUse --size <name> to select a model size")
        return

    # Initialize model
    initialize_model(
        size=args.size,
        output_dir=args.output_dir,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        use_flash_attention=not args.no_flash_attention,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
