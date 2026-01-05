"""Simplified tokenization for demo"""
import os
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

def create_simple_tokenizer():
    """Use a pre-existing small tokenizer for demo"""
    print("Loading GPT-2 tokenizer for demo...")

    # Use GPT-2 tokenizer as base (it's small and works)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|bos|>",
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>"]
    }

    tokenizer.add_special_tokens(special_tokens)

    # Save tokenizer
    os.makedirs("configs/tokenizer", exist_ok=True)
    tokenizer.save_pretrained("configs/tokenizer")

    print(f"✓ Tokenizer ready: vocab_size={len(tokenizer)}")
    return tokenizer

def simple_pack_sequences():
    """Simple sequence packing for demo"""
    print("\nPacking sequences...")

    # Load processed data
    if not os.path.exists("data/processed/pretraining_demo_clean.parquet"):
        # If cleaning wasn't run, use raw data
        print("  Using raw data (skipping cleaning for demo)")
        df = pd.read_parquet("data/raw/pretraining_demo.parquet")
    else:
        df = pd.read_parquet("data/processed/pretraining_demo_clean.parquet")

    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    ctx_len = 128  # Short context for demo
    packed_dataset = []

    print(f"  Packing {len(df)} documents into {ctx_len}-token sequences...")

    for text in df['text']:
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=ctx_len, truncation=True)

        # Pad to ctx_len
        if len(tokens) < ctx_len:
            tokens = tokens + [tokenizer.pad_token_id] * (ctx_len - len(tokens))

        packed_dataset.append(tokens)

    # Save packed data
    os.makedirs("data/packed", exist_ok=True)
    packed_array = np.array(packed_dataset, dtype=np.int32)
    np.save("data/packed/pretrain_demo.npy", packed_array)

    print(f"  ✓ Saved {len(packed_dataset)} packed sequences")
    print(f"    Shape: {packed_array.shape}")

    return packed_array

if __name__ == "__main__":
    print("Demo Tokenization Pipeline\n")
    create_simple_tokenizer()
    simple_pack_sequences()
    print("\n✓ Tokenization complete!")
