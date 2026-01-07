"""Download and prepare production-scale training data"""
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
from tqdm import tqdm

def download_and_prepare_production_data(
    max_samples=100000,  # 100K samples for production
    seq_length=512,
    output_file="data/packed/production_pretrain.npy"
):
    """Download WikiText-103 and tokenize for production training"""

    print("="*60)
    print("PRODUCTION DATA PREPARATION")
    print("="*60)
    print(f"Target samples: {max_samples:,}")
    print(f"Sequence length: {seq_length}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Download WikiText-103 (larger production dataset)
    print("\nDownloading WikiText-103 dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        print(f"✓ Downloaded {len(dataset):,} documents")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to WikiText-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        print(f"✓ Downloaded {len(dataset):,} documents")

    # Tokenize and pack sequences
    print(f"\nTokenizing and packing into {seq_length}-token sequences...")
    all_tokens = []

    for i, example in enumerate(tqdm(dataset, desc="Processing")):
        text = example['text'].strip()
        if len(text) < 10:  # Skip very short texts
            continue

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

        # Stop if we have enough tokens
        if len(all_tokens) >= max_samples * seq_length:
            print(f"\n✓ Collected enough tokens ({len(all_tokens):,})")
            break

    print(f"Total tokens collected: {len(all_tokens):,}")

    # Pack into sequences
    print(f"\nPacking into {seq_length}-token sequences...")
    sequences = []
    total_possible = min(max_samples, (len(all_tokens) - seq_length) // seq_length)

    for i in tqdm(range(0, len(all_tokens) - seq_length, seq_length),
                  desc="Packing sequences", total=total_possible, unit="seq"):
        seq = all_tokens[i:i+seq_length]
        if len(seq) == seq_length:
            sequences.append(seq)

        if len(sequences) >= max_samples:
            break

    # Convert to numpy array
    data = np.array(sequences, dtype=np.int32)

    # Verify token range
    if data.max() >= len(tokenizer):
        print(f"⚠️  Warning: Token range exceeds vocab size!")
        print(f"   Max token: {data.max()}, Vocab size: {len(tokenizer)}")
        # Clip to vocab size
        data = np.clip(data, 0, len(tokenizer) - 1)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, data)

    print(f"\n{'='*60}")
    print("PRODUCTION DATA READY")
    print(f"{'='*60}")
    print(f"✓ Samples: {len(data):,}")
    print(f"✓ Shape: {data.shape}")
    print(f"✓ Token range: {data.min()} - {data.max()}")
    print(f"✓ Memory: {data.nbytes / 1024**2:.1f} MB")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*60}")

    return output_file

if __name__ == "__main__":
    # Prepare 100K samples for production training
    download_and_prepare_production_data(max_samples=100000, seq_length=512)
