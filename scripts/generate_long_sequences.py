"""Generate longer sequence training data (512 tokens) for maximum GPU utilization"""
import numpy as np
from transformers import AutoTokenizer
import random

def generate_long_sequences(seq_length=512, num_sequences=200):
    """Generate synthetic long sequences for GPU stress testing"""
    print(f"Generating {num_sequences} sequences of {seq_length} tokens...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")
    vocab_size = len(tokenizer)

    # Generate random token sequences
    # Use a mix of common tokens and varied content
    sequences = []

    for i in range(num_sequences):
        # Create varied sequences to simulate real text
        seq = []
        for _ in range(seq_length):
            # Bias towards lower token IDs (more common words)
            if random.random() < 0.7:
                token = random.randint(100, min(5000, vocab_size-1))
            else:
                token = random.randint(13, vocab_size-1)
            seq.append(token)
        sequences.append(seq)

    # Convert to numpy array
    data = np.array(sequences, dtype=np.int32)

    # Save
    output_path = "data/packed/pretrain_long.npy"
    np.save(output_path, data)

    print(f"✓ Generated {len(data)} sequences")
    print(f"  Shape: {data.shape}")
    print(f"  Token range: {data.min()} - {data.max()}")
    print(f"  Saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    generate_long_sequences(seq_length=512, num_sequences=200)
