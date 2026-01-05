import os
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

def train_tokenizer():
    """Train custom BPE tokenizer on mixed data"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|bos|>", "<|eos|>",
                        "<|user|>", "<|assistant|>", "<|system|>"]
    )

    # Gather text files
    files = [f"data/processed/{f}" for f in os.listdir("data/processed")]

    # Train
    tokenizer.train(files, trainer)
    tokenizer.save("configs/tokenizer.json")

    # Convert to HuggingFace format
    from transformers import PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="configs/tokenizer.json")
    fast_tokenizer.pad_token = "<|pad|>"
    fast_tokenizer.eos_token = "<|endoftext|>"
    fast_tokenizer.save_pretrained("configs/tokenizer")

    print(f"✓ Tokenizer trained: vocab_size={len(fast_tokenizer)}")

def pack_sequences(max_seq_len=2048, context_lengths=[512, 1024, 2048]):
    """Pack tokenized sequences for efficient training"""
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    for ctx_len in context_lengths:
        print(f"Packing sequences for context length {ctx_len}...")

        packed_dataset = []
        current_pack = []
        current_length = 0

        for file in os.listdir("data/processed"):
            df = pd.read_parquet(f"data/processed/{file}")

            for text in df['text']:
                tokens = tokenizer.encode(text, add_special_tokens=True)

                if current_length + len(tokens) <= ctx_len:
                    current_pack.extend(tokens)
                    current_length += len(tokens)
                else:
                    # Pad and save current pack
                    if current_length > 0:
                        current_pack.extend([tokenizer.pad_token_id] * (ctx_len - current_length))
                        packed_dataset.append(current_pack)

                    # Start new pack
                    current_pack = tokens[:ctx_len]
                    current_length = len(current_pack)

        # Save packed data
        np.save(f"data/packed/pretrain_ctx{ctx_len}.npy", np.array(packed_dataset, dtype=np.int32))
        print(f"  Saved {len(packed_dataset)} packed sequences")

if __name__ == "__main__":
    os.makedirs("data/packed", exist_ok=True)
    train_tokenizer()
    pack_sequences()
