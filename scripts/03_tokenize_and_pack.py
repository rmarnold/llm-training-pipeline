import os
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Chunk size for streaming - save every N sequences to avoid OOM
CHUNK_SIZE = 50000

# Batch size for tokenization (much faster than one-by-one)
TOKENIZE_BATCH_SIZE = 10000


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

def pack_sequences_streaming(ctx_len: int, tokenizer, input_dir: str = "data/processed", output_dir: str = "data/packed", num_workers: int = None):
    """Pack tokenized sequences with streaming to avoid OOM.

    Saves chunks to disk as we go instead of accumulating everything in memory.
    Final output is a HuggingFace Dataset for compatibility with Trainer.

    Uses batch tokenization for 10-50x speedup over single-document tokenization.
    """
    chunk_dir = Path(output_dir) / f"chunks_ctx{ctx_len}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing chunks (resume support)
    existing_chunks = sorted(chunk_dir.glob("chunk_*.npy"))
    if existing_chunks:
        print(f"  Found {len(existing_chunks)} existing chunks, resuming...")

    current_pack = []
    current_length = 0
    packed_buffer = []
    chunk_idx = len(existing_chunks)
    total_sequences = 0

    # Use multiple workers for tokenization if available
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.parquet')])

    for file in files:
        print(f"  Processing {file}...")
        df = pd.read_parquet(f"{input_dir}/{file}")
        texts = df['text'].tolist()
        total_docs = len(texts)

        # Process in batches for much faster tokenization
        print(f"    Tokenizing {total_docs:,} documents in batches of {TOKENIZE_BATCH_SIZE}...")

        processed_docs = 0
        for batch_start in tqdm(range(0, total_docs, TOKENIZE_BATCH_SIZE),
                                desc="    Tokenizing batches",
                                unit="batch"):
            batch_end = min(batch_start + TOKENIZE_BATCH_SIZE, total_docs)
            batch_texts = texts[batch_start:batch_end]

            # Batch tokenization is MUCH faster (10-50x)
            batch_encodings = tokenizer(
                batch_texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            # Pack each document's tokens
            for tokens in batch_encodings['input_ids']:
                if current_length + len(tokens) <= ctx_len:
                    current_pack.extend(tokens)
                    current_length += len(tokens)
                else:
                    # Pad and save current pack
                    if current_length > 0:
                        current_pack.extend([tokenizer.pad_token_id] * (ctx_len - current_length))
                        packed_buffer.append(current_pack)

                        # Save chunk when buffer is full
                        if len(packed_buffer) >= CHUNK_SIZE:
                            chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.npy"
                            np.save(chunk_path, np.array(packed_buffer, dtype=np.int32))
                            print(f"\n    [Saved chunk {chunk_idx}: {len(packed_buffer)} sequences]")
                            total_sequences += len(packed_buffer)
                            packed_buffer = []
                            chunk_idx += 1

                    # Start new pack with truncated tokens if needed
                    current_pack = tokens[:ctx_len]
                    current_length = len(current_pack)

            processed_docs += len(batch_texts)

    # Save final pack and remaining buffer
    if current_length > 0:
        current_pack.extend([tokenizer.pad_token_id] * (ctx_len - current_length))
        packed_buffer.append(current_pack)

    if packed_buffer:
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.npy"
        np.save(chunk_path, np.array(packed_buffer, dtype=np.int32))
        print(f"    [Saved final chunk {chunk_idx}: {len(packed_buffer)} sequences]")
        total_sequences += len(packed_buffer)

    return chunk_dir, total_sequences


def merge_chunks_to_dataset(chunk_dir: Path, output_path: str, ctx_len: int, keep_chunks: bool = False):
    """Merge chunk files into a HuggingFace Dataset.

    Args:
        chunk_dir: Directory containing .npy chunk files
        output_path: Output path for HuggingFace dataset
        ctx_len: Context length (for logging)
        keep_chunks: If True, don't delete chunk files after merge (for recovery)
    """
    print(f"  Merging chunks into HuggingFace Dataset...")

    chunk_files = sorted(chunk_dir.glob("chunk_*.npy"))
    if not chunk_files:
        print(f"  No chunks found in {chunk_dir}")
        return None

    # Load chunks one at a time and concatenate
    all_input_ids = []
    all_attention_mask = []

    for chunk_path in tqdm(chunk_files, desc="    Loading chunks"):
        chunk_data = np.load(chunk_path)
        for seq in chunk_data:
            all_input_ids.append(seq.tolist())
            # Attention mask: 1 for real tokens, 0 for padding
            all_attention_mask.append([1 if t != 0 else 0 for t in seq])

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'labels': all_input_ids,  # For causal LM, labels = input_ids
    })

    # Save to disk
    dataset.save_to_disk(output_path)
    print(f"    Saved dataset with {len(dataset)} sequences to {output_path}")

    # Cleanup chunks (unless keep_chunks is True)
    if not keep_chunks:
        for chunk_path in chunk_files:
            chunk_path.unlink()
        chunk_dir.rmdir()
        print(f"    Cleaned up chunk files")
    else:
        print(f"    Kept chunk files for recovery (--keep-chunks)")

    return dataset


def recover_from_chunks(output_dir: str, ctx_len: int = 2048):
    """Recover a HuggingFace dataset from existing .npy chunk files.

    Use this when tokenization completed but dataset creation was interrupted.
    """
    import shutil

    chunk_dir = Path(output_dir) / f"chunks_ctx{ctx_len}"
    train_path = f"{output_dir}/train"
    val_path = f"{output_dir}/val"

    # Check for chunks
    if not chunk_dir.exists():
        print(f"No chunk directory found at {chunk_dir}")
        print(f"Looking for alternative chunk locations...")

        # Check for chunks in parent directory
        for potential_dir in Path(output_dir).glob("chunks_*"):
            if list(potential_dir.glob("chunk_*.npy")):
                chunk_dir = potential_dir
                print(f"Found chunks at {chunk_dir}")
                break
        else:
            print("No chunks found. Need to re-run tokenization from scratch.")
            return False

    chunk_files = sorted(chunk_dir.glob("chunk_*.npy"))
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        return False

    print(f"Found {len(chunk_files)} chunk files to recover")

    # Remove corrupted dataset if exists
    if os.path.exists(train_path):
        print(f"Removing corrupted dataset at {train_path}")
        shutil.rmtree(train_path)
    if os.path.exists(val_path):
        print(f"Removing old validation set at {val_path}")
        shutil.rmtree(val_path)

    # Merge chunks into dataset (keep chunks in case it fails again)
    dataset = merge_chunks_to_dataset(chunk_dir, train_path, ctx_len, keep_chunks=True)

    if dataset is None:
        print("Failed to merge chunks")
        return False

    # Create validation split
    print(f"Creating validation split...")
    full_dataset = Dataset.load_from_disk(train_path)
    val_size = max(1, min(1000, len(full_dataset) // 100))
    splits = full_dataset.train_test_split(test_size=val_size, seed=42)

    # Save to temp paths first
    train_temp = f"{output_dir}/train_temp"
    splits['train'].save_to_disk(train_temp)
    splits['test'].save_to_disk(val_path)

    # Delete original and rename temp
    del full_dataset, splits
    shutil.rmtree(train_path)
    shutil.move(train_temp, train_path)

    # Now clean up chunks
    print("Cleaning up chunk files...")
    for chunk_path in chunk_files:
        chunk_path.unlink()
    try:
        chunk_dir.rmdir()
    except OSError:
        pass  # Directory might not be empty

    # Verify
    final_train = Dataset.load_from_disk(train_path)
    final_val = Dataset.load_from_disk(val_path)
    print(f"\n✓ Recovery complete!")
    print(f"  Train: {len(final_train)} sequences")
    print(f"  Val: {len(final_val)} sequences")

    return True


def pack_sequences(context_lengths=[2048], input_dir="data/processed", output_dir="data/packed"):
    """Pack tokenized sequences for efficient training.

    Uses streaming approach to avoid OOM with large datasets.
    Output format: HuggingFace Dataset (compatible with Trainer)
    """
    # Check if input directory exists and has data
    if not os.path.exists(input_dir):
        print(f"\n✗ ERROR: Input directory not found: {input_dir}")
        print("  Run the data cleaning step first (02_clean_deduplicate_optimized.py)")
        return

    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"\n✗ ERROR: No parquet files found in {input_dir}")
        print("  Run the data cleaning step first (02_clean_deduplicate_optimized.py)")
        return

    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")
    os.makedirs(output_dir, exist_ok=True)

    for ctx_len in context_lengths:
        print(f"\nPacking sequences for context length {ctx_len}...")

        # Check if already done
        train_path = f"{output_dir}/train"
        if os.path.exists(train_path):
            print(f"  {train_path} already exists, skipping...")
            continue

        # Stream pack sequences
        chunk_dir, total_seqs = pack_sequences_streaming(
            ctx_len=ctx_len,
            tokenizer=tokenizer,
            input_dir=input_dir,
            output_dir=output_dir
        )

        # Check if any sequences were created
        if total_seqs == 0:
            print(f"  ✗ No sequences created. Check if input data is valid.")
            continue

        # Merge into final dataset
        dataset = merge_chunks_to_dataset(chunk_dir, train_path, ctx_len)

        # Check if merge was successful
        if dataset is None:
            print(f"  ✗ Failed to merge chunks. Check the errors above.")
            continue

        # Create small validation set (last 1% or 1000 sequences)
        print(f"  Creating validation split...")
        full_dataset = Dataset.load_from_disk(train_path)
        val_size = max(1, min(1000, len(full_dataset) // 100))
        splits = full_dataset.train_test_split(test_size=val_size, seed=42)

        # Save to temp paths first (can't overwrite loaded dataset)
        train_temp = f"{output_dir}/train_temp"
        val_path = f"{output_dir}/val"

        splits['train'].save_to_disk(train_temp)
        splits['test'].save_to_disk(val_path)

        # Delete original and rename temp
        import shutil
        del full_dataset, splits  # Release the loaded dataset
        shutil.rmtree(train_path)
        shutil.move(train_temp, train_path)

        # Reload to get counts
        final_train = Dataset.load_from_disk(train_path)
        final_val = Dataset.load_from_disk(val_path)
        print(f"  Train: {len(final_train)} sequences")
        print(f"  Val: {len(final_val)} sequences")

    print(f"\n✓ Packing complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tokenize and pack sequences')
    parser.add_argument('--ctx-len', type=int, default=2048, help='Context length (default: 2048)')
    parser.add_argument('--input-dir', type=str, default='data/processed', help='Input directory')
    parser.add_argument('--output-dir', type=str, default='data/packed', help='Output directory')
    parser.add_argument('--skip-tokenizer', action='store_true', help='Skip tokenizer training')
    parser.add_argument('--recover', action='store_true',
                        help='Recover from existing .npy chunks (use if dataset creation was interrupted)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-tokenization even if output exists')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Recovery mode - rebuild dataset from existing chunks
    if args.recover:
        print("=" * 50)
        print("RECOVERY MODE: Rebuilding dataset from chunks")
        print("=" * 50)
        success = recover_from_chunks(args.output_dir, args.ctx_len)
        if success:
            print("\nRecovery successful! You can now run pretraining.")
        else:
            print("\nRecovery failed. You may need to re-run tokenization with --force")
        exit(0 if success else 1)

    # Force mode - delete existing output and re-run
    if args.force:
        import shutil
        train_path = f"{args.output_dir}/train"
        val_path = f"{args.output_dir}/val"
        if os.path.exists(train_path):
            print(f"Removing existing {train_path}")
            shutil.rmtree(train_path)
        if os.path.exists(val_path):
            print(f"Removing existing {val_path}")
            shutil.rmtree(val_path)

    if not args.skip_tokenizer and not os.path.exists("configs/tokenizer"):
        train_tokenizer()
    else:
        print("Tokenizer already exists, skipping training...")

    pack_sequences(
        context_lengths=[args.ctx_len],
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
