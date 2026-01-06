"""Optimized data cleaning with caching, checkpoints, and GPU acceleration."""
from __future__ import annotations

import os
import json
import hashlib
import pandas as pd
from datasketch import MinHash, MinHashLSH
from ftfy import fix_text
from detoxify import Detoxify
import re
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count
from functools import partial


def _clean_text_worker(text: str) -> str:
    """Worker function for parallel text cleaning."""
    if pd.isna(text) or text is None:
        return ""
    text = fix_text(str(text))
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    return text.strip()


def parallel_clean_texts_streaming(
    texts: list[str],
    n_workers: int = None,
    chunk_callback=None,
    chunk_size: int = 500000
):
    """Clean texts in parallel with streaming output to avoid memory buildup.

    Instead of accumulating all results in memory, yields chunks of results
    and calls chunk_callback to save them incrementally.

    Args:
        texts: List of texts to clean
        n_workers: Number of CPU workers (default: cpu_count // 2)
        chunk_callback: Function to call with each chunk of results
        chunk_size: Number of results per chunk (default: 500K)

    Yields:
        Chunks of cleaned text results
    """
    if n_workers is None:
        # With streaming, we can use more workers since results don't accumulate
        # Each worker adds ~500MB-1GB overhead, but that's manageable
        n_workers = max(1, cpu_count() // 2)

    n_workers = min(n_workers, cpu_count())
    n_texts = len(texts)

    if n_texts == 0:
        return

    pool_chunk_size = 5000
    print(f"    Using {n_workers} CPU workers (pool_chunk={pool_chunk_size:,}, save_chunk={chunk_size:,})...")

    current_chunk = []
    chunk_idx = 0

    with Pool(processes=n_workers) as pool:
        with tqdm(total=n_texts, desc="    Cleaning text") as pbar:
            for cleaned in pool.imap(_clean_text_worker, texts, chunksize=pool_chunk_size):
                current_chunk.append(cleaned)
                pbar.update(1)

                # When chunk is full, save it and clear memory
                if len(current_chunk) >= chunk_size:
                    if chunk_callback:
                        chunk_callback(current_chunk, chunk_idx)
                    yield current_chunk
                    chunk_idx += 1
                    current_chunk = []  # Clear memory

    # Yield remaining results
    if current_chunk:
        if chunk_callback:
            chunk_callback(current_chunk, chunk_idx)
        yield current_chunk


def parallel_clean_texts(
    texts: list[str],
    n_workers: int = None,
    checkpoint_callback=None,
    checkpoint_interval: int = 500000
) -> list[str]:
    """Clean texts in parallel with optional incremental checkpointing.

    Note: For large datasets (>2M docs), use parallel_clean_texts_streaming instead
    to avoid memory issues.
    """
    if n_workers is None:
        # Default to quarter of CPU cores to avoid memory issues
        n_workers = max(1, cpu_count() // 4)

    n_workers = min(n_workers, cpu_count())
    n_texts = len(texts)

    if n_workers <= 1 or n_texts < 1000:
        return [_clean_text_worker(t) for t in tqdm(texts, desc="    Cleaning text")]

    chunk_size = 5000
    print(f"    Using {n_workers} CPU workers (chunk_size={chunk_size:,})...")

    results = []
    last_checkpoint = 0

    with Pool(processes=n_workers) as pool:
        with tqdm(total=n_texts, desc="    Cleaning text") as pbar:
            for cleaned in pool.imap(_clean_text_worker, texts, chunksize=chunk_size):
                results.append(cleaned)
                pbar.update(1)

                # Incremental checkpoint every N items
                if checkpoint_callback and len(results) - last_checkpoint >= checkpoint_interval:
                    checkpoint_callback(results, len(results))
                    last_checkpoint = len(results)

    return results


class DataCleaner:
    """GPU-accelerated data cleaner with batched processing."""

    # Class-level model cache to avoid reloading
    _model_cache: dict = {}

    def __init__(self, toxicity_threshold: float = 0.7, use_gpu: bool = True, batch_size: int = 128):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.toxicity_threshold = toxicity_threshold
        self.batch_size = batch_size
        self.lsh = MinHashLSH(threshold=0.85, num_perm=128)

        # Use cached model if available (avoid re-downloading)
        cache_key = f"detoxify_{self.device}"
        if cache_key not in DataCleaner._model_cache:
            print(f"Loading toxicity model on {self.device}...")
            DataCleaner._model_cache[cache_key] = Detoxify('original', device=self.device)
        self.toxicity_model = DataCleaner._model_cache[cache_key]

    def clean_text(self, text: str) -> str:
        """Clean a single text document."""
        if pd.isna(text):
            return ""
        text = fix_text(str(text))
        text = re.sub(r'\s+', ' ', text)
        # Remove PII patterns
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        return text.strip()

    def filter_quality_batch(self, texts: list[str], min_words: int = 50, max_words: int = 10000) -> list[bool]:
        """Batch quality filtering."""
        results = []
        for text in texts:
            word_count = len(text.split())
            if word_count < min_words or word_count > max_words:
                results.append(False)
                continue
            unique_chars = len(set(text.lower()))
            if unique_chars < 20:
                results.append(False)
                continue
            results.append(True)
        return results

    def is_toxic_batch(self, texts: list[str], show_progress: bool = True) -> list[bool]:
        """Batch toxicity detection with progress bar."""
        if not texts:
            return []

        all_results = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="    Toxicity check", unit="batch")

        for i in iterator:
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                results = self.toxicity_model.predict(batch)
            for j in range(len(batch)):
                is_toxic = any(results[key][j] > self.toxicity_threshold for key in results.keys())
                all_results.append(is_toxic)

        return all_results

    def compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash for deduplication."""
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    def deduplicate_batch(self, texts: list[str], doc_ids: list[str], show_progress: bool = True) -> list[bool]:
        """Batch deduplication with progress bar."""
        keep_mask = []
        iterator = zip(texts, doc_ids)
        if show_progress:
            iterator = tqdm(list(iterator), desc="    Deduplicating", unit="doc")

        for text, doc_id in iterator:
            m = self.compute_minhash(text)
            if self.lsh.query(m):
                keep_mask.append(False)
            else:
                self.lsh.insert(doc_id, m)
                keep_mask.append(True)
        return keep_mask


class CheckpointManager:
    """Manage intermediate checkpoints for resumable processing."""

    def __init__(self, cache_dir: str = "data/.cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_file_hash(self, filepath: str) -> str:
        """Get hash of file for cache invalidation."""
        stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()[:12]

    def get_checkpoint_path(self, filename: str, step: str, file_hash: str) -> Path:
        """Get path for a checkpoint file."""
        return self.cache_dir / f"{filename}_{step}_{file_hash}.parquet"

    def get_state_path(self, filename: str, file_hash: str) -> Path:
        """Get path for processing state."""
        return self.cache_dir / f"{filename}_{file_hash}_state.json"

    def save_checkpoint(self, df: pd.DataFrame, filename: str, step: str, file_hash: str) -> None:
        """Save intermediate checkpoint."""
        path = self.get_checkpoint_path(filename, step, file_hash)
        df.to_parquet(path, index=False)

    def load_checkpoint(self, filename: str, step: str, file_hash: str) -> Optional[pd.DataFrame]:
        """Load checkpoint if it exists."""
        path = self.get_checkpoint_path(filename, step, file_hash)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_state(self, filename: str, file_hash: str, state: dict) -> None:
        """Save processing state."""
        path = self.get_state_path(filename, file_hash)
        with open(path, 'w') as f:
            json.dump(state, f)

    def load_state(self, filename: str, file_hash: str) -> Optional[dict]:
        """Load processing state if it exists."""
        path = self.get_state_path(filename, file_hash)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def cleanup(self, filename: str, file_hash: str) -> None:
        """Remove checkpoints after successful completion."""
        for step in ['clean', 'quality', 'toxicity', 'dedup']:
            path = self.get_checkpoint_path(filename, step, file_hash)
            if path.exists():
                path.unlink()
        state_path = self.get_state_path(filename, file_hash)
        if state_path.exists():
            state_path.unlink()


def process_single_file(
    filename: str,
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    use_gpu: bool = True,
    use_cache: bool = True,
    batch_size: int = 128,
    n_workers: int = None,
) -> tuple[str, int, int]:
    """Process a single file with checkpointing support."""

    input_path = os.path.join(input_dir, filename)
    output_filename = filename.replace('.parquet', '_clean.parquet')
    output_path = os.path.join(output_dir, output_filename)

    # Skip if already fully processed
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already processed)")
        # Read to get stats
        df = pd.read_parquet(output_path)
        return filename, len(df), len(df)

    print(f"\nProcessing {filename}...")

    # Initialize checkpoint manager
    checkpoint = CheckpointManager() if use_cache else None
    file_hash = checkpoint.get_file_hash(input_path) if checkpoint else ""

    try:
        # Check for existing state
        state = checkpoint.load_state(filename, file_hash) if checkpoint else None
        completed_steps = state.get('completed', []) if state else []
        original_count = state.get('original_count', 0) if state else 0

        # Step 1: Load and clean text (PARALLEL - uses all CPU cores)
        if 'clean' in completed_steps and checkpoint:
            print(f"  Loading cached clean data...")
            df = checkpoint.load_checkpoint(filename, 'clean', file_hash)
        else:
            df = pd.read_parquet(input_path)
            original_count = len(df)

            # Check for partial progress using append-only chunk files
            chunk_pattern = f"{filename}_clean_chunk_*_{file_hash}.parquet"
            chunk_dir = checkpoint.cache_dir if checkpoint else None
            start_idx = 0
            existing_chunks = []

            if chunk_dir:
                # Find all existing chunk files - count docs without loading into memory
                existing_chunks = sorted(chunk_dir.glob(chunk_pattern))
                if existing_chunks:
                    print(f"  Found {len(existing_chunks)} existing checkpoint chunk(s)...")
                    # Count rows without loading full data into memory
                    for chunk_path in existing_chunks:
                        chunk_df = pd.read_parquet(chunk_path, columns=[])  # Just get row count
                        start_idx += len(chunk_df)
                    print(f"    {start_idx:,} documents already cleaned")

            remaining = original_count - start_idx
            # Track chunks for append-only saves
            current_chunk_idx = len(existing_chunks)

            if remaining <= 0:
                print(f"  All documents already cleaned from chunks")
            else:
                print(f"  Cleaning {remaining:,} documents (of {original_count:,} total)...")

                # Streaming chunk callback - saves each chunk to disk immediately
                def save_chunk_streaming(chunk_data, idx):
                    nonlocal current_chunk_idx
                    if checkpoint:
                        actual_idx = current_chunk_idx + idx
                        chunk_path = chunk_dir / f"{filename}_clean_chunk_{actual_idx:04d}_{file_hash}.parquet"
                        print(f"\n    [Saving chunk {actual_idx}: {len(chunk_data):,} docs...]", end="", flush=True)
                        chunk_df = pd.DataFrame({'text': chunk_data})
                        chunk_df.to_parquet(chunk_path, index=False)
                        print(f" done]")

                # Use streaming approach - processes and saves in chunks, doesn't accumulate in memory
                texts_to_clean = df['text'].tolist()[start_idx:]

                chunks_generated = 0
                for chunk in parallel_clean_texts_streaming(
                    texts_to_clean,
                    n_workers=n_workers,
                    chunk_callback=save_chunk_streaming if checkpoint else None,
                    chunk_size=500000
                ):
                    chunks_generated += 1
                    # Chunk is saved to disk by callback, we don't keep it in memory

                current_chunk_idx += chunks_generated

            # Now load all chunks and combine for final output
            print(f"  Combining {current_chunk_idx} chunks into final output...")
            all_chunk_files = sorted(chunk_dir.glob(chunk_pattern)) if chunk_dir else []
            all_cleaned = []
            for chunk_path in tqdm(all_chunk_files, desc="    Loading chunks"):
                chunk_df = pd.read_parquet(chunk_path)
                all_cleaned.extend(chunk_df['text'].tolist())

            df['text'] = all_cleaned

            # Save final checkpoint and cleanup chunk files
            if checkpoint:
                print(f"  Saving final cleaned data...")
                checkpoint.save_checkpoint(df, filename, 'clean', file_hash)
                checkpoint.save_state(filename, file_hash, {'completed': ['clean'], 'original_count': original_count})
                # Remove chunk files after successful merge
                for chunk_path in all_chunk_files:
                    chunk_path.unlink()
                if all_chunk_files:
                    print(f"    Cleaned up {len(all_chunk_files)} chunk files")

        # Step 2: Quality filtering
        if 'quality' in completed_steps and checkpoint:
            print(f"  Loading cached quality-filtered data...")
            df = checkpoint.load_checkpoint(filename, 'quality', file_hash)
        else:
            print(f"  Filtering by quality...")
            # Vectorized quality checks
            word_counts = df['text'].str.split().str.len()
            unique_chars = df['text'].str.lower().apply(lambda x: len(set(x)))
            quality_mask = (word_counts >= 50) & (word_counts <= 10000) & (unique_chars >= 20)
            df = df[quality_mask].reset_index(drop=True)
            print(f"    After quality filter: {len(df)}/{original_count} ({len(df)/original_count*100:.1f}%)")

            if checkpoint:
                checkpoint.save_checkpoint(df, filename, 'quality', file_hash)
                checkpoint.save_state(filename, file_hash, {'completed': ['clean', 'quality'], 'original_count': original_count})

        # Step 3: Toxicity filtering (GPU-accelerated)
        if 'toxicity' in completed_steps and checkpoint:
            print(f"  Loading cached toxicity-filtered data...")
            df = checkpoint.load_checkpoint(filename, 'toxicity', file_hash)
        else:
            print(f"  Filtering toxicity ({len(df)} docs, batch_size={batch_size})...")
            cleaner = DataCleaner(use_gpu=use_gpu, batch_size=batch_size)
            toxic_mask = cleaner.is_toxic_batch(df['text'].tolist(), show_progress=True)
            df = df[~np.array(toxic_mask)].reset_index(drop=True)
            print(f"    After toxicity filter: {len(df)}/{original_count} ({len(df)/original_count*100:.1f}%)")

            if checkpoint:
                checkpoint.save_checkpoint(df, filename, 'toxicity', file_hash)
                checkpoint.save_state(filename, file_hash, {'completed': ['clean', 'quality', 'toxicity'], 'original_count': original_count})

        # Step 4: Deduplication
        print(f"  Deduplicating {len(df)} docs...")
        cleaner = DataCleaner(use_gpu=False)  # Dedup is CPU-only
        doc_ids = [f"{filename}_{i}" for i in range(len(df))]
        keep_mask = cleaner.deduplicate_batch(df['text'].tolist(), doc_ids, show_progress=True)
        df = df[keep_mask].reset_index(drop=True)
        final_count = len(df)
        print(f"    After deduplication: {final_count}/{original_count} ({final_count/original_count*100:.1f}%)")

        # Add source column and save
        df['source'] = filename
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved {output_filename}: {final_count}/{original_count} documents")

        # Cleanup checkpoints on success
        if checkpoint:
            checkpoint.cleanup(filename, file_hash)

        return filename, final_count, original_count

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return filename, 0, 0


def process_all_files(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    file_pattern: str = "pretraining_",
    use_gpu: bool = True,
    use_cache: bool = True,
    batch_size: int = 128,
    n_workers: int = None,
) -> None:
    """Process all matching files sequentially (best for GPU)."""

    os.makedirs(output_dir, exist_ok=True)

    files_to_process = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith(file_pattern) and f.endswith('.parquet')
    ])

    if n_workers is None:
        n_workers = max(1, cpu_count() // 2)

    print(f"\n{'='*60}")
    print(f"Found {len(files_to_process)} files to process")
    print(f"GPU acceleration: {use_gpu and torch.cuda.is_available()}")
    print(f"Caching enabled: {use_cache}")
    print(f"Batch size: {batch_size}")
    print(f"CPU workers: {n_workers}")
    print(f"{'='*60}\n")

    results = []
    for filename in files_to_process:
        result = process_single_file(
            filename, input_dir, output_dir,
            use_gpu=use_gpu, use_cache=use_cache, batch_size=batch_size,
            n_workers=n_workers
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_kept = sum(r[1] for r in results)
    total_original = sum(r[2] for r in results if r[2] > 0)
    for filename, kept, original in results:
        if original > 0:
            print(f"  {filename}: {kept}/{original} ({kept/original*100:.1f}%)")
    if total_original > 0:
        print(f"\nTotal: {total_kept}/{total_original} ({total_kept/total_original*100:.1f}% kept)")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean and deduplicate data with GPU acceleration and caching')
    parser.add_argument('--pattern', type=str, default='pretraining_',
                       help='File pattern to match (default: pretraining_)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable checkpoint caching')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for toxicity detection (default: 128)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of CPU workers for text cleaning (default: all cores)')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory (default: data/raw)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory (default: data/processed)')

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available() and not args.no_gpu
    # Default to half of CPU cores - streaming approach prevents memory buildup
    n_workers = args.workers if args.workers else max(1, cpu_count() // 2)

    print(f"Starting optimized data cleaning:")
    print(f"  - GPU acceleration: {use_gpu}")
    print(f"  - Checkpoint caching: {not args.no_cache}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - CPU workers: {n_workers}")
    print(f"  - File pattern: {args.pattern}")
    print()

    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.pattern,
        use_gpu=use_gpu,
        use_cache=not args.no_cache,
        batch_size=args.batch_size,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
