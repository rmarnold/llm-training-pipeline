import os
import pandas as pd
from datasketch import MinHash, MinHashLSH
from ftfy import fix_text
from detoxify import Detoxify
import re
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

class DataCleaner:
    def __init__(self, toxicity_threshold=0.7, use_gpu=True, batch_size=32):
        # Use GPU if available
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.toxicity_model = Detoxify('original', device=device)
        self.toxicity_threshold = toxicity_threshold
        self.batch_size = batch_size
        self.lsh = MinHashLSH(threshold=0.85, num_perm=128)

    def clean_text(self, text):
        """Vectorized text cleaning"""
        if pd.isna(text):
            return ""
        # Fix encoding issues
        text = fix_text(str(text))
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove PII patterns
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        return text.strip()

    def filter_quality_batch(self, texts, min_words=50, max_words=10000):
        """Vectorized quality filtering"""
        results = []
        for text in texts:
            word_count = len(text.split())
            if word_count < min_words or word_count > max_words:
                results.append(False)
                continue

            # Check character variety (detect gibberish)
            unique_chars = len(set(text.lower()))
            if unique_chars < 20:
                results.append(False)
                continue

            results.append(True)
        return results

    def is_toxic_batch(self, texts):
        """Batch toxicity detection - MUCH faster than one-by-one"""
        if not texts:
            return []

        # Process in batches for memory efficiency
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results = self.toxicity_model.predict(batch)
            # Check if any toxicity score exceeds threshold
            for j in range(len(batch)):
                is_toxic = any(results[key][j] > self.toxicity_threshold
                             for key in results.keys())
                all_results.append(is_toxic)

        return all_results

    def compute_minhash(self, text):
        """Compute MinHash for a text"""
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    def deduplicate_batch(self, texts, doc_ids):
        """Batch deduplication using MinHash LSH"""
        keep_mask = []

        for text, doc_id in zip(texts, doc_ids):
            m = self.compute_minhash(text)

            # Check if duplicate
            result = self.lsh.query(m)
            if result:
                keep_mask.append(False)
            else:
                self.lsh.insert(doc_id, m)
                keep_mask.append(True)

        return keep_mask

def process_single_file(args):
    """Process a single file - designed for parallel execution"""
    filename, input_dir, output_dir = args

    input_path = os.path.join(input_dir, filename)
    output_filename = filename.replace('.parquet', '_clean.parquet')
    output_path = os.path.join(output_dir, output_filename)

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"✓ Skipping {filename} (already processed)")
        return filename, 0, 0

    print(f"Processing {filename}...")

    try:
        # Read data
        df = pd.read_parquet(input_path)
        original_count = len(df)

        # Initialize cleaner (GPU will be used automatically)
        cleaner = DataCleaner(use_gpu=True, batch_size=64)

        # Step 1: Clean text (vectorized)
        print(f"  Cleaning text...")
        df['text'] = df['text'].apply(cleaner.clean_text)

        # Step 2: Quality filtering (vectorized)
        print(f"  Filtering quality...")
        quality_mask = cleaner.filter_quality_batch(df['text'].tolist())
        df = df[quality_mask].reset_index(drop=True)
        print(f"    After quality filter: {len(df)}/{original_count} ({len(df)/original_count*100:.1f}%)")

        # Step 3: Toxicity filtering (batched GPU inference)
        print(f"  Filtering toxicity (GPU batched)...")
        toxic_mask = cleaner.is_toxic_batch(df['text'].tolist())
        df = df[~np.array(toxic_mask)].reset_index(drop=True)
        print(f"    After toxicity filter: {len(df)}/{original_count} ({len(df)/original_count*100:.1f}%)")

        # Step 4: Deduplication (batched)
        print(f"  Deduplicating...")
        doc_ids = [f"{filename}_{i}" for i in range(len(df))]
        keep_mask = cleaner.deduplicate_batch(df['text'].tolist(), doc_ids)
        df = df[keep_mask].reset_index(drop=True)
        final_count = len(df)
        print(f"    After deduplication: {final_count}/{original_count} ({final_count/original_count*100:.1f}%)")

        # Add source column
        df['source'] = filename

        # Save
        df.to_parquet(output_path, index=False)
        print(f"✓ Saved {output_filename}: {final_count}/{original_count} documents ({final_count/original_count*100:.1f}%)")

        return filename, final_count, original_count

    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        return filename, 0, 0

def process_all_files_parallel(input_dir="data/raw", output_dir="data/processed",
                               file_pattern="pretraining_", max_workers=None):
    """Process multiple files in parallel using all CPU cores"""

    os.makedirs(output_dir, exist_ok=True)

    # Get list of files to process
    files_to_process = [
        f for f in os.listdir(input_dir)
        if f.startswith(file_pattern) and f.endswith('.parquet')
    ]

    print(f"\n{'='*60}")
    print(f"Found {len(files_to_process)} files to process")
    print(f"Using up to {max_workers or cpu_count()} parallel workers")
    print(f"{'='*60}\n")

    # Prepare arguments for parallel processing
    args_list = [(f, input_dir, output_dir) for f in files_to_process]

    # Process files in parallel
    if max_workers == 1:
        # Sequential processing (for debugging)
        results = [process_single_file(args) for args in args_list]
    else:
        # Parallel processing
        with Pool(processes=max_workers) as pool:
            results = pool.map(process_single_file, args_list)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_kept = sum(r[1] for r in results)
    total_original = sum(r[2] for r in results)
    for filename, kept, original in results:
        if original > 0:
            print(f"  {filename}: {kept}/{original} ({kept/original*100:.1f}%)")
    print(f"\nTotal: {total_kept}/{total_original} ({total_kept/total_original*100:.1f}% kept)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Clean and deduplicate data with GPU acceleration')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--pattern', type=str, default='pretraining_',
                       help='File pattern to match (default: pretraining_)')

    args = parser.parse_args()

    # Note: For GPU, we process files in parallel but each file uses GPU sequentially
    # This is optimal since GPU can only handle one toxicity model at a time efficiently
    workers = args.workers if args.workers else min(3, cpu_count())  # Limit to 3 for GPU memory

    print(f"Starting optimized data cleaning with:")
    print(f"  - GPU acceleration: {torch.cuda.is_available()}")
    print(f"  - Parallel workers: {workers}")
    print(f"  - Batch processing: Enabled")
    print(f"  - File pattern: {args.pattern}")
    print()

    process_all_files_parallel(max_workers=workers, file_pattern=args.pattern)
