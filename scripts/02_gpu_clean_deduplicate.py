#!/usr/bin/env python3
"""GPU-accelerated data cleaning and deduplication pipeline.

Hybrid approach combining:
- RAPIDS cuDF for text cleaning (100-150x faster)
- NeMo Curator for deduplication (16-107x faster)
- Existing CPU quality filters (well-parallelized)
- Existing GPU toxicity detection (already optimized)

Usage:
    # Auto-detect GPU capabilities
    python scripts/02_gpu_clean_deduplicate.py

    # Force GPU mode
    python scripts/02_gpu_clean_deduplicate.py --force-gpu

    # Benchmark mode
    python scripts/02_gpu_clean_deduplicate.py --benchmark

    # Fast mode (skip toxicity)
    python scripts/02_gpu_clean_deduplicate.py --fast-quality --no-toxicity
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GPU utilities
from scripts.gpu_text_utils import (
    gpu_clean_texts,
    gpu_load_parquet,
    gpu_save_parquet,
    is_gpu_available as is_gpu_text_available,
    RAPIDS_AVAILABLE,
)
from scripts.gpu_dedup import (
    gpu_fuzzy_dedup,
    is_gpu_dedup_available,
    NEMO_AVAILABLE,
)

# Import existing utilities from original script
try:
    from scripts.gpu_utils import detect_gpu_type, get_optimal_settings
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Import datatrove quality filters
try:
    from datatrove.pipeline.filters import (
        GopherQualityFilter,
        GopherRepetitionFilter,
        FineWebQualityFilter,
    )
    DATATROVE_AVAILABLE = True
except ImportError:
    DATATROVE_AVAILABLE = False

# Import toxicity detection from original script
try:
    from scripts.gpu_utils import detect_gpu_type
except ImportError:
    pass


class GPUDataPipeline:
    """GPU-accelerated data cleaning pipeline."""

    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
        cache_dir: str = ".gpu_cache",
        use_gpu: Optional[bool] = None,
        fast_quality: bool = False,
        skip_toxicity: bool = False,
        skip_dedup: bool = False,
        toxicity_threshold: float = 0.7,
        dedup_threshold: float = 0.85,
        batch_size: int = 500_000,
        show_progress: bool = True,
    ):
        """Initialize GPU data pipeline.

        Args:
            input_dir: Directory with raw parquet files
            output_dir: Output directory for processed data
            cache_dir: Cache directory for intermediate results
            use_gpu: Force GPU (True) or CPU (False). None = auto-detect.
            fast_quality: Skip expensive repetition filters
            skip_toxicity: Skip toxicity filtering
            skip_dedup: Skip deduplication
            toxicity_threshold: Threshold for toxicity detection
            dedup_threshold: Jaccard similarity threshold for deduplication
            batch_size: Batch size for GPU processing
            show_progress: Show progress bars
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.fast_quality = fast_quality
        self.skip_toxicity = skip_toxicity
        self.skip_dedup = skip_dedup
        self.toxicity_threshold = toxicity_threshold
        self.dedup_threshold = dedup_threshold
        self.batch_size = batch_size
        self.show_progress = show_progress

        # Auto-detect GPU capabilities
        if use_gpu is None:
            self.use_gpu_text = is_gpu_text_available()
            self.use_gpu_dedup = is_gpu_dedup_available()
        else:
            self.use_gpu_text = use_gpu and RAPIDS_AVAILABLE
            self.use_gpu_dedup = use_gpu and NEMO_AVAILABLE

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize quality filters
        self._init_quality_filters()

        # Initialize toxicity model (lazy load)
        self._toxicity_model = None

    def _init_quality_filters(self):
        """Initialize datatrove quality filters."""
        self.quality_filters = []

        if DATATROVE_AVAILABLE:
            # Fast filters (always enabled)
            self.quality_filters.append(
                ('gopher_quality', GopherQualityFilter(
                    min_doc_words=50,
                    max_doc_words=100000,
                    min_avg_word_length=3,
                    max_avg_word_length=10,
                    min_stop_words=2,
                    max_symbol_word_ratio=0.1,
                ))
            )

            if not self.fast_quality:
                # Medium speed filter
                self.quality_filters.append(
                    ('fineweb', FineWebQualityFilter())
                )
                # Slowest filter (n-gram repetition)
                self.quality_filters.append(
                    ('repetition', GopherRepetitionFilter())
                )

    def _get_toxicity_model(self):
        """Lazy load toxicity model."""
        if self._toxicity_model is None and not self.skip_toxicity:
            try:
                # Import from the existing optimized script
                sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
                from importlib import import_module
                mod = import_module("02_clean_deduplicate_optimized")
                DataCleaner = mod.DataCleaner
                self._toxicity_model = DataCleaner(
                    toxicity_threshold=self.toxicity_threshold,
                    use_gpu=True,
                    lazy_load=True
                )
            except ImportError:
                print("Warning: Could not import DataCleaner for toxicity detection")
                self._toxicity_model = None
        return self._toxicity_model

    def process(self) -> dict:
        """Run the full GPU pipeline.

        Returns:
            Dict with processing statistics
        """
        import gc

        stats = {
            'start_time': time.time(),
            'input_docs': 0,
            'after_cleaning': 0,
            'after_quality': 0,
            'after_toxicity': 0,
            'after_dedup': 0,
            'gpu_text': self.use_gpu_text,
            'gpu_dedup': self.use_gpu_dedup,
        }

        print("=" * 60)
        print("GPU DATA PIPELINE (Memory-Optimized)")
        print("=" * 60)
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"GPU Text Cleaning: {self.use_gpu_text} (RAPIDS)")
        print(f"GPU Deduplication: {self.use_gpu_dedup} (NeMo)")
        print(f"Quality Filters: {[f[0] for f in self.quality_filters]}")
        print(f"Toxicity Filter: {not self.skip_toxicity}")
        print(f"Deduplication: {not self.skip_dedup}")
        print("=" * 60)

        # Stage 1: Load, clean, and save file-by-file (streaming to avoid OOM)
        print("\n[Stage 1/4] Loading and cleaning text (streaming mode)...")
        stage1_start = time.time()

        # Find input files - ONLY pretraining data (not SFT/DPO which have different formats)
        parquet_files = list(self.input_dir.glob("**/*.parquet"))

        # Filter to only pretraining files (have 'text' column with raw text)
        pretraining_prefixes = ['pretraining_', 'slimpajama', 'wikipedia', 'openwebtext',
                                'the-stack', 'arxiv', 'pg19', 'c4', 'pile', 'fineweb']
        pretraining_files = [
            f for f in parquet_files
            if any(f.stem.lower().startswith(p) or p in f.stem.lower() for p in pretraining_prefixes)
        ]

        # Exclude SFT/DPO/reasoning files which have different column structures
        exclude_prefixes = ['reasoning_', 'function_calling_', 'instruction_tuning_',
                           'preference_data_', 'logic_', 'sft_', 'dpo_']
        pretraining_files = [
            f for f in pretraining_files
            if not any(f.stem.lower().startswith(p) for p in exclude_prefixes)
        ]

        if not pretraining_files:
            # Fall back to all files if no pretraining files found
            pretraining_files = parquet_files
            print(f"  No pretraining-specific files found, using all {len(parquet_files)} files")
        else:
            print(f"  Found {len(pretraining_files)} pretraining files (filtered from {len(parquet_files)} total)")

        # Create temp directory for cleaned files
        cleaned_dir = self.cache_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        total_input = 0
        total_cleaned = 0

        # Process each file individually with TRUE STREAMING to avoid memory accumulation
        # Use pyarrow for chunked reading - never load full file into memory
        import pyarrow.parquet as pq

        for file_idx, pq_file in enumerate(tqdm(pretraining_files, desc="  Processing files", disable=not self.show_progress)):
            try:
                file_size_mb = pq_file.stat().st_size / (1024 * 1024)

                # Use very small row groups for streaming (10K rows at a time)
                STREAM_CHUNK_SIZE = 10_000

                # Open parquet file for streaming
                try:
                    parquet_file = pq.ParquetFile(pq_file)
                except Exception as e:
                    print(f"  Skipping {pq_file.name}: cannot open ({e})")
                    continue

                # Check for 'text' column
                if 'text' not in parquet_file.schema.names:
                    print(f"  Skipping {pq_file.name}: no 'text' column")
                    continue

                file_input = 0
                file_cleaned = 0
                chunk_idx = 0

                # Stream through the file in small batches
                for batch in parquet_file.iter_batches(batch_size=STREAM_CHUNK_SIZE, columns=['text']):
                    batch_df = batch.to_pandas()
                    texts = batch_df['text'].fillna('').tolist()
                    batch_input = len(texts)
                    file_input += batch_input
                    total_input += batch_input

                    del batch_df, batch
                    gc.collect()

                    # GPU clean this small chunk
                    cleaned_chunk = gpu_clean_texts(
                        texts,
                        batch_size=min(self.batch_size, 50_000),  # Smaller GPU batches
                        show_progress=False,
                        use_gpu=self.use_gpu_text
                    )

                    del texts
                    gc.collect()

                    # Filter empty and create IDs
                    cleaned_texts = []
                    all_ids = []
                    for i, t in enumerate(cleaned_chunk):
                        if t and len(t.strip()) > 0:
                            cleaned_texts.append(t)
                            all_ids.append(f"{pq_file.stem}_{chunk_idx}_{i}")

                    del cleaned_chunk
                    gc.collect()

                    # Save each chunk immediately to disk (don't accumulate in memory)
                    if cleaned_texts:
                        cleaned_df = pd.DataFrame({'id': all_ids, 'text': cleaned_texts})
                        cleaned_file = cleaned_dir / f"cleaned_{file_idx:03d}_{chunk_idx:04d}_{pq_file.stem[:30]}.parquet"
                        cleaned_df.to_parquet(cleaned_file, compression='snappy')
                        file_cleaned += len(cleaned_df)
                        total_cleaned += len(cleaned_df)
                        del cleaned_df

                    del cleaned_texts, all_ids
                    gc.collect()

                    chunk_idx += 1

                del parquet_file
                gc.collect()

                print(f"    {pq_file.name}: {file_input:,} -> {file_cleaned:,} docs ({chunk_idx} chunks)")

            except Exception as e:
                print(f"  Warning: Failed to process {pq_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        stats['input_docs'] = total_input
        stats['after_cleaning'] = total_cleaned
        stage1_time = time.time() - stage1_start
        print(f"  Stage 1 complete: {stats['after_cleaning']:,} docs from {stats['input_docs']:,} ({stage1_time:.1f}s)")

        # Stage 2: Quality filtering (streaming from cleaned files)
        print("\n[Stage 2/4] Quality filtering (streaming mode)...")
        stage2_start = time.time()

        quality_dir = self.cache_dir / "quality_filtered"
        quality_dir.mkdir(parents=True, exist_ok=True)

        cleaned_files = sorted(cleaned_dir.glob("*.parquet"))
        total_after_quality = 0

        if DATATROVE_AVAILABLE and self.quality_filters:
            # Process all files with a single progress bar
            print(f"  Processing {len(cleaned_files)} chunk files...")

            for file_idx, cf in enumerate(tqdm(cleaned_files, desc="  Quality filtering", disable=not self.show_progress)):
                try:
                    df = pd.read_parquet(cf)
                    texts = df['text'].tolist()
                    ids = df['id'].tolist()
                    del df

                    # Single-threaded quality filtering (fast for 10K chunk files)
                    quality_mask = self._apply_quality_filters(texts)
                    filtered_texts = [t for t, m in zip(texts, quality_mask) if m]
                    filtered_ids = [i for i, m in zip(ids, quality_mask) if m]

                    del texts, ids, quality_mask

                    if filtered_texts:
                        quality_df = pd.DataFrame({'id': filtered_ids, 'text': filtered_texts})
                        quality_file = quality_dir / f"quality_{file_idx:05d}.parquet"
                        quality_df.to_parquet(quality_file, compression='snappy')
                        total_after_quality += len(quality_df)
                        del quality_df

                    del filtered_texts, filtered_ids

                    # Only gc every 100 files to reduce overhead
                    if file_idx % 100 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"  Warning: Quality filter failed on {cf}: {e}")
        else:
            print("  Skipping quality filters (datatrove not available)")
            # Just copy files
            for cf in cleaned_files:
                df = pd.read_parquet(cf)
                total_after_quality += len(df)
                df.to_parquet(quality_dir / cf.name, compression='snappy')
                del df

        stats['after_quality'] = total_after_quality
        stage2_time = time.time() - stage2_start
        print(f"  Stage 2 complete: {stats['after_quality']:,} docs ({stage2_time:.1f}s)")

        # Stage 3: Toxicity filtering (streaming)
        print("\n[Stage 3/4] Toxicity filtering...")
        stage3_start = time.time()

        toxicity_dir = self.cache_dir / "toxicity_filtered"
        toxicity_dir.mkdir(parents=True, exist_ok=True)

        quality_files = sorted(quality_dir.glob("*.parquet"))
        total_after_toxicity = 0

        if not self.skip_toxicity:
            toxicity_model = self._get_toxicity_model()
            if toxicity_model:
                for file_idx, qf in enumerate(tqdm(quality_files, desc="  Toxicity filtering", disable=not self.show_progress)):
                    try:
                        df = pd.read_parquet(qf)
                        texts = df['text'].tolist()
                        ids = df['id'].tolist()
                        del df
                        gc.collect()

                        toxic_mask = toxicity_model.is_toxic_batch(texts, show_progress=False)
                        filtered_texts = [t for t, is_toxic in zip(texts, toxic_mask) if not is_toxic]
                        filtered_ids = [i for i, is_toxic in zip(ids, toxic_mask) if not is_toxic]

                        del texts, ids, toxic_mask
                        gc.collect()

                        if filtered_texts:
                            toxicity_df = pd.DataFrame({'id': filtered_ids, 'text': filtered_texts})
                            toxicity_file = toxicity_dir / f"toxicity_{file_idx:03d}.parquet"
                            toxicity_df.to_parquet(toxicity_file, compression='snappy')
                            total_after_toxicity += len(toxicity_df)
                            del toxicity_df

                        del filtered_texts, filtered_ids
                        gc.collect()

                    except Exception as e:
                        print(f"  Warning: Toxicity filter failed on {qf}: {e}")
            else:
                print("  Skipping (model not available)")
                for qf in quality_files:
                    df = pd.read_parquet(qf)
                    total_after_toxicity += len(df)
                    df.to_parquet(toxicity_dir / qf.name, compression='snappy')
                    del df
        else:
            print("  Skipping (disabled)")
            for qf in quality_files:
                df = pd.read_parquet(qf)
                total_after_toxicity += len(df)
                df.to_parquet(toxicity_dir / qf.name, compression='snappy')
                del df

        stats['after_toxicity'] = total_after_toxicity
        stage3_time = time.time() - stage3_start
        print(f"  Stage 3 complete: {stats['after_toxicity']:,} docs ({stage3_time:.1f}s)")

        # Combine files for deduplication (streaming concatenation to avoid OOM)
        print("\n  Combining files for deduplication (streaming mode)...")
        intermediate_path = self.cache_dir / "pre_dedup"
        intermediate_path.mkdir(parents=True, exist_ok=True)
        intermediate_file = intermediate_path / "data.parquet"

        toxicity_files = sorted(toxicity_dir.glob("*.parquet"))

        if toxicity_files:
            # Use pyarrow to stream-concatenate without loading all into memory
            import pyarrow as pa
            import pyarrow.parquet as pq_writer

            # First pass: collect all record batches
            writer = None
            total_rows = 0

            for tf in tqdm(toxicity_files, desc="  Combining files", disable=not self.show_progress):
                try:
                    table = pq.read_table(tf)
                    total_rows += table.num_rows

                    if writer is None:
                        writer = pq_writer.ParquetWriter(str(intermediate_file), table.schema)

                    writer.write_table(table)
                    del table
                    gc.collect()
                except Exception as e:
                    print(f"  Warning: Failed to read {tf}: {e}")

            if writer:
                writer.close()
                print(f"  Combined {total_rows:,} documents from {len(toxicity_files)} files")
            else:
                print("  Warning: No documents after filtering!")
                empty_df = pd.DataFrame({'id': [], 'text': []})
                gpu_save_parquet(empty_df, str(intermediate_file))
        else:
            print("  Warning: No documents after filtering!")
            empty_df = pd.DataFrame({'id': [], 'text': []})
            gpu_save_parquet(empty_df, str(intermediate_file))

        # Stage 4: Deduplication
        print("\n[Stage 4/4] Deduplication...")
        stage4_start = time.time()

        if not self.skip_dedup:
            dedup_output = self.output_dir / "deduplicated"
            gpu_fuzzy_dedup(
                input_path=str(intermediate_file),
                output_path=str(dedup_output),
                text_column='text',
                id_column='id',
                similarity_threshold=self.dedup_threshold,
                cache_path=str(self.cache_dir / "dedup_cache"),
                use_gpu=self.use_gpu_dedup,
                show_progress=self.show_progress,
            )

            # Count final documents
            final_files = list(dedup_output.glob("*.parquet"))
            final_count = sum(len(pd.read_parquet(f)) for f in final_files)
            stats['after_dedup'] = final_count

            # Copy to final output
            final_output = self.output_dir / "processed.parquet"
            if final_files:
                dfs = [pd.read_parquet(f) for f in final_files]
                final_df = pd.concat(dfs, ignore_index=True)
                gpu_save_parquet(final_df, str(final_output))
        else:
            print("  Skipping (disabled)")
            # Copy intermediate to final
            final_output = self.output_dir / "processed.parquet"
            import shutil
            shutil.copy(intermediate_file, final_output)
            intermediate_df = pd.read_parquet(intermediate_file)
            stats['after_dedup'] = len(intermediate_df)
            del intermediate_df

        stage4_time = time.time() - stage4_start
        print(f"  Stage 4 complete: {stats['after_dedup']:,} docs ({stage4_time:.1f}s)")

        # Summary
        stats['end_time'] = time.time()
        stats['total_time'] = stats['end_time'] - stats['start_time']

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Input documents:    {stats['input_docs']:,}")
        print(f"After cleaning:     {stats['after_cleaning']:,} ({100*stats['after_cleaning']/max(1,stats['input_docs']):.1f}%)")
        print(f"After quality:      {stats['after_quality']:,} ({100*stats['after_quality']/max(1,stats['input_docs']):.1f}%)")
        print(f"After toxicity:     {stats['after_toxicity']:,} ({100*stats['after_toxicity']/max(1,stats['input_docs']):.1f}%)")
        print(f"After dedup:        {stats['after_dedup']:,} ({100*stats['after_dedup']/max(1,stats['input_docs']):.1f}%)")
        print(f"Total time:         {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} min)")
        print(f"Throughput:         {stats['input_docs']/stats['total_time']:.0f} docs/sec")
        print(f"Output:             {self.output_dir / 'processed.parquet'}")
        print("=" * 60)

        return stats

    def _apply_quality_filters(self, texts: list[str]) -> list[bool]:
        """Apply quality filters to texts.

        Args:
            texts: List of text documents

        Returns:
            Boolean mask (True = keep)
        """
        # For streaming mode with small chunks (10K), single-threaded is actually faster
        # because it avoids process spawn overhead and CUDA initialization in workers
        # Multiprocessing only helps for batches > 50K docs
        n_texts = len(texts)

        if n_texts < 50_000:
            # Single-threaded - faster for small batches, no spawn overhead
            return [self._filter_single(t) for t in texts]

        # Only use multiprocessing for very large batches
        from multiprocessing import get_context, cpu_count

        n_workers = max(1, min(4, int(cpu_count() * 0.5)))  # Fewer workers, less overhead

        # Use 'spawn' to avoid CUDA issues in child processes
        ctx = get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(self._filter_single, texts, chunksize=10000)

        return results

    def _filter_single(self, text: str) -> bool:
        """Filter a single document through all quality filters."""
        if not text or len(text.strip()) < 50:
            return False

        if not DATATROVE_AVAILABLE:
            return True

        try:
            from datatrove.data import Document as DatatroveDocument
            doc = DatatroveDocument(text=text, id="0")

            for _, filter_obj in self.quality_filters:
                result = filter_obj.filter(doc)
                # Handle tuple return (bool, reason) or just bool
                passed = result[0] if isinstance(result, tuple) else result
                if not passed:
                    return False

            return True
        except Exception:
            return False


def run_benchmark(n_samples: int = 100_000):
    """Run GPU vs CPU benchmark."""
    from scripts.gpu_text_utils import benchmark_gpu_vs_cpu

    print("=" * 60)
    print("GPU TEXT CLEANING BENCHMARK")
    print("=" * 60)

    results = benchmark_gpu_vs_cpu(n_samples=n_samples)

    print(f"\nResults for {results['n_samples']:,} samples:")
    print(f"  CPU: {results['cpu_time']:.2f}s ({results['cpu_docs_per_sec']:,.0f} docs/sec)")

    if results['gpu_time']:
        print(f"  GPU: {results['gpu_time']:.2f}s ({results['gpu_docs_per_sec']:,.0f} docs/sec)")
        print(f"  Speedup: {results['speedup']:.1f}x")
    else:
        print("  GPU: Not available")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated data cleaning and deduplication pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory with parquet files"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--cache",
        default=".gpu_cache",
        help="Cache directory"
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Force GPU mode (fail if not available)"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode (ignore GPU)"
    )
    parser.add_argument(
        "--fast-quality",
        action="store_true",
        help="Skip expensive quality filters (repetition detection)"
    )
    parser.add_argument(
        "--no-toxicity",
        action="store_true",
        help="Skip toxicity filtering"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--toxicity-threshold",
        type=float,
        default=0.7,
        help="Toxicity detection threshold (default: 0.7)"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Deduplication similarity threshold (default: 0.85)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Batch size for GPU processing (default: 500000)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode"
    )
    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=100_000,
        help="Number of samples for benchmark (default: 100000)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars"
    )

    args = parser.parse_args()

    # Benchmark mode
    if args.benchmark:
        run_benchmark(n_samples=args.benchmark_samples)
        return

    # Determine GPU mode
    use_gpu = None
    if args.force_gpu:
        use_gpu = True
    elif args.force_cpu:
        use_gpu = False

    # Run pipeline
    pipeline = GPUDataPipeline(
        input_dir=args.input,
        output_dir=args.output,
        cache_dir=args.cache,
        use_gpu=use_gpu,
        fast_quality=args.fast_quality,
        skip_toxicity=args.no_toxicity,
        skip_dedup=args.no_dedup,
        toxicity_threshold=args.toxicity_threshold,
        dedup_threshold=args.dedup_threshold,
        batch_size=args.batch_size,
        show_progress=not args.quiet,
    )

    stats = pipeline.process()
    return stats


if __name__ == "__main__":
    main()
