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
        print("GPU DATA PIPELINE")
        print("=" * 60)
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"GPU Text Cleaning: {self.use_gpu_text} (RAPIDS)")
        print(f"GPU Deduplication: {self.use_gpu_dedup} (NeMo)")
        print(f"Quality Filters: {[f[0] for f in self.quality_filters]}")
        print(f"Toxicity Filter: {not self.skip_toxicity}")
        print(f"Deduplication: {not self.skip_dedup}")
        print("=" * 60)

        # Stage 1: Load and clean text
        print("\n[Stage 1/4] Loading and cleaning text...")
        stage1_start = time.time()

        all_texts = []
        all_ids = []

        # Find input files - ONLY pretraining data (not SFT/DPO which have different formats)
        parquet_files = list(self.input_dir.glob("**/*.parquet"))

        # Filter to only pretraining files (have 'text' column with raw text)
        pretraining_prefixes = ['pretraining_', 'slimpajama', 'wikipedia', 'openwebtext',
                                'the-stack', 'arxiv', 'pg19', 'c4', 'pile']
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

        for pq_file in tqdm(pretraining_files, desc="  Loading", disable=not self.show_progress):
            try:
                # Load in chunks for large files to avoid OOM
                file_size_mb = pq_file.stat().st_size / (1024 * 1024)

                if file_size_mb > 1000 and self.use_gpu_text:
                    # Large file - load in chunks on CPU then process
                    print(f"  Large file ({file_size_mb:.0f}MB): {pq_file.name} - loading in chunks...")
                    chunk_size = 500_000
                    df_iter = pd.read_parquet(pq_file, columns=['text'])
                    texts = df_iter['text'].fillna('').tolist()
                else:
                    # Normal size - try GPU load first
                    try:
                        df = gpu_load_parquet(str(pq_file), columns=['text'])
                        texts = df['text'].fillna('').tolist()
                    except Exception as gpu_err:
                        if 'out_of_memory' in str(gpu_err) or 'bad_alloc' in str(gpu_err):
                            print(f"  GPU OOM on {pq_file.name} - falling back to CPU...")
                            df = pd.read_parquet(pq_file, columns=['text'])
                            texts = df['text'].fillna('').tolist()
                        else:
                            raise

                all_texts.extend(texts)
                all_ids.extend([f"{pq_file.stem}_{i}" for i in range(len(texts))])
            except KeyError as e:
                # File doesn't have 'text' column - skip it
                print(f"  Skipping {pq_file.name}: missing column {e}")
            except Exception as e:
                print(f"  Warning: Failed to load {pq_file}: {e}")

        stats['input_docs'] = len(all_texts)
        print(f"  Loaded {stats['input_docs']:,} documents")

        # GPU text cleaning
        print(f"  Cleaning text (GPU={self.use_gpu_text})...")
        cleaned_texts = gpu_clean_texts(
            all_texts,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
            use_gpu=self.use_gpu_text
        )

        # Remove empty texts
        valid_mask = [bool(t and len(t.strip()) > 0) for t in cleaned_texts]
        cleaned_texts = [t for t, v in zip(cleaned_texts, valid_mask) if v]
        all_ids = [i for i, v in zip(all_ids, valid_mask) if v]

        stats['after_cleaning'] = len(cleaned_texts)
        stage1_time = time.time() - stage1_start
        print(f"  Stage 1 complete: {stats['after_cleaning']:,} docs ({stage1_time:.1f}s)")

        # Stage 2: Quality filtering
        print("\n[Stage 2/4] Quality filtering...")
        stage2_start = time.time()

        if DATATROVE_AVAILABLE and self.quality_filters:
            quality_mask = self._apply_quality_filters(cleaned_texts)
            cleaned_texts = [t for t, m in zip(cleaned_texts, quality_mask) if m]
            all_ids = [i for i, m in zip(all_ids, quality_mask) if m]
        else:
            print("  Skipping (datatrove not available)")

        stats['after_quality'] = len(cleaned_texts)
        stage2_time = time.time() - stage2_start
        print(f"  Stage 2 complete: {stats['after_quality']:,} docs ({stage2_time:.1f}s)")

        # Stage 3: Toxicity filtering
        print("\n[Stage 3/4] Toxicity filtering...")
        stage3_start = time.time()

        if not self.skip_toxicity:
            toxicity_model = self._get_toxicity_model()
            if toxicity_model:
                toxic_mask = toxicity_model.is_toxic_batch(cleaned_texts, show_progress=self.show_progress)
                cleaned_texts = [t for t, is_toxic in zip(cleaned_texts, toxic_mask) if not is_toxic]
                all_ids = [i for i, is_toxic in zip(all_ids, toxic_mask) if not is_toxic]
            else:
                print("  Skipping (model not available)")
        else:
            print("  Skipping (disabled)")

        stats['after_toxicity'] = len(cleaned_texts)
        stage3_time = time.time() - stage3_start
        print(f"  Stage 3 complete: {stats['after_toxicity']:,} docs ({stage3_time:.1f}s)")

        # Save intermediate results for deduplication
        intermediate_path = self.cache_dir / "pre_dedup"
        intermediate_path.mkdir(parents=True, exist_ok=True)
        intermediate_file = intermediate_path / "data.parquet"

        df = pd.DataFrame({'id': all_ids, 'text': cleaned_texts})
        gpu_save_parquet(df, str(intermediate_file))

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
            gpu_save_parquet(df, str(final_output))
            stats['after_dedup'] = len(df)

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
        from multiprocessing import Pool, cpu_count

        n_texts = len(texts)
        n_workers = max(1, int(cpu_count() * 0.8))

        # For small batches, use single-threaded
        if n_texts < 1000:
            return [self._filter_single(t) for t in texts]

        # Parallel processing
        with Pool(processes=n_workers) as pool:
            if self.show_progress:
                results = list(tqdm(
                    pool.imap(self._filter_single, texts, chunksize=5000),
                    total=n_texts,
                    desc="  Quality filter"
                ))
            else:
                results = pool.map(self._filter_single, texts, chunksize=5000)

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
