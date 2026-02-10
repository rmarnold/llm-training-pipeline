"""Optimized data cleaning with caching, checkpoints, and GPU acceleration."""
from __future__ import annotations

# Disable HuggingFace tokenizer internal parallelism - we use multiprocessing instead
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CRITICAL FIX: Use 'spawn' to avoid CUDA context inheritance in forked workers
import multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

# CRITICAL: Also set spawn for 'multiprocess' library (used by HuggingFace datasets)
try:
    import multiprocess
    if multiprocess.get_start_method(allow_none=True) is None:
        multiprocess.set_start_method('spawn')
except ImportError:
    pass

import shutil
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

import pipeline_lib.text_cleaning.cleaning as _cleaning_mod  # For setting CLEANING_MODE
from pipeline_lib.text_cleaning.checkpointing import CheckpointManager, StageManager
from pipeline_lib.text_cleaning.cleaning import (
    clean_text_fast,
    clean_text_full,
    parallel_clean_texts_streaming,
)
from pipeline_lib.text_cleaning.quality_filter import (
    DATATROVE_AVAILABLE,
    DatatroveQualityFilter,
    apply_quality_filter_parallel,
    configure_datatrove_filters,
)
from pipeline_lib.text_cleaning.toxicity import DataCleaner

# Datatrove pipeline components (optional - for native pipeline mode)
try:
    from datatrove.data import Document
    from datatrove.pipeline.filters import (
        GopherQualityFilter,
        GopherRepetitionFilter,
        FineWebQualityFilter,
    )
    from datatrove.pipeline.filters.base_filter import BaseFilter
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter
    from datatrove.pipeline.dedup import (
        MinhashDedupSignature,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.dedup.minhash import MinhashConfig, HashConfig
    DATATROVE_PIPELINE_AVAILABLE = True
except ImportError:
    DATATROVE_PIPELINE_AVAILABLE = False


# =============================================================================
# NATIVE DATATROVE PIPELINE - Optimized execution using datatrove's executors
# =============================================================================

if DATATROVE_PIPELINE_AVAILABLE:
    import time as _time
    from datatrove.pipeline.base import PipelineStep

    class PIICleanerFilter(BaseFilter):
        """Custom datatrove filter for PII removal and text normalization.

        This filter modifies text in-place (removes PII, normalizes whitespace)
        and always returns True (keeps the document).
        """
        name = "pii_cleaner"

        def __init__(self, use_full_clean: bool = False):
            super().__init__()
            self.use_full_clean = use_full_clean

        def filter(self, doc: Document) -> bool:
            """Clean PII from document text."""
            if self.use_full_clean:
                doc.text = clean_text_full(doc.text)
            else:
                doc.text = clean_text_fast(doc.text)
            return True  # Always keep, just modifies text

    class ProgressTracker(PipelineStep):
        """Track and report progress through the pipeline.

        Logs progress every N documents to show pipeline is working.
        """
        name = "progress_tracker"
        type = "🔢"

        def __init__(self, log_every: int = 10000, stage_name: str = "Processing"):
            super().__init__()
            self.log_every = log_every
            self.stage_name = stage_name
            self.count = 0
            self.passed = 0
            self.start_time = None

        def run(self, data, rank: int = 0, world_size: int = 1):
            """Pass through documents while tracking progress."""
            if self.start_time is None:
                self.start_time = _time.time()

            for doc in data:
                self.count += 1
                self.passed += 1

                if self.count % self.log_every == 0:
                    elapsed = _time.time() - self.start_time
                    rate = self.count / elapsed if elapsed > 0 else 0
                    print(f"    [{self.stage_name}] Rank {rank}: {self.count:,} docs processed ({rate:.0f} docs/sec)")

                yield doc

            # Final count
            elapsed = _time.time() - self.start_time if self.start_time else 0
            rate = self.count / elapsed if elapsed > 0 else 0
            print(f"    [{self.stage_name}] Rank {rank}: COMPLETE - {self.count:,} docs in {elapsed:.1f}s ({rate:.0f} docs/sec)")


def process_with_datatrove_pipeline(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    file_pattern: str = "pretraining_",
    n_workers: int = None,
    use_gopher_quality: bool = True,
    use_fineweb: bool = True,
    use_gopher_rep: bool = True,
    use_full_clean: bool = False,
    use_gpu: bool = True,
    toxicity_batch_size: int = None,
    keep_temp_files: bool = False,
    skip_toxicity: bool = False,
) -> dict:
    """Process data using datatrove's native pipeline architecture.

    This is significantly faster than the legacy approach because:
    1. Native parallel execution without Pool overhead
    2. Streaming through pipeline (lower memory)
    3. Optimized MinhashDedup instead of custom LSH
    4. Built-in task tracking and resumption

    Pipeline stages:
    1. ParquetReader -> PIICleaner -> QualityFilters -> ParquetWriter (temp)
    2. Load temp -> ToxicityFilter (GPU) -> ParquetWriter (temp2)
    3. MinhashDedup 4-stage pipeline -> Final output

    Args:
        input_dir: Directory containing input parquet files
        output_dir: Directory for output files
        file_pattern: Pattern to match input files
        n_workers: Number of parallel workers (default: 80% of CPU cores)
        use_gopher_quality: Enable GopherQualityFilter
        use_fineweb: Enable FineWebQualityFilter
        use_gopher_rep: Enable GopherRepetitionFilter
        use_full_clean: Use full Unicode cleaning (slower but fixes mojibake)
        use_gpu: Use GPU for toxicity detection
        toxicity_batch_size: Batch size for toxicity (auto-tuned if None)
        keep_temp_files: Keep intermediate files for recovery (default: False)

    Returns:
        Dict with processing statistics
    """
    if not DATATROVE_PIPELINE_AVAILABLE:
        raise ImportError("datatrove pipeline components not available. Install with: pip install 'datatrove[io,processing]'")

    n_workers = n_workers or max(1, int(cpu_count() * 0.8))
    os.makedirs(output_dir, exist_ok=True)

    # Check if output already exists (skip if already processed)
    existing_output = [f for f in os.listdir(output_dir) if f.endswith('.parquet') and not f.startswith('.')]
    if existing_output:
        print(f"\n{'='*60}")
        print(f"SKIPPING - Cleaned data already exists")
        print(f"{'='*60}")
        print(f"Found {len(existing_output)} parquet files in {output_dir}")
        print(f"Delete these files to re-run cleaning pipeline.")
        print(f"{'='*60}\n")

        # Count existing documents for stats
        total_docs = 0
        for f in existing_output:
            pf = pq.ParquetFile(os.path.join(output_dir, f))
            total_docs += pf.metadata.num_rows

        return {
            'files_processed': len(existing_output),
            'original_docs': total_docs,
            'after_quality': total_docs,
            'after_toxicity': total_docs,
            'after_dedup': total_docs,
            'skipped': True,
        }

    # Find input files
    input_files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith(file_pattern) and f.endswith('.parquet')
    ])

    if not input_files:
        print(f"No files matching '{file_pattern}*.parquet' in {input_dir}")
        return {'files_processed': 0}

    print(f"\n{'='*60}")
    print(f"DATATROVE NATIVE PIPELINE")
    print(f"{'='*60}")
    print(f"Input files: {len(input_files)}")
    print(f"Workers: {n_workers}")
    print(f"Filters: GopherQuality={use_gopher_quality}, FineWeb={use_fineweb}, GopherRep={use_gopher_rep}")
    print(f"{'='*60}\n")

    # Temp directories for intermediate results
    # IMPORTANT: Use local storage for temp files to avoid corruption from Drive sync issues
    # In Colab, /content is local SSD while Drive is async-synced and can corrupt on session end
    if os.path.exists("/content") and os.path.islink(output_dir):
        # Running in Colab with Drive symlink - use local SSD for temp
        temp_dir = Path("/content/datatrove_temp")
        print(f"  Using local SSD for temp files: {temp_dir}")
    else:
        # Local development or no symlink - use output dir
        temp_dir = Path(output_dir) / ".datatrove_temp"

    stage1_dir = temp_dir / "stage1_filtered"
    stage2_dir = temp_dir / "stage2_toxicity"
    dedup_sig_dir = temp_dir / "dedup_signatures"
    dedup_buckets_dir = temp_dir / "dedup_buckets"
    dedup_clusters_dir = temp_dir / "dedup_clusters"

    for d in [stage1_dir, stage2_dir, dedup_sig_dir, dedup_buckets_dir, dedup_clusters_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        'files_processed': len(input_files),
        'original_docs': 0,
        'after_quality': 0,
        'after_toxicity': 0,
        'after_dedup': 0,
    }

    # Count original documents first
    print("Counting input documents...")
    for f in input_files:
        pf = pq.ParquetFile(os.path.join(input_dir, f))
        stats['original_docs'] += pf.metadata.num_rows
    print(f"  Total input documents: {stats['original_docs']:,}")

    # Check for resume from existing temp files
    # IMPORTANT: Validate files before using - corrupted files from interrupted runs crash the pipeline
    def validate_parquet_files(file_list, stage_name):
        """Validate parquet files and return (valid_files, total_rows)."""
        valid_files = []
        total_rows = 0
        for f in file_list:
            try:
                pf = pq.ParquetFile(f)
                total_rows += pf.metadata.num_rows
                valid_files.append(f)
            except Exception as e:
                print(f"  WARNING: Corrupted file {f.name} in {stage_name}, removing...")
                try:
                    f.unlink()
                except Exception:
                    pass
        return valid_files, total_rows

    stage1_files = list(stage1_dir.glob("*.parquet"))
    stage2_files = list(stage2_dir.glob("*.parquet"))
    resume_from_stage = None

    if stage2_files:
        stage2_files, stage2_rows = validate_parquet_files(stage2_files, "Stage 2")
        if stage2_files:
            print(f"\n  RESUMING: Found {len(stage2_files)} valid files from Stage 2 (toxicity)")
            resume_from_stage = 3  # Skip to deduplication
            stats['after_toxicity'] = stage2_rows
            stats['after_quality'] = stats['after_toxicity']  # Approximate
            print(f"  Skipping to Stage 3 (deduplication)")
        else:
            print(f"\n  No valid Stage 2 files found, starting fresh")

    if not resume_from_stage and stage1_files:
        stage1_files, stage1_rows = validate_parquet_files(stage1_files, "Stage 1")
        if stage1_files:
            print(f"\n  RESUMING: Found {len(stage1_files)} valid files from Stage 1 (quality)")
            resume_from_stage = 2  # Skip to toxicity
            stats['after_quality'] = stage1_rows
            print(f"  Skipping to Stage 2 (toxicity)")
        else:
            print(f"\n  No valid Stage 1 files found, starting fresh")

    # =========================================================================
    # STAGE 1: Quality Filtering with Native Pipeline
    # =========================================================================
    if resume_from_stage and resume_from_stage > 1:
        print("\nSTAGE 1: Quality Filtering - SKIPPED (using cached results)")
        print("-" * 50)
        print(f"  Documents from cache: {stats['after_quality']:,}")
    else:
        print("\nSTAGE 1: Quality Filtering (native datatrove pipeline)")
        print("-" * 50)

    # Only run Stage 1 if not resuming from a later stage
    if not resume_from_stage or resume_from_stage <= 1:
        # Build filter pipeline with progress tracking
        filters = [
            PIICleanerFilter(use_full_clean=use_full_clean),
            ProgressTracker(log_every=50000, stage_name="Quality"),  # Log every 50K docs
        ]

        if use_gopher_quality:
            filters.append(GopherQualityFilter(
                min_doc_words=50,
                max_doc_words=100000,
                min_avg_word_length=3,
                max_avg_word_length=10,
                min_stop_words=2,
                max_symbol_word_ratio=0.1,
            ))

        if use_fineweb:
            filters.append(FineWebQualityFilter())

        if use_gopher_rep:
            filters.append(GopherRepetitionFilter())

        # Add progress tracker after filters to show how many passed
        filters.append(ProgressTracker(log_every=50000, stage_name="Passed"))

        # Set tasks higher than file count to enable intra-file parallelization
        # Datatrove will shard each file across multiple tasks for better CPU utilization
        n_tasks = max(len(input_files), n_workers * 2)  # At least 2x workers for good load balancing
        print(f"  Tasks: {n_tasks} (2x workers for load balancing)")

        pipeline_stage1 = [
            ParquetReader(
                data_folder=input_dir,
                glob_pattern=f"{file_pattern}*.parquet",
                text_key="text",
            ),
            *filters,
            ParquetWriter(
                output_folder=str(stage1_dir),
                output_filename="${rank}.parquet",
            ),
        ]

        executor1 = LocalPipelineExecutor(
            pipeline=pipeline_stage1,
            logging_dir=str(temp_dir / "logs_stage1"),
            tasks=n_tasks,
            workers=n_workers,
        )

        import time as _time
        start_time = _time.time()
        print(f"  Running with {n_tasks} tasks, {n_workers} workers...")
        print(f"  Processing {stats['original_docs']:,} documents...")
        executor1.run()
        elapsed = _time.time() - start_time
        print(f"  Stage 1 completed in {elapsed:.1f}s")

        # Count results from stage 1
        stage1_files = list(stage1_dir.glob("*.parquet"))
        for f in stage1_files:
            pf = pq.ParquetFile(f)
            stats['after_quality'] += pf.metadata.num_rows
        print(f"  After quality filtering: {stats['after_quality']:,} docs")

    # =========================================================================
    # STAGE 2: Toxicity Filtering (GPU - can't be in datatrove pipeline)
    # =========================================================================
    if skip_toxicity:
        print("\nSTAGE 2: Toxicity Filtering - SKIPPED (--no-toxicity flag)")
        print("-" * 50)
        # Just copy/link files from stage1 to stage2
        toxicity_kept = 0
        for stage1_file in stage1_files:
            df = pd.read_parquet(stage1_file)
            if len(df) > 0:
                output_file = stage2_dir / stage1_file.name
                shutil.copy2(stage1_file, output_file)
                toxicity_kept += len(df)
        stats['after_toxicity'] = toxicity_kept
        print(f"  Passed through: {toxicity_kept:,} docs (no filtering)")
    elif resume_from_stage and resume_from_stage > 2:
        print("\nSTAGE 2: Toxicity Filtering - SKIPPED (using cached results)")
        print("-" * 50)
        print(f"  Documents from cache: {stats['after_toxicity']:,}")
    else:
        print("\nSTAGE 2: Toxicity Filtering (GPU)")
        print("-" * 50)

        # Only run Stage 2 if not resuming from a later stage
        if not resume_from_stage or resume_from_stage <= 2:
            # Initialize toxicity model
            cleaner = DataCleaner(use_gpu=use_gpu, batch_size=toxicity_batch_size)

            toxicity_kept = 0
            for stage1_file in tqdm(stage1_files, desc="  Processing files"):
                df = pd.read_parquet(stage1_file)
                if len(df) == 0:
                    continue

                # Run toxicity detection
                toxic_mask = cleaner.is_toxic_batch(df['text'].tolist(), show_progress=False)
                df = df[~np.array(toxic_mask)].reset_index(drop=True)

                if len(df) > 0:
                    output_file = stage2_dir / stage1_file.name
                    df.to_parquet(output_file, index=False)
                    toxicity_kept += len(df)

            stats['after_toxicity'] = toxicity_kept
            print(f"  After toxicity filtering: {toxicity_kept:,} docs")

    # =========================================================================
    # STAGE 3: Deduplication with Native MinhashDedup
    # =========================================================================
    print("\nSTAGE 3: Deduplication (native MinhashDedup)")
    print("-" * 50)
    print(f"  Input: {stats['after_toxicity']:,} documents")

    stage3_start = _time.time()

    # MinhashDedup configuration (similar to FineWeb)
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )

    stage2_files = list(stage2_dir.glob("*.parquet"))
    # Use more tasks for better parallelization
    n_dedup_tasks = max(len(stage2_files), n_workers)

    # Stage 3a: Compute signatures
    print("  3a: Computing MinHash signatures...")
    step_start = _time.time()
    sig_pipeline = [
        ParquetReader(
            data_folder=str(stage2_dir),
            glob_pattern="*.parquet",
            text_key="text",
        ),
        ProgressTracker(log_every=50000, stage_name="Signatures"),
        MinhashDedupSignature(
            output_folder=str(dedup_sig_dir),
            config=minhash_config,
        ),
    ]

    executor_sig = LocalPipelineExecutor(
        pipeline=sig_pipeline,
        logging_dir=str(temp_dir / "logs_dedup_sig"),
        tasks=n_dedup_tasks,
        workers=n_workers,
    )
    executor_sig.run()
    print(f"      Completed in {_time.time() - step_start:.1f}s")

    # Stage 3b: Find duplicate pairs in buckets
    print("  3b: Finding duplicate pairs...")
    step_start = _time.time()
    bucket_pipeline = [
        MinhashDedupBuckets(
            input_folder=str(dedup_sig_dir),
            output_folder=str(dedup_buckets_dir),
            config=minhash_config,
        ),
    ]

    executor_buckets = LocalPipelineExecutor(
        pipeline=bucket_pipeline,
        logging_dir=str(temp_dir / "logs_dedup_buckets"),
        tasks=minhash_config.num_buckets,  # One task per bucket
        workers=min(n_workers, minhash_config.num_buckets),
    )
    executor_buckets.run()
    print(f"      Completed in {_time.time() - step_start:.1f}s")

    # Stage 3c: Cluster duplicates
    print("  3c: Clustering duplicates...")
    step_start = _time.time()
    cluster_pipeline = [
        MinhashDedupCluster(
            input_folder=str(dedup_buckets_dir),
            output_folder=str(dedup_clusters_dir),
        ),
    ]

    executor_cluster = LocalPipelineExecutor(
        pipeline=cluster_pipeline,
        logging_dir=str(temp_dir / "logs_dedup_cluster"),
        tasks=1,  # Clustering is single-task
        workers=1,
    )
    executor_cluster.run()
    print(f"      Completed in {_time.time() - step_start:.1f}s")

    # Stage 3d: Filter out duplicates and write final output
    # Output directly to output_dir (not a subdirectory) for compatibility with tokenize script
    print("  3d: Filtering duplicates and writing output...")
    step_start = _time.time()
    final_output_dir = Path(output_dir)  # Write directly to output_dir

    filter_pipeline = [
        ParquetReader(
            data_folder=str(stage2_dir),
            glob_pattern="*.parquet",
            text_key="text",
        ),
        MinhashDedupFilter(
            input_folder=str(dedup_clusters_dir),
        ),
        ProgressTracker(log_every=50000, stage_name="Dedup"),
        ParquetWriter(
            output_folder=str(final_output_dir),
            output_filename="cleaned_${rank}.parquet",  # Prefix to identify native pipeline output
        ),
    ]

    executor_filter = LocalPipelineExecutor(
        pipeline=filter_pipeline,
        logging_dir=str(temp_dir / "logs_dedup_filter"),
        tasks=n_dedup_tasks,
        workers=n_workers,
    )
    executor_filter.run()
    print(f"      Completed in {_time.time() - step_start:.1f}s")

    # Count final results
    final_files = list(final_output_dir.glob("*.parquet"))
    for f in final_files:
        pf = pq.ParquetFile(f)
        stats['after_dedup'] += pf.metadata.num_rows

    stage3_elapsed = _time.time() - stage3_start
    print(f"  Stage 3 total: {stage3_elapsed:.1f}s")
    print(f"  After deduplication: {stats['after_dedup']:,} docs ({stats['after_dedup']/stats['after_toxicity']*100:.1f}% kept)")

    # =========================================================================
    # Cleanup temp files (optional - keep for recovery if requested)
    # =========================================================================
    if keep_temp_files:
        print(f"\nKeeping temporary files for recovery: {temp_dir}")
        print("  To clean up manually: rm -rf " + str(temp_dir))
    else:
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  After quality filtering: {stats['after_quality']:,}")
    print(f"  After toxicity filtering: {stats['after_toxicity']:,}")
    print(f"  After deduplication: {stats['after_dedup']:,}")
    print(f"  Final output: {final_output_dir}")
    print(f"{'='*60}\n")

    return stats


def process_single_file(
    filename: str,
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    cache_dir: str = "data/.cache",
    use_gpu: bool = True,
    use_cache: bool = True,
    batch_size: int = 128,
    n_workers: int = None,
    drive_dir: str = None,
    sync_threads: int = 10,
    auto_sync: bool = True,
) -> tuple[str, int, int]:
    """Process a single file with checkpointing and stage-based recovery.

    Args:
        filename: Input filename
        input_dir: Directory containing input files
        output_dir: Directory for output files
        cache_dir: Directory for intermediate checkpoint chunks
        use_gpu: Use GPU for toxicity detection
        use_cache: Enable checkpoint caching
        batch_size: Batch size for toxicity detection
        n_workers: Number of CPU workers
        drive_dir: Google Drive directory for sync (None to disable)
        sync_threads: Number of threads for Drive sync
        auto_sync: Auto-sync to Drive after each stage
    """
    # Initialize checkpoint manager first (needed for stage manager)
    checkpoint = CheckpointManager(cache_dir) if use_cache else None
    checkpoint_dir = str(checkpoint.cache_dir) if checkpoint else cache_dir

    # Initialize stage manager for recovery (pass checkpoint_dir for proper file syncing)
    stage_mgr = StageManager(output_dir, drive_dir, checkpoint_dir=checkpoint_dir) if auto_sync else None
    if stage_mgr:
        stage_mgr.print_status()

    input_path = os.path.join(input_dir, filename)
    output_filename = filename.replace('.parquet', '_clean.parquet')
    output_path = os.path.join(output_dir, output_filename)

    # Skip if already fully processed (check local first, then Drive)
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already processed)")
        # Read to get stats
        df = pd.read_parquet(output_path)
        return filename, len(df), len(df)

    # Check Drive for completed output if local doesn't exist
    # This handles Colab restarts where local SSD is wiped but Drive has the data
    if drive_dir and not os.path.exists(output_path):
        drive_output_path = os.path.join(drive_dir, output_filename)
        if os.path.exists(drive_output_path):
            print(f"Restoring {output_filename} from Google Drive...")
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(drive_output_path, output_path)
            if os.path.exists(output_path):
                print(f"Skipping {filename} (restored from Drive)")
                df = pd.read_parquet(output_path)
                return filename, len(df), len(df)

    print(f"\nProcessing {filename}...")
    file_hash = checkpoint.get_file_hash(input_path) if checkpoint else ""

    try:
        # Check for existing state
        state = checkpoint.load_state(filename, file_hash) if checkpoint else None
        completed_steps = state.get('completed', []) if state else []
        original_count = state.get('original_count', 0) if state else 0

        # Step 1: Load and clean text (PARALLEL - uses all CPU cores)

        # Get row count from metadata (instant, no data loading)
        parquet_file = pq.ParquetFile(input_path)
        original_count = parquet_file.metadata.num_rows
        print(f"  Source file: {original_count:,} documents")

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
                for chunk_path in existing_chunks:
                    chunk_pf = pq.ParquetFile(chunk_path)
                    start_idx += chunk_pf.metadata.num_rows
                print(f"    {start_idx:,} documents already cleaned")

        remaining = original_count - start_idx
        # Track chunks for append-only saves
        current_chunk_idx = len(existing_chunks)

        if remaining <= 0:
            print(f"  All documents already cleaned from chunks")
        else:
            print(f"  Cleaning {remaining:,} documents (of {original_count:,} total)...")

            # Stream parquet in batches - never load entire file at once
            BATCH_SIZE = 100000  # Read 100K rows at a time from parquet
            rows_seen = 0
            texts_buffer = []
            chunks_generated = 0

            # Streaming chunk callback - saves each chunk to disk immediately
            # Also syncs to Drive periodically (every SYNC_EVERY_N_CHUNKS chunks)
            SYNC_EVERY_N_CHUNKS = 5  # Sync every 5 chunks (~2.5M docs) to avoid data loss
            last_sync_chunk = [0]  # Use list to allow mutation in nested function

            def save_chunk_streaming(chunk_data, idx):
                nonlocal current_chunk_idx
                if checkpoint:
                    actual_idx = current_chunk_idx + idx
                    chunk_path = chunk_dir / f"{filename}_clean_chunk_{actual_idx:04d}_{file_hash}.parquet"
                    # DEBUG: Clear line and show chunk save progress clearly
                    print(f"")  # Force newline to escape tqdm
                    print(f"    [Saving chunk {actual_idx}: {len(chunk_data):,} docs to {chunk_path.name}...]", end="", flush=True)
                    chunk_df = pd.DataFrame({'text': chunk_data})
                    chunk_df.to_parquet(chunk_path, index=False)
                    # Verify file was created
                    if chunk_path.exists():
                        print(f" done ({chunk_path.stat().st_size / 1024 / 1024:.1f} MB)]")
                    else:
                        print(f" FAILED - file not created!]")

                    # Periodic sync to Drive to preserve progress
                    chunks_since_sync = actual_idx - last_sync_chunk[0]
                    if stage_mgr and chunks_since_sync >= SYNC_EVERY_N_CHUNKS:
                        print(f" [syncing to Drive...]", end="", flush=True)
                        synced = stage_mgr.sync_to_drive('text_clean', max_workers=sync_threads)
                        last_sync_chunk[0] = actual_idx
                        if synced > 0:
                            print(f" {synced} files]", end="")

            print(f"    Streaming from parquet (batch_size={BATCH_SIZE:,})...")

            # Stream batches from parquet file
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=['text']):
                batch_texts = batch.column('text').to_pylist()
                batch_start = rows_seen
                batch_end = rows_seen + len(batch_texts)

                # Skip already-processed rows
                if batch_end <= start_idx:
                    rows_seen = batch_end
                    continue

                # Partial skip (batch straddles start_idx)
                if batch_start < start_idx:
                    batch_texts = batch_texts[start_idx - batch_start:]

                texts_buffer.extend(batch_texts)
                rows_seen = batch_end

                # When buffer is large enough, process it through cleaning
                if len(texts_buffer) >= 500000:
                    print(f"\n    [DEBUG: Processing buffer of {len(texts_buffer):,} texts, current_chunk_idx={current_chunk_idx}]")
                    batch_chunks = 0
                    for chunk in parallel_clean_texts_streaming(
                        texts_buffer,
                        n_workers=n_workers,
                        chunk_callback=save_chunk_streaming if checkpoint else None,
                        chunk_size=500000
                    ):
                        batch_chunks += 1
                    # CRITICAL: Update current_chunk_idx AFTER each streaming call
                    # so the next call uses correct chunk indices (fixes overwrite bug)
                    print(f"    [DEBUG: Streaming call produced {batch_chunks} chunk(s), updating current_chunk_idx {current_chunk_idx} -> {current_chunk_idx + batch_chunks}]")
                    current_chunk_idx += batch_chunks
                    chunks_generated += batch_chunks
                    texts_buffer = []

            # Process remaining texts in buffer
            if texts_buffer:
                batch_chunks = 0
                for chunk in parallel_clean_texts_streaming(
                    texts_buffer,
                    n_workers=n_workers,
                    chunk_callback=save_chunk_streaming if checkpoint else None,
                    chunk_size=500000
                ):
                    batch_chunks += 1
                current_chunk_idx += batch_chunks
                chunks_generated += batch_chunks

        # Stage 1 complete: Text cleaning done
        if stage_mgr:
            stage_mgr.stage_complete_callback(
                'text_clean',
                stats={'chunks': current_chunk_idx, 'docs_cleaned': original_count},
                sync_threads=sync_threads,
                cleanup=False  # Don't cleanup - need chunks for next stage
            )

        # Process each chunk through quality + toxicity filters (memory-efficient)
        # Instead of combining all chunks then filtering, we filter each chunk
        print(f"  Processing {current_chunk_idx} chunks through quality/toxicity filters...")
        all_chunk_files = sorted(chunk_dir.glob(chunk_pattern)) if chunk_dir else []

        # DEBUG: Verify chunk files exist
        print(f"    [DEBUG: Looking for pattern '{chunk_pattern}' in {chunk_dir}]")
        print(f"    [DEBUG: Found {len(all_chunk_files)} chunk files]")
        if all_chunk_files:
            for cf in all_chunk_files[:5]:  # Show first 5
                print(f"      - {cf.name} ({cf.stat().st_size / 1024 / 1024:.1f} MB)")
            if len(all_chunk_files) > 5:
                print(f"      ... and {len(all_chunk_files) - 5} more")
        if len(all_chunk_files) != current_chunk_idx:
            print(f"    [DEBUG: WARNING! Expected {current_chunk_idx} chunks but found {len(all_chunk_files)}!]")

        # Initialize toxicity model once
        cleaner = DataCleaner(use_gpu=use_gpu, batch_size=batch_size)

        filtered_chunks = []
        total_after_quality = 0
        total_after_toxicity = 0

        # Initialize datatrove quality filter once (if available)
        quality_filter = DatatroveQualityFilter(n_workers=n_workers) if DATATROVE_AVAILABLE else None
        filter_type = "datatrove (Gopher+FineWeb) parallel" if quality_filter else "basic"
        print(f"    Using {filter_type} quality filters ({n_workers} workers)")

        # Aggregate stats across all chunks
        total_quality_stats = {
            'passed': 0,
            'failed_repetition': 0,
            'failed_quality': 0,
            'failed_fineweb': 0,
            'failed_error': 0,
        }

        for i, chunk_path in enumerate(all_chunk_files):
            # Load one chunk at a time
            chunk_df = pd.read_parquet(chunk_path)
            chunk_size = len(chunk_df)
            print(f"    Chunk {i+1}/{len(all_chunk_files)}: {chunk_size:,} docs")

            # Quality filter (datatrove production filters or basic fallback)
            if quality_filter:
                # Datatrove: Gopher repetition + quality + FineWeb filters (parallel with stats)
                quality_mask, chunk_stats = quality_filter.filter_batch_with_stats(
                    chunk_df['text'].tolist(), show_progress=True
                )
                # Accumulate stats
                for key in total_quality_stats:
                    total_quality_stats[key] += chunk_stats.get(key, 0)
                # Print per-chunk rejection breakdown
                print(f"      Rejection breakdown:")
                print(f"        - Passed: {chunk_stats['passed']:,} ({chunk_stats['passed']/chunk_size*100:.1f}%)")
                if chunk_stats['failed_repetition'] > 0:
                    print(f"        - Failed repetition: {chunk_stats['failed_repetition']:,} ({chunk_stats['failed_repetition']/chunk_size*100:.1f}%)")
                if chunk_stats['failed_quality'] > 0:
                    print(f"        - Failed quality: {chunk_stats['failed_quality']:,} ({chunk_stats['failed_quality']/chunk_size*100:.1f}%)")
                if chunk_stats['failed_fineweb'] > 0:
                    print(f"        - Failed fineweb: {chunk_stats['failed_fineweb']:,} ({chunk_stats['failed_fineweb']/chunk_size*100:.1f}%)")
                if chunk_stats['failed_error'] > 0:
                    print(f"        - Failed error: {chunk_stats['failed_error']:,} ({chunk_stats['failed_error']/chunk_size*100:.1f}%)")
            else:
                # Basic fallback: word count + unique chars (parallel)
                quality_mask = apply_quality_filter_parallel(
                    chunk_df['text'].tolist(),
                    use_datatrove=False,
                    n_workers=n_workers,
                    show_progress=True
                )
            chunk_df = chunk_df[quality_mask].reset_index(drop=True)
            total_after_quality += len(chunk_df)
            print(f"      After quality: {len(chunk_df):,} docs ({len(chunk_df)/chunk_size*100:.1f}% kept)")

            # Toxicity filter (GPU-accelerated)
            if len(chunk_df) > 0:
                toxic_mask = cleaner.is_toxic_batch(chunk_df['text'].tolist(), show_progress=True)
                chunk_df = chunk_df[~np.array(toxic_mask)].reset_index(drop=True)
            total_after_toxicity += len(chunk_df)
            print(f"      After toxicity: {len(chunk_df):,} docs")

            # Save filtered chunk
            if len(chunk_df) > 0:
                filtered_path = chunk_dir / f"{filename}_filtered_chunk_{i:04d}_{file_hash}.parquet"
                chunk_df.to_parquet(filtered_path, index=False)
                filtered_chunks.append(filtered_path)

                # Periodic sync to Drive (every 5 chunks) to preserve filter progress
                if stage_mgr and (i + 1) % 5 == 0:
                    print(f"      [Syncing filtered chunks to Drive...]", end="", flush=True)
                    synced = stage_mgr.sync_to_drive('quality_filter', max_workers=sync_threads)
                    if synced > 0:
                        print(f" {synced} files synced]")
                    else:
                        print(f" up to date]")

            # Delete original chunk to free disk space
            chunk_path.unlink()

        print(f"\n    Quality filter summary:")
        print(f"      Total passed: {total_after_quality:,}/{original_count:,} ({total_after_quality/original_count*100:.1f}%)")
        if quality_filter and sum(total_quality_stats.values()) > 0:
            total_filtered = original_count
            # Check for filter initialization issues
            if total_quality_stats.get('no_filters', 0) > 0:
                print(f"      WARNING: Filters not initialized in {total_quality_stats['no_filters']:,} docs!")
                print(f"      This is likely a multiprocessing issue. Try --fast-quality or check datatrove installation.")
            print(f"      Rejection reasons:")
            if total_quality_stats['failed_repetition'] > 0:
                print(f"        - Repetition (spam/duplicates): {total_quality_stats['failed_repetition']:,} ({total_quality_stats['failed_repetition']/total_filtered*100:.1f}%)")
            if total_quality_stats['failed_quality'] > 0:
                print(f"        - Quality (word count/stop words): {total_quality_stats['failed_quality']:,} ({total_quality_stats['failed_quality']/total_filtered*100:.1f}%)")
            if total_quality_stats['failed_fineweb'] > 0:
                print(f"        - FineWeb (line structure): {total_quality_stats['failed_fineweb']:,} ({total_quality_stats['failed_fineweb']/total_filtered*100:.1f}%)")
            if total_quality_stats['failed_error'] > 0:
                print(f"        - Errors: {total_quality_stats['failed_error']:,} ({total_quality_stats['failed_error']/total_filtered*100:.1f}%)")
        print(f"    After toxicity filter: {total_after_toxicity:,}/{original_count:,} ({total_after_toxicity/original_count*100:.1f}%)")

        # Stage 2 & 3 complete: Quality + Toxicity filtering done
        if stage_mgr:
            stage_mgr.stage_complete_callback(
                'quality_filter',
                stats={'after_quality': total_after_quality, 'quality_stats': total_quality_stats},
                sync_threads=sync_threads,
                cleanup=True  # Cleanup text_clean stage
            )
            stage_mgr.stage_complete_callback(
                'toxicity_filter',
                stats={'after_toxicity': total_after_toxicity},
                sync_threads=sync_threads,
                cleanup=True  # Cleanup quality_filter stage
            )

        # Stream deduplication - process chunks one at a time, write output incrementally
        # This avoids loading all filtered data into memory at once
        print(f"  Streaming deduplication across {len(filtered_chunks)} filtered chunks...")

        dedup_cleaner = DataCleaner(use_gpu=False)  # Dedup is CPU-only
        doc_counter = 0
        kept_buffer = []
        final_count = 0
        WRITE_BATCH = 100000  # Write every 100K docs to avoid memory buildup

        schema = pa.schema([('text', pa.string()), ('source', pa.string())])
        temp_output_path = output_path + '.tmp'
        writer = None

        for chunk_idx, p in enumerate(tqdm(filtered_chunks, desc="    Deduplicating")):
            chunk_df = pd.read_parquet(p)

            for text in chunk_df['text']:
                doc_id = f"{filename}_{doc_counter}"
                doc_counter += 1

                m = dedup_cleaner.compute_minhash(text)
                if not dedup_cleaner.lsh.query(m):
                    dedup_cleaner.lsh.insert(doc_id, m)
                    kept_buffer.append({'text': text, 'source': filename})

                    # Write batch when buffer is full
                    if len(kept_buffer) >= WRITE_BATCH:
                        batch_df = pd.DataFrame(kept_buffer)
                        table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temp_output_path, schema)
                        writer.write_table(table)
                        final_count += len(kept_buffer)
                        print(f"\n      [Written {final_count:,} docs so far]", end="", flush=True)
                        kept_buffer = []

            # Delete filtered chunk after processing
            p.unlink()

        # Write final batch
        if kept_buffer:
            batch_df = pd.DataFrame(kept_buffer)
            table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_output_path, schema)
            writer.write_table(table)
            final_count += len(kept_buffer)

        if writer:
            writer.close()

        print(f"\n    After deduplication: {final_count}/{original_count} ({final_count/original_count*100:.1f}%)")

        # Rename temp file to final output
        if os.path.exists(temp_output_path):
            os.rename(temp_output_path, output_path)
            print(f"Saved {output_filename}: {final_count}/{original_count} documents")
        else:
            # No docs passed all filters
            df = pd.DataFrame({'text': [], 'source': []})
            df.to_parquet(output_path, index=False)
            print(f"Saved {output_filename}: 0/{original_count} documents (all filtered)")

        # Ensure output directory exists (for cases where streaming wrote the file)
        os.makedirs(output_dir, exist_ok=True)

        # Stage 4 & 5 complete: Deduplication done, final output ready
        if stage_mgr:
            stage_mgr.stage_complete_callback(
                'dedup',
                stats={'after_dedup': final_count, 'dedup_ratio': final_count/original_count if original_count > 0 else 0},
                sync_threads=sync_threads,
                cleanup=True  # Cleanup toxicity_filter stage
            )
            stage_mgr.stage_complete_callback(
                'final',
                stats={'final_count': final_count, 'original_count': original_count},
                sync_threads=sync_threads,
                cleanup=True  # Cleanup dedup stage
            )

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
    cache_dir: str = "data/.cache",
    file_pattern: str = "pretraining_",
    use_gpu: bool = True,
    use_cache: bool = True,
    batch_size: int = 128,
    n_workers: int = None,
    drive_dir: str = None,
    sync_threads: int = 10,
    auto_sync: bool = True,
) -> None:
    """Process all matching files sequentially (best for GPU).

    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        cache_dir: Directory for intermediate checkpoint chunks
        file_pattern: Pattern to match input files
        use_gpu: Use GPU for toxicity detection
        use_cache: Enable checkpoint caching
        batch_size: Batch size for toxicity detection
        n_workers: Number of CPU workers
        drive_dir: Google Drive directory for sync (None to disable)
        sync_threads: Number of threads for Drive sync
        auto_sync: Auto-sync to Drive after each stage
    """
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith(file_pattern) and f.endswith('.parquet')
    ])

    if n_workers is None:
        n_workers = max(1, int(cpu_count() * 0.8))

    print(f"\n{'='*60}")
    print(f"Found {len(files_to_process)} files to process")
    print(f"GPU acceleration: {use_gpu and torch.cuda.is_available()}")
    print(f"Caching enabled: {use_cache}")
    print(f"Batch size: {batch_size}")
    print(f"CPU workers: {n_workers}")
    if drive_dir and auto_sync:
        print(f"Drive sync: ENABLED ({sync_threads} threads)")
        print(f"Drive path: {drive_dir}")
    else:
        print(f"Drive sync: DISABLED")
    print(f"{'='*60}\n")

    results = []
    for filename in files_to_process:
        result = process_single_file(
            filename, input_dir, output_dir, cache_dir=cache_dir,
            use_gpu=use_gpu, use_cache=use_cache, batch_size=batch_size,
            n_workers=n_workers, drive_dir=drive_dir, sync_threads=sync_threads,
            auto_sync=auto_sync
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
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for toxicity detection (default: auto-tune based on GPU memory)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of CPU workers for text cleaning (default: all cores)')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory (default: data/raw)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory (default: data/processed)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Cache directory for intermediate chunks (default: auto - uses .cache sibling to output-dir)')
    parser.add_argument('--fast-clean', action='store_true', default=True,
                       help='Use fast cleaning (skip Unicode fixing) [default: enabled]')
    parser.add_argument('--full-clean', action='store_true',
                       help='Use full cleaning with plsfix (Rust-based, 10x faster than ftfy)')
    # Datatrove filter options
    parser.add_argument('--no-repetition-filter', action='store_true',
                       help='Skip GopherRepetitionFilter (2x faster, but misses spam)')
    parser.add_argument('--no-fineweb-filter', action='store_true',
                       help='Skip FineWebQualityFilter (15% faster, less strict)')
    parser.add_argument('--fast-quality', action='store_true',
                       help='Only use GopherQualityFilter (fastest, basic filtering)')
    parser.add_argument('--no-toxicity', action='store_true',
                       help='Skip toxicity filtering (faster, use for pretraining data)')
    parser.add_argument('--native-pipeline', action='store_true',
                       help='Use native datatrove pipeline (faster, uses MinhashDedup)')
    parser.add_argument('--legacy', action='store_true',
                       help='Force legacy pipeline mode (default if --native-pipeline not specified)')
    parser.add_argument('--keep-temp-files', action='store_true',
                       help='Keep intermediate temp files for recovery (native pipeline only)')
    parser.add_argument('--fresh', action='store_true',
                       help='Force fresh start by removing all intermediate/cached files')
    # Google Drive sync options (for Colab)
    parser.add_argument('--drive-dir', type=str, default=None,
                       help='Google Drive directory for sync (e.g., /content/drive/MyDrive/llm-training-pipeline/data/processed)')
    parser.add_argument('--sync-threads', type=int, default=10,
                       help='Number of threads for Drive sync (default: 10)')
    parser.add_argument('--no-auto-sync', action='store_true',
                       help='Disable automatic sync to Drive after each stage')

    args = parser.parse_args()

    # Auto-derive cache-dir from output-dir if not specified
    # This ensures cache is on same storage as output (e.g., local SSD)
    if args.cache_dir is None:
        output_parent = Path(args.output_dir).parent
        args.cache_dir = str(output_parent / '.cache')

    # Set cleaning mode globally (affects worker processes via module-level variable)
    if args.full_clean:
        _cleaning_mod.CLEANING_MODE = 'full'
        print("Cleaning mode: FULL (with plsfix Unicode/mojibake fixing)")
    else:
        _cleaning_mod.CLEANING_MODE = 'fast'
        print("Cleaning mode: FAST (skip Unicode fixing, ~5-10x faster)")

    # Configure datatrove filters
    if args.fast_quality:
        # Fastest: only basic quality checks
        configure_datatrove_filters(
            use_gopher_quality=True,
            use_fineweb=False,
            use_gopher_rep=False
        )
        print("Quality filters: FAST (GopherQuality only)")
    else:
        use_rep = not args.no_repetition_filter
        use_fineweb = not args.no_fineweb_filter
        configure_datatrove_filters(
            use_gopher_quality=True,
            use_fineweb=use_fineweb,
            use_gopher_rep=use_rep
        )
        filters_enabled = ["GopherQuality"]
        if use_fineweb: filters_enabled.append("FineWeb")
        if use_rep: filters_enabled.append("GopherRepetition")
        print(f"Quality filters: {', '.join(filters_enabled)}")

    use_gpu = torch.cuda.is_available() and not args.no_gpu
    # Default to 80% of CPU cores - streaming approach prevents memory buildup
    n_workers = args.workers if args.workers else max(1, int(cpu_count() * 0.8))

    # Determine which pipeline to use
    use_native = args.native_pipeline and DATATROVE_PIPELINE_AVAILABLE and not args.legacy

    if use_native:
        print("Pipeline mode: NATIVE DATATROVE (optimized)")
        print(f"Starting native datatrove pipeline:")
        print(f"  - GPU acceleration: {use_gpu}")
        print(f"  - CPU workers: {n_workers}")
        print(f"  - File pattern: {args.pattern}")
        print(f"  - Toxicity filtering: {'DISABLED' if args.no_toxicity else 'ENABLED'}")
        print()

        # Determine filter settings
        if args.fast_quality:
            use_gopher_quality = True
            use_fineweb = False
            use_gopher_rep = False
        else:
            use_gopher_quality = True
            use_fineweb = not args.no_fineweb_filter
            use_gopher_rep = not args.no_repetition_filter

        # Handle --fresh flag: remove all intermediate files
        if args.fresh:
            temp_dir = Path(args.output_dir) / ".datatrove_temp"
            if temp_dir.exists():
                print(f"  --fresh: Removing intermediate files in {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            # Also remove any output files to force complete reprocessing
            for f in Path(args.output_dir).glob("*.parquet"):
                print(f"  --fresh: Removing {f.name}")
                f.unlink()

        process_with_datatrove_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            n_workers=n_workers,
            use_gopher_quality=use_gopher_quality,
            use_fineweb=use_fineweb,
            use_gopher_rep=use_gopher_rep,
            use_full_clean=args.full_clean,
            use_gpu=use_gpu,
            toxicity_batch_size=args.batch_size,
            keep_temp_files=args.keep_temp_files,
            skip_toxicity=args.no_toxicity,
        )
    else:
        if args.native_pipeline and not DATATROVE_PIPELINE_AVAILABLE:
            print("Warning: --native-pipeline requested but datatrove pipeline not available")
            print("Falling back to legacy pipeline. Install with: pip install 'datatrove[io,processing]'")

        print("Pipeline mode: LEGACY (with multiprocessing)")
        print(f"Starting optimized data cleaning:")
        print(f"  - GPU acceleration: {use_gpu}")
        print(f"  - Checkpoint caching: {not args.no_cache}")
        print(f"  - Cache dir: {args.cache_dir}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - CPU workers: {n_workers}")
        print(f"  - File pattern: {args.pattern}")
        if args.drive_dir and not args.no_auto_sync:
            print(f"  - Drive sync: ENABLED ({args.sync_threads} threads)")
            print(f"  - Drive path: {args.drive_dir}")
        else:
            print(f"  - Drive sync: DISABLED")
        print()

        process_all_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            file_pattern=args.pattern,
            use_gpu=use_gpu,
            use_cache=not args.no_cache,
            batch_size=args.batch_size,
            n_workers=n_workers,
            drive_dir=args.drive_dir,
            sync_threads=args.sync_threads,
            auto_sync=not args.no_auto_sync,
        )


if __name__ == "__main__":
    main()
