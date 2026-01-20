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
        backup_dir: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        fast_quality: bool = False,
        skip_quality: bool = False,
        skip_toxicity: bool = False,
        skip_dedup: bool = False,
        cleanup_intermediate: bool = False,
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
            backup_dir: Optional backup directory (e.g., Google Drive) for incremental sync
            use_gpu: Force GPU (True) or CPU (False). None = auto-detect.
            fast_quality: Skip expensive repetition filters
            skip_quality: Skip ALL quality filtering (fastest)
            skip_toxicity: Skip toxicity filtering
            skip_dedup: Skip deduplication
            cleanup_intermediate: Delete intermediate directories after each stage to save disk space
            toxicity_threshold: Threshold for toxicity detection
            dedup_threshold: Jaccard similarity threshold for deduplication
            batch_size: Batch size for GPU processing
            show_progress: Show progress bars
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.fast_quality = fast_quality
        self.skip_quality = skip_quality
        self.skip_toxicity = skip_toxicity
        self.skip_dedup = skip_dedup
        self.cleanup_intermediate = cleanup_intermediate
        self.toxicity_threshold = toxicity_threshold
        self.dedup_threshold = dedup_threshold
        self.batch_size = batch_size
        self.show_progress = show_progress

        # Create backup subdirectories if backup_dir specified
        if self.backup_dir:
            (self.backup_dir / "cleaned").mkdir(parents=True, exist_ok=True)
            (self.backup_dir / "quality_filtered").mkdir(parents=True, exist_ok=True)
            (self.backup_dir / "toxicity_filtered").mkdir(parents=True, exist_ok=True)

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

    def _check_stage_complete(self, stage: int) -> bool:
        """Check if a stage has already been completed.

        Args:
            stage: Stage number (1-4)

        Returns:
            True if stage is complete and can be skipped
        """
        if stage == 1:
            cleaned_dir = self.cache_dir / "cleaned"
            if cleaned_dir.exists():
                files = list(cleaned_dir.glob("*.parquet"))
                if len(files) > 0:
                    # Check if we have a reasonable number of cleaned files
                    return True
            return False

        elif stage == 2:
            quality_dir = self.cache_dir / "quality_filtered"
            if quality_dir.exists():
                files = list(quality_dir.glob("*.parquet"))
                if len(files) > 0:
                    return True
            return False

        elif stage == 3:
            toxicity_dir = self.cache_dir / "toxicity_filtered"
            if toxicity_dir.exists():
                files = list(toxicity_dir.glob("*.parquet"))
                if len(files) > 0:
                    return True
            return False

        elif stage == 4:
            final_output = self.output_dir / "processed.parquet"
            return final_output.exists()

        return False

    def _count_docs_in_dir(self, dir_path: Path) -> int:
        """Count total documents in all parquet files in a directory."""
        total = 0
        for f in dir_path.glob("*.parquet"):
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(f)
                total += pf.metadata.num_rows
            except Exception:
                pass
        return total

    def _backup_file(self, local_file: Path, stage: str) -> None:
        """Backup a single file to the backup directory."""
        if not self.backup_dir:
            return
        try:
            import shutil
            backup_path = self.backup_dir / stage / local_file.name
            shutil.copy2(local_file, backup_path)
        except Exception as e:
            print(f"  Warning: Backup failed for {local_file.name}: {e}")

    def _restore_from_backup(self, local_dir: Path, stage: str) -> int:
        """Restore files from backup directory to local cache.

        Returns:
            Number of files restored
        """
        if not self.backup_dir:
            return 0

        backup_stage_dir = self.backup_dir / stage
        if not backup_stage_dir.exists():
            return 0

        local_dir.mkdir(parents=True, exist_ok=True)
        restored = 0

        for backup_file in backup_stage_dir.glob("*.parquet"):
            local_file = local_dir / backup_file.name
            if not local_file.exists():
                try:
                    import shutil
                    shutil.copy2(backup_file, local_file)
                    restored += 1
                except Exception:
                    pass

        return restored

    def _get_processed_files(self, local_dir: Path, stage: str) -> set:
        """Get set of already-processed file prefixes from local + backup."""
        processed = set()

        # Check local cache
        for f in local_dir.glob("*.parquet"):
            # Extract the source file prefix (e.g., "pretraining_fineweb-edu-sample")
            # Filename format: cleaned_{file_idx:03d}_{chunk_idx:04d}_{stem[:30]}.parquet
            parts = f.stem.split('_', 3)
            if len(parts) >= 4:
                processed.add(parts[3])  # The stem[:30] part

        # Check backup
        if self.backup_dir:
            backup_dir = self.backup_dir / stage
            if backup_dir.exists():
                for f in backup_dir.glob("*.parquet"):
                    parts = f.stem.split('_', 3)
                    if len(parts) >= 4:
                        processed.add(parts[3])

        return processed

    def process(self, resume: bool = True) -> dict:
        """Run the full GPU pipeline.

        Args:
            resume: If True, skip completed stages (default: True)

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
            'resumed_from_stage': None,
        }

        print("=" * 60)
        print("GPU DATA PIPELINE (Memory-Optimized)")
        print("=" * 60)
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"GPU Text Cleaning: {self.use_gpu_text} (RAPIDS)")
        print(f"GPU Deduplication: {self.use_gpu_dedup} (NeMo)")
        print(f"Quality Filters: {'SKIPPED' if self.skip_quality else [f[0] for f in self.quality_filters]}")
        print(f"Toxicity Filter: {not self.skip_toxicity}")
        print(f"Deduplication: {not self.skip_dedup}")
        print(f"Resume Mode: {resume}")
        print("=" * 60)

        # Check for resume points
        if resume:
            for stage in [4, 3, 2, 1]:
                if self._check_stage_complete(stage):
                    stats['resumed_from_stage'] = stage + 1
                    print(f"\n*** RESUMING: Stage {stage} complete, starting from Stage {stage + 1} ***")
                    break

        # Define stage directories
        cleaned_dir = self.cache_dir / "cleaned"
        quality_dir = self.cache_dir / "quality_filtered"
        toxicity_dir = self.cache_dir / "toxicity_filtered"

        # Stage 1: Load, clean, and save file-by-file (streaming to avoid OOM)
        skip_stage1 = resume and self._check_stage_complete(1)
        stage1_start = time.time()

        if skip_stage1:
            print("\n[Stage 1/4] SKIPPED (already complete)")
            stats['after_cleaning'] = self._count_docs_in_dir(cleaned_dir)
            # When resuming, input_docs is unknown - use after_cleaning as baseline
            stats['input_docs'] = stats['after_cleaning']
            print(f"  Found {stats['after_cleaning']:,} cleaned docs in cache")
        else:
            print("\n[Stage 1/4] Loading and cleaning text (streaming mode)...")
            cleaned_dir.mkdir(parents=True, exist_ok=True)

            # Restore any existing files from backup (if backup_dir specified)
            if self.backup_dir:
                restored = self._restore_from_backup(cleaned_dir, "cleaned")
                if restored > 0:
                    print(f"  Restored {restored} chunk files from backup")

            # Get already-processed file prefixes (for resume within stage)
            processed_prefixes = self._get_processed_files(cleaned_dir, "cleaned")
            if processed_prefixes:
                print(f"  Found {len(processed_prefixes)} already-processed file prefixes")

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

            total_input = 0
            total_cleaned = 0

            # Process each file individually with TRUE STREAMING to avoid memory accumulation
            # Use pyarrow for chunked reading - never load full file into memory
            import pyarrow.parquet as pq

            for file_idx, pq_file in enumerate(tqdm(pretraining_files, desc="  Processing files", disable=not self.show_progress)):
                try:
                    # Check if this file was already processed (resume support)
                    file_prefix = pq_file.stem[:30]
                    if file_prefix in processed_prefixes:
                        # Count existing chunks for this file
                        existing_chunks = list(cleaned_dir.glob(f"cleaned_{file_idx:03d}_*_{file_prefix}.parquet"))
                        if existing_chunks:
                            existing_docs = sum(len(pd.read_parquet(f)) for f in existing_chunks)
                            total_cleaned += existing_docs
                            print(f"    {pq_file.name}: SKIPPED (already processed, {existing_docs:,} docs)")
                            continue

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
                    file_chunk_files = []  # Track chunk files for this input file

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
                            file_chunk_files.append(cleaned_file)
                            file_cleaned += len(cleaned_df)
                            total_cleaned += len(cleaned_df)
                            del cleaned_df

                        del cleaned_texts, all_ids
                        gc.collect()

                        chunk_idx += 1

                    del parquet_file
                    gc.collect()

                    # Backup all chunks for this file after processing completes
                    if self.backup_dir and file_chunk_files:
                        for chunk_file in file_chunk_files:
                            self._backup_file(chunk_file, "cleaned")

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
        skip_stage2 = resume and self._check_stage_complete(2)
        stage2_start = time.time()

        if skip_stage2:
            print("\n[Stage 2/4] SKIPPED (already complete)")
            stats['after_quality'] = self._count_docs_in_dir(quality_dir)
            print(f"  Found {stats['after_quality']:,} quality-filtered docs in cache")
        elif self.skip_quality:
            # Fast mode: skip quality filtering entirely, use symlinks to save disk space
            print("\n[Stage 2/4] SKIPPED (--skip-quality mode) - using symlinks")
            quality_dir.mkdir(parents=True, exist_ok=True)
            cleaned_files = sorted(cleaned_dir.glob("*.parquet"))
            total_after_quality = 0
            import shutil
            import os as os_link
            import pyarrow.parquet as pq_meta
            for cf in tqdm(cleaned_files, desc="  Linking files", disable=not self.show_progress):
                target = quality_dir / cf.name
                if not target.exists():
                    try:
                        target.symlink_to(cf.resolve())
                    except OSError:
                        try:
                            os_link.link(str(cf.resolve()), str(target))
                        except OSError:
                            # Last resort: copy the file
                            shutil.copy2(cf, target)
                # Count rows without loading into memory
                total_after_quality += pq_meta.read_metadata(cf).num_rows
            stats['after_quality'] = total_after_quality
            stage2_time = time.time() - stage2_start
            print(f"  Stage 2 complete: {stats['after_quality']:,} docs linked ({stage2_time:.1f}s)")
        else:
            print("\n[Stage 2/4] Quality filtering (streaming mode)...")
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
                print("  Skipping quality filters (datatrove not available) - using symlinks")
                # Use symlinks instead of copying to save disk space
                import shutil
                import os as os_link
                import pyarrow.parquet as pq_meta
                for cf in cleaned_files:
                    target = quality_dir / cf.name
                    if not target.exists():
                        try:
                            target.symlink_to(cf.resolve())
                        except OSError:
                            try:
                                os_link.link(str(cf.resolve()), str(target))
                            except OSError:
                                shutil.copy2(cf, target)
                    total_after_quality += pq_meta.read_metadata(cf).num_rows

            stats['after_quality'] = total_after_quality
            stage2_time = time.time() - stage2_start
            print(f"  Stage 2 complete: {stats['after_quality']:,} docs ({stage2_time:.1f}s)")

        # Stage 3: Toxicity filtering (streaming)
        skip_stage3 = resume and self._check_stage_complete(3)
        stage3_start = time.time()

        if skip_stage3:
            print("\n[Stage 3/4] SKIPPED (already complete)")
            stats['after_toxicity'] = self._count_docs_in_dir(toxicity_dir)
            print(f"  Found {stats['after_toxicity']:,} toxicity-filtered docs in cache")
        else:
            print("\n[Stage 3/4] Toxicity filtering...")
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
                    print("  Skipping (model not available) - using symlinks")
                    import shutil
                    import os as os_link
                    for qf in quality_files:
                        target = toxicity_dir / qf.name
                        if not target.exists():
                            try:
                                target.symlink_to(qf.resolve())
                            except OSError:
                                try:
                                    os_link.link(str(qf.resolve()), str(target))
                                except OSError:
                                    shutil.copy2(qf, target)
                        import pyarrow.parquet as pq_meta
                        total_after_toxicity += pq_meta.read_metadata(qf).num_rows
            else:
                print("  Skipping (disabled) - using symlinks to save disk space")
                import shutil
                import os as os_link
                for qf in quality_files:
                    # Use symlink instead of copying to save disk space
                    target = toxicity_dir / qf.name
                    if not target.exists():
                        try:
                            target.symlink_to(qf.resolve())
                        except OSError:
                            # Fallback to hardlink if symlink fails
                            try:
                                os_link.link(str(qf.resolve()), str(target))
                            except OSError:
                                # Last resort: copy the file
                                shutil.copy2(qf, target)
                    # Count rows without loading into memory
                    import pyarrow.parquet as pq_meta
                    total_after_toxicity += pq_meta.read_metadata(qf).num_rows

            stats['after_toxicity'] = total_after_toxicity
            stage3_time = time.time() - stage3_start
            print(f"  Stage 3 complete: {stats['after_toxicity']:,} docs ({stage3_time:.1f}s)")

        # Stage 4: Deduplication
        skip_stage4 = resume and self._check_stage_complete(4)
        stage4_start = time.time()

        if skip_stage4:
            print("\n[Stage 4/4] SKIPPED (already complete)")
            final_output = self.output_dir / "processed.parquet"
            if final_output.exists():
                stats['after_dedup'] = len(pd.read_parquet(final_output))
            print(f"  Found {stats['after_dedup']:,} deduplicated docs")
        else:
            print("\n[Stage 4/4] Deduplication...")

            # For GPU deduplication, pass the directory of files instead of combining
            # This allows the workflow to process files in chunks and avoid OOM
            toxicity_files = sorted(toxicity_dir.glob("*.parquet"))
            total_rows = sum(len(pd.read_parquet(f)) for f in toxicity_files[:3]) if len(toxicity_files) >= 3 else 0
            # Estimate total from sample
            if toxicity_files and total_rows > 0:
                avg_per_file = total_rows / min(3, len(toxicity_files))
                total_rows = int(avg_per_file * len(toxicity_files))
            print(f"  Found {len(toxicity_files)} files (~{total_rows:,} docs) for deduplication")

            # Run deduplication
            if not self.skip_dedup:
                dedup_output = self.output_dir / "deduplicated"
                # Pass the DIRECTORY of files, not a single combined file
                # This allows NeMo Curator to process files in chunks based on input_blocksize
                gpu_fuzzy_dedup(
                    input_path=str(toxicity_dir),  # Directory of chunk files
                    output_path=str(dedup_output),
                    text_column='text',
                    id_column='id',
                    similarity_threshold=self.dedup_threshold,
                    cache_path=str(self.cache_dir / "dedup_cache"),
                    use_gpu=self.use_gpu_dedup,
                    show_progress=self.show_progress,
                )

                # Count final documents (use metadata to avoid loading into memory)
                final_files = list(dedup_output.glob("*.parquet"))
                import pyarrow.parquet as pq_count
                final_count = sum(pq_count.read_metadata(f).num_rows for f in final_files)
                stats['after_dedup'] = final_count

                # Move to final output (don't copy to avoid OOM with 11M+ docs)
                final_output = self.output_dir / "processed.parquet"
                if final_files:
                    if len(final_files) == 1:
                        # Single file - just move it
                        import shutil
                        shutil.move(str(final_files[0]), str(final_output))
                    else:
                        # Multiple files - stream and merge without loading all into memory
                        import pyarrow.parquet as pq_merge
                        writer = None
                        for ff in final_files:
                            table = pq_merge.read_table(ff)
                            if writer is None:
                                writer = pq_merge.ParquetWriter(str(final_output), table.schema)
                            writer.write_table(table)
                            del table
                        if writer:
                            writer.close()
            else:
                print("  Skipping dedup (disabled)")
                # Combine toxicity files into final output
                final_output = self.output_dir / "processed.parquet"
                if toxicity_files:
                    import pyarrow.parquet as pq
                    import pyarrow.parquet as pq_writer
                    writer = None
                    total_rows = 0
                    for tf in toxicity_files:
                        table = pq.read_table(tf)
                        total_rows += table.num_rows
                        if writer is None:
                            writer = pq_writer.ParquetWriter(str(final_output), table.schema)
                        writer.write_table(table)
                        del table
                    if writer:
                        writer.close()
                    stats['after_dedup'] = total_rows
                else:
                    empty_df = pd.DataFrame({'id': [], 'text': []})
                    gpu_save_parquet(empty_df, str(final_output))
                    stats['after_dedup'] = 0

            stage4_time = time.time() - stage4_start
            print(f"  Stage 4 complete: {stats['after_dedup']:,} docs ({stage4_time:.1f}s)")

        # Cleanup intermediate files if requested
        if self.cleanup_intermediate:
            self._cleanup_intermediate_files(stats)

        # Summary
        stats['end_time'] = time.time()
        stats['total_time'] = stats['end_time'] - stats['start_time']

        # Use a reasonable baseline for percentages (first non-zero count)
        baseline = stats['input_docs']
        if baseline == 0:
            # Find the first non-zero count to use as baseline
            baseline = stats['after_cleaning'] or stats['after_quality'] or stats['after_toxicity'] or 1

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Input documents:    {stats['input_docs']:,}")
        print(f"After cleaning:     {stats['after_cleaning']:,} ({100*stats['after_cleaning']/baseline:.1f}%)")
        print(f"After quality:      {stats['after_quality']:,} ({100*stats['after_quality']/baseline:.1f}%)")
        print(f"After toxicity:     {stats['after_toxicity']:,} ({100*stats['after_toxicity']/baseline:.1f}%)")
        print(f"After dedup:        {stats['after_dedup']:,} ({100*stats['after_dedup']/baseline:.1f}%)")
        print(f"Total time:         {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} min)")
        print(f"Throughput:         {baseline/max(1, stats['total_time']):.0f} docs/sec")
        print(f"Output:             {self.output_dir / 'processed.parquet'}")
        print("=" * 60)

        return stats

    def _cleanup_intermediate_files(self, stats: dict) -> None:
        """Clean up intermediate files to save disk space.

        Called after all stages complete successfully.
        Only deletes directories that are no longer needed.
        """
        import shutil

        print("\n[Cleanup] Removing intermediate files to save disk space...")
        total_freed = 0

        # Directories to clean up (in order of size, largest first)
        cleanup_targets = [
            (self.cache_dir / "dedup_cache", "dedup cache"),
            (self.cache_dir / "cleaned", "cleaned text cache"),
            (self.cache_dir / "quality_filtered", "quality filter cache"),
            (self.cache_dir / "toxicity_filtered", "toxicity filter cache"),
        ]

        for path, name in cleanup_targets:
            if path.exists():
                try:
                    # Calculate size before deletion
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    size_gb = size / (1024**3)

                    # Delete the directory
                    shutil.rmtree(path)
                    total_freed += size
                    print(f"  Deleted {name}: {size_gb:.2f} GB freed")
                except Exception as e:
                    print(f"  Warning: Could not delete {name}: {e}")

        # Also clean up HuggingFace datasets cache if it exists
        hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
        if hf_cache.exists():
            try:
                # Only delete temporary files, not downloaded datasets
                temp_dirs = list(hf_cache.glob("**/tmp*"))
                for temp_dir in temp_dirs:
                    if temp_dir.is_dir():
                        size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
                        shutil.rmtree(temp_dir)
                        total_freed += size
            except Exception:
                pass

        total_freed_gb = total_freed / (1024**3)
        print(f"  Total freed: {total_freed_gb:.2f} GB")
        stats['disk_freed_gb'] = total_freed_gb

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
        "--backup-dir",
        default=None,
        help="Backup directory (e.g., Google Drive) for incremental sync and resume"
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
        "--skip-quality",
        action="store_true",
        help="Skip ALL quality filtering (fastest mode, just clean + dedup)"
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (re-run all stages from scratch)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate files after each stage completes to save disk space"
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
        backup_dir=args.backup_dir,
        use_gpu=use_gpu,
        fast_quality=args.fast_quality,
        skip_quality=args.skip_quality,
        skip_toxicity=args.no_toxicity,
        skip_dedup=args.no_dedup,
        cleanup_intermediate=args.cleanup,
        toxicity_threshold=args.toxicity_threshold,
        dedup_threshold=args.dedup_threshold,
        batch_size=args.batch_size,
        show_progress=not args.quiet,
    )

    stats = pipeline.process(resume=not args.no_resume)
    return stats


if __name__ == "__main__":
    main()
