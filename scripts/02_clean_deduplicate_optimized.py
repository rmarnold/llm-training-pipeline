"""Optimized data cleaning with caching, checkpoints, and GPU acceleration."""
from __future__ import annotations

import os
import json
import hashlib
import pandas as pd
from datasketch import MinHash, MinHashLSH
from detoxify import Detoxify
import re
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count
from functools import partial

# Datatrove quality filters (production-grade, used by FineWeb/LLaMA)
try:
    from datatrove.data import Document
    from datatrove.pipeline.filters import (
        GopherQualityFilter,
        GopherRepetitionFilter,
        FineWebQualityFilter,
        LanguageFilter,
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
    DATATROVE_AVAILABLE = True
    DATATROVE_PIPELINE_AVAILABLE = True
except ImportError:
    DATATROVE_AVAILABLE = False
    DATATROVE_PIPELINE_AVAILABLE = False
    print("Warning: datatrove not installed, using basic quality filters")

# =============================================================================
# FAST TEXT CLEANING - Pre-compiled patterns for 5-10x speedup
# =============================================================================

# Pre-compile all regex patterns ONCE (not per-call)
# Combined pattern for single-pass substitution where possible
_WHITESPACE_PATTERN = re.compile(r'\s+')
_EMAIL_PATTERN = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
_SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_PHONE_PATTERN = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b')
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# Combined PII pattern for single-pass (faster than multiple re.sub calls)
_PII_COMBINED_PATTERN = re.compile(
    r'(?P<email>\b[\w\.-]+@[\w\.-]+\.\w+\b)|'
    r'(?P<ssn>\b\d{3}-\d{2}-\d{4}\b)|'
    r'(?P<phone>\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)|'
    r'(?P<url>https?://\S+|www\.\S+)'
)

# Cleaning mode: 'fast' skips ftfy, 'full' uses ftfy
_CLEANING_MODE = os.environ.get('CLEANING_MODE', 'fast')


def _pii_replacer(match) -> str:
    """Replace PII matches with placeholders."""
    if match.group('email'):
        return '[EMAIL]'
    elif match.group('ssn'):
        return '[SSN]'
    elif match.group('phone'):
        return '[PHONE]'
    elif match.group('url'):
        return '[URL]'
    return match.group(0)


def _clean_text_fast(text: str) -> str:
    """Fast text cleaning - skips ftfy, uses compiled regex."""
    if not text or pd.isna(text):
        return ""

    text = str(text)

    # Single-pass PII removal (5x faster than multiple re.sub)
    text = _PII_COMBINED_PATTERN.sub(_pii_replacer, text)

    # Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(' ', text)

    return text.strip()


def _clean_text_full(text: str) -> str:
    """Full text cleaning with plsfix (Rust-based, 10x faster than ftfy)."""
    if not text or pd.isna(text):
        return ""

    try:
        # plsfix is a Rust-based ftfy replacement (~10x faster)
        from plsfix import fix_text
        text = fix_text(str(text))
    except ImportError:
        # Fallback to ftfy if plsfix not installed
        try:
            from ftfy import fix_text
            text = fix_text(str(text))
        except ImportError:
            # No Unicode fixer available, just use basic normalization
            import unicodedata
            text = unicodedata.normalize('NFKC', str(text))

    # Single-pass PII removal
    text = _PII_COMBINED_PATTERN.sub(_pii_replacer, text)

    # Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(' ', text)

    return text.strip()


def _clean_text_worker(text: str) -> str:
    """Worker function for parallel text cleaning."""
    if _CLEANING_MODE == 'fast':
        return _clean_text_fast(text)
    else:
        return _clean_text_full(text)


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
        n_workers = max(1, int(cpu_count() * 0.8))

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
        # Default to 80% of CPU cores for optimal throughput
        n_workers = max(1, int(cpu_count() * 0.8))

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

    def __init__(self, toxicity_threshold: float = 0.7, use_gpu: bool = True, batch_size: int = None):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.toxicity_threshold = toxicity_threshold

        # Auto-tune batch size based on GPU memory
        # A100 (40GB/80GB) can handle much larger batches than default 128
        if batch_size is None:
            if self.device == 'cuda':
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if gpu_mem >= 70:  # A100-80GB or similar
                        batch_size = 512
                    elif gpu_mem >= 35:  # A100-40GB or similar
                        batch_size = 384
                    elif gpu_mem >= 20:  # RTX 3090/4090
                        batch_size = 256
                    else:
                        batch_size = 128
                    print(f"    Auto-tuned toxicity batch_size={batch_size} for {gpu_mem:.0f}GB GPU")
                except Exception:
                    batch_size = 128
            else:
                batch_size = 64  # CPU is slower, use smaller batches

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


# Global filter instances for multiprocessing workers
_datatrove_filters = None
_datatrove_filter_config = {
    'use_gopher_quality': True,  # Fast: word count, stop words, symbol ratio
    'use_fineweb': True,         # Medium: line structure, punctuation
    'use_gopher_rep': True,      # Slow: n-gram repetition analysis
}


def configure_datatrove_filters(
    use_gopher_quality: bool = True,
    use_fineweb: bool = True,
    use_gopher_rep: bool = True
):
    """Configure which datatrove filters to use.

    For faster processing, disable expensive filters:
    - use_gopher_rep=False: Skip n-gram analysis (biggest speedup, ~2x faster)
    - use_fineweb=False: Skip line structure checks (~15% faster)

    Args:
        use_gopher_quality: Enable word count/stop word checks (fast, recommended)
        use_fineweb: Enable line structure checks (medium speed)
        use_gopher_rep: Enable repetition/spam detection (slow but catches spam)
    """
    global _datatrove_filter_config, _datatrove_filters
    _datatrove_filter_config = {
        'use_gopher_quality': use_gopher_quality,
        'use_fineweb': use_fineweb,
        'use_gopher_rep': use_gopher_rep,
    }
    # Reset filters so they get re-initialized with new config
    _datatrove_filters = None


def _worker_init(config: dict):
    """Initialize datatrove filters in worker process.

    This is called once per worker when the Pool starts, ensuring
    filters are properly initialized in each worker process.
    """
    global _datatrove_filter_config, _datatrove_filters
    _datatrove_filter_config = config
    _datatrove_filters = None  # Force re-initialization with passed config
    _init_datatrove_filters()


def _init_datatrove_filters():
    """Initialize datatrove filters for the current process."""
    global _datatrove_filters
    if _datatrove_filters is None and DATATROVE_AVAILABLE:
        _datatrove_filters = {}

        if _datatrove_filter_config.get('use_gopher_quality', True):
            _datatrove_filters['gopher_quality'] = GopherQualityFilter(
                min_doc_words=50,
                max_doc_words=100000,
                min_avg_word_length=3,
                max_avg_word_length=10,
                min_stop_words=2,
                max_symbol_word_ratio=0.1,
            )

        if _datatrove_filter_config.get('use_fineweb', True):
            _datatrove_filters['fineweb'] = FineWebQualityFilter()

        if _datatrove_filter_config.get('use_gopher_rep', True):
            _datatrove_filters['gopher_rep'] = GopherRepetitionFilter()

    return _datatrove_filters


def _filter_single_text_datatrove(text: str) -> bool:
    """Filter a single text using datatrove filters (for multiprocessing).
    Returns True if passed, False if rejected.

    OPTIMIZED: Runs filters in order of speed (fastest first = fail-fast):
    1. GopherQuality (fast - simple word stats)
    2. FineWeb (medium - line analysis)
    3. GopherRepetition (slow - n-gram analysis)
    """
    filters = _init_datatrove_filters()
    if not filters:
        return True  # No filters available/enabled, keep doc

    try:
        doc = Document(text=text, id="0")

        # Apply filters in order of speed (fastest first for fail-fast)
        # 1. GopherQuality is fastest (simple word count, stop words, etc.)
        if 'gopher_quality' in filters and not filters['gopher_quality'].filter(doc):
            return False
        # 2. FineWeb is medium speed (line analysis)
        if 'fineweb' in filters and not filters['fineweb'].filter(doc):
            return False
        # 3. GopherRepetition is slowest (n-gram analysis) - run last
        if 'gopher_rep' in filters and not filters['gopher_rep'].filter(doc):
            return False

        return True
    except Exception:
        return False


def _filter_single_text_datatrove_with_reason(text: str) -> str:
    """Filter a single text and return rejection reason (for stats).
    Returns: 'passed', 'quality', 'fineweb', 'repetition', or 'error'

    OPTIMIZED: Same order as _filter_single_text_datatrove (fastest first)
    """
    filters = _init_datatrove_filters()
    if not filters:
        return 'passed'

    try:
        doc = Document(text=text, id="0")

        # Run in order of speed (fastest first)
        if 'gopher_quality' in filters and not filters['gopher_quality'].filter(doc):
            return 'quality'
        if 'fineweb' in filters and not filters['fineweb'].filter(doc):
            return 'fineweb'
        if 'gopher_rep' in filters and not filters['gopher_rep'].filter(doc):
            return 'repetition'

        return 'passed'
    except Exception:
        return 'error'


def _filter_single_text_basic(text: str) -> bool:
    """Basic quality filter (fast fallback)."""
    word_count = len(text.split())
    if word_count < 50 or word_count > 10000:
        return False
    unique_chars = len(set(text.lower()))
    if unique_chars < 20:
        return False
    return True


def _filter_single_text_basic_with_reason(text: str) -> str:
    """Basic quality filter with rejection reason."""
    word_count = len(text.split())
    if word_count < 50:
        return 'too_short'
    if word_count > 10000:
        return 'too_long'
    unique_chars = len(set(text.lower()))
    if unique_chars < 20:
        return 'low_diversity'
    return 'passed'


class DatatroveQualityFilter:
    """Production-grade quality filtering using datatrove (FineWeb/Gopher filters).

    Much more sophisticated than basic word count/unique char filters:
    - GopherRepetitionFilter: Detects repeated n-grams, lines, paragraphs
    - GopherQualityFilter: Word count, stop words, alpha ratio, etc.
    - FineWebQualityFilter: Line-ending punctuation, short lines, etc.

    Uses multiprocessing for parallel filtering on large batches.
    """

    def __init__(self, n_workers: int = None):
        if not DATATROVE_AVAILABLE:
            raise ImportError("datatrove not installed. Run: pip install datatrove")

        self.n_workers = n_workers or max(1, int(cpu_count() * 0.8))
        # Store config for passing to workers
        self.filter_config = dict(_datatrove_filter_config)
        # Initialize filters in main process too (for small batches)
        _init_datatrove_filters()
        # Log which filters are enabled
        enabled = [k for k, v in self.filter_config.items() if v]
        print(f"      Datatrove filters enabled: {enabled}")

    def filter_batch(self, texts: list[str], show_progress: bool = False) -> list[bool]:
        """Apply all quality filters to a batch of texts using parallel processing.

        Returns a list of booleans (True = keep, False = filter out).
        """
        n_texts = len(texts)

        # For small batches, use single-threaded
        if n_texts < 1000:
            return [_filter_single_text_datatrove(t) for t in texts]

        # Parallel processing with initializer to ensure filters are set up in workers
        with Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(self.filter_config,)
        ) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(_filter_single_text_datatrove, texts, chunksize=5000),
                    total=n_texts,
                    desc="      Quality filter"
                ))
            else:
                results = pool.map(_filter_single_text_datatrove, texts, chunksize=5000)

        return results

    def filter_batch_with_stats(self, texts: list[str], show_progress: bool = False) -> tuple[list[bool], dict]:
        """Apply quality filters and return rejection statistics.

        Returns:
            tuple: (mask list, stats dict)
            - mask: List of booleans (True = keep, False = filter out)
            - stats: Dict with counts for each rejection reason
        """
        n_texts = len(texts)

        # Get rejection reasons for each text
        if n_texts < 1000:
            reasons = [_filter_single_text_datatrove_with_reason(t) for t in texts]
        else:
            # Use initializer to ensure filters are properly set up in each worker
            with Pool(
                processes=self.n_workers,
                initializer=_worker_init,
                initargs=(self.filter_config,)
            ) as pool:
                if show_progress:
                    reasons = list(tqdm(
                        pool.imap(_filter_single_text_datatrove_with_reason, texts, chunksize=5000),
                        total=n_texts,
                        desc="      Quality filter"
                    ))
                else:
                    reasons = pool.map(_filter_single_text_datatrove_with_reason, texts, chunksize=5000)

        # Convert reasons to boolean mask and count stats
        mask = [r == 'passed' for r in reasons]
        stats = {
            'passed': reasons.count('passed'),
            'failed_repetition': reasons.count('repetition'),
            'failed_quality': reasons.count('quality'),
            'failed_fineweb': reasons.count('fineweb'),
            'failed_error': reasons.count('error'),
        }

        return mask, stats


def apply_quality_filter_parallel(
    texts: list[str],
    use_datatrove: bool = True,
    n_workers: int = None,
    show_progress: bool = False
) -> list[bool]:
    """Apply quality filtering to texts with parallel processing.

    Args:
        texts: List of text documents
        use_datatrove: If True, use production datatrove filters. If False, use basic filters.
        n_workers: Number of parallel workers (default: cpu_count // 2)
        show_progress: Show progress bar

    Returns:
        List of booleans (True = keep, False = filter out)
    """
    n_workers = n_workers or max(1, int(cpu_count() * 0.8))
    n_texts = len(texts)

    if use_datatrove and DATATROVE_AVAILABLE:
        filter_fn = _filter_single_text_datatrove
        # Use initializer for datatrove filters
        pool_kwargs = {
            'initializer': _worker_init,
            'initargs': (dict(_datatrove_filter_config),)
        }
    else:
        filter_fn = _filter_single_text_basic
        pool_kwargs = {}

    # For small batches, use single-threaded
    if n_texts < 1000:
        return [filter_fn(t) for t in texts]

    # Parallel processing for large batches
    with Pool(processes=n_workers, **pool_kwargs) as pool:
        if show_progress:
            results = list(tqdm(
                pool.imap(filter_fn, texts, chunksize=5000),
                total=n_texts,
                desc="      Quality filter"
            ))
        else:
            results = pool.map(filter_fn, texts, chunksize=5000)

    return results


# =============================================================================
# NATIVE DATATROVE PIPELINE - Optimized execution using datatrove's executors
# =============================================================================

if DATATROVE_PIPELINE_AVAILABLE:
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
                doc.text = _clean_text_full(doc.text)
            else:
                doc.text = _clean_text_fast(doc.text)
            return True  # Always keep, just modifies text


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

    Returns:
        Dict with processing statistics
    """
    if not DATATROVE_PIPELINE_AVAILABLE:
        raise ImportError("datatrove pipeline components not available. Install with: pip install 'datatrove[io,processing]'")

    n_workers = n_workers or max(1, int(cpu_count() * 0.8))
    os.makedirs(output_dir, exist_ok=True)

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

    # =========================================================================
    # STAGE 1: Quality Filtering with Native Pipeline
    # =========================================================================
    print("STAGE 1: Quality Filtering (native datatrove pipeline)")
    print("-" * 50)

    # Build filter pipeline
    filters = [PIICleanerFilter(use_full_clean=use_full_clean)]

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

    # Count tasks = number of input files for optimal parallelization
    n_tasks = len(input_files)

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

    print(f"  Running with {n_tasks} tasks, {n_workers} workers...")
    executor1.run()

    # Count results from stage 1
    stage1_files = list(stage1_dir.glob("*.parquet"))
    for f in stage1_files:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(f)
        stats['after_quality'] += pf.metadata.num_rows
    print(f"  After quality filtering: {stats['after_quality']:,} docs")

    # =========================================================================
    # STAGE 2: Toxicity Filtering (GPU - can't be in datatrove pipeline)
    # =========================================================================
    print("\nSTAGE 2: Toxicity Filtering (GPU)")
    print("-" * 50)

    # Initialize toxicity model
    cleaner = DataCleaner(use_gpu=use_gpu, batch_size=toxicity_batch_size)

    import pyarrow.parquet as pq
    import pyarrow as pa

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

    # MinhashDedup configuration (similar to FineWeb)
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )

    stage2_files = list(stage2_dir.glob("*.parquet"))
    n_dedup_tasks = max(1, len(stage2_files))

    # Stage 3a: Compute signatures
    print("  3a: Computing MinHash signatures...")
    sig_pipeline = [
        ParquetReader(
            data_folder=str(stage2_dir),
            glob_pattern="*.parquet",
            text_key="text",
        ),
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

    # Stage 3b: Find duplicate pairs in buckets
    print("  3b: Finding duplicate pairs...")
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

    # Stage 3c: Cluster duplicates
    print("  3c: Clustering duplicates...")
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

    # Stage 3d: Filter out duplicates and write final output
    print("  3d: Filtering duplicates and writing output...")
    final_output_dir = Path(output_dir) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    filter_pipeline = [
        ParquetReader(
            data_folder=str(stage2_dir),
            glob_pattern="*.parquet",
            text_key="text",
        ),
        MinhashDedupFilter(
            input_folder=str(dedup_clusters_dir),
        ),
        ParquetWriter(
            output_folder=str(final_output_dir),
            output_filename="${rank}.parquet",
        ),
    ]

    executor_filter = LocalPipelineExecutor(
        pipeline=filter_pipeline,
        logging_dir=str(temp_dir / "logs_dedup_filter"),
        tasks=n_dedup_tasks,
        workers=n_workers,
    )
    executor_filter.run()

    # Count final results
    final_files = list(final_output_dir.glob("*.parquet"))
    for f in final_files:
        pf = pq.ParquetFile(f)
        stats['after_dedup'] += pf.metadata.num_rows

    print(f"  After deduplication: {stats['after_dedup']:,} docs")

    # =========================================================================
    # Cleanup temp files
    # =========================================================================
    print("\nCleaning up temporary files...")
    import shutil
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
        # Use streaming to avoid loading entire file into memory
        import pyarrow.parquet as pq

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
            def save_chunk_streaming(chunk_data, idx):
                nonlocal current_chunk_idx
                if checkpoint:
                    actual_idx = current_chunk_idx + idx
                    chunk_path = chunk_dir / f"{filename}_clean_chunk_{actual_idx:04d}_{file_hash}.parquet"
                    print(f"\n    [Saving chunk {actual_idx}: {len(chunk_data):,} docs...]", end="", flush=True)
                    chunk_df = pd.DataFrame({'text': chunk_data})
                    chunk_df.to_parquet(chunk_path, index=False)
                    print(f" done]")

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
                    for chunk in parallel_clean_texts_streaming(
                        texts_buffer,
                        n_workers=n_workers,
                        chunk_callback=save_chunk_streaming if checkpoint else None,
                        chunk_size=500000
                    ):
                        chunks_generated += 1
                    texts_buffer = []

            # Process remaining texts in buffer
            if texts_buffer:
                for chunk in parallel_clean_texts_streaming(
                    texts_buffer,
                    n_workers=n_workers,
                    chunk_callback=save_chunk_streaming if checkpoint else None,
                    chunk_size=500000
                ):
                    chunks_generated += 1

            current_chunk_idx += chunks_generated

        # Process each chunk through quality + toxicity filters (memory-efficient)
        # Instead of combining all chunks then filtering, we filter each chunk
        print(f"  Processing {current_chunk_idx} chunks through quality/toxicity filters...")
        all_chunk_files = sorted(chunk_dir.glob(chunk_pattern)) if chunk_dir else []

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

            # Delete original chunk to free disk space
            chunk_path.unlink()

        print(f"\n    Quality filter summary:")
        print(f"      Total passed: {total_after_quality:,}/{original_count:,} ({total_after_quality/original_count*100:.1f}%)")
        if quality_filter and sum(total_quality_stats.values()) > 0:
            total_filtered = original_count
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

        # Stream deduplication - process chunks one at a time, write output incrementally
        # This avoids loading all filtered data into memory at once
        print(f"  Streaming deduplication across {len(filtered_chunks)} filtered chunks...")

        import pyarrow.parquet as pq
        import pyarrow as pa

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
        n_workers = max(1, int(cpu_count() * 0.8))

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
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for toxicity detection (default: auto-tune based on GPU memory)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of CPU workers for text cleaning (default: all cores)')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory (default: data/raw)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory (default: data/processed)')
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
    parser.add_argument('--native-pipeline', action='store_true',
                       help='Use native datatrove pipeline (faster, uses MinhashDedup)')
    parser.add_argument('--legacy', action='store_true',
                       help='Force legacy pipeline mode (default if --native-pipeline not specified)')

    args = parser.parse_args()

    # Set cleaning mode globally
    global _CLEANING_MODE
    if args.full_clean:
        _CLEANING_MODE = 'full'
        print("Cleaning mode: FULL (with plsfix Unicode/mojibake fixing)")
    else:
        _CLEANING_MODE = 'fast'
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
        filters_enabled = []
        if True: filters_enabled.append("GopherQuality")
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
        )
    else:
        if args.native_pipeline and not DATATROVE_PIPELINE_AVAILABLE:
            print("Warning: --native-pipeline requested but datatrove pipeline not available")
            print("Falling back to legacy pipeline. Install with: pip install 'datatrove[io,processing]'")

        print("Pipeline mode: LEGACY (with multiprocessing)")
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
