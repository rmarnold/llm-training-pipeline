"""Quality filtering using datatrove production filters (Gopher, FineWeb)."""
from __future__ import annotations

from multiprocessing import Pool, cpu_count

from tqdm import tqdm

# Datatrove quality filters (production-grade, used by FineWeb/LLaMA)
try:
    from datatrove.data import Document
    from datatrove.pipeline.filters import (
        GopherQualityFilter,
        GopherRepetitionFilter,
        FineWebQualityFilter,
    )
    DATATROVE_AVAILABLE = True
except ImportError:
    DATATROVE_AVAILABLE = False

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
    """Initialize datatrove filters in worker process."""
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

    OPTIMIZED: Runs filters in order of speed (fastest first = fail-fast).
    """
    filters = _init_datatrove_filters()
    if not filters:
        return True

    try:
        doc = Document(text=text, id="0")

        if 'gopher_quality' in filters and not filters['gopher_quality'].filter(doc):
            return False
        if 'fineweb' in filters and not filters['fineweb'].filter(doc):
            return False
        if 'gopher_rep' in filters and not filters['gopher_rep'].filter(doc):
            return False

        return True
    except Exception:
        return False


def _filter_single_text_datatrove_with_reason(text: str) -> str:
    """Filter a single text and return rejection reason (for stats).
    Returns: 'passed', 'quality', 'fineweb', 'repetition', or 'error'
    """
    filters = _init_datatrove_filters()
    if not filters:
        return 'no_filters'

    try:
        doc = Document(text=text, id="0")

        if 'gopher_quality' in filters:
            result = filters['gopher_quality'].filter(doc)
            passed = result[0] if isinstance(result, tuple) else result
            if not passed:
                return 'quality'

        if 'fineweb' in filters:
            result = filters['fineweb'].filter(doc)
            passed = result[0] if isinstance(result, tuple) else result
            if not passed:
                return 'fineweb'

        if 'gopher_rep' in filters:
            result = filters['gopher_rep'].filter(doc)
            passed = result[0] if isinstance(result, tuple) else result
            if not passed:
                return 'repetition'

        return 'passed'
    except Exception as e:
        return f'error:{str(e)[:50]}'


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
        self.filter_config = dict(_datatrove_filter_config)
        _init_datatrove_filters()
        enabled = [k for k, v in self.filter_config.items() if v]
        print(f"      Datatrove filters enabled: {enabled}")

    def filter_batch(self, texts: list[str], show_progress: bool = False) -> list[bool]:
        """Apply all quality filters to a batch of texts using parallel processing."""
        n_texts = len(texts)

        if n_texts < 1000:
            return [_filter_single_text_datatrove(t) for t in texts]

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
        """Apply quality filters and return rejection statistics."""
        n_texts = len(texts)

        if n_texts < 1000:
            reasons = [_filter_single_text_datatrove_with_reason(t) for t in texts]
        else:
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

        mask = [r == 'passed' for r in reasons]

        error_reasons = [r for r in reasons if r.startswith('error:')]
        stats = {
            'passed': reasons.count('passed'),
            'failed_repetition': reasons.count('repetition'),
            'failed_quality': reasons.count('quality'),
            'failed_fineweb': reasons.count('fineweb'),
            'failed_error': len(error_reasons),
            'no_filters': reasons.count('no_filters'),
        }

        if error_reasons:
            unique_errors = set(error_reasons[:10])
            print(f"        Debug: Sample errors: {unique_errors}")

        if stats['no_filters'] > 0:
            print(f"        WARNING: {stats['no_filters']} docs had no filters applied!")
            print(f"        This indicates filters failed to initialize in worker processes")

        return mask, stats


def apply_quality_filter_parallel(
    texts: list[str],
    use_datatrove: bool = True,
    n_workers: int = None,
    show_progress: bool = False
) -> list[bool]:
    """Apply quality filtering to texts with parallel processing."""
    n_workers = n_workers or max(1, int(cpu_count() * 0.8))
    n_texts = len(texts)

    if use_datatrove and DATATROVE_AVAILABLE:
        filter_fn = _filter_single_text_datatrove
        pool_kwargs = {
            'initializer': _worker_init,
            'initargs': (dict(_datatrove_filter_config),)
        }
    else:
        filter_fn = _filter_single_text_basic
        pool_kwargs = {}

    if n_texts < 1000:
        return [filter_fn(t) for t in texts]

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
