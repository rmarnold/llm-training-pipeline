"""Fast parallel text cleaning with pre-compiled PII patterns."""
from __future__ import annotations

import os
import re
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

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
CLEANING_MODE = os.environ.get('CLEANING_MODE', 'fast')


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


def clean_text_fast(text: str) -> str:
    """Fast text cleaning - skips ftfy, uses compiled regex."""
    if not text or pd.isna(text):
        return ""

    text = str(text)

    # Single-pass PII removal (5x faster than multiple re.sub)
    text = _PII_COMBINED_PATTERN.sub(_pii_replacer, text)

    # Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(' ', text)

    return text.strip()


def clean_text_full(text: str) -> str:
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
    if CLEANING_MODE == 'fast':
        return clean_text_fast(text)
    else:
        return clean_text_full(text)


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
        n_workers: Number of CPU workers (default: cpu_count * 0.8)
        chunk_callback: Function to call with each chunk of results
        chunk_size: Number of results per chunk (default: 500K)

    Yields:
        Chunks of cleaned text results
    """
    if n_workers is None:
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
