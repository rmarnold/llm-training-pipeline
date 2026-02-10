"""MinHash computation - shingle extraction and GPU-accelerated signatures."""
from __future__ import annotations

import numpy as np

from dedup.common import XXHASH_AVAILABLE, CUDA_AVAILABLE

if XXHASH_AVAILABLE:
    import xxhash


def fast_shingle_hash(text: str, char_ngrams: int = 5) -> np.ndarray:
    """Fast shingle extraction and hashing using xxhash.

    ~10-100x faster than Python's built-in hash().
    """
    if not text or len(text) < char_ngrams:
        return np.array([], dtype=np.uint64)

    n_shingles = len(text) - char_ngrams + 1

    if XXHASH_AVAILABLE:
        # xxhash is much faster than Python hash
        hashes = np.array([
            xxhash.xxh64_intdigest(text[i:i+char_ngrams])
            for i in range(n_shingles)
        ], dtype=np.uint64)
    else:
        # Fallback to Python hash
        hashes = np.array([
            hash(text[i:i+char_ngrams]) & 0xFFFFFFFFFFFFFFFF
            for i in range(n_shingles)
        ], dtype=np.uint64)

    return hashes


def compute_minhash_signatures_gpu(
    all_shingles: list[np.ndarray],
    num_perm: int,
    device: str = 'cuda',
    batch_size: int = 10000,
) -> np.ndarray:
    """Compute MinHash signatures for multiple documents on GPU.

    Fully vectorized - processes thousands of documents per GPU kernel.

    Args:
        all_shingles: List of shingle hash arrays (one per document)
        num_perm: Number of MinHash permutations
        device: 'cuda' or 'cpu'
        batch_size: Documents per GPU batch

    Returns:
        Array of shape (n_docs, num_perm) with uint64 signatures
    """
    import torch

    n_docs = len(all_shingles)
    MAX_HASH = np.iinfo(np.uint64).max

    # Pre-allocate output
    signatures = np.full((n_docs, num_perm), MAX_HASH, dtype=np.uint64)

    # Generate hash coefficients (consistent across calls)
    torch.manual_seed(42)
    PRIME = 2**61 - 1

    # Create coefficients on GPU
    a = torch.randint(1, PRIME, (num_perm,), dtype=torch.int64, device=device)
    b = torch.randint(0, PRIME, (num_perm,), dtype=torch.int64, device=device)

    # Process in batches
    for batch_start in range(0, n_docs, batch_size):
        batch_end = min(batch_start + batch_size, n_docs)

        # Collect all shingles for this batch with document boundaries
        batch_shingles = []
        doc_boundaries = [0]

        for i in range(batch_start, batch_end):
            shingles = all_shingles[i]
            if shingles is not None and len(shingles) > 0:
                batch_shingles.extend(shingles.tolist())
            doc_boundaries.append(len(batch_shingles))

        if not batch_shingles:
            continue

        # Move to GPU
        shingles_tensor = torch.tensor(batch_shingles, dtype=torch.int64, device=device)

        # Compute all hashes at once: (num_perm, n_shingles)
        # h(x) = (a*x + b) mod p
        hash_values = (a.unsqueeze(1) * shingles_tensor.unsqueeze(0) + b.unsqueeze(1)) % PRIME

        hash_values_cpu = hash_values.cpu().numpy()

        for j, i in enumerate(range(batch_start, batch_end)):
            start = doc_boundaries[j]
            end = doc_boundaries[j + 1]
            if end > start:
                signatures[i] = hash_values_cpu[:, start:end].min(axis=1)

        del shingles_tensor, hash_values, hash_values_cpu
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    return signatures
