"""Locality-Sensitive Hashing (LSH) for finding duplicate candidates."""
from __future__ import annotations

import numpy as np


def build_lsh_index(
    signatures: np.ndarray,
    num_bands: int,
) -> list[dict]:
    """Build LSH index using band hashing.

    Much faster than datasketch - uses numpy vectorization.

    Args:
        signatures: Array of shape (n_docs, num_perm)
        num_bands: Number of LSH bands

    Returns:
        List of hash tables (one per band), mapping band_hash -> list of doc indices
    """
    n_docs, num_perm = signatures.shape
    rows_per_band = num_perm // num_bands

    # Initialize hash tables for each band
    hash_tables = [{} for _ in range(num_bands)]

    # Process all documents at once per band
    for band_idx in range(num_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band

        # Extract band for all documents: (n_docs, rows_per_band)
        band_data = signatures[:, start_row:end_row]

        # Hash each document's band (using tuple as key)
        for doc_idx in range(n_docs):
            band_hash = hash(band_data[doc_idx].tobytes())
            if band_hash not in hash_tables[band_idx]:
                hash_tables[band_idx][band_hash] = []
            hash_tables[band_idx][band_hash].append(doc_idx)

    return hash_tables


def find_duplicates_lsh(
    hash_tables: list[dict],
    signatures: np.ndarray,
    similarity_threshold: float,
) -> set[int]:
    """Find duplicate documents using LSH + verification.

    Returns indices of documents to REMOVE (keeping first occurrence).
    """
    # Track which documents to remove
    duplicates_to_remove: set[int] = set()

    # Find candidate pairs from LSH buckets
    for table in hash_tables:
        for bucket_hash, doc_indices in table.items():
            if len(doc_indices) < 2:
                continue

            # All documents in same bucket are candidates
            # Keep the first one, mark others for removal after verification
            doc_indices = sorted(doc_indices)  # Keep lowest index
            first_doc = doc_indices[0]

            if first_doc in duplicates_to_remove:
                continue

            for other_doc in doc_indices[1:]:
                if other_doc in duplicates_to_remove:
                    continue

                # Verify similarity using Jaccard estimation from MinHash
                sig1 = signatures[first_doc]
                sig2 = signatures[other_doc]
                estimated_jaccard = np.mean(sig1 == sig2)

                if estimated_jaccard >= similarity_threshold:
                    duplicates_to_remove.add(other_doc)

    return duplicates_to_remove
