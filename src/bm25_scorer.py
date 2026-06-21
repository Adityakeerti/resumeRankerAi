"""
src/bm25_scorer.py — BM25 lexical scoring for exact-term recall.

BM25 is kept at a low weight (10%) because:
  1. The JD explicitly warns keyword matching is a trap.
  2. The bi-encoder catches semantic matches that BM25 misses.

BUT BM25 is still useful for:
  - Exact technology names (Pinecone, Qdrant) that bi-encoders can confuse.
  - Candidates who literally use the JD's exact terminology.

Pre-computation: Builds bm25_index.pkl (stored offline).
Online: Queries with JD tokens → returns raw BM25 scores → min-max normalised.
"""

from __future__ import annotations
import pickle
import os

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def build_bm25_index(
    tokenized_corpus: list[list[str]],
    output_path: str,
) -> BM25Okapi:
    """
    Build and persist a BM25 index from a tokenized corpus.

    Args:
        tokenized_corpus: List of token lists, one per candidate (in order)
        output_path: File path to save the pickled BM25 index

    Returns:
        Fitted BM25Okapi object
    """
    print(f"[BM25] Building index over {len(tokenized_corpus):,} documents ...")
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(bm25, f, protocol=4)

    print(f"[BM25] Index saved to {output_path}")
    return bm25


def load_bm25_index(index_path: str) -> BM25Okapi:
    """
    Load a pre-built BM25 index from disk.

    Args:
        index_path: Path to the pickled BM25 index

    Returns:
        BM25Okapi object
    """
    with open(index_path, 'rb') as f:
        bm25 = pickle.load(f)
    print(f"[BM25] Index loaded from {index_path}")
    return bm25


def query_bm25(
    bm25: BM25Okapi,
    query_tokens: list[str],
) -> np.ndarray:
    """
    Query the BM25 index and return min-max normalised scores.

    Args:
        bm25: Fitted BM25Okapi object
        query_tokens: Tokenized query (e.g., JD tokens)

    Returns:
        np.ndarray of float32 scores, shape (n_candidates,), normalised to [0, 1]
    """
    raw_scores = bm25.get_scores(query_tokens).astype(np.float32)

    # Min-max normalisation
    min_val = raw_scores.min()
    max_val = raw_scores.max()

    if max_val - min_val < 1e-9:
        return np.zeros_like(raw_scores)

    return (raw_scores - min_val) / (max_val - min_val)
