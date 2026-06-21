"""
src/semantic_scorer.py — Semantic scoring via bi-encoder (pre-computed) and cross-encoder (online, top-500 only).

Bi-encoder (all-MiniLM-L6-v2):
  - Offline: embed all 100K candidates → embeddings.npy (150 MB)
  - Online:  embed JD → cosine_sim(100K) → ~0.5 sec (pure NumPy matmul)
  - Weight: 25% of final score

Cross-encoder (ms-marco-MiniLM-L-6-v2):
  - Online only: top-500 candidates × JD → relevance logits → ~25 sec
  - Weight: 20% of final score (maximises NDCG@10)
  - Must NOT be applied to all 100K (would take ~83 minutes)
"""

from __future__ import annotations
import os
import warnings

import numpy as np
from tqdm import tqdm

from config import (
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    CANDIDATE_TEXT_MAX_CHARS,
    JD_TEXT_FOR_CROSS_ENCODER,
    CAND_TEXT_FOR_CROSS_ENCODER,
    CROSS_ENCODER_BATCH,
)

# Suppress sentence-transformers verbose logging
warnings.filterwarnings('ignore', category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Bi-Encoder — Offline Embedding Computation
# ─────────────────────────────────────────────────────────────────────────────

def build_bi_encoder_embeddings(
    candidate_texts: list[str],
    output_path: str,
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute and save bi-encoder embeddings for all candidates using multi-processing.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[BiEncoder] Loading model: {BI_ENCODER_MODEL}")
    model = SentenceTransformer(BI_ENCODER_MODEL)

    print(f"[BiEncoder] Starting multi-process pool ...")
    pool = model.start_multi_process_pool()

    print(f"[BiEncoder] Encoding {len(candidate_texts):,} candidates via multi-process (batch={batch_size}) ...")
    embeddings = model.encode_multi_process(
        candidate_texts,
        pool,
        batch_size=batch_size,
    ).astype(np.float32)

    model.stop_multi_process_pool(pool)

    # L2-normalise manually since encode_multi_process doesn't normalise by default
    print("[BiEncoder] L2-normalising embeddings ...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"[BiEncoder] Saved embeddings: {output_path}  shape={embeddings.shape}")
    return embeddings


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """
    Load pre-computed candidate embeddings from disk.

    Args:
        embeddings_path: Path to .npy file

    Returns:
        np.ndarray, shape (N, D)
    """
    embs = np.load(embeddings_path, mmap_mode='r')
    print(f"[BiEncoder] Loaded embeddings: {embeddings_path}  shape={embs.shape}")
    return embs


def embed_query(query_text: str) -> np.ndarray:
    """
    Encode a single query string (the JD) with the bi-encoder at runtime.

    Args:
        query_text: JD or query text

    Returns:
        np.ndarray of float32, shape (384,), L2-normalised
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BI_ENCODER_MODEL)
    vec   = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    return vec


def cosine_scores(
    query_vec: np.ndarray,
    candidate_embs: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between query and all candidate embeddings.
    Since embeddings are L2-normalised, this is just a dot product.

    Args:
        query_vec: Shape (D,)
        candidate_embs: Shape (N, D)

    Returns:
        np.ndarray of float32, shape (N,), values in [-1, 1]
        Clipped to [0, 1] and returned (negative cosine = clearly wrong domain).
    """
    raw = candidate_embs @ query_vec          # (N,)
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder — Online Reranking (top-500 only)
# ─────────────────────────────────────────────────────────────────────────────

def cross_encoder_scores(
    jd_text: str,
    candidate_texts: list[str],
    batch_size: int = CROSS_ENCODER_BATCH,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Score (JD, candidate) pairs using a cross-encoder for precise relevance.

    Applied ONLY to top-500 candidates after Stage 1 fusion.
    Runtime: ~25 seconds for 500 candidates on CPU.

    Args:
        jd_text: Job description text (truncated to JD_TEXT_FOR_CROSS_ENCODER chars)
        candidate_texts: List of candidate text strings (one per candidate)
        batch_size: Inference batch size
        show_progress: Show tqdm

    Returns:
        np.ndarray of float32, shape (N,), values in [0, 1]
    """
    from sentence_transformers import CrossEncoder

    print(f"[CrossEncoder] Loading model: {CROSS_ENCODER_MODEL}")
    model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)

    jd_trunc   = jd_text[:JD_TEXT_FOR_CROSS_ENCODER]
    cand_trunc = [t[:CAND_TEXT_FOR_CROSS_ENCODER] for t in candidate_texts]

    pairs = [(jd_trunc, ct) for ct in cand_trunc]

    print(f"[CrossEncoder] Scoring {len(pairs)} pairs ...")
    raw_logits = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )

    # Sigmoid to get probabilities in [0, 1]
    scores = 1.0 / (1.0 + np.exp(-raw_logits))
    return scores.astype(np.float32)
