"""
src/fusion.py — Two-stage score fusion.

Stage 1: Fast filter to top-500
  score = 0.30 × semantic + 0.12 × lexical + 0.58 × structured
  Apply honeypot penalty and disqualifier penalty.
  Select top-500 by penalised score.

Stage 2: Final ranking of top-500 (with cross-encoder)
  score = 0.25 × semantic + 0.10 × lexical + 0.45 × structured + 0.20 × cross_encoder
  Apply penalties again (in case cross-encoder changes relative order).
  Select top-100 with tie-break by ascending candidate_id.

Score normalisation:
  All component scores should already be in [0, 1].
  Final scores are min-max normalised to [0.20, 0.99] to produce meaningful
  non-uniform scores (more informative than the sample's linear 0.008 decrement).
"""

from __future__ import annotations
import numpy as np

from config import (
    STAGE1_WEIGHTS,
    STAGE2_WEIGHTS,
    TOP_N_STAGE1,
    TOP_N_FINAL,
)


def stage1_fusion(
    semantic_scores:    np.ndarray,   # (N,) float32
    lexical_scores:     np.ndarray,   # (N,) float32
    structured_scores:  np.ndarray,   # (N,) float32
    honeypot_penalties: np.ndarray,   # (N,) float32, 1.0 = clean
    disq_penalties:     np.ndarray,   # (N,) float32, 1.0 = no disq
    top_n: int = TOP_N_STAGE1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Stage 1 fused scores and return indices of top-N candidates.

    Args:
        semantic_scores:    Bi-encoder cosine sim scores
        lexical_scores:     BM25 normalised scores
        structured_scores:  Weighted structured sub-component scores
        honeypot_penalties: Honeypot multipliers (0.0 = definite honeypot)
        disq_penalties:     Disqualifier multipliers
        top_n:              Number of candidates to pass to Stage 2

    Returns:
        (stage1_scores, top_n_indices)
        stage1_scores: Raw (not normalised) Stage 1 scores for all N candidates
        top_n_indices: Indices of top-N candidates sorted by descending score
    """
    w = STAGE1_WEIGHTS
    base_scores = (
        w['semantic']   * semantic_scores   +
        w['lexical']    * lexical_scores    +
        w['structured'] * structured_scores
    )

    # Apply penalties multiplicatively
    penalised = base_scores * honeypot_penalties * disq_penalties

    # Sort descending, take top_n
    # argsort ascending → reverse for descending
    sorted_idx = np.argsort(-penalised)
    top_n_idx  = sorted_idx[:top_n]

    return penalised, top_n_idx


def stage2_fusion(
    top_n_indices:        np.ndarray,   # (top_n,) int indices into full arrays
    semantic_scores:      np.ndarray,   # (N,)
    lexical_scores:       np.ndarray,   # (N,)
    structured_scores:    np.ndarray,   # (N,)
    cross_encoder_scores: np.ndarray,   # (top_n,) — indexed by position in top_n_indices
    honeypot_penalties:   np.ndarray,   # (N,)
    disq_penalties:       np.ndarray,   # (N,)
    candidate_ids:        list[str],    # (N,) — for tie-breaking
    top_n_final: int = TOP_N_FINAL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Stage 2 fused scores for top-N candidates and return final top-100 ordering.

    Args:
        top_n_indices:        Indices of Stage 1 top-N candidates
        semantic_scores:      Bi-encoder scores (full N)
        lexical_scores:       BM25 scores (full N)
        structured_scores:    Structured scores (full N)
        cross_encoder_scores: Cross-encoder scores for top-N candidates only
        honeypot_penalties:   Full N honeypot multipliers
        disq_penalties:       Full N disqualifier multipliers
        candidate_ids:        All N candidate IDs (for lexicographic tie-breaking)
        top_n_final:          Final number of candidates to return

    Returns:
        (final_indices, final_scores)
        final_indices: Indices into full arrays of the top-100 candidates, in order
        final_scores:  Normalised scores (non-increasing), shape (top_n_final,)
    """
    w = STAGE2_WEIGHTS

    # Extract top-N component scores
    sem_top   = semantic_scores[top_n_indices]
    lex_top   = lexical_scores[top_n_indices]
    str_top   = structured_scores[top_n_indices]
    hp_top    = honeypot_penalties[top_n_indices]
    dq_top    = disq_penalties[top_n_indices]

    base_scores = (
        w['semantic']      * sem_top            +
        w['lexical']       * lex_top            +
        w['structured']    * str_top            +
        w['cross_encoder'] * cross_encoder_scores
    )

    penalised = base_scores * hp_top * dq_top

    # Tie-break by ascending candidate_id (lexicographic)
    # Create a secondary sort key: convert candidate_id to integer suffix
    ids_top = [candidate_ids[i] for i in top_n_indices]
    id_keys = _id_sort_keys(ids_top)

    # Sort: primary = descending score, secondary = ascending candidate_id
    # Negate score for ascending argsort
    sort_order = np.lexsort((id_keys, -penalised))
    top_sorted  = sort_order[:top_n_final]

    # Map back to original full-array indices
    final_indices_in_top = top_sorted
    final_indices        = top_n_indices[final_indices_in_top]
    raw_final_scores     = penalised[final_indices_in_top]

    # Normalise to [0.20, 0.99] (non-uniform, reflects actual fit gaps)
    normalised_scores = _normalise_scores(raw_final_scores, low=0.20, high=0.99)

    return final_indices, normalised_scores


def _id_sort_keys(ids: list[str]) -> np.ndarray:
    """
    Convert CAND_XXXXXXX IDs to integer keys for stable tie-breaking.
    Ascending lexicographic order on the ID = ascending integer suffix.

    Args:
        ids: List of candidate ID strings like "CAND_0000042"

    Returns:
        np.ndarray of int64 sort keys
    """
    keys = []
    for cid in ids:
        try:
            keys.append(int(cid.split('_')[-1]))
        except (ValueError, IndexError):
            keys.append(0)
    return np.array(keys, dtype=np.int64)


def _normalise_scores(
    scores: np.ndarray,
    low: float = 0.20,
    high: float = 0.99,
) -> np.ndarray:
    """
    Min-max normalise scores to [low, high] range.
    Preserves relative ordering (monotone transformation).

    If all scores are equal (degenerate case), returns linearly spaced values.

    Args:
        scores: Raw scores, already in descending order
        low:    Minimum output score
        high:   Maximum output score

    Returns:
        Normalised scores in [low, high], same order as input
    """
    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s < 1e-9:
        # Degenerate case: space them linearly
        return np.linspace(high, low, len(scores)).astype(np.float32)

    normed = (scores - min_s) / (max_s - min_s)  # [0, 1]
    normed = normed * (high - low) + low            # [low, high]

    # Ensure non-increasing (may have tiny floating point violations)
    for i in range(1, len(normed)):
        if normed[i] > normed[i - 1]:
            normed[i] = normed[i - 1]

    return normed.astype(np.float32)
