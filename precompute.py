#!/usr/bin/env python3
"""
precompute.py — Offline pre-computation script (run ONCE before rank.py).

Produces inside --out directory:
  embeddings.npy       → float32 (N, 384) bi-encoder embeddings
  candidate_ids.npy    → str (N,) candidate IDs in order
  features.npy         → float32 (N, 8) structured feature matrix
  bm25_index.pkl       → BM25Okapi index
  honeypot_flags.npy   → float32 (N,) penalty multipliers
  disq_flags.npy       → float32 (N,) disqualifier penalty multipliers

Usage:
  python precompute.py --candidates ./candidates.jsonl --out ./data/

Runtime: ~45 minutes on CPU for 100K candidates.
After this, rank.py runs in < 1 minute.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# Optimize CPU thread allocation for multi-processing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CANDIDATE_TEXT_MAX_CHARS
from src.text_builder import build_candidate_text, build_bm25_tokens
from src.semantic_scorer import build_bi_encoder_embeddings
from src.bm25_scorer import build_bm25_index
from src.structured_scorer import batch_compute_structured
from src.honeypot_detector import batch_detect_honeypots
from src.disqualifiers import batch_compute_disqualifiers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Offline Pre-computation"
    )
    parser.add_argument(
        '--candidates', '-c',
        required=True,
        help='Path to candidates.jsonl (100K candidate file)'
    )
    parser.add_argument(
        '--out', '-o',
        default='./data',
        help='Output directory for pre-computed artifacts (default: ./data)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=256,
        help='Batch size for bi-encoder embedding (default: 256)'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip bi-encoder embedding step (use if already done)'
    )
    parser.add_argument(
        '--skip-bm25',
        action='store_true',
        help='Skip BM25 index building (use if already done)'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip structured feature extraction (use if already done)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=None,
        help='Only process first N candidates (for debugging)'
    )
    return parser.parse_args()


def load_candidates(jsonl_path: str, limit: int | None = None) -> list[dict]:
    """Stream candidates from JSONL file."""
    candidates = []
    print(f"[Load] Reading candidates from {jsonl_path} ...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading", unit=" lines")):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {i}: {e}")
    print(f"[Load] Loaded {len(candidates):,} candidates")
    return candidates


def compute_cdfs(candidates: list[dict]) -> dict:
    import numpy as np

    response_rates = []
    response_times = []
    interview_rates = []
    offer_rates = []
    github_scores = []
    completeness_scores = []

    for c in candidates:
        sig = c.get('redrob_signals', {})
        response_rates.append(sig.get('recruiter_response_rate', 0.0))
        response_times.append(sig.get('avg_response_time_hours', 0.0))
        interview_rates.append(sig.get('interview_completion_rate', 0.0))

        o_rate = sig.get('offer_acceptance_rate', -1)
        if o_rate != -1:
            offer_rates.append(o_rate)

        gh_score = sig.get('github_activity_score', -1)
        if gh_score != -1:
            github_scores.append(gh_score)

        completeness_scores.append(sig.get('profile_completeness_score', 0.0))

    cdfs = {
        'recruiter_response_rate': np.sort(response_rates) if response_rates else np.array([0.0]),
        'avg_response_time_hours': np.sort(response_times) if response_times else np.array([0.0]),
        'interview_completion_rate': np.sort(interview_rates) if interview_rates else np.array([0.0]),
        'offer_acceptance_rate': np.sort(offer_rates) if offer_rates else np.array([0.0]),
        'github_activity_score': np.sort(github_scores) if github_scores else np.array([0.0]),
        'profile_completeness_score': np.sort(completeness_scores) if completeness_scores else np.array([0.0])
    }
    return cdfs


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    today   = date.today()

    # ── 1. Load candidates ────────────────────────────────────────────────
    t0         = time.time()
    candidates = load_candidates(args.candidates, limit=args.limit)
    N          = len(candidates)
    print(f"[Load] Done in {time.time()-t0:.1f}s")

    # ── 2. Extract candidate IDs ──────────────────────────────────────────
    candidate_ids = [c.get('candidate_id', f'CAND_{i:07d}') for i, c in enumerate(candidates)]
    ids_path      = out_dir / 'candidate_ids.npy'
    np.save(str(ids_path), np.array(candidate_ids, dtype=object))
    print(f"[IDs] Saved {N} candidate IDs → {ids_path}")

    # ── 2.5 Compute and save CDFs ─────────────────────────────────────────
    import pickle
    print(f"[CDF] Computing cumulative distribution functions ...")
    cdfs = compute_cdfs(candidates)
    cdf_path = out_dir / 'cdfs.pkl'
    with open(cdf_path, 'wb') as f:
        pickle.dump(cdfs, f)
    print(f"[CDF] Saved CDF distributions to {cdf_path}")

    # ── 3. Build candidate texts (for embeddings + BM25) ─────────────────
    print(f"[Text] Building candidate text blobs ...")
    t0 = time.time()
    candidate_texts = [
        build_candidate_text(c, max_chars=CANDIDATE_TEXT_MAX_CHARS, cdfs=cdfs)
        for c in tqdm(candidates, desc="Text builder", unit=" cands")
    ]
    print(f"[Text] Done in {time.time()-t0:.1f}s")

    # ── 4. Bi-encoder embeddings ──────────────────────────────────────────
    emb_path = out_dir / 'embeddings.npy'
    if not args.skip_embeddings:
        t0 = time.time()
        build_bi_encoder_embeddings(
            candidate_texts=candidate_texts,
            output_path=str(emb_path),
            batch_size=args.batch_size,
            show_progress=True,
        )
        print(f"[BiEncoder] Embeddings done in {time.time()-t0:.1f}s")
    else:
        print(f"[BiEncoder] Skipped (using existing {emb_path})")

    # ── 5. BM25 index ─────────────────────────────────────────────────────
    bm25_path = out_dir / 'bm25_index.pkl'
    if not args.skip_bm25:
        t0 = time.time()
        print(f"[BM25] Tokenizing {N:,} documents ...")
        tokenized = [
            build_bm25_tokens(c)
            for c in tqdm(candidates, desc="Tokenizing", unit=" cands")
        ]
        build_bm25_index(
            tokenized_corpus=tokenized,
            output_path=str(bm25_path),
        )
        print(f"[BM25] Index done in {time.time()-t0:.1f}s")
    else:
        print(f"[BM25] Skipped (using existing {bm25_path})")

    # ── 6. Structured features ────────────────────────────────────────────
    feat_path = out_dir / 'features.npy'
    if not args.skip_features:
        t0 = time.time()
        print(f"[Structured] Extracting features for {N:,} candidates ...")
        scores, sub_list = batch_compute_structured(candidates, today=today)
        features = np.array(scores, dtype=np.float32)
        np.save(str(feat_path), features)
        print(f"[Structured] Done in {time.time()-t0:.1f}s → {feat_path}")
    else:
        print(f"[Structured] Skipped (using existing {feat_path})")

    # ── 7. Honeypot detection ─────────────────────────────────────────────
    hp_path = out_dir / 'honeypot_flags.npy'
    t0 = time.time()
    print(f"[Honeypot] Screening {N:,} candidates ...")
    hp_penalties, hp_flags_list = batch_detect_honeypots(candidates)
    hp_array = np.array(hp_penalties, dtype=np.float32)
    np.save(str(hp_path), hp_array)

    n_definite = sum(1 for p in hp_penalties if p == 0.0)
    n_likely   = sum(1 for p in hp_penalties if p == 0.05)
    n_suspect  = sum(1 for p in hp_penalties if p == 0.40)
    print(f"[Honeypot] Done in {time.time()-t0:.1f}s → {hp_path}")
    print(f"[Honeypot] Definite: {n_definite} | Likely: {n_likely} | Suspect: {n_suspect}")

    # ── 8. Disqualifier penalties ─────────────────────────────────────────
    dq_path = out_dir / 'disq_flags.npy'
    t0 = time.time()
    print(f"[Disqualifiers] Processing {N:,} candidates ...")
    dq_penalties, dq_triggered = batch_compute_disqualifiers(candidates, today=today)
    dq_array = np.array(dq_penalties, dtype=np.float32)
    np.save(str(dq_path), dq_array)

    n_disq = sum(1 for p in dq_penalties if p < 1.0)
    print(f"[Disqualifiers] Done in {time.time()-t0:.1f}s → {dq_path}")
    print(f"[Disqualifiers] Candidates with any penalty: {n_disq:,}")

    # ── 9. Save candidate texts (needed by cross-encoder at rank time) ────
    texts_path = out_dir / 'candidate_texts.npy'
    np.save(str(texts_path), np.array(candidate_texts, dtype=object))
    print(f"[Texts] Saved candidate texts → {texts_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Pre-computation complete in {total_time/60:.1f} minutes")
    print(f"Artifacts saved to: {out_dir.resolve()}")
    print(f"  embeddings.npy:       {emb_path.stat().st_size / 1e6:.1f} MB" if emb_path.exists() else "")
    print(f"  bm25_index.pkl:       {bm25_path.stat().st_size / 1e6:.1f} MB" if bm25_path.exists() else "")
    print(f"  features.npy:         {feat_path.stat().st_size / 1e6:.1f} MB" if feat_path.exists() else "")
    print(f"  candidate_ids.npy:    {ids_path.stat().st_size / 1e6:.1f} MB")
    print(f"  honeypot_flags.npy:   {hp_path.stat().st_size / 1e6:.1f} MB")
    print(f"  disq_flags.npy:       {dq_path.stat().st_size / 1e6:.1f} MB")
    print(f"  candidate_texts.npy:  {texts_path.stat().st_size / 1e6:.1f} MB")
    print(f"\nNext step:  python rank.py --candidates {args.candidates} "
          f"--jd ./job_description.docx --precomputed {out_dir} --out ./submission.csv")


if __name__ == '__main__':
    main()
