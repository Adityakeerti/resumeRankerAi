#!/usr/bin/env python3
"""
rank.py — REPRODUCE COMMAND entrypoint for Redrob AI Hackathon submission.

Usage:
  python rank.py --candidates ./candidates.jsonl \\
                 --jd ./job_description.docx \\
                 --precomputed ./data/ \\
                 --out ./submission.csv

Requirements:
  - precompute.py must have been run first to produce ./data/ artifacts
  - No network access (--precomputed artifacts are fully self-contained)
  - CPU-only (no GPU required)

Runtime target: < 60 seconds on CPU with 16 GB RAM.

Output: submission.csv with columns: candidate_id, rank, score, reasoning
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Intelligent Candidate Ranking"
    )
    parser.add_argument(
        '--candidates', '-c',
        required=True,
        help='Path to candidates.jsonl'
    )
    parser.add_argument(
        '--jd', '-j',
        required=True,
        help='Path to job_description.docx (or .txt)'
    )
    parser.add_argument(
        '--precomputed', '-p',
        default='./data',
        help='Directory containing pre-computed artifacts (default: ./data)'
    )
    parser.add_argument(
        '--out', '-o',
        default='./submission.csv',
        help='Output CSV path (default: ./submission.csv)'
    )
    parser.add_argument(
        '--top-n-stage1',
        type=int,
        default=None,
        help='Override number of candidates passed to cross-encoder (default from config)'
    )
    parser.add_argument(
        '--no-cross-encoder',
        action='store_true',
        help='Skip cross-encoder step (faster, slightly lower quality)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Run validate_submission.py after generation (default: True)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug info for top-20 candidates'
    )
    return parser.parse_args()


def load_jd_text(jd_path: str) -> str:
    """
    Load job description text from .docx or .txt file.

    Args:
        jd_path: Path to JD file

    Returns:
        Raw JD text string
    """
    path = Path(jd_path)

    if path.suffix.lower() == '.docx':
        try:
            import docx
            doc  = docx.Document(str(path))
            text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
            return text
        except ImportError:
            print("[WARN] python-docx not installed; falling back to config JD text")
        except Exception as e:
            print(f"[WARN] Could not read .docx ({e}); falling back to config JD text")

    elif path.suffix.lower() == '.txt':
        with open(str(path), 'r', encoding='utf-8') as f:
            return f.read()

    # Fallback: use the config JD summary
    from config import JD_TEXT_SUMMARY
    print("[WARN] Using JD summary from config.py — for best results, provide actual JD file")
    return JD_TEXT_SUMMARY


def stream_candidates_by_ids(jsonl_path: str, wanted_ids: set[str]) -> dict[str, dict]:
    """
    Stream candidates.jsonl and collect only those with wanted IDs.
    Used to load the final top-100 candidate dicts for reasoning generation.

    Args:
        jsonl_path: Path to candidates.jsonl
        wanted_ids: Set of candidate IDs to retrieve

    Returns:
        Dict mapping candidate_id → candidate dict
    """
    result  = {}
    found   = 0
    needed  = len(wanted_ids)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if found >= needed:
                break
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                cid  = cand.get('candidate_id', '')
                if cid in wanted_ids:
                    result[cid] = cand
                    found += 1
            except json.JSONDecodeError:
                continue

    return result


def main():
    args  = parse_args()
    today = date.today()
    t_start = time.time()

    # ── Import modules ────────────────────────────────────────────────────
    from config import (
        TOP_N_STAGE1, TOP_N_FINAL,
        CANDIDATE_TEXT_MAX_CHARS,
        JD_TEXT_SUMMARY,
        NLI_MODEL,
        NLI_PREMISE_MODE,
        NLI_PREMISE_RELOC,
        PREFERRED_LOCATIONS,
    )
    from src.text_builder import build_candidate_text, build_jd_text, build_jd_tokens
    from src.semantic_scorer import load_embeddings, embed_query, cosine_scores, cross_encoder_scores, nli_constraint_penalties
    from src.bm25_scorer import load_bm25_index, query_bm25
    from src.fusion import stage1_fusion, stage2_fusion
    from src.reasoning import build_reasoning
    import pickle

    top_n_stage1 = args.top_n_stage1 or TOP_N_STAGE1
    data_dir     = Path(args.precomputed)

    # ── Step 1: Load pre-computed artifacts ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"Redrob AI Hackathon — Intelligent Candidate Ranking")
    print(f"{'='*60}")
    print(f"\n[1/7] Loading pre-computed artifacts from {data_dir} ...")
    t0 = time.time()

    emb_path   = data_dir / 'embeddings.npy'
    ids_path   = data_dir / 'candidate_ids.npy'
    feat_path  = data_dir / 'features.npy'
    bm25_path  = data_dir / 'bm25_index.pkl'
    hp_path    = data_dir / 'honeypot_flags.npy'
    dq_path    = data_dir / 'disq_flags.npy'
    texts_path = data_dir / 'candidate_texts.npy'
    cdf_path   = data_dir / 'cdfs.pkl'

    for p in [emb_path, ids_path, feat_path, bm25_path, hp_path, dq_path]:
        if not p.exists():
            print(f"[ERROR] Missing artifact: {p}")
            print(f"        Run precompute.py first!")
            sys.exit(1)

    candidate_ids     = list(np.load(str(ids_path), allow_pickle=True))
    candidate_embs    = load_embeddings(str(emb_path))   # mmap'd — no full RAM load
    structured_scores = np.load(str(feat_path))
    honeypot_pen      = np.load(str(hp_path))
    disq_pen          = np.load(str(dq_path))
    bm25              = load_bm25_index(str(bm25_path))

    # Load CDFs if available
    if cdf_path.exists():
        with open(cdf_path, 'rb') as f:
            cdfs = pickle.load(f)
        print(f"[1/7] Loaded CDF distribution data")
    else:
        cdfs = None

    # Candidate texts (for cross-encoder)
    if texts_path.exists() and not args.no_cross_encoder:
        candidate_texts_all = list(np.load(str(texts_path), allow_pickle=True))
    else:
        candidate_texts_all = None

    N = len(candidate_ids)
    print(f"[1/7] Loaded {N:,} candidates in {time.time()-t0:.1f}s")

    # ── Step 2: Load & embed JD ───────────────────────────────────────────
    print(f"\n[2/7] Loading and embedding job description ...")
    t0 = time.time()

    raw_jd_text = load_jd_text(args.jd)
    jd_text     = build_jd_text(raw_jd_text)
    jd_vec      = embed_query(jd_text)
    jd_tokens   = build_jd_tokens(raw_jd_text)

    print(f"[2/7] JD embedded in {time.time()-t0:.1f}s  ({len(jd_tokens)} tokens)")

    # ── Step 2.5: L1 Shortlist (Top 2,000) ─────────────────────────────────
    print(f"\n[2.5/7] L1 Shortlist (Top 2,000) ...")
    t0 = time.time()

    # Compute cosine similarities on the full 384-dimensional embeddings
    # Since candidate_embs and jd_vec are already L2-normalized, this is just a dot product.
    scores_full = candidate_embs @ jd_vec

    # Sort and select top-2000 indices
    l1_indices = np.argsort(-scores_full)[:2000]
    l1_indices = np.array(l1_indices)

    print(f"[2.5/7] L1 shortlist generated in {time.time()-t0:.1f}s (Top 2,000 selected)")

    # ── Step 2.6: NLI Logical Gate (on Top-2000) ──────────────────────────
    print(f"\n[2.6/7] NLI Logical Gate (on Top-2000) ...")
    t0 = time.time()

    l1_ids = [candidate_ids[idx] for idx in l1_indices]
    print(f"[NLI] Loading profiles for Top-2000 candidates from candidates.jsonl ...")
    l1_cands_map = stream_candidates_by_ids(args.candidates, set(l1_ids))
    l1_cands_list = [l1_cands_map[cid] for cid in l1_ids if cid in l1_cands_map]

    # Ensure mapping matches the order of l1_indices
    # Fallback to empty candidates if stream missed some profiles
    ordered_l1_cands = []
    for cid in l1_ids:
        if cid in l1_cands_map:
            ordered_l1_cands.append(l1_cands_map[cid])
        else:
            ordered_l1_cands.append({'candidate_id': cid, 'profile': {}, 'redrob_signals': {}, 'career_history': [], 'skills': []})

    nli_penalties = nli_constraint_penalties(
        candidates_list=ordered_l1_cands,
        nli_model_name=NLI_MODEL,
        work_mode_premise=NLI_PREMISE_MODE,
        relocation_premise=NLI_PREMISE_RELOC,
        preferred_locations=PREFERRED_LOCATIONS,
        batch_size=32
    )

    print(f"[2.6/7] NLI Logical Gate done in {time.time()-t0:.1f}s")

    # ── Step 3: Compute semantic & lexical scores on top-2000 ─────────────
    print(f"\n[3/7] Scoring Top-2,000 shortlisted candidates ...")
    t0 = time.time()

    # Full 768-d semantic cosine similarities for the top-2000
    semantic_sc_full = cosine_scores(jd_vec, candidate_embs[l1_indices])

    # BM25 lexical scores for the top-2000
    lexical_sc_full  = query_bm25(bm25, jd_tokens)[l1_indices]

    # Expand back to global arrays (rest of candidates set to 0.0)
    semantic_sc = np.zeros(N, dtype=np.float32)
    semantic_sc[l1_indices] = semantic_sc_full

    lexical_sc = np.zeros(N, dtype=np.float32)
    lexical_sc[l1_indices] = lexical_sc_full

    # Update global disqualifier penalties to include NLI logic check
    disq_pen_nli = disq_pen.copy()
    disq_pen_nli[l1_indices] *= nli_penalties

    print(f"[3/7] Semantic + lexical scoring done in {time.time()-t0:.1f}s")

    # -- Step 4: Stage 1 fusion -> top-500 ---------------------------------
    print(f"\n[4/7] Stage 1 fusion -> top-{top_n_stage1} candidates ...")
    t0 = time.time()

    _, top_n_indices = stage1_fusion(
        semantic_scores=semantic_sc,
        lexical_scores=lexical_sc,
        structured_scores=structured_scores,
        honeypot_penalties=honeypot_pen,
        disq_penalties=disq_pen_nli,
        top_n=top_n_stage1,
    )

    print(f"[4/7] Stage 1 done in {time.time()-t0:.1f}s  "
          f"(top-{top_n_stage1} selected from {N:,})")

    # ── Step 5: Cross-encoder reranking ───────────────────────────────────
    if args.no_cross_encoder:
        print(f"\n[5/7] Cross-encoder SKIPPED (--no-cross-encoder flag)")
        ce_scores_top = np.zeros(len(top_n_indices), dtype=np.float32)
    else:
        print(f"\n[5/7] Cross-encoder reranking top-{top_n_stage1} candidates ...")
        t0 = time.time()

        if candidate_texts_all is not None:
            top_texts = [candidate_texts_all[i] for i in top_n_indices]
        else:
            # Fallback: rebuild texts on the fly for top-N
            print("[5/7] Candidate texts not cached; rebuilding for top candidates ...")
            top_texts = _rebuild_texts_for_indices(
                jsonl_path=args.candidates,
                target_ids=[candidate_ids[i] for i in top_n_indices],
                max_chars=CANDIDATE_TEXT_MAX_CHARS,
            )

        ce_scores_top = cross_encoder_scores(
            jd_text=jd_text,
            candidate_texts=top_texts,
            show_progress=True,
        )
        print(f"[5/7] Cross-encoder done in {time.time()-t0:.1f}s")

    # ── Step 6: Stage 2 fusion -> final top-100 ───────────────────────────
    print(f"\n[6/7] Stage 2 fusion -> final top-{TOP_N_FINAL} ...")
    t0 = time.time()

    final_indices, final_scores = stage2_fusion(
        top_n_indices=top_n_indices,
        semantic_scores=semantic_sc,
        lexical_scores=lexical_sc,
        structured_scores=structured_scores,
        cross_encoder_scores=ce_scores_top,
        honeypot_penalties=honeypot_pen,
        disq_penalties=disq_pen_nli,
        candidate_ids=candidate_ids,
        top_n_final=TOP_N_FINAL,
    )

    final_cand_ids = [candidate_ids[i] for i in final_indices]
    print(f"[6/7] Stage 2 done in {time.time()-t0:.1f}s")

    # ── Step 7: Load final candidates + generate reasoning ────────────────
    print(f"\n[7/7] Slicing top-{TOP_N_FINAL} candidate profiles for reasoning ...")
    t0 = time.time()

    # Reuse the already loaded profiles from Step 2.6 to avoid disk I/O
    top_cand_map = {}
    missing_ids = set()
    for cid in final_cand_ids:
        if cid in l1_cands_map:
            top_cand_map[cid] = l1_cands_map[cid]
        else:
            missing_ids.add(cid)

    if missing_ids:
        print(f"[WARN] Retrieving {len(missing_ids)} profiles missing from L1 cache ...")
        fallback_map = stream_candidates_by_ids(args.candidates, missing_ids)
        top_cand_map.update(fallback_map)

    # Build rows
    rows = []
    for rank_idx, (cid, score) in enumerate(zip(final_cand_ids, final_scores), start=1):
        cand = top_cand_map.get(cid)
        if cand is None:
            reasoning = f"Rank {rank_idx} candidate (profile unavailable)"
        else:
            reasoning = build_reasoning(cand, rank=rank_idx, today=today)

        rows.append({
            'candidate_id': cid,
            'rank':         rank_idx,
            'score':        f"{float(score):.4f}",
            'reasoning':    reasoning,
        })

        if args.debug and rank_idx <= 20:
            orig_idx = final_indices[rank_idx - 1]
            print(
                f"  Rank {rank_idx:3d} | {cid} | "
                f"sem={semantic_sc[orig_idx]:.3f} "
                f"lex={lexical_sc[orig_idx]:.3f} "
                f"str={structured_scores[orig_idx]:.3f} "
                f"ce={ce_scores_top[list(top_n_indices).index(orig_idx)] if not args.no_cross_encoder else 0:.3f} "
                f"hp={honeypot_pen[orig_idx]:.2f} "
                f"dq={disq_pen_nli[orig_idx]:.2f} "
                f"-> {float(score):.4f}"
            )

    print(f"[7/7] Reasoning generated in {time.time()-t0:.1f}s")

    # ── Write submission.csv ──────────────────────────────────────────────
    out_path = Path(args.out)
    with open(str(out_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['candidate_id', 'rank', 'score', 'reasoning'],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"submission.csv written: {out_path.resolve()}")
    print(f"Total runtime: {total_time:.1f}s")
    print(f"{'='*60}")

    # Quick sanity check output
    print(f"\nTop-5 candidates:")
    for row in rows[:5]:
        print(f"  {row['rank']:3d}. {row['candidate_id']}  score={row['score']}")
        print(f"       {row['reasoning'][:100]}...")

    # ── Auto-validate ─────────────────────────────────────────────────────
    if args.validate:
        _run_validation(str(out_path))

    return rows


def _rebuild_texts_for_indices(
    jsonl_path: str,
    target_ids: list[str],
    max_chars: int = 2000,
) -> list[str]:
    """
    Fallback: rebuild candidate texts from JSONL for specific candidate IDs.
    Returns texts in the same order as target_ids.

    Args:
        jsonl_path: Path to candidates.jsonl
        target_ids: Ordered list of candidate IDs
        max_chars: Text truncation limit

    Returns:
        List of text strings in same order as target_ids
    """
    from src.text_builder import build_candidate_text

    id_to_text = {}
    needed     = set(target_ids)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Rebuilding texts", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            if len(id_to_text) >= len(needed):
                break
            try:
                cand = json.loads(line)
                cid  = cand.get('candidate_id', '')
                if cid in needed:
                    id_to_text[cid] = build_candidate_text(cand, max_chars=max_chars)
            except json.JSONDecodeError:
                continue

    return [id_to_text.get(cid, '') for cid in target_ids]


def _run_validation(csv_path: str):
    """Run validate_submission.py if it exists."""
    validate_script = Path(__file__).parent / 'validate_submission.py'
    if validate_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(validate_script), csv_path],
            capture_output=True, text=True
        )
        print(f"\n[Validate] {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"[Validate ERROR] {result.stderr.strip()}")
    else:
        print(f"\n[Validate] validate_submission.py not found — run manually to verify")


if __name__ == '__main__':
    main()
