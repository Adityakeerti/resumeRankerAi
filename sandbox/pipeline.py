"""
sandbox/pipeline.py — Instrumented, stage-by-stage version of the ranking pipeline for the demo UI.

Same logic as rank.py (reuses every scorer in src/), but:
  * models are passed in (so the UI can cache them across runs), and
  * run_pipeline() is a generator that yields after every stage with that
    stage's timing + artefacts, so the UI can render the pipeline live.

No scoring logic lives here; if a number in the UI disagrees with rank.py,
the bug is in this file's plumbing, not in src/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from config import (
    CANDIDATE_TEXT_MAX_CHARS,
    JD_TEXT_FOR_CROSS_ENCODER,
    CAND_TEXT_FOR_CROSS_ENCODER,
    STAGE1_WEIGHTS,
    STAGE2_WEIGHTS,
)
from src.text_builder import build_candidate_text, build_jd_text, build_jd_tokens, build_bm25_tokens
from src.semantic_scorer import cosine_scores
from src.structured_scorer import batch_compute_structured
from src.honeypot_detector import batch_detect_honeypots
from src.disqualifiers import batch_compute_disqualifiers
from src.fusion import stage1_fusion, stage2_fusion
from src.reasoning import build_reasoning


# Stage registry — order matters, the UI draws the stepper from this.
STAGES = [
    ("ingest",     "Ingest",        "Parse & validate profiles"),
    ("text",       "Represent",     "Build model-ready text"),
    ("semantic",   "Semantic",      "Bi-encoder similarity"),
    ("lexical",    "Lexical",       "BM25 keyword match"),
    ("structured", "JD-fit",        "8 structured signals"),
    ("trust",      "Trust",         "Honeypots & disqualifiers"),
    ("stage1",     "Shortlist",     "Stage-1 fusion"),
    ("rerank",     "Rerank",        "Cross-encoder"),
    ("final",      "Rank & explain", "Stage-2 fusion + reasoning"),
]
STAGE_KEYS = [s[0] for s in STAGES]


@dataclass
class StageEvent:
    key: str
    seconds: float
    headline: str                     # one-line, human summary for the live log
    data: dict = field(default_factory=dict)


def run_pipeline(
    candidates: list[dict],
    jd_raw: str,
    *,
    bi_encoder,
    cross_encoder,                    # may be None when reranking is disabled
    shortlist_k: int,
    final_n: int,
    today: date,
):
    """Generator: yields one StageEvent per stage, then returns the full trace via StopIteration.value."""
    N = len(candidates)
    trace: dict = {"N": N, "today": today.isoformat()}

    # ── 1. Ingest ───────────────────────────────────────────────────────────
    t = time.perf_counter()
    ids = [c.get("candidate_id") or f"CAND_{i:07d}" for i, c in enumerate(candidates)]
    missing = {
        "career_history": sum(1 for c in candidates if not c.get("career_history")),
        "skills":         sum(1 for c in candidates if not c.get("skills")),
        "redrob_signals": sum(1 for c in candidates if not c.get("redrob_signals")),
    }
    yoe = np.array([float(c.get("profile", {}).get("years_of_experience") or 0) for c in candidates])
    dupes = N - len(set(ids))
    trace["ids"] = ids
    yield StageEvent(
        "ingest", time.perf_counter() - t,
        f"{N} profiles parsed · {dupes} duplicate IDs · median {np.median(yoe):.1f} yrs experience",
        {"missing": missing, "dupes": dupes, "yoe": yoe},
    )

    # ── 2. Text representations ─────────────────────────────────────────────
    t = time.perf_counter()
    texts = [build_candidate_text(c, CANDIDATE_TEXT_MAX_CHARS) for c in candidates]
    tokens = [build_bm25_tokens(c) for c in candidates]
    jd_text = build_jd_text(jd_raw)
    jd_tokens = build_jd_tokens(jd_raw)
    trace.update(texts=texts, tokens=tokens, jd_text=jd_text, jd_tokens=jd_tokens)
    yield StageEvent(
        "text", time.perf_counter() - t,
        f"{N} text blobs (avg {np.mean([len(x) for x in texts]):.0f} chars) · "
        f"{len(set(jd_tokens))} unique JD query terms",
        {"avg_chars": float(np.mean([len(x) for x in texts])),
         "avg_tokens": float(np.mean([len(x) for x in tokens])),
         "jd_text": jd_text, "jd_terms": len(set(jd_tokens)), "example_text": texts[0] if texts else ""},
    )

    # ── 3. Semantic (bi-encoder) ────────────────────────────────────────────
    t = time.perf_counter()
    jd_vec = bi_encoder.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    embs = bi_encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                             batch_size=64, show_progress_bar=False).astype(np.float32)
    sem = cosine_scores(jd_vec, embs)
    trace["sem"] = sem
    yield StageEvent(
        "semantic", time.perf_counter() - t,
        f"Embedded JD + {N} profiles into {embs.shape[1]}-d space · cosine range "
        f"{sem.min():.2f}–{sem.max():.2f}",
        {"dim": int(embs.shape[1]), "sem": sem},
    )

    # ── 4. Lexical (BM25) ───────────────────────────────────────────────────
    t = time.perf_counter()
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokens)                     # in-memory; no pickle needed for the demo
    raw = bm25.get_scores(jd_tokens).astype(np.float32)
    lex = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() - raw.min() > 1e-9 else np.zeros_like(raw)
    # Which JD terms are most discriminative in *this* pool (high IDF and present)?
    jd_unique = sorted(set(jd_tokens))
    term_idf = sorted(((bm25.idf.get(tk, 0.0), tk) for tk in jd_unique if tk in bm25.idf), reverse=True)
    trace["lex"] = lex
    yield StageEvent(
        "lexical", time.perf_counter() - t,
        f"BM25 over {N} docs · {sum(1 for _, tk in term_idf)} of {len(jd_unique)} JD terms appear in the pool",
        {"term_idf": term_idf[:25], "lex": lex},
    )

    # ── 5. Structured JD-fit ────────────────────────────────────────────────
    t = time.perf_counter()
    str_scores, sub_scores = batch_compute_structured(candidates, today=today)
    st_arr = np.array(str_scores, dtype=np.float32)
    trace.update(struct=st_arr, sub_scores=sub_scores)
    yield StageEvent(
        "structured", time.perf_counter() - t,
        f"8 sub-scores per profile (skills, career, experience, location, notice, behaviour, "
        f"technical, demand) · mean JD-fit {st_arr.mean():.2f}",
        {"struct": st_arr, "sub_scores": sub_scores},
    )

    # ── 6. Trust: honeypots + disqualifiers ─────────────────────────────────
    t = time.perf_counter()
    hp_pen, hp_flags = batch_detect_honeypots(candidates)
    dq_pen, dq_flags = batch_compute_disqualifiers(candidates, today=today)
    hp = np.array(hp_pen, dtype=np.float32)
    dq = np.array(dq_pen, dtype=np.float32)
    trace.update(hp=hp, dq=dq, hp_flags=hp_flags, dq_flags=dq_flags)
    n_hp = int((hp < 1).sum())
    n_dq = int((dq < 1).sum())
    yield StageEvent(
        "trust", time.perf_counter() - t,
        f"{n_hp} likely honeypot / inconsistent profiles · {n_dq} hit a JD disqualifier "
        f"(penalties are multiplicative, not hard filters)",
        {"n_hp": n_hp, "n_dq": n_dq, "hp": hp, "dq": dq, "hp_flags": hp_flags, "dq_flags": dq_flags},
    )

    # ── 7. Stage-1 fusion → shortlist ───────────────────────────────────────
    t = time.perf_counter()
    k = min(shortlist_k, N)
    s1, top_idx = stage1_fusion(sem, lex, st_arr, hp, dq, top_n=k)
    trace.update(s1=s1, top_idx=top_idx)
    yield StageEvent(
        "stage1", time.perf_counter() - t,
        f"{N} → {k} shortlisted using {fmt_weights(STAGE1_WEIGHTS)} × penalties",
        {"k": k, "s1": s1, "top_idx": top_idx},
    )

    # ── 8. Cross-encoder rerank ─────────────────────────────────────────────
    t = time.perf_counter()
    if cross_encoder is not None:
        jd_trunc = jd_text[:JD_TEXT_FOR_CROSS_ENCODER]
        pairs = [(jd_trunc, texts[i][:CAND_TEXT_FOR_CROSS_ENCODER]) for i in top_idx]
        logits = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
        ce = (1.0 / (1.0 + np.exp(-np.asarray(logits)))).astype(np.float32)
        head = f"Scored {len(pairs)} (JD, profile) pairs jointly · relevance {ce.min():.2f}–{ce.max():.2f}"
    else:
        ce = np.zeros(len(top_idx), dtype=np.float32)
        head = "Skipped (disabled) — cross-encoder term contributes 0 for everyone"
    trace["ce"] = ce
    yield StageEvent("rerank", time.perf_counter() - t, head, {"enabled": cross_encoder is not None, "ce": ce})

    # ── 9. Stage-2 fusion + reasoning ───────────────────────────────────────
    t = time.perf_counter()
    n_out = min(final_n, len(top_idx))
    final_idx, final_sc = stage2_fusion(
        top_n_indices=top_idx, semantic_scores=sem, lexical_scores=lex,
        structured_scores=st_arr, cross_encoder_scores=ce,
        honeypot_penalties=hp, disq_penalties=dq, candidate_ids=ids, top_n_final=n_out,
    )
    pos_in_short = {int(ci): p for p, ci in enumerate(top_idx)}
    w = STAGE2_WEIGHTS
    rows = []
    for rank, (ci, score) in enumerate(zip(final_idx, final_sc), start=1):
        ci = int(ci)
        p = pos_in_short[ci]
        contrib = {
            "Semantic":      w["semantic"] * float(sem[ci]),
            "Lexical":       w["lexical"] * float(lex[ci]),
            "JD-fit":        w["structured"] * float(st_arr[ci]),
            "Cross-encoder": w["cross_encoder"] * float(ce[p]),
        }
        rows.append({
            "rank": rank, "idx": ci, "id": ids[ci], "score": float(score),
            "stage1_rank": p + 1, "contrib": contrib,
            "penalty": float(hp[ci] * dq[ci]),
            "reasoning": build_reasoning(candidates[ci], rank=rank, today=today),
        })
    trace["final"] = rows
    moved = sum(1 for r in rows if r["stage1_rank"] != r["rank"])
    yield StageEvent(
        "final", time.perf_counter() - t,
        f"Top {n_out} ranked · {moved} changed position after reranking · reasoning generated for each",
        {"rows": rows},
    )
    return trace


def fmt_weights(w: dict) -> str:
    names = {"semantic": "sem", "lexical": "lex", "structured": "fit", "cross_encoder": "ce"}
    return " + ".join(f"{v:.2f}·{names.get(k, k)}" for k, v in w.items())
