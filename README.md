---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: sandbox/app.py
pinned: false
---

# Redrob AI Hackathon — Intelligent Candidate Ranking System

**Team**: Team Aditya | **Role**: Senior AI Engineer — Founding Team @ Redrob AI

---

## Architecture Overview

A **Hybrid Multi-Stage Ranking Pipeline** that combines semantic understanding,
lexical matching, structured JD-fit scoring, behavioral signals, and cross-encoder
reranking to produce a ranked top-100 from 100,000 candidate profiles.

```
═══════════════════════════════════════════════════════════
  OFFLINE (precompute.py — run once, ~45 min on CPU)
═══════════════════════════════════════════════════════════

  candidates.jsonl ──► Text Builder ──► BiEncoder ──► embeddings.npy
                   ──► BM25 Tokenizer ──────────── ──► bm25_index.pkl
                   ──► Structured Extractor ──────── ──► features.npy
                   ──► Honeypot Detector ─────────── ──► honeypot_flags.npy
                   ──► Disqualifier Scorer ────────── ──► disq_flags.npy

═══════════════════════════════════════════════════════════
  ONLINE (rank.py — < 60 seconds on CPU)
═══════════════════════════════════════════════════════════

  JD ──► embed ──► cosine_sim(100K) ──► semantic_scores
      ──► BM25 query ────────────── ──► lexical_scores
  Load features.npy ─────────────── ──► structured_scores

  Stage 1 Fusion (0.30×sem + 0.12×lex + 0.58×str) × penalties
       └──► Top-500 selected

  Cross-Encoder (ms-marco-MiniLM-L-6-v2) on top-500  [~25s]
       └──► cross_encoder_scores

  Stage 2 Fusion (0.25×sem + 0.10×lex + 0.45×str + 0.20×ce) × penalties
       └──► Final top-100 with tie-break by ascending candidate_id

  Reasoning Generator ──► submission.csv
```

---

## Key Differentiators vs. Naive Approaches

| Decision | Naive | Our Approach |
|---|---|---|
| Skills matching | Count AI keywords | Trust-weighted: proficiency × duration × endorsements × assessment |
| Career history | Ignored | Product vs consulting split + tenure stability + company size progression |
| Anti-stuffing | None | Career descriptions weighted above skills in text representation |
| Behavioral signals | 1-2 fields | 6-dimension behavioral score (recency, responsiveness, reliability) |
| Honeypot defense | None | 7 independent detection checks |
| Text semantics | BM25 only | Bi-encoder on career descriptions (catches "plain-language Tier 5s") |
| Final reranking | None | Cross-encoder on top-500 → maximises NDCG@10 (= 50% of total score) |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-compute artifacts (run once, ~45 min)
```bash
python precompute.py \
  --candidates ./candidates.jsonl \
  --out ./data/
```

### 3. Rank candidates (< 1 min)
```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --jd ./job_description.docx \
  --precomputed ./data/ \
  --out ./submission.csv
```

### 4. Validate output
```bash
python validate_submission.py submission.csv
```

---

## Project Structure

```
├── rank.py                    ← REPRODUCE COMMAND entrypoint
├── precompute.py              ← Offline pre-computation (run once)
├── config.py                  ← JD-extracted params, skill lists, weights
├── src/
│   ├── text_builder.py        ← build_candidate_text() + BM25 tokenizer
│   ├── semantic_scorer.py     ← Bi-encoder + Cross-encoder
│   ├── bm25_scorer.py         ← BM25 index build + query
│   ├── structured_scorer.py   ← 8 sub-component JD-fit scoring
│   ├── honeypot_detector.py   ← 7-check honeypot detection
│   ├── disqualifiers.py       ← 9 multiplicative penalty categories
│   ├── fusion.py              ← Stage 1 + Stage 2 score fusion
│   └── reasoning.py           ← Fact-grounded reasoning string generator
├── sandbox/
│   └── app.py                 ← Streamlit demo (HuggingFace Spaces)
├── data/                      ← Pre-computed artifacts (gitignored)
├── submission_metadata.yaml
├── requirements.txt
└── README.md
```

---

## Scoring Components

### Semantic Score (25%)
`all-MiniLM-L6-v2` bi-encoder over candidate career descriptions (not skills lists).
Career descriptions are placed first in the text representation to ensure semantic
matching captures "what they built" rather than "what keywords they listed."

### Lexical Score (10%)
BM25 over the full candidate profile. Lower weight because the JD explicitly warns
against keyword matching; used primarily for exact technology names (Pinecone, Qdrant).

### Structured JD-Fit Score (45%)
Eight sub-components:
1. **Skills Trust** (30%) — proficiency × duration × endorsements × assessment score
2. **Career Quality** (25%) — product vs consulting ratio, stability, company progression
3. **Experience Range** (10%) — YoE fit vs JD's stated 5–9y (ideal 6–8y)
4. **Location Fit** (8%) — Pune/Noida → preferred metros → relocate-willing → elsewhere
5. **Notice Period** (5%) — sub-30d ideal; exponential penalty above 60d
6. **Behavioral Availability** (12%) — recency, response rate, interview completion, offer history
7. **Technical Credibility** (7%) — GitHub activity + platform skill assessments
8. **Platform Demand** (3%) — recruiter saves + search appearances

### Cross-Encoder Score (20%)
`ms-marco-MiniLM-L-6-v2` applied to top-500 candidates only (~25s on CPU).
Maximises NDCG@10 (= 50% of the competition score metric).

---

## Constraints Compliance

| Constraint | Status |
|---|---|
| CPU-only inference | ✅ No GPU libraries |
| No network during ranking | ✅ All models loaded from local cache |
| ≤ 5 min ranking runtime | ✅ ~55 seconds observed |
| ≤ 16 GB RAM | ✅ Peak ~4 GB (embeddings memory-mapped) |
| ≤ 5 GB disk | ✅ ~400 MB total artifacts |

---

## Honeypot Detection

Seven independent checks flag implausible profiles:
1. YoE inflation (claimed YoE >> career history total)
2. Expert skills with zero duration
3. Mass expert claims with zero endorsements
4. Assessment score contradicts declared proficiency
5. Single-job career with enormous YoE gap
6. Skill duration exceeds claimed total YoE
7. Behavioral impossibility (mass applications + near-zero completion + minimal profile)

Penalty: 3+ flags → 0.0 (definite honeypot), 2 → 0.05, 1 → 0.40, 0 → 1.0

---

## Runtime Budget

| Step | Time |
|---|---|
| Load artifacts | ~8s |
| Embed JD | ~1s |
| Bi-encoder cosine (100K) | ~0.5s |
| BM25 scoring (100K) | ~3s |
| Stage 1 fusion + top-500 | ~1s |
| Cross-encoder (top-500) | ~25s |
| Final sort + reasoning + CSV | ~5s |
| **Total** | **~45s** |
