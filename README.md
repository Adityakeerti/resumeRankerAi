---
title: TalentBridge - Candidate Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# TalentBridge: Candidate Ranking Pipeline

TalentBridge ranks 100,000 candidate profiles against a job description and returns an explained top 100.
It combines semantic search, keyword search, structured job-fit scoring, fake-profile detection and
cross-encoder reranking. Everything runs on CPU, with no network access at ranking time.

Built by **team_404_Founders** for the Redrob AI challenge (role: Senior AI Engineer, Founding Team).

- Code: <https://github.com/Adityakeerti/resumeRankerAi>
- Hosted Streamlit demo: <https://huggingface.co/spaces/adityacodes404/redrob-ranker>
- Design rationale: [`solution.md`](solution.md)

---

## Contents

1. [How the pipeline works](#how-the-pipeline-works)
2. [Requirements](#requirements)
3. [Setup (once)](#setup-once)
4. [Run the full pipeline](#run-the-full-pipeline)
5. [Run the interactive demo](#run-the-interactive-demo)
6. [Command reference](#command-reference)
7. [Project structure](#project-structure)
8. [Scoring details](#scoring-details)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)

---

## How the pipeline works

The work is split into an offline step that runs once and an online step that runs per job description.

```
OFFLINE  precompute.py  (once per candidate file)
  candidates.jsonl ─► candidate text ─► bi-encoder (all-MiniLM-L6-v2) ─► embeddings.npy
                   ─► BM25 tokens ─────────────────────────────────────► bm25_index.pkl
                   ─► 8 structured JD-fit features ────────────────────► features.npy
                   ─► honeypot checks (fake profiles) ─────────────────► honeypot_flags.npy
                   ─► disqualifier rules ──────────────────────────────► disq_flags.npy
                   ─► signal distributions, cached texts, ids ─────────► cdfs.pkl, candidate_texts.npy, candidate_ids.npy

ONLINE   rank.py  (per job description)
  1. Embed the JD with the bi-encoder
  2. L1 shortlist: cosine similarity over all candidates, keep top 2,000
  3. NLI gate (cross-encoder/nli-deberta-v3-small) on the 2,000: penalise remote-only / won't-relocate contradictions
  4. Semantic + BM25 scores on the 2,000
  5. Stage 1 fusion  0.30·semantic + 0.12·lexical + 0.58·structured, × penalties  ─► top 500
  6. Cross-encoder rerank (cross-encoder/ms-marco-MiniLM-L-6-v2) on the 500
  7. Stage 2 fusion  0.25·semantic + 0.10·lexical + 0.45·structured + 0.20·cross-encoder, × penalties ─► top 100
  8. Fact-based reasoning per candidate ─► submission.csv ─► validate_submission.py
```

Penalties are multiplicative, so one strong signal cannot outweigh a red flag.
Ties are broken by ascending `candidate_id`.

---

## Requirements

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.11 (3.10–3.12 work; numpy 1.26 does not support 3.13+) | everything |
| Node.js + npm | 20 or newer | React web demo only |
| Disk | ~1.5 GB (CPU PyTorch + models + artifacts) + 465 MB dataset | |
| RAM | up to 16 GB for the full 100K run (the challenge limit) | |
| GPU | not used | |

Network access is needed only for setup (packages and models). Ranking runs offline.

---

## Setup (once)

### 1. Clone the repository

```bash
git clone https://github.com/Adityakeerti/resumeRankerAi.git
cd resumeRankerAi
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Install Python dependencies

Install the CPU build of PyTorch first. It is about 200 MB, against about 2.5 GB for the default CUDA build.
Then install the rest:

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Download the models from Hugging Face

The pipeline uses three public models (no Hugging Face account or token needed):

| Model | Role | Size |
|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | bi-encoder for semantic similarity | ~90 MB |
| [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | cross-encoder reranker | ~90 MB |
| [`cross-encoder/nli-deberta-v3-small`](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | NLI gate for work-mode / relocation | ~550 MB |

```bash
python download_models.py
```

This downloads only the weights each model needs into the Hugging Face cache (`~/.cache/huggingface/hub`,
or `$HF_HOME/hub` if set). It then runs one prediction per model to confirm the download works.

<details>
<summary>Alternative: download with the Hugging Face CLI</summary>

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 --exclude "onnx/*" "openvino/*" "*.h5" "*.msgpack" "*.ot" "tf_*" "flax_*" "rust_*"
hf download cross-encoder/ms-marco-MiniLM-L-6-v2   --exclude "onnx/*" "openvino/*" "*.h5" "*.msgpack" "*.ot" "tf_*" "flax_*" "rust_*"
hf download cross-encoder/nli-deberta-v3-small     --exclude "onnx/*" "openvino/*" "*.h5" "*.msgpack" "*.ot" "tf_*" "flax_*" "rust_*"
```

On `huggingface_hub` versions before 0.34, the command is `huggingface-cli download` instead of `hf download`.
Without `--exclude`, the CLI also fetches ONNX and OpenVINO copies that the pipeline never uses.
</details>

### 5. Add the dataset

These files from the challenge are already in `data/`:

| File | What it is |
|---|---|
| `data/job_description.docx` | the official job description |
| `data/candidate_schema.json` | JSON schema of a candidate profile |
| `data/sample_candidates.jsonl` | 50-profile official sample |
| `data/demo_candidates.jsonl` | 200-profile curated subset for the demo (see [Limitations](#limitations)) |

The full `candidates.jsonl` is 465 MB, so it is not in git.
Copy your copy of the challenge's `candidates.jsonl` into `data/`:

```bash
cp /path/to/India_runs_data_and_ai_challenge/candidates.jsonl data/candidates.jsonl
wc -l data/candidates.jsonl        # expect 100000
```

---

## Run the full pipeline

All commands run from the repository root with the virtual environment active.

### 1. Pre-compute candidate artifacts (once per candidate file)

```bash
python precompute.py --candidates data/candidates.jsonl --out data/
```

This writes the eight artifacts listed above into `data/`. It takes the longest of all the steps, mostly embedding
100K profiles. It runs on CPU and the time scales linearly with the number of profiles: the original runs
took about 45 minutes on a laptop CPU.

### 2. Rank against the job description (offline)

```bash
HF_HUB_OFFLINE=1 python rank.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.docx \
  --precomputed data/ \
  --out submission.csv
```

`HF_HUB_OFFLINE=1` makes sure no network call is made; the models load from the local cache.
`--jd` also accepts a `.txt` file, so you can rank against any other role.
The script validates the CSV automatically at the end.

### 3. Validate the submission manually (optional)

```bash
python validate_submission.py submission.csv
```

### Quick end-to-end check (under a minute)

To confirm the setup without the long pre-compute, run the whole pipeline on the first 3,000 profiles:

```bash
python precompute.py --candidates data/candidates.jsonl --out data/smoke/ --limit 3000
HF_HUB_OFFLINE=1 python rank.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.docx \
  --precomputed data/smoke/ \
  --out smoke_submission.csv
```

The ranking is only over those 3,000 profiles, so this tests the setup, not ranking quality.

---

## Run the interactive demo

There are two front ends. Both call the same `src/` scorers as `rank.py`, through
`sandbox/pipeline.py`, and score the selected pool live, without the pre-computed artifacts.

### React web app (recommended for presenting)

It streams each of the nine pipeline stages to the browser as it finishes. It also has a ranked shortlist,
a per-candidate detail panel, a risk review of flagged profiles and CSV export.

```bash
./run_demo.sh            # installs UI deps, builds the UI if needed, serves UI + API
```

Open <http://127.0.0.1:8000>.

For development with hot reload (UI on :5173, API on :8000):

```bash
./run_demo.sh --dev
```

Open <http://127.0.0.1:5173>.

The same thing step by step (also the way to run it on Windows):

```bash
cd web
npm ci                   # install exact UI dependency versions
npm run build            # type-check and build into web/dist/
cd ..
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Candidate pools you can pick in the app:
- the curated demo set (200 profiles)
- a random sample of up to 3,000 from `data/candidates.jsonl`
- the 50-profile official sample
- your own JSONL or JSON upload

### Streamlit app (lightweight; also what the Hugging Face Space runs)

```bash
streamlit run sandbox/app.py
```

Open <http://localhost:8501>.

### Docker (Streamlit app, as deployed on Hugging Face Spaces)

```bash
docker build -t talentbridge .
docker run --rm -p 7860:7860 talentbridge
```

Open <http://localhost:7860>. The container downloads the models on first use.

### Rebuild the curated demo set (optional)

```bash
python sandbox/make_demo_set.py
```

This needs `data/candidates.jsonl`.
It picks 60 strong fits, 20 likely-fake profiles, 30 disqualified profiles and 90 random ones, with seed 7.

---

## Command reference

| Purpose | Command |
|---|---|
| Install CPU PyTorch | `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu` |
| Install dependencies | `pip install -r requirements.txt` |
| Download models | `python download_models.py` |
| Pre-compute artifacts | `python precompute.py --candidates data/candidates.jsonl --out data/` |
| Rank | `HF_HUB_OFFLINE=1 python rank.py --candidates data/candidates.jsonl --jd data/job_description.docx --precomputed data/ --out submission.csv` |
| Validate output | `python validate_submission.py submission.csv` |
| Web demo | `./run_demo.sh` (or `./run_demo.sh --dev`) |
| API server only | `python -m uvicorn api.server:app --host 127.0.0.1 --port 8000` |
| Build web UI | `cd web && npm ci && npm run build` |
| Type-check web UI | `cd web && npm run typecheck` |
| Streamlit demo | `streamlit run sandbox/app.py` |
| Docker demo | `docker build -t talentbridge . && docker run --rm -p 7860:7860 talentbridge` |
| Rebuild demo set | `python sandbox/make_demo_set.py` |

`precompute.py` options:

| Option | Default | Meaning |
|---|---|---|
| `--candidates`, `-c` | required | path to `candidates.jsonl` |
| `--out`, `-o` | `./data` | output directory for artifacts |
| `--batch-size`, `-b` | `256` | bi-encoder batch size |
| `--limit`, `-n` | all | only process the first N profiles (testing) |
| `--skip-embeddings` | off | reuse existing `embeddings.npy` |
| `--skip-bm25` | off | reuse existing `bm25_index.pkl` |
| `--skip-features` | off | reuse existing `features.npy` |

`rank.py` options:

| Option | Default | Meaning |
|---|---|---|
| `--candidates`, `-c` | required | path to `candidates.jsonl` (profiles are read back for the NLI gate and the reasoning) |
| `--jd`, `-j` | required | job description, `.docx` or `.txt` |
| `--precomputed`, `-p` | `./data` | artifact directory from `precompute.py` |
| `--out`, `-o` | `./submission.csv` | output CSV |
| `--top-n-stage1` | `500` | how many candidates go to the cross-encoder |
| `--no-cross-encoder` | off | skip reranking (faster, lower quality) |
| `--debug` | off | print score details for the top 20 |

---

## Project structure

```
├── rank.py                   online ranking entry point (reproduce command)
├── precompute.py             offline artifact builder
├── download_models.py        fetches + smoke-tests the three Hugging Face models
├── validate_submission.py    checks submission.csv against the challenge format
├── config.py                 JD parameters, skill lists, weights, model ids
├── src/
│   ├── text_builder.py       candidate/JD text representation, BM25 tokenizer
│   ├── semantic_scorer.py    bi-encoder, cross-encoder, NLI gate
│   ├── bm25_scorer.py        BM25 index build + query
│   ├── structured_scorer.py  8-part JD-fit score
│   ├── honeypot_detector.py  7 fake-profile checks
│   ├── disqualifiers.py      9 multiplicative penalty rules
│   ├── fusion.py             stage 1 + stage 2 fusion
│   └── reasoning.py          fact-based reasoning strings
├── sandbox/
│   ├── pipeline.py           instrumented pipeline that reports each stage (used by both demos)
│   ├── app.py                Streamlit demo
│   └── make_demo_set.py      builds data/demo_candidates.jsonl
├── api/server.py             FastAPI backend for the web demo (streams stages as NDJSON)
├── web/                      React + TypeScript + Vite front end
├── run_demo.sh               one-command web demo launcher
├── data/                     JD, schema, sample + demo sets; artifacts and the full dataset are gitignored
├── Dockerfile                Hugging Face Spaces image (Streamlit)
├── solution.md               architecture and design decisions
├── submission.csv            final ranked output
└── submission_metadata.yaml  challenge metadata
```

---

## Scoring details

**Semantic (stage 2 weight 0.25).**
The `all-MiniLM-L6-v2` bi-encoder (384-d) embeds each candidate's text. Career descriptions come first in that
text, so the match reflects what candidates built, not the keywords they listed.

**Lexical (0.10).**
BM25 over the full profile. It gets a low weight because the JD warns against keyword matching. It mainly helps
with exact technology names.

**Structured JD-fit (0.45).** Eight sub-scores:

| Sub-score | Weight | What it measures |
|---|---|---|
| Skills trust | 30% | proficiency × duration × endorsements × assessment |
| Career quality | 25% | product vs consulting, tenure stability, company progression |
| Experience range | 10% | years of experience against the JD's 5–9 (ideal 6–8) |
| Behavioural availability | 12% | recency, response rate, interview completion, offer history |
| Location | 8% | Pune/Noida, then preferred metros, then willing to relocate |
| Technical credibility | 7% | GitHub activity and platform assessments |
| Notice period | 5% | under 30 days ideal, steep penalty above 60 |
| Platform demand | 3% | recruiter saves and search appearances |

**Cross-encoder (0.20).**
`ms-marco-MiniLM-L-6-v2` reads the JD and the candidate text together. It is applied to the top 500 only.

**Honeypot checks** (penalty: 3+ flags → 0.0, 2 → 0.05, 1 → 0.40):
1. Claimed experience far above the career history
2. Expert skills with zero duration
3. Many expert claims with zero endorsements
4. Assessment scores that contradict the claimed proficiency
5. A single job with a large experience gap
6. A skill duration longer than total experience
7. Implausible platform behaviour

**Disqualifiers** (multiplicative):

| Rule | Multiplier |
|---|---|
| Consulting-only career | 0.10 |
| Title does not match description | 0.20 |
| Wrong-domain expert | 0.30 |
| Long inactive with a low response rate | 0.35 |
| Low interview completion | 0.60 |
| Job hopping | 0.65 |
| Offer acceptance under 15% | 0.75 |
| Notice period over 90 days | 0.80 |
| Unverified identity | 0.92 |

**Challenge constraints:**

| Constraint | How it is met |
|---|---|
| CPU-only inference | no GPU code paths; CPU PyTorch build |
| No network during ranking | models loaded from local cache (`HF_HUB_OFFLINE=1`) |
| ≤ 5 min ranking | heavy work moved to `precompute.py`; the cross-encoder only sees 500 profiles |
| ≤ 16 GB RAM | embeddings are memory-mapped |

---

## Limitations

- **The JD is truncated to 1,500 characters before embedding**, in `src/text_builder.build_jd_text`.
  With the full official JD, the models mostly see its introduction. The web demo shows a warning when the JD is
  longer than this.
- **There are no ground-truth labels.** The weights were set by judgement and checked by inspection, not tuned
  against labelled relevance data. Treat the scores as a ranking aid, not a verdict.
- **The demo set is not an evaluation set.** `data/demo_candidates.jsonl` was selected using this same scorer,
  so results on it will look better than on a random sample. Use the "Full dataset sample" pool in the web
  demo for a realistic base rate.
- **Penalties are strong.** On the demo set, no flagged profile reached the final shortlist. That is intended for
  likely-fake profiles, but it can also hide a good candidate who has one noisy signal.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing artifact: data/embeddings.npy` | run `precompute.py` first, or point `--precomputed` at the right directory |
| `numpy.dtype size changed` or pyarrow errors in Streamlit | keep the pins: `numpy==1.26.4`, `pyarrow==16.1.0` (pyarrow ≥ 17 needs numpy 2) |
| Model download hangs or fails | re-run `python download_models.py`; set `HF_HOME` to use a different cache directory |
| `OfflineModeIsEnabled` / model not found when ranking | models are not cached yet; run `python download_models.py` without `HF_HUB_OFFLINE` |
| Port 8000 already in use | `python -m uvicorn api.server:app --port 8010`, or stop the other process |
| `npm ci` fails | install Node.js 20 or newer |
