"""
api/server.py — HTTP API behind the React UI (web/).

Endpoints
  GET  /api/config                       sources, JD presets, weights, stage list
  POST /api/upload                       multipart JSONL/JSON → upload token
  POST /api/run                          NDJSON stream: start → 9 × stage → done
  GET  /api/runs/{run_id}/candidates/{i} full record + model view for one candidate

In production the built UI (web/dist) is served from "/" by this same process:
  uvicorn api.server:app --host 0.0.0.0 --port 8000

All scoring comes from sandbox/pipeline.py → src/*; this file only moves data.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
import uuid
from collections import OrderedDict
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    JD_TEXT_SUMMARY,
    STAGE1_WEIGHTS,
    STAGE2_WEIGHTS,
    STRUCTURED_SUB_WEIGHTS,
    TIER1_SKILLS,
)
from sandbox.pipeline import STAGES, run_pipeline  # noqa: E402

DATA = ROOT / "data"
DIST = ROOT / "web" / "dist"
MAX_POOL = 3000
JD_WINDOW = 1500          # build_jd_text() truncates to this many characters
MAX_RUNS = 8              # finished runs kept in memory for drill-down
MAX_UPLOADS = 8

SOURCES = {
    "demo":   (DATA / "demo_candidates.jsonl", "Curated demo set",
               "Real profiles selected to exercise every stage: strong fits, likely-fake profiles, "
               "JD disqualifiers and a random background. Not an evaluation set."),
    "full":   (DATA / "candidates.jsonl", "Full dataset sample",
               "Uniform random sample from the complete 100,000-profile dataset. Realistic base rate."),
    "sample": (DATA / "sample_candidates.jsonl", "Official sample file",
               "The 50-profile sample distributed with the challenge."),
}

app = FastAPI(title="TalentBridge API", version="1.0.0")

_runs: "OrderedDict[str, dict]" = OrderedDict()
_uploads: "OrderedDict[str, list[dict]]" = OrderedDict()
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BI_ENCODER_MODEL)


@lru_cache(maxsize=1)
def cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)


def parse_lines(lines) -> tuple[list[dict], int]:
    out, bad = [], 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("profile"), dict):
            out.append(obj)
        else:
            bad += 1
    return out, bad


@lru_cache(maxsize=4)
def load_file(path: str, mtime: float) -> tuple[list[dict], int]:
    with open(path, encoding="utf-8") as f:
        return parse_lines(f)


@lru_cache(maxsize=16)
def sample_file(path: str, mtime: float, n: int, seed: int) -> tuple[list[dict], int]:
    """Reservoir-sample n raw lines; only the chosen lines are JSON-parsed."""
    rng = random.Random(seed)
    res: list[str] = []
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i < n:
                res.append(ln)
            else:
                j = rng.randint(0, i)
                if j < n:
                    res[j] = ln
    return parse_lines(res)


@lru_cache(maxsize=4)
def count_lines(path: str, mtime: float) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


@lru_cache(maxsize=2)
def official_jd(path: str, mtime: float) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(path).paragraphs if p.text.strip())


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def arr(x, nd: int = 4) -> list:
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def summary(i: int, c: dict) -> dict:
    p, s = c.get("profile", {}) or {}, c.get("redrob_signals", {}) or {}
    skills = [
        sk.get("name") for sk in c.get("skills", []) or []
        if (sk.get("name") or "").lower().strip() in TIER1_SKILLS
        and sk.get("proficiency") in ("advanced", "expert")
    ]
    return {
        "i": i,
        "id": c.get("candidate_id") or f"CAND_{i:07d}",
        "name": p.get("anonymized_name") or "",
        "title": p.get("current_title") or "",
        "company": p.get("current_company") or "",
        "location": p.get("location") or "",
        "yoe": float(p.get("years_of_experience") or 0),
        "notice": s.get("notice_period_days"),
        "open": bool(s.get("open_to_work_flag")),
        "skills": skills[:6],
    }


def stage_payload(key: str, data: dict, cands: list[dict]) -> dict:
    if key == "ingest":
        return {"missing": data["missing"], "dupes": data["dupes"], "yoe": arr(data["yoe"], 1),
                "candidates": [summary(i, c) for i, c in enumerate(cands)]}
    if key == "text":
        return {k: data[k] for k in ("avg_chars", "avg_tokens", "jd_text", "jd_terms", "example_text")}
    if key == "semantic":
        return {"dim": data["dim"], "sem": arr(data["sem"])}
    if key == "lexical":
        return {"lex": arr(data["lex"]),
                "terms": [{"term": t, "idf": round(float(v), 3)} for v, t in data["term_idf"]]}
    if key == "structured":
        return {"fit": arr(data["struct"]),
                "subs": [{k: round(float(v), 3) for k, v in s.items()} for s in data["sub_scores"]]}
    if key == "trust":
        return {"hp": arr(data["hp"], 3), "dq": arr(data["dq"], 3),
                "hp_flags": data["hp_flags"], "dq_flags": data["dq_flags"]}
    if key == "stage1":
        return {"k": data["k"], "s1": arr(data["s1"], 5), "top": [int(i) for i in data["top_idx"]]}
    if key == "rerank":
        return {"enabled": data["enabled"], "ce": arr(data["ce"])}
    if key == "final":
        return {"rows": [{
            "rank": r["rank"], "i": r["idx"], "id": r["id"], "score": round(r["score"], 4),
            "s1_rank": r["stage1_rank"], "penalty": round(r["penalty"], 3),
            "contrib": {k: round(v, 4) for k, v in r["contrib"].items()},
            "reasoning": r["reasoning"],
        } for r in data["rows"]]}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    sources = []
    for sid, (path, label, desc) in SOURCES.items():
        if not path.exists():
            continue
        n = count_lines(str(path), mtime(path))
        sources.append({"id": sid, "label": label, "description": desc, "count": n,
                        "sampled": sid == "full"})
    presets = [{"id": "condensed", "label": "Condensed", "text": JD_TEXT_SUMMARY.strip()}]
    jd_path = DATA / "job_description.docx"
    if jd_path.exists():
        presets.append({"id": "official", "label": "Official JD", "text": official_jd(str(jd_path), mtime(jd_path))})
    return {
        "sources": sources,
        "jd_presets": presets,
        "jd_window": JD_WINDOW,
        "max_pool": MAX_POOL,
        "weights": {"stage1": STAGE1_WEIGHTS, "stage2": STAGE2_WEIGHTS, "structured": STRUCTURED_SUB_WEIGHTS},
        "models": {"bi_encoder": BI_ENCODER_MODEL, "cross_encoder": CROSS_ENCODER_MODEL},
        "stages": [{"key": k, "title": t, "description": d} for k, t, d in STAGES],
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = (await file.read()).decode("utf-8", errors="replace")
    if (file.filename or "").endswith(".json"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON: {e}") from e
        if not isinstance(data, list):
            raise HTTPException(400, "Expected a JSON array of candidate objects.")
        cands, bad = parse_lines(json.dumps(d) for d in data)
    else:
        cands, bad = parse_lines(raw.splitlines())
    if not cands:
        raise HTTPException(400, "No valid candidate records found (each needs a 'profile' object).")
    truncated = len(cands) > MAX_POOL
    cands = cands[:MAX_POOL]
    token = uuid.uuid4().hex[:12]
    with _lock:
        _uploads[token] = cands
        while len(_uploads) > MAX_UPLOADS:
            _uploads.popitem(last=False)
    return {"token": token, "count": len(cands), "skipped": bad, "truncated": truncated,
            "filename": file.filename}


class RunRequest(BaseModel):
    source: str
    sample_n: int = Field(500, ge=10, le=MAX_POOL)
    seed: int = 42
    upload_token: str | None = None
    jd_text: str = Field(..., min_length=20)
    shortlist_k: int = Field(50, ge=5, le=500)
    final_n: int = Field(25, ge=5, le=100)
    use_cross_encoder: bool = True


def resolve_candidates(req: RunRequest) -> list[dict]:
    if req.source == "upload":
        cands = _uploads.get(req.upload_token or "")
        if cands is None:
            raise HTTPException(404, "Upload expired — please upload the file again.")
        return cands
    if req.source not in SOURCES or not SOURCES[req.source][0].exists():
        raise HTTPException(400, f"Unknown or unavailable source '{req.source}'.")
    path = SOURCES[req.source][0]
    if req.source == "full":
        return sample_file(str(path), mtime(path), req.sample_n, req.seed)[0]
    return load_file(str(path), mtime(path))[0]


@app.post("/api/run")
def run(req: RunRequest):
    cands = resolve_candidates(req)
    run_id = uuid.uuid4().hex[:12]

    def stream():
        def line(obj) -> bytes:
            return (json.dumps(obj, separators=(",", ":")) + "\n").encode()

        yield line({"type": "start", "run_id": run_id, "n": len(cands)})
        try:
            t0 = time.perf_counter()
            gen = run_pipeline(
                cands, req.jd_text, bi_encoder=bi_encoder(),
                cross_encoder=cross_encoder() if req.use_cross_encoder else None,
                shortlist_k=req.shortlist_k, final_n=req.final_n, today=date.today(),
            )
            trace = None
            while True:
                try:
                    ev = next(gen)
                except StopIteration as stop:
                    trace = stop.value
                    break
                yield line({"type": "stage", "key": ev.key, "seconds": round(ev.seconds, 4),
                            "headline": ev.headline, "data": stage_payload(ev.key, ev.data, cands)})
            with _lock:
                _runs[run_id] = {"cands": cands, "trace": trace}
                while len(_runs) > MAX_RUNS:
                    _runs.popitem(last=False)
            yield line({"type": "done", "run_id": run_id, "wall_seconds": round(time.perf_counter() - t0, 3),
                        "today": trace["today"]})
        except Exception as e:  # surface pipeline errors to the UI instead of a dropped stream
            yield line({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/candidates/{i}")
def candidate(run_id: str, i: int):
    r = _runs.get(run_id)
    if r is None:
        raise HTTPException(404, "Run not found (the server keeps only recent runs).")
    cands, tr = r["cands"], r["trace"]
    if not 0 <= i < len(cands):
        raise HTTPException(404, "Candidate index out of range.")
    c = cands[i]
    sig = c.get("redrob_signals", {}) or {}
    assess = sig.get("skill_assessment_scores") or {}
    final = next((row for row in tr["final"] if row["idx"] == i), None)
    return {
        "i": i,
        "id": tr["ids"][i],
        "profile": c.get("profile", {}),
        "career": [{k: j.get(k) for k in ("title", "company", "industry", "company_size", "start_date",
                                          "end_date", "duration_months", "is_current", "description")}
                   for j in c.get("career_history", []) or []],
        "education": c.get("education", []) or [],
        "skills": [{"name": s.get("name"), "proficiency": s.get("proficiency"),
                    "months": s.get("duration_months"), "endorsements": s.get("endorsements"),
                    "assessed": assess.get(s.get("name")),
                    "tier1": (s.get("name") or "").lower().strip() in TIER1_SKILLS}
                   for s in c.get("skills", []) or []],
        "signals": {k: sig.get(k) for k in (
            "recruiter_response_rate", "interview_completion_rate", "offer_acceptance_rate",
            "notice_period_days", "last_active_date", "preferred_work_mode", "willing_to_relocate",
            "github_activity_score", "open_to_work_flag", "avg_response_time_hours",
            "verified_email", "verified_phone", "profile_completeness_score")},
        "model_text": tr["texts"][i],
        "matched_terms": sorted(set(tr["jd_tokens"]) & set(tr["tokens"][i])),
        "reasoning": final["reasoning"] if final else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static UI (production build)
# ─────────────────────────────────────────────────────────────────────────────
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
