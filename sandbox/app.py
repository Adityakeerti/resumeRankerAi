"""
sandbox/app.py — TalentBridge interactive demo (Streamlit).

An end-to-end product view of the ranking pipeline: pick a role and a talent
pool, run the pipeline, and watch every stage execute live — then inspect the
shortlist, each stage's intermediate outputs, any single candidate's score
breakdown, the trust/risk exclusions, and export the submission CSV.

Run locally (from the repo root):   streamlit run sandbox/app.py
Data it looks for in ./data/:       demo_candidates.jsonl, candidates.jsonl,
                                    sample_candidates.jsonl, job_description.docx
"""

from __future__ import annotations

import csv
import html
import io
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
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
from sandbox.pipeline import STAGES, STAGE_KEYS, run_pipeline, fmt_weights  # noqa: E402

DATA = ROOT / "data"
MAX_POOL = 3000          # bi-encoder on CPU ≈ 10 ms/profile → keeps a live run under ~30 s

# Categorical slots 1–4 of the validated reference palette (adjacent-pair CVD-safe, light mode).
COMP_COLORS = {
    "Semantic":      "#2a78d6",
    "Lexical":       "#eb6834",
    "JD-fit":        "#1baf7a",
    "Cross-encoder": "#eda100",
}
NEUTRAL = "#a3a9b5"

st.set_page_config(
    page_title="TalentBridge — Candidate Ranking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# Styling
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
:root{--tb-ink:#0f172a;--tb-muted:#5b6475;--tb-line:#e3e6ee;--tb-card:#ffffff;--tb-soft:#f1f4fa;
--tb-accent:#2a78d6;--tb-ok:#137a2f;--tb-warn:#9a5b00;--tb-bad:#b42318;}
.block-container{padding-top:1.4rem;max-width:1320px}
h1,h2,h3{letter-spacing:-.01em}
.tb-hero{background:linear-gradient(130deg,#0d2547 0%,#174a8b 55%,#2a78d6 100%);color:#fff;border-radius:18px;padding:26px 30px;margin-bottom:6px}
.tb-brand{font-weight:700;letter-spacing:.14em;text-transform:uppercase;font-size:.78rem;opacity:.85}
.tb-hero h1{color:#fff;font-size:2.05rem;line-height:1.2;margin:.35rem 0 .45rem;padding:0}
.tb-hero p{opacity:.92;margin:0;max-width:820px;font-size:1.02rem}
.tb-pill{display:inline-block;border:1px solid rgba(255,255,255,.38);border-radius:999px;padding:3px 11px;font-size:.78rem;margin:14px 6px 0 0}
.tb-sec{display:flex;align-items:center;gap:10px;margin:18px 0 6px}
.tb-sec .n{width:26px;height:26px;border-radius:50%;background:var(--tb-accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem}
.tb-sec .t{font-weight:700;font-size:1.12rem;color:var(--tb-ink)}
.tb-sec .d{color:var(--tb-muted);font-size:.9rem}
/* stepper */
.tb-steps{display:grid;grid-template-columns:repeat(9,1fr);background:var(--tb-card);border:1px solid var(--tb-line);border-radius:14px;padding:18px 8px 14px;margin:8px 0 10px}
.tb-step{position:relative;text-align:center;padding:0 4px}
.tb-step::before{content:"";position:absolute;top:17px;left:-50%;width:100%;height:3px;background:var(--tb-line);z-index:0}
.tb-step:first-child::before{display:none}
.tb-step.done::before,.tb-step.active::before{background:var(--tb-accent)}
.tb-dot{width:36px;height:36px;border-radius:50%;margin:0 auto 6px;display:flex;align-items:center;justify-content:center;background:#fff;border:2px solid var(--tb-line);color:var(--tb-muted);font-weight:700;position:relative;z-index:1;font-size:.9rem}
.tb-step.done .tb-dot{background:var(--tb-accent);border-color:var(--tb-accent);color:#fff}
.tb-step.active .tb-dot{border-color:var(--tb-accent);color:var(--tb-accent);animation:tbpulse 1s infinite}
.tb-step-t{font-weight:650;font-size:.86rem;color:var(--tb-ink)}
.tb-step-s{font-size:.74rem;color:var(--tb-muted);line-height:1.25}
.tb-step.done .tb-step-s b{color:var(--tb-accent)}
@keyframes tbpulse{0%{box-shadow:0 0 0 0 rgba(42,120,214,.45)}100%{box-shadow:0 0 0 11px rgba(42,120,214,0)}}
/* KPI funnel */
.tb-kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:6px 0 14px}
.tb-kpi{background:var(--tb-card);border:1px solid var(--tb-line);border-radius:12px;padding:12px 14px}
.tb-kpi .v{font-size:1.55rem;font-weight:750;color:var(--tb-ink);line-height:1.15}
.tb-kpi .l{color:var(--tb-muted);font-size:.78rem;margin-top:2px}
.tb-kpi.bad .v{color:var(--tb-bad)} .tb-kpi.warn .v{color:var(--tb-warn)} .tb-kpi.acc .v{color:var(--tb-accent)}
/* candidate cards */
.tb-card{background:var(--tb-card);border:1px solid var(--tb-line);border-radius:14px;padding:14px 16px;margin-bottom:12px}
.tb-card-top{display:flex;gap:14px;align-items:flex-start}
.tb-rank{min-width:46px;height:46px;border-radius:12px;background:var(--tb-soft);color:var(--tb-accent);font-weight:800;font-size:1.05rem;display:flex;align-items:center;justify-content:center}
.tb-who{flex:1;min-width:0}
.tb-name{font-weight:700;font-size:1.02rem;color:var(--tb-ink)}
.tb-id{font-weight:500;font-size:.75rem;color:var(--tb-muted);margin-left:6px;font-family:ui-monospace,monospace}
.tb-role{color:var(--tb-ink);font-size:.9rem}
.tb-meta{color:var(--tb-muted);font-size:.8rem;margin-top:2px}
.tb-score{text-align:right;min-width:92px}
.tb-score .v{font-size:1.45rem;font-weight:800;color:var(--tb-ink);line-height:1}
.tb-score .l{font-size:.7rem;color:var(--tb-muted)}
.tb-move{font-size:.72rem;font-weight:650;margin-top:4px}
.tb-move.up{color:var(--tb-ok)} .tb-move.down{color:var(--tb-bad)} .tb-move.same{color:var(--tb-muted)}
.tb-bar{display:flex;gap:2px;height:10px;border-radius:5px;background:var(--tb-soft);margin:12px 0 8px;overflow:hidden}
.tb-bar span{display:block;height:100%}
.tb-bar span:first-child{border-radius:5px 0 0 5px}
.tb-why{font-size:.86rem;color:#2a3242;line-height:1.45}
.tb-chips{margin-top:8px}
.tb-chip{display:inline-block;font-size:.72rem;padding:2px 9px;border-radius:999px;background:var(--tb-soft);color:#28406a;margin:0 5px 5px 0;border:1px solid #dbe4f3}
.tb-chip.warn{background:#fff6e6;color:var(--tb-warn);border-color:#f3dcae}
.tb-chip.bad{background:#fdecec;color:var(--tb-bad);border-color:#f5c6c3}
.tb-legend{font-size:.8rem;color:var(--tb-muted);margin:2px 0 10px}
.tb-legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 5px 0 12px;vertical-align:-1px}
/* walkthrough */
.tb-stage-h{display:flex;align-items:baseline;gap:10px;margin-top:6px}
.tb-stage-h .k{font-size:.75rem;font-weight:700;color:var(--tb-accent);letter-spacing:.08em;text-transform:uppercase}
.tb-stage-h .t{font-size:1.15rem;font-weight:700;color:var(--tb-ink)}
.tb-stage-h .s{font-size:.8rem;color:var(--tb-muted);margin-left:auto}
.tb-explain{font-size:.88rem;color:#3a4356;line-height:1.5;margin:4px 0 10px}
.tb-mono{font-family:ui-monospace,monospace;font-size:.78rem;background:var(--tb-soft);border:1px solid var(--tb-line);border-radius:10px;padding:10px 12px;white-space:pre-wrap;color:#253047;max-height:220px;overflow:auto}
@media (max-width: 900px){
 .tb-steps{grid-template-columns:repeat(3,1fr);row-gap:14px}
 .tb-step::before{display:none}
 .tb-kpis{grid-template-columns:repeat(2,1fr)}
 .tb-hero h1{font-size:1.5rem}
}
</style>
""",
    unsafe_allow_html=True,
)

esc = html.escape


# ═════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading bi-encoder (all-MiniLM-L6-v2) …")
def get_bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BI_ENCODER_MODEL)


@st.cache_resource(show_spinner="Loading cross-encoder (ms-marco-MiniLM-L-6-v2) …")
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)


def _parse_lines(lines: list[str]) -> tuple[list[dict], int]:
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
        if isinstance(obj, dict) and "profile" in obj:
            out.append(obj)
        else:
            bad += 1
    return out, bad


@st.cache_data(show_spinner="Reading candidates …")
def load_jsonl(path: str, mtime: float, limit: int | None = None) -> tuple[list[dict], int]:
    if path.endswith(".json"):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data[:limit] if limit else data, 0
    with open(path, encoding="utf-8") as f:
        lines = []
        for i, ln in enumerate(f):
            if limit and i >= limit:
                break
            lines.append(ln)
    return _parse_lines(lines)


@st.cache_data(show_spinner="Sampling from the full 100K dataset …")
def sample_jsonl(path: str, mtime: float, n: int, seed: int) -> tuple[list[dict], int]:
    """Reservoir-sample n raw lines (parse only the chosen ones — the full file is ~490 MB)."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i < n:
                reservoir.append(ln)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = ln
    return _parse_lines(reservoir)


@st.cache_data
def load_docx_jd(path: str, mtime: float) -> str:
    import docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Small render helpers
# ═════════════════════════════════════════════════════════════════════════════
def section(n: str, title: str, desc: str = ""):
    st.markdown(
        f'<div class="tb-sec"><div class="n">{n}</div><div class="t">{esc(title)}</div>'
        f'<div class="d">{esc(desc)}</div></div>',
        unsafe_allow_html=True,
    )


def fmt_t(sec: float) -> str:
    return f"{sec * 1000:.0f} ms" if sec < 1 else f"{sec:.1f} s"


def stepper_html(states: dict, times: dict) -> str:
    cells = []
    for i, (key, title, desc) in enumerate(STAGES, start=1):
        stt = states.get(key, "pending")
        dot = "✓" if stt == "done" else str(i)
        if stt == "done":
            sub = f"<b>{fmt_t(times[key])}</b>"
        elif stt == "active":
            sub = "running…"
        else:
            sub = esc(desc)
        cells.append(
            f'<div class="tb-step {stt}"><div class="tb-dot">{dot}</div>'
            f'<div class="tb-step-t">{esc(title)}</div><div class="tb-step-s">{sub}</div></div>'
        )
    return '<div class="tb-steps">' + "".join(cells) + "</div>"


def contrib_bar_html(contrib: dict) -> str:
    segs = "".join(
        f'<span style="width:{v * 100:.1f}%;background:{COMP_COLORS[k]}" title="{k}: {v:.3f}"></span>'
        for k, v in contrib.items() if v > 0
    )
    return f'<div class="tb-bar">{segs}</div>'


def legend_html() -> str:
    return '<div class="tb-legend">Score build-up (share of a perfect 1.0, before penalties):' + "".join(
        f'<i style="background:{c}"></i>{k} ({STAGE2_WEIGHTS[w]:.0%})'
        for (k, c), w in zip(COMP_COLORS.items(), ["semantic", "lexical", "structured", "cross_encoder"])
    ) + "</div>"


def matched_skills(c: dict, k: int = 6) -> list[str]:
    out = []
    for s in c.get("skills", []):
        name = (s.get("name") or "").strip()
        if name.lower() in TIER1_SKILLS and s.get("proficiency") in ("advanced", "expert"):
            out.append(name)
    return out[:k]


def short_flag(f: str) -> str:
    return f.split(":")[0].replace("_", " ")


def chart_cfg(ch: alt.Chart) -> alt.Chart:
    return ch.configure_view(strokeWidth=0).configure_axis(
        gridColor="#eceff4", domainColor="#cfd5df", labelColor="#5b6475", titleColor="#5b6475",
        labelFontSize=11, titleFontSize=11, titleFontWeight="normal",
    ).configure_legend(labelColor="#3a4356", titleColor="#5b6475", orient="top")


def hist(values, title: str, color: str = "#2a78d6", bins: int = 30, height: int = 170):
    df = pd.DataFrame({"v": np.asarray(values, dtype=float)})
    ch = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=bins), title=title),
        y=alt.Y("count():Q", title="profiles"),
        tooltip=[alt.Tooltip("v:Q", bin=alt.Bin(maxbins=bins), title=title), alt.Tooltip("count():Q", title="profiles")],
    ).properties(height=height)
    st.altair_chart(chart_cfg(ch), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar — about / honest caveats
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### About TalentBridge")
    st.markdown(
        "A CPU-only, offline-capable ranking pipeline that turns a job description and a "
        "pool of candidate profiles into a ranked, explained shortlist."
    )
    st.markdown("**Models**")
    st.caption(f"Bi-encoder: `{BI_ENCODER_MODEL}`  \nCross-encoder: `{CROSS_ENCODER_MODEL}`")
    st.markdown("**Fusion weights**")
    st.caption(f"Stage 1: {fmt_weights(STAGE1_WEIGHTS)}  \nStage 2: {fmt_weights(STAGE2_WEIGHTS)}")
    st.markdown("**Caveats**")
    st.caption(
        "• The curated demo set was *selected using this scorer* (strong fits + honeypots + "
        "disqualified + random) so every stage has something to show. It is a demo prop, "
        "not an evaluation set.  \n"
        "• Displayed scores are min-max rescaled to [0.20, 0.99] within the output list — "
        "they rank, they are not probabilities.  \n"
        "• Full 100K runs use precomputed embeddings via `precompute.py` + `rank.py`."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Hero
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="tb-hero">
  <div class="tb-brand">🎯 TalentBridge</div>
  <h1>From 100,000 profiles to a shortlist you can defend.</h1>
  <p>Hybrid semantic + keyword retrieval, structured JD-fit scoring, trust checks for fake or
  inconsistent profiles, and cross-encoder reranking, with a written reason for every rank.</p>
  <span class="tb-pill">9-stage pipeline</span><span class="tb-pill">CPU-only</span>
  <span class="tb-pill">Offline models</span><span class="tb-pill">Explainable by design</span>
</div>
""",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1 + 2 — Inputs
# ═════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.15, 1], gap="large")

with left:
    section("1", "Define the role", "what are we hiring for?")
    with st.container(border=True):
        jd_docx = DATA / "job_description.docx"
        jd_opts = ["Condensed JD (signal-dense summary)"]
        if jd_docx.exists():
            jd_opts.append("Official JD (job_description.docx)")
        jd_opts.append("Write / paste my own")
        jd_choice = st.radio("Job description source", jd_opts, horizontal=True, label_visibility="collapsed")
        if jd_choice.startswith("Official"):
            default_jd = load_docx_jd(str(jd_docx), mtime(jd_docx))
        elif jd_choice.startswith("Condensed"):
            default_jd = JD_TEXT_SUMMARY.strip()
        else:
            default_jd = ""
        jd_input = st.text_area(
            "Job description", value=default_jd, height=210, key=f"jd_{jd_choice}",
            placeholder="Paste the job description here …", label_visibility="collapsed",
        )
        if len(jd_input) > 1500:
            st.caption(
                f"⚠️ {len(jd_input):,} characters — the embedding models only read the first 1,500 "
                "(≈ the opening paragraphs). The condensed JD keeps the requirements inside that window."
            )
        else:
            st.caption(f"{len(jd_input):,} characters · all of it is read by the models")

with right:
    section("2", "Choose the talent pool", "who are we screening?")
    with st.container(border=True):
        sources = {}
        demo_p, full_p, samp_p = DATA / "demo_candidates.jsonl", DATA / "candidates.jsonl", DATA / "sample_candidates.jsonl"
        if demo_p.exists():
            sources["Curated demo set · 200 real profiles"] = "demo"
        if full_p.exists():
            sources["Random sample · full 100K dataset"] = "full"
        if samp_p.exists():
            sources["Official sample file · 50 profiles"] = "sample"
        sources["Upload my own JSONL"] = "upload"
        src_label = st.radio("Pool", list(sources), label_visibility="collapsed")
        src = sources[src_label]

        candidates: list[dict] = []
        bad_lines = 0
        if src == "demo":
            candidates, bad_lines = load_jsonl(str(demo_p), mtime(demo_p))
            st.caption("Mix of strong fits, honeypots, disqualified and random profiles from the real dataset.")
        elif src == "sample":
            candidates, bad_lines = load_jsonl(str(samp_p), mtime(samp_p))
        elif src == "full":
            c1, c2 = st.columns([3, 1])
            n_pool = c1.slider("Profiles to sample", 100, MAX_POOL, 500, step=100)
            seed = c2.number_input("Seed", 0, 9999, 42)
            candidates, bad_lines = sample_jsonl(str(full_p), mtime(full_p), n_pool, int(seed))
            st.caption("Uniform random sample: expect very few strong fits (the realistic base rate).")
        else:
            up = st.file_uploader("candidates.jsonl", type=["jsonl", "json"], label_visibility="collapsed")
            if up is not None:
                raw = up.getvalue().decode("utf-8", errors="replace")
                if up.name.endswith(".json"):
                    try:
                        data = json.loads(raw)
                        candidates = [d for d in data if isinstance(d, dict)][:MAX_POOL]
                    except json.JSONDecodeError:
                        st.error("Could not parse JSON file.")
                else:
                    candidates, bad_lines = _parse_lines(raw.splitlines()[:MAX_POOL])
                st.caption(f"Capped at {MAX_POOL:,} profiles for an interactive run.")

        if candidates:
            yoes = [float(c.get("profile", {}).get("years_of_experience") or 0) for c in candidates]
            m1, m2, m3 = st.columns(3)
            m1.metric("Profiles loaded", f"{len(candidates):,}")
            m2.metric("Median experience", f"{np.median(yoes):.1f} yrs")
            m3.metric("Malformed rows skipped", bad_lines)

with st.expander("⚙️  Pipeline settings", expanded=False):
    s1, s2, s3, s4 = st.columns(4)
    shortlist_k = s1.slider("Stage-1 shortlist size (sent to cross-encoder)", 10, 500, 50, step=10)
    final_n = s2.slider("Final ranked list size", 5, 100, 25, step=5)
    use_ce = s3.toggle("Cross-encoder rerank", value=True)
    pace = s4.toggle("Presentation pacing", value=True,
                     help="Adds a short pause between stages so an audience can follow along. "
                          "Reported stage timings are real compute time and exclude the pause.")

run = st.button(
    "▶  Run ranking pipeline", type="primary", use_container_width=True,
    disabled=not (candidates and jd_input.strip()),
)

# ═════════════════════════════════════════════════════════════════════════════
# 3 — Live run
# ═════════════════════════════════════════════════════════════════════════════
if run:
    st.session_state.pop("result", None)
    section("3", "Pipeline running", "each stage lights up as it completes")
    bi = get_bi_encoder()
    ce_model = get_cross_encoder() if use_ce else None

    stepper = st.empty()
    log = st.container()
    states = {k: "pending" for k in STAGE_KEYS}
    times: dict = {}
    events: dict = {}
    today = date.today()
    gen = run_pipeline(candidates, jd_input, bi_encoder=bi, cross_encoder=ce_model,
                       shortlist_k=shortlist_k, final_n=final_n, today=today)
    t_all = time.perf_counter()
    for key, title, desc in STAGES:
        states[key] = "active"
        stepper.markdown(stepper_html(states, times), unsafe_allow_html=True)
        status = log.status(f"**{title}** — {desc} …", state="running", expanded=False)
        ev = next(gen)
        if pace:
            time.sleep(0.45)
        states[key] = "done"
        times[key] = ev.seconds
        events[key] = ev
        status.update(label=f"**{title}** · {ev.headline}  ·  _{fmt_t(ev.seconds)}_", state="complete")
    try:
        next(gen)
    except StopIteration as stop:
        trace = stop.value
    stepper.markdown(stepper_html(states, times), unsafe_allow_html=True)
    st.session_state.result = {
        "trace": trace, "events": events, "times": times, "candidates": candidates,
        "jd": jd_input, "source": src_label, "k": shortlist_k, "n": final_n, "ce": use_ce,
        "wall": time.perf_counter() - t_all,
    }
    st.toast(f"Ranked {len(candidates):,} profiles in {sum(times.values()):.1f} s of compute", icon="✅")


# ═════════════════════════════════════════════════════════════════════════════
# 4 — Results
# ═════════════════════════════════════════════════════════════════════════════
res = st.session_state.get("result")
if not res:
    if not run:
        st.info("Pick a role and a talent pool above, then press **Run ranking pipeline** to watch it work.")
    st.stop()

tr, ev, cands = res["trace"], res["events"], res["candidates"]
N, rows = tr["N"], tr["final"]
ids = tr["ids"]
s1_rank = np.empty(N, dtype=int)
s1_rank[np.argsort(-tr["s1"])] = np.arange(1, N + 1)
final_rank = {r["idx"]: r["rank"] for r in rows}
short_set = set(int(i) for i in tr["top_idx"])
ce_by_idx = {int(i): float(tr["ce"][p]) for p, i in enumerate(tr["top_idx"])}

if not run:
    section("3", "Pipeline run", f"{res['source']} · last run")
    st.markdown(stepper_html({k: "done" for k in STAGE_KEYS}, res["times"]), unsafe_allow_html=True)

n_hp = int((tr["hp"] < 1).sum())
n_dq = int((tr["dq"] < 1).sum())
st.markdown(
    '<div class="tb-kpis">'
    f'<div class="tb-kpi"><div class="v">{N:,}</div><div class="l">profiles screened</div></div>'
    f'<div class="tb-kpi bad"><div class="v">{n_hp:,}</div><div class="l">flagged as likely fake / inconsistent</div></div>'
    f'<div class="tb-kpi warn"><div class="v">{n_dq:,}</div><div class="l">hit a JD disqualifier (penalised)</div></div>'
    f'<div class="tb-kpi"><div class="v">{len(tr["top_idx"]):,}</div><div class="l">shortlisted for reranking</div></div>'
    f'<div class="tb-kpi acc"><div class="v">{len(rows)}</div><div class="l">ranked &amp; explained</div></div>'
    f'<div class="tb-kpi"><div class="v">{sum(res["times"].values()):.1f}s</div><div class="l">compute time (CPU)</div></div>'
    "</div>",
    unsafe_allow_html=True,
)

section("4", "Results", "explore the shortlist and every intermediate step")
tab_short, tab_walk, tab_cand, tab_risk, tab_exp = st.tabs(
    ["🏆 Ranked shortlist", "🔬 Pipeline walkthrough", "👤 Candidate deep-dive", "🛡️ Risk & exclusions", "📤 Export"]
)


def card_html(r: dict) -> str:
    c = cands[r["idx"]]
    p, sig = c.get("profile", {}), c.get("redrob_signals", {})
    mv = r["stage1_rank"] - r["rank"]
    if not res["ce"]:
        move = ""
    elif mv > 0:
        move = f'<div class="tb-move up">▲ {mv} after rerank</div>'
    elif mv < 0:
        move = f'<div class="tb-move down">▼ {-mv} after rerank</div>'
    else:
        move = '<div class="tb-move same">— held position</div>'
    meta = [f"📍 {esc(p.get('location') or '—')}", f"{float(p.get('years_of_experience') or 0):.1f} yrs exp"]
    if sig.get("notice_period_days") is not None:
        meta.append(f"{sig['notice_period_days']}-day notice")
    if sig.get("open_to_work_flag"):
        meta.append("open to work")
    chips = "".join(f'<span class="tb-chip">{esc(s)}</span>' for s in matched_skills(c))
    chips += "".join(f'<span class="tb-chip bad">⚠ {esc(short_flag(f))}</span>' for f in tr["hp_flags"][r["idx"]])
    chips += "".join(f'<span class="tb-chip warn">⚑ {esc(short_flag(f))}</span>' for f in tr["dq_flags"][r["idx"]])
    return (
        '<div class="tb-card"><div class="tb-card-top">'
        f'<div class="tb-rank">#{r["rank"]}</div>'
        f'<div class="tb-who"><div class="tb-name">{esc(p.get("anonymized_name") or r["id"])}'
        f'<span class="tb-id">{esc(r["id"])}</span></div>'
        f'<div class="tb-role">{esc(p.get("current_title") or "—")} · {esc(p.get("current_company") or "—")}</div>'
        f'<div class="tb-meta">{" · ".join(meta)}</div></div>'
        f'<div class="tb-score"><div class="v">{r["score"]:.2f}</div><div class="l">rank score</div>{move}</div>'
        "</div>"
        f"{contrib_bar_html(r['contrib'])}"
        f'<div class="tb-why">{esc(r["reasoning"])}</div>'
        f'<div class="tb-chips">{chips}</div></div>'
    )


# ── Shortlist ─────────────────────────────────────────────────────────────────
with tab_short:
    st.markdown(legend_html(), unsafe_allow_html=True)
    n_cards = min(10, len(rows))
    cc = st.columns(2, gap="medium")
    for i, r in enumerate(rows[:n_cards]):
        cc[i % 2].markdown(card_html(r), unsafe_allow_html=True)

    st.markdown(f"##### Full ranked list ({len(rows)})")
    table = pd.DataFrame([{
        "Rank": r["rank"],
        "Candidate": r["id"],
        "Title": cands[r["idx"]]["profile"].get("current_title"),
        "Company": cands[r["idx"]]["profile"].get("current_company"),
        "YoE": cands[r["idx"]]["profile"].get("years_of_experience"),
        "Score": r["score"],
        "Stage-1 rank": r["stage1_rank"],
        "Penalty ×": round(r["penalty"], 2),
        "Reasoning": r["reasoning"],
    } for r in rows])
    st.dataframe(
        table, hide_index=True, use_container_width=True, height=min(38 * len(rows) + 40, 560),
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.3f"),
            "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        },
    )

# ── Walkthrough ───────────────────────────────────────────────────────────────
STAGE_DOC = {
    "ingest": ("Every profile is parsed and checked for the fields the scorers rely on.",
               "Missing career history or platform signals silently drag scores down, so we surface it up-front."),
    "text": ("Each profile is flattened into one text blob (career descriptions first, then headline, summary, "
             "strong skills, field of study) and a token list for keyword search.",
             "Career descriptions are the most evidence-dense part of a profile; skill lists are easy to pad."),
    "semantic": ("A bi-encoder embeds the JD and every profile independently; cosine similarity measures topical fit.",
                 "Catches paraphrases (“vector search” ≈ “ANN retrieval”) that keyword match misses. Cheap enough for 100K."),
    "lexical": ("BM25 scores exact term overlap between JD and profile, weighted by how rare each term is in this pool.",
                "Rare, specific terms (faiss, ndcg, qdrant) are strong evidence; embeddings tend to blur them."),
    "structured": ("Eight hand-designed signals derived from the JD's explicit requirements, combined with fixed weights.",
                   "Most of the JD is constraints (years, product vs consulting, location, notice) that text similarity cannot see."),
    "trust": ("Honeypot checks catch internally inconsistent profiles; disqualifiers encode the JD's explicit exclusions.",
              "Both are multiplicative penalties rather than hard filters, so a single noisy signal can't silently delete a strong candidate."),
    "stage1": ("Semantic, lexical and JD-fit scores are fused, penalties applied, and the top-K kept.",
               "The cross-encoder is ~100× more expensive per profile, so it only sees a shortlist."),
    "rerank": ("A cross-encoder reads the JD and each shortlisted profile together and outputs a relevance probability.",
               "Joint attention over both texts is more precise than comparing two separately-computed embeddings."),
    "final": ("Stage-2 fusion adds the cross-encoder score, re-applies penalties, breaks ties by ID and writes a "
              "fact-grounded reason for every rank.",
              "Reasons are templated from the record itself, so they cannot invent facts."),
}


def stage_header(i: int, key: str):
    title = dict((k, t) for k, t, _ in STAGES)[key]
    st.markdown(
        f'<div class="tb-stage-h"><span class="k">Stage {i}</span><span class="t">{esc(title)}</span>'
        f'<span class="s">{fmt_t(res["times"][key])}</span></div>',
        unsafe_allow_html=True,
    )
    what, why = STAGE_DOC[key]
    st.markdown(f'<div class="tb-explain"><b>What:</b> {esc(what)}<br><b>Why:</b> {esc(why)}<br>'
                f'<b>Result:</b> {esc(ev[key].headline)}</div>', unsafe_allow_html=True)


def top_by(arr, k=5) -> pd.DataFrame:
    idx = np.argsort(-np.asarray(arr))[:k]
    return pd.DataFrame([{
        "Candidate": ids[i],
        "Title": cands[i]["profile"].get("current_title"),
        "Score": round(float(arr[i]), 3),
        "Final rank": final_rank.get(int(i)),
    } for i in idx])


with tab_walk:
    top1 = rows[0]["idx"] if rows else 0

    with st.container(border=True):
        stage_header(1, "ingest")
        a, b = st.columns([1, 1.4])
        miss = ev["ingest"].data["missing"]
        a.dataframe(pd.DataFrame({"Field": list(miss), "Profiles missing it": list(miss.values())}),
                    hide_index=True, use_container_width=True)
        with b:
            hist(ev["ingest"].data["yoe"], "years of experience", bins=25, height=150)

    with st.container(border=True):
        stage_header(2, "text")
        a, b = st.columns(2)
        a.markdown("**What the models read for the #1 candidate**")
        a.markdown(f'<div class="tb-mono">{esc(tr["texts"][top1])}</div>', unsafe_allow_html=True)
        b.markdown("**What the models read for the job**")
        b.markdown(f'<div class="tb-mono">{esc(tr["jd_text"])}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        stage_header(3, "semantic")
        a, b = st.columns([1.2, 1])
        with a:
            hist(tr["sem"], "cosine similarity to JD")
        b.markdown("**Top 5 by semantic score alone**")
        b.dataframe(top_by(tr["sem"]), hide_index=True, use_container_width=True)

    with st.container(border=True):
        stage_header(4, "lexical")
        a, b = st.columns([1.2, 1])
        terms = ev["lexical"].data["term_idf"][:15]
        if terms:
            tdf = pd.DataFrame({"term": [t for _, t in terms], "idf": [v for v, _ in terms]})
            ch = alt.Chart(tdf).mark_bar(color=COMP_COLORS["Lexical"], cornerRadiusTopRight=3,
                                         cornerRadiusBottomRight=3).encode(
                x=alt.X("idf:Q", title="rarity in this pool (IDF) — rarer = stronger evidence"),
                y=alt.Y("term:N", sort="-x", title=None),
                tooltip=["term", alt.Tooltip("idf:Q", format=".2f")],
            ).properties(height=320, title="Most discriminative JD terms present in the pool")
            a.altair_chart(chart_cfg(ch), use_container_width=True)
        b.markdown("**Top 5 by keyword score alone**")
        b.dataframe(top_by(tr["lex"]), hide_index=True, use_container_width=True)

    with st.container(border=True):
        stage_header(5, "structured")
        sub_keys = list(STRUCTURED_SUB_WEIGHTS)
        hm = []
        for r in rows[:15]:
            for k in sub_keys:
                hm.append({"candidate": f"#{r['rank']} {r['id']}", "signal": f"{k} ({STRUCTURED_SUB_WEIGHTS[k]:.0%})",
                           "value": float(tr["sub_scores"][r["idx"]][k]), "order": r["rank"]})
        hdf = pd.DataFrame(hm)
        heat = alt.Chart(hdf).mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=3).encode(
            x=alt.X("signal:N", sort=[f"{k} ({STRUCTURED_SUB_WEIGHTS[k]:.0%})" for k in sub_keys], title=None,
                    axis=alt.Axis(labelAngle=0, orient="top", labelLimit=120)),
            y=alt.Y("candidate:N", sort=alt.EncodingSortField("order"), title=None),
            color=alt.Color("value:Q", scale=alt.Scale(domain=[0, 1], range=["#cde2fb", "#0d366b"]),
                            legend=alt.Legend(title="sub-score", orient="right")),
            tooltip=["candidate", "signal", alt.Tooltip("value:Q", format=".2f")],
        ).properties(height=34 * min(15, len(rows)) + 30)
        st.caption("JD-fit sub-scores for the top 15 (weight in brackets). Darker = stronger fit.")
        st.altair_chart(chart_cfg(heat), use_container_width=True)

    with st.container(border=True):
        stage_header(6, "trust")
        reasons = {}
        for fl in tr["hp_flags"]:
            for f in fl:
                reasons[("Honeypot", short_flag(f))] = reasons.get(("Honeypot", short_flag(f)), 0) + 1
        for fl in tr["dq_flags"]:
            for f in fl:
                reasons[("Disqualifier", short_flag(f))] = reasons.get(("Disqualifier", short_flag(f)), 0) + 1
        if reasons:
            rdf = pd.DataFrame([{"type": t, "reason": r, "profiles": n} for (t, r), n in reasons.items()])
            ch = alt.Chart(rdf).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
                x=alt.X("profiles:Q", title="profiles triggered"),
                y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color("type:N", scale=alt.Scale(domain=["Honeypot", "Disqualifier"],
                                                          range=["#e34948", "#eda100"]), title=None),
                tooltip=["type", "reason", "profiles"],
            ).properties(height=26 * len(rdf) + 30)
            st.altair_chart(chart_cfg(ch), use_container_width=True)
            st.caption("See the 🛡️ Risk & exclusions tab for every flagged profile and its exact reason.")
        else:
            st.success("No profile in this pool triggered a trust check.")

    with st.container(border=True):
        stage_header(7, "stage1")
        grp = np.where(tr["hp"] * tr["dq"] < 1, "Penalised", "Clean")
        grp = np.where(np.isin(np.arange(N), list(short_set)), "Shortlisted", grp)
        sdf = pd.DataFrame({"Semantic": tr["sem"], "JD-fit": tr["struct"], "group": grp, "candidate": ids,
                            "stage-1 score": tr["s1"]})
        sc = alt.Chart(sdf).mark_circle(size=60, opacity=0.8, stroke="#ffffff", strokeWidth=1).encode(
            x=alt.X("Semantic:Q", scale=alt.Scale(zero=False), title="semantic similarity"),
            y=alt.Y("JD-fit:Q", scale=alt.Scale(zero=False), title="structured JD-fit"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Shortlisted", "Penalised", "Clean"],
                                                       range=["#2a78d6", "#eb6834", NEUTRAL]), title=None),
            order=alt.Order("group:N", sort="descending"),
            tooltip=["candidate", "group", alt.Tooltip("Semantic:Q", format=".2f"),
                     alt.Tooltip("JD-fit:Q", format=".2f"), alt.Tooltip("stage-1 score:Q", format=".3f")],
        ).properties(height=330)
        st.caption("Every profile in the pool. Shortlisted profiles sit top-right; penalised ones can score "
                   "well on text and still be pushed out by the trust stage.")
        st.altair_chart(chart_cfg(sc.interactive()), use_container_width=True)

    with st.container(border=True):
        stage_header(8, "rerank")
        if res["ce"]:
            mdf = pd.DataFrame([{"candidate": r["id"], "Stage 1": r["stage1_rank"], "Final": r["rank"],
                                 "move": "Up" if r["stage1_rank"] > r["rank"] else
                                 ("Down" if r["stage1_rank"] < r["rank"] else "Same")} for r in rows])
            long = mdf.melt(id_vars=["candidate", "move"], value_vars=["Stage 1", "Final"],
                            var_name="stage", value_name="rank")
            base = alt.Chart(long).encode(
                x=alt.X("stage:N", sort=["Stage 1", "Final"], title=None, axis=alt.Axis(labelAngle=0, orient="top")),
                y=alt.Y("rank:Q", scale=alt.Scale(reverse=True, domainMin=1), title="rank (1 = best)"),
                color=alt.Color("move:N", scale=alt.Scale(domain=["Up", "Down", "Same"],
                                                          range=["#2a78d6", "#eb6834", NEUTRAL]),
                                title="moved after rerank"),
                detail="candidate:N",
                tooltip=["candidate", "stage", "rank", "move"],
            )
            slope = (base.mark_line(strokeWidth=2) + base.mark_circle(size=70, stroke="#ffffff", strokeWidth=1.5))
            a, b = st.columns([1.1, 1])
            with a:
                st.altair_chart(chart_cfg(slope.properties(height=420)), use_container_width=True)
            with b:
                st.markdown("**Biggest movers**")
                mv = pd.DataFrame([{"Candidate": r["id"], "Title": cands[r["idx"]]["profile"].get("current_title"),
                                    "Stage-1 → Final": f"{r['stage1_rank']} → {r['rank']}",
                                    "Cross-encoder": round(ce_by_idx[r["idx"]], 3)} for r in
                                   sorted(rows, key=lambda r: -abs(r["stage1_rank"] - r["rank"]))[:8]])
                st.dataframe(mv, hide_index=True, use_container_width=True)
        else:
            st.info("Cross-encoder was disabled for this run, so the stage-1 order carries through unchanged.")

    with st.container(border=True):
        stage_header(9, "final")
        cdf = pd.DataFrame([{"candidate": f"#{r['rank']} {r['id']}", "rank": r["rank"], "component": k,
                             "contribution": v, "order": list(COMP_COLORS).index(k)}
                            for r in rows for k, v in r["contrib"].items()])
        stack = alt.Chart(cdf).mark_bar(stroke="#ffffff", strokeWidth=2).encode(
            y=alt.Y("candidate:N", sort=alt.EncodingSortField("rank"), title=None),
            x=alt.X("contribution:Q", title="weighted contribution to the stage-2 score (before penalties)",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("component:N", scale=alt.Scale(domain=list(COMP_COLORS), range=list(COMP_COLORS.values())),
                            title=None),
            order=alt.Order("order:Q"),
            tooltip=["candidate", "component", alt.Tooltip("contribution:Q", format=".3f")],
        ).properties(height=22 * len(rows) + 40)
        st.altair_chart(chart_cfg(stack), use_container_width=True)

# ── Candidate deep-dive ───────────────────────────────────────────────────────
with tab_cand:
    order = [r["idx"] for r in rows] + [int(i) for i in np.argsort(-tr["s1"]) if int(i) not in final_rank]

    def label(i: int) -> str:
        p = cands[i]["profile"]
        tag = f"#{final_rank[i]}" if i in final_rank else f"not ranked (stage-1 #{s1_rank[i]})"
        return f"{tag} · {ids[i]} · {p.get('current_title') or '—'} @ {p.get('current_company') or '—'}"

    labels = [label(i) for i in order]
    sel = order[labels.index(st.selectbox("Pick any candidate in the pool", labels))]
    c = cands[sel]
    p, sig = c.get("profile", {}), c.get("redrob_signals", {})

    if sel in final_rank:
        r = next(r for r in rows if r["idx"] == sel)
        st.markdown(card_html(r), unsafe_allow_html=True)
        verdict = f"Ranked **#{r['rank']}** of {N:,}."
    else:
        why_not = []
        if tr["hp"][sel] < 1:
            why_not.append("flagged by the honeypot checks")
        if tr["dq"][sel] < 1:
            why_not.append("penalised by a JD disqualifier")
        if sel not in short_set:
            why_not.append(f"stage-1 rank {s1_rank[sel]} is outside the top {len(tr['top_idx'])} shortlist")
        else:
            why_not.append("dropped below the cut after cross-encoder reranking")
        verdict = "**Not in the final list** — " + "; ".join(why_not) + "."
    st.markdown(verdict)

    a, b = st.columns([1, 1.15], gap="large")
    with a:
        st.markdown("##### Score components")
        comp = pd.DataFrame([
            {"Component": "Semantic (bi-encoder)", "Value": float(tr["sem"][sel]), "Weight": STAGE2_WEIGHTS["semantic"]},
            {"Component": "Lexical (BM25)", "Value": float(tr["lex"][sel]), "Weight": STAGE2_WEIGHTS["lexical"]},
            {"Component": "JD-fit (structured)", "Value": float(tr["struct"][sel]), "Weight": STAGE2_WEIGHTS["structured"]},
            {"Component": "Cross-encoder", "Value": ce_by_idx.get(sel, np.nan), "Weight": STAGE2_WEIGHTS["cross_encoder"]},
            {"Component": "Honeypot multiplier", "Value": float(tr["hp"][sel]), "Weight": np.nan},
            {"Component": "Disqualifier multiplier", "Value": float(tr["dq"][sel]), "Weight": np.nan},
        ])
        st.dataframe(comp, hide_index=True, use_container_width=True,
                     column_config={"Value": st.column_config.NumberColumn(format="%.3f"),
                                    "Weight": st.column_config.NumberColumn(format="%.2f")})
        if sel not in ce_by_idx:
            st.caption("Cross-encoder only runs on the stage-1 shortlist.")
        subs = tr["sub_scores"][sel]
        sdf = pd.DataFrame({"signal": list(subs), "score": list(subs.values()),
                            "weight": [STRUCTURED_SUB_WEIGHTS[k] for k in subs]})
        ch = alt.Chart(sdf).mark_bar(color=COMP_COLORS["JD-fit"], cornerRadiusTopRight=3,
                                     cornerRadiusBottomRight=3).encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="sub-score"),
            y=alt.Y("signal:N", sort=list(subs), title=None),
            tooltip=["signal", alt.Tooltip("score:Q", format=".2f"), alt.Tooltip("weight:Q", format=".0%")],
        ).properties(height=230, title="JD-fit breakdown")
        st.altair_chart(chart_cfg(ch), use_container_width=True)
        flags = [("🚩 Honeypot", f) for f in tr["hp_flags"][sel]] + [("⚑ Disqualifier", f) for f in tr["dq_flags"][sel]]
        if flags:
            st.markdown("##### Trust findings")
            for t, f in flags:
                st.markdown(f"- **{t}:** {f}")

    with b:
        st.markdown("##### Profile")
        st.markdown(
            f"**{esc(p.get('anonymized_name') or ids[sel])}** — {esc(p.get('headline') or '')}  \n"
            f"📍 {esc(p.get('location') or '—')}, {esc(p.get('country') or '')} · "
            f"{p.get('years_of_experience', '—')} yrs · {esc(p.get('current_industry') or '—')}"
        )
        hist_rows = [{"Title": j.get("title"), "Company": j.get("company"), "Industry": j.get("industry"),
                      "Months": j.get("duration_months"), "Current": "●" if j.get("is_current") else ""}
                     for j in c.get("career_history", [])]
        if hist_rows:
            st.markdown("**Career history**")
            st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)
        assess = sig.get("skill_assessment_scores") or {}
        sk = [{"Skill": s.get("name"), "Proficiency": s.get("proficiency"), "Months": s.get("duration_months"),
               "Endorsements": s.get("endorsements"), "Assessed": assess.get(s.get("name"), None),
               "JD tier-1": "✓" if (s.get("name") or "").lower() in TIER1_SKILLS else ""}
              for s in c.get("skills", [])]
        if sk:
            st.markdown("**Skills (claims vs evidence)**")
            st.dataframe(pd.DataFrame(sk), hide_index=True, use_container_width=True, height=230)
        st.markdown("**Platform signals**")
        sigs = {
            "Response rate": f"{sig.get('recruiter_response_rate', 0):.0%}" if sig.get("recruiter_response_rate") is not None else "—",
            "Interview completion": f"{sig.get('interview_completion_rate', 0):.0%}" if sig.get("interview_completion_rate") is not None else "—",
            "Notice period": f"{sig.get('notice_period_days', '—')} days",
            "Last active": sig.get("last_active_date", "—"),
            "Work mode": sig.get("preferred_work_mode", "—"),
            "Relocate": "yes" if sig.get("willing_to_relocate") else "no",
            "GitHub score": sig.get("github_activity_score", "—"),
        }
        st.dataframe(pd.DataFrame({"Signal": list(sigs), "Value": [str(v) for v in sigs.values()]}),
                     hide_index=True, use_container_width=True)

    with st.expander("What the models actually saw for this candidate"):
        st.markdown(f'<div class="tb-mono">{esc(tr["texts"][sel])}</div>', unsafe_allow_html=True)
        overlap = sorted(set(tr["jd_tokens"]) & set(tr["tokens"][sel]))
        st.markdown(f"**JD terms matched by BM25 ({len(overlap)}):** " + (", ".join(f"`{t}`" for t in overlap) or "none"))

# ── Risk & exclusions ─────────────────────────────────────────────────────────
with tab_risk:
    flagged = [i for i in range(N) if tr["hp"][i] < 1 or tr["dq"][i] < 1]
    st.markdown(
        f"**{len(flagged):,} of {N:,}** profiles triggered at least one check. Penalties multiply the score "
        "(1.0 = untouched), so a flag lowers a candidate rather than silently deleting them."
    )
    if flagged:
        view = st.radio("Show", ["All flagged", "Honeypots only", "Disqualifiers only"], horizontal=True)
        if view == "Honeypots only":
            flagged = [i for i in flagged if tr["hp"][i] < 1]
        elif view == "Disqualifiers only":
            flagged = [i for i in flagged if tr["dq"][i] < 1]
        rdf = pd.DataFrame([{
            "Candidate": ids[i],
            "Title": cands[i]["profile"].get("current_title"),
            "Company": cands[i]["profile"].get("current_company"),
            "Honeypot ×": round(float(tr["hp"][i]), 2),
            "Disqualifier ×": round(float(tr["dq"][i]), 2),
            "Reasons": " | ".join(tr["hp_flags"][i] + tr["dq_flags"][i]),
            "Final rank": final_rank.get(i),
        } for i in flagged]).sort_values(["Honeypot ×", "Disqualifier ×"])
        st.dataframe(rdf, hide_index=True, use_container_width=True, height=480,
                     column_config={"Reasons": st.column_config.TextColumn(width="large")})

# ── Export ────────────────────────────────────────────────────────────────────
with tab_exp:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"], quoting=csv.QUOTE_ALL)
    w.writeheader()
    for r in rows:
        w.writerow({"candidate_id": r["id"], "rank": r["rank"], "score": f"{r['score']:.4f}", "reasoning": r["reasoning"]})
    full = pd.DataFrame({
        "candidate_id": ids, "semantic": tr["sem"], "lexical": tr["lex"], "jd_fit": tr["struct"],
        "honeypot_mult": tr["hp"], "disq_mult": tr["dq"], "stage1_score": tr["s1"], "stage1_rank": s1_rank,
        "cross_encoder": [ce_by_idx.get(i, np.nan) for i in range(N)],
        "final_rank": [final_rank.get(i, np.nan) for i in range(N)],
        "flags": [" | ".join(tr["hp_flags"][i] + tr["dq_flags"][i]) for i in range(N)],
    })
    a, b = st.columns(2)
    a.download_button("⬇  submission.csv (ranked list)", buf.getvalue(), "submission.csv", "text/csv",
                      use_container_width=True, type="primary")
    b.download_button("⬇  pipeline_trace.csv (every score, every profile)", full.to_csv(index=False),
                      "pipeline_trace.csv", "text/csv", use_container_width=True)
    st.markdown("**Preview — submission format**")
    st.code(buf.getvalue()[:1500] + ("\n…" if len(buf.getvalue()) > 1500 else ""), language="csv")
    st.caption(f"Run config · pool: {res['source']} · shortlist K={res['k']} · final N={res['n']} · "
               f"cross-encoder: {'on' if res['ce'] else 'off'} · reference date: {tr['today']}")
