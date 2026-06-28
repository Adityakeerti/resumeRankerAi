"""
sandbox/app.py — Streamlit demo for HuggingFace Spaces / Replit.

Accepts a JSONL file with ≤ 100 candidates and a JD text, then ranks them
and displays the results. This is the sandbox_link required in submission_metadata.yaml.

Run locally: streamlit run sandbox/app.py
Deploy: Push to HuggingFace Spaces with requirements.txt (see top-level requirements.txt)
"""

import json
import sys
import io
import csv
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st

# ── Path setup (works both locally and in Spaces) ──────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TOP_N_FINAL,
    CANDIDATE_TEXT_MAX_CHARS,
    JD_TEXT_SUMMARY,
    TIER1_SKILLS,
)
from src.text_builder import build_candidate_text, build_jd_text, build_jd_tokens
from src.semantic_scorer import embed_query, cosine_scores, cross_encoder_scores
from src.bm25_scorer import build_bm25_index, query_bm25
from src.structured_scorer import batch_compute_structured
from src.honeypot_detector import batch_detect_honeypots
from src.disqualifiers import batch_compute_disqualifiers
from src.fusion import stage1_fusion, stage2_fusion
from src.reasoning import build_reasoning


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI - Candidate Ranker",
    page_icon="💼",
    layout="wide",
)

st.title("Redrob AI - Candidate Ranking System")
st.caption(
    "Multi-Stage Retrieval and Reranking Pipeline: Bi-Encoder Semantics + BM25 Lexical + "
    "Structured JD-Fit Scoring + Cross-Encoder Reranking"
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    use_cross_encoder = st.checkbox("Use Cross-Encoder (slower, more accurate)", value=True)
    top_n_display     = st.slider("Show top N results", 10, 100, 50)
    st.divider()
    st.subheader("Scoring Weights")
    st.write("• Structured (JD-fit):  45%")
    st.write("• Semantic (bi-encoder): 25%")
    st.write("• Cross-Encoder:         20%")
    st.write("• Lexical (BM25):        10%")
    st.divider()
    st.info(
        "For the full 100K ranking, run `precompute.py` + `rank.py` locally. "
        "This demo accepts up to 100 candidates for interactive exploration."
    )

# ── Main layout ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    jd_input = st.text_area(
        "Paste job description text:",
        value=JD_TEXT_SUMMARY.strip(),
        height=200,
        help="The ranker will score candidates against this JD."
    )

with col2:
    st.subheader("Candidate Profiles")
    uploaded_file = st.file_uploader(
        "Upload candidates.jsonl (≤ 100 candidates)",
        type=['jsonl', 'json'],
        help="Each line should be a JSON object matching the candidate schema."
    )
    st.caption("Or use sample data:")
    use_sample = st.button("Generate & use 20 sample candidates")


# ── Sample data generator ────────────────────────────────────────────────────

def make_sample_candidates(n: int = 20) -> list[dict]:
    """Generate minimal synthetic candidates for demo purposes."""
    import random
    random.seed(42)

    titles = [
        "Senior ML Engineer", "AI Engineer", "Data Scientist",
        "NLP Engineer", "Software Engineer", "Marketing Manager",
        "HR Manager", "Search Engineer", "Recommendation Systems Lead",
        "ML Platform Engineer",
    ]
    companies = [
        "Flipkart", "Swiggy", "Razorpay", "Meesho", "CRED",
        "TCS", "Infosys", "Zomato", "PhonePe", "Juspay",
    ]
    skill_pools = {
        "ai":      ["python", "faiss", "sentence-transformers", "elasticsearch", "ndcg", "qdrant"],
        "general": ["java", "sql", "marketing", "excel", "communication"],
    }

    candidates = []
    for i in range(n):
        cid       = f"CAND_{i:07d}"
        is_ai     = random.random() > 0.3
        title     = random.choice(titles[:5]) if is_ai else random.choice(titles[5:])
        company   = random.choice(companies)
        yoe       = random.uniform(2, 15)
        pool      = skill_pools["ai"] if is_ai else skill_pools["general"]
        skills    = [
            {
                "name": s,
                "proficiency": random.choice(["intermediate", "advanced", "expert"]) if is_ai else "beginner",
                "endorsements": random.randint(0, 30) if is_ai else 0,
                "duration_months": random.randint(12, 60) if is_ai else random.randint(0, 6),
            }
            for s in random.sample(pool, min(3, len(pool)))
        ]
        cand = {
            "candidate_id": cid,
            "profile": {
                "anonymized_name": f"Candidate {i}",
                "headline": f"{title} with {yoe:.0f}y exp" if is_ai else f"{title}",
                "summary":  (
                    f"Experienced in building ranking and retrieval systems. "
                    f"Shipped production ML pipelines at scale." if is_ai
                    else f"Professional with {yoe:.0f} years of experience."
                ),
                "location": random.choice(["Pune, Maharashtra", "Noida, UP", "Bengaluru, Karnataka", "Chennai, TN"]),
                "country": "India",
                "years_of_experience": round(yoe, 1),
                "current_title": title,
                "current_company": company,
                "current_company_size": random.choice(["501-1000", "1001-5000", "5001-10000"]),
                "current_industry": "Technology" if is_ai else "Consulting",
            },
            "career_history": [
                {
                    "company": company,
                    "title": title,
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": int(yoe * 12),
                    "is_current": True,
                    "industry": "Technology" if is_ai else "IT Services",
                    "company_size": "1001-5000",
                    "description": (
                        "Built end-to-end ranking and recommendation systems using "
                        "dense retrieval, FAISS vector search, and NDCG evaluation." if is_ai
                        else "Managed client deliverables and team operations."
                    ),
                }
            ],
            "education": [
                {
                    "institution": random.choice(["IIT Bombay", "NIT Trichy", "VIT"]),
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science" if is_ai else "Electronics",
                    "start_year": 2014,
                    "end_year": 2018,
                    "grade": "8.5" if is_ai else "7.0",
                    "tier": "tier_1" if "IIT" in company else "tier_2",
                }
            ],
            "skills": skills,
            "certifications": [],
            "languages": [{"language": "English", "proficiency": "professional"}],
            "redrob_signals": {
                "profile_completeness_score": random.uniform(60, 100),
                "signup_date": "2023-01-01",
                "last_active_date": "2026-05-15" if is_ai else "2025-10-01",
                "open_to_work_flag": is_ai,
                "notice_period_days": random.choice([15, 30, 60, 90]),
                "applications_submitted_30d": random.randint(1, 20),
                "recruiter_response_rate": random.uniform(0.5, 1.0) if is_ai else random.uniform(0.1, 0.5),
                "avg_response_time_hours": random.uniform(2, 48),
                "interview_completion_rate": random.uniform(0.7, 1.0) if is_ai else random.uniform(0.3, 0.7),
                "offer_acceptance_rate": random.uniform(0.5, 1.0) if is_ai else -1,
                "profile_views_received_30d": random.randint(10, 200),
                "search_appearance_30d": random.randint(20, 500),
                "saved_by_recruiters_30d": random.randint(0, 15) if is_ai else 0,
                "connection_count": random.randint(100, 2000),
                "endorsements_received": random.randint(5, 100) if is_ai else 0,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": is_ai,
                "github_activity_score": random.uniform(40, 95) if is_ai else -1,
                "skill_assessment_scores": {s["name"]: random.randint(60, 95) for s in skills[:2]} if is_ai else {},
                "expected_salary_range_inr_lpa": {"min": 20, "max": 40} if is_ai else {"min": 10, "max": 20},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": random.choice([True, False]),
            },
        }
        candidates.append(cand)
    return candidates


# ── Session state ────────────────────────────────────────────────────────────
if 'candidates' not in st.session_state:
    st.session_state.candidates = []

if use_sample:
    st.session_state.candidates = make_sample_candidates(20)
    st.success(f"Generated 20 sample candidates")

if uploaded_file is not None:
    raw = uploaded_file.read().decode('utf-8')
    parsed = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if line:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                st.warning(f"Skipped malformed line")
    if len(parsed) > 100:
        st.warning("Capped at 100 candidates for the demo")
        parsed = parsed[:100]
    st.session_state.candidates = parsed
    st.success(f"Loaded {len(parsed)} candidates")


# ── Ranking ──────────────────────────────────────────────────────────────────

if st.button("Rank Candidates", type="primary", disabled=not st.session_state.candidates):
    candidates = st.session_state.candidates
    N          = len(candidates)
    today      = date.today()

    progress_bar = st.progress(0, text="Starting ranking pipeline ...")

    with st.spinner(""):

        # Step 1: Build texts
        progress_bar.progress(10, "Building candidate text representations ...")
        cand_texts = [build_candidate_text(c, CANDIDATE_TEXT_MAX_CHARS) for c in candidates]
        cand_ids   = [c.get('candidate_id', f'CAND_{i:07d}') for i, c in enumerate(candidates)]

        # Step 2: Embed JD
        progress_bar.progress(20, "Embedding job description ...")
        jd_text   = build_jd_text(jd_input)
        jd_vec    = embed_query(jd_text)
        jd_tokens = build_jd_tokens(jd_input)

        # Step 3: Bi-encoder
        progress_bar.progress(30, "Computing semantic similarities ...")
        from sentence_transformers import SentenceTransformer
        bi_model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs      = bi_model.encode(
            cand_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        sem_sc    = cosine_scores(jd_vec, embs)

        # Step 4: BM25
        progress_bar.progress(45, "BM25 lexical scoring ...")
        from src.text_builder import build_bm25_tokens
        tok_corpus = [build_bm25_tokens(c) for c in candidates]
        bm25 = build_bm25_index(tok_corpus, output_path="/tmp/bm25_sandbox.pkl")
        lex_sc = query_bm25(bm25, jd_tokens)

        # Step 5: Structured scoring
        progress_bar.progress(60, "Structured JD-fit scoring ...")
        str_scores, _ = batch_compute_structured(candidates, today=today)
        str_sc        = np.array(str_scores, dtype=np.float32)

        # Step 6: Honeypot + disqualifiers
        progress_bar.progress(70, "Honeypot detection + disqualifier penalties ...")
        hp_pen, hp_flags   = batch_detect_honeypots(candidates)
        dq_pen, dq_flags   = batch_compute_disqualifiers(candidates, today=today)
        hp_arr = np.array(hp_pen, dtype=np.float32)
        dq_arr = np.array(dq_pen, dtype=np.float32)

        # Step 7: Stage 1 fusion
        progress_bar.progress(75, "Stage 1 fusion ...")
        _, top_idx = stage1_fusion(sem_sc, lex_sc, str_sc, hp_arr, dq_arr, top_n=min(N, 50))

        # Step 8: Cross-encoder
        if use_cross_encoder and N <= 50:
            progress_bar.progress(80, "Cross-encoder reranking ...")
            top_texts  = [cand_texts[i] for i in top_idx]
            ce_scores  = cross_encoder_scores(jd_text, top_texts, show_progress=False)
        else:
            ce_scores = np.zeros(len(top_idx), dtype=np.float32)

        # Step 9: Stage 2 fusion
        progress_bar.progress(90, "Final ranking ...")
        final_n = min(N, top_n_display)
        final_idx, final_sc = stage2_fusion(
            top_n_indices=top_idx,
            semantic_scores=sem_sc,
            lexical_scores=lex_sc,
            structured_scores=str_sc,
            cross_encoder_scores=ce_scores,
            honeypot_penalties=hp_arr,
            disq_penalties=dq_arr,
            candidate_ids=cand_ids,
            top_n_final=final_n,
        )

        progress_bar.progress(100, "Done!")

    # ── Results display ──────────────────────────────────────────────────
    st.success(f"Ranked {N} candidates -> showing top {final_n}")

    rows = []
    for rank_i, (idx, score) in enumerate(zip(final_idx, final_sc), start=1):
        cand = candidates[idx]
        rows.append({
            "Rank":       rank_i,
            "ID":         cand.get('candidate_id', '?'),
            "Title":      cand['profile'].get('current_title', '?'),
            "Company":    cand['profile'].get('current_company', '?'),
            "YoE":        cand['profile'].get('years_of_experience', 0),
            "Score":      round(float(score), 4),
            "Reasoning":  build_reasoning(cand, rank=rank_i, today=today),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    # Highlight top-10
    st.dataframe(
        df,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
        },
        use_container_width=True,
        hide_index=True,
    )

    # ── CSV download ─────────────────────────────────────────────────────
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=['candidate_id', 'rank', 'score', 'reasoning'],
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            'candidate_id': row['ID'],
            'rank':         row['Rank'],
            'score':        f"{row['Score']:.4f}",
            'reasoning':    row['Reasoning'],
        })

    st.download_button(
        label="Download submission.csv",
        data=output.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )

    # ── Score breakdown chart ─────────────────────────────────────────────
    st.subheader("Score Distribution")
    import pandas as pd
    score_df = pd.DataFrame({
        "Rank": list(range(1, len(rows) + 1)),
        "Score": [r["Score"] for r in rows],
    })
    st.line_chart(score_df.set_index("Rank"))

elif not st.session_state.candidates:
    st.info("Upload a candidates.jsonl file or click 'Generate sample candidates' to get started.")
