"""
src/text_builder.py — Builds a candidate's text representation for embedding/BM25.

Integrates the Statistical CDF and NLI Categorical Hypotheses into the candidate text blob.
"""

import numpy as np
from config import TIER1_SKILLS, TIER2_SKILLS

def get_cdf_category(percentile: float) -> str:
    if percentile <= 10:
        return "extremely low"
    elif percentile <= 25:
        return "below average"
    elif percentile <= 75:
        return "average"
    elif percentile <= 90:
        return "high"
    else:
        return "exceptional"

def build_cdf_sentences(candidate: dict, cdfs: dict) -> str:
    """
    Generate semantically rich sentences from raw platform signals using CDF percentiles.
    """
    sig = candidate.get('redrob_signals', {})
    sentences = []

    # 1. Recruiter Response Rate
    val = sig.get('recruiter_response_rate', 0.0)
    pct = (np.searchsorted(cdfs['recruiter_response_rate'], val, side='right') / len(cdfs['recruiter_response_rate'])) * 100
    cat = get_cdf_category(pct)
    sentences.append(f"Candidate shows {cat} engagement with recruiters.")

    # 2. Avg Response Time (lower is better)
    val = sig.get('avg_response_time_hours', 0.0)
    pct = (np.searchsorted(cdfs['avg_response_time_hours'], val, side='right') / len(cdfs['avg_response_time_hours'])) * 100
    pct = 100.0 - pct  # invert so lower is better
    cat = get_cdf_category(pct)
    sentences.append(f"Candidate average response time is {cat}.")

    # 3. Interview Completion Rate
    val = sig.get('interview_completion_rate', 0.0)
    pct = (np.searchsorted(cdfs['interview_completion_rate'], val, side='right') / len(cdfs['interview_completion_rate'])) * 100
    cat = get_cdf_category(pct)
    sentences.append(f"Candidate interview completion rate is {cat}.")

    # 4. Offer Acceptance Rate
    val = sig.get('offer_acceptance_rate', -1)
    if val == -1:
        sentences.append("Candidate has no offer acceptance history.")
    else:
        pct = (np.searchsorted(cdfs['offer_acceptance_rate'], val, side='right') / len(cdfs['offer_acceptance_rate'])) * 100
        cat = get_cdf_category(pct)
        sentences.append(f"Candidate offer acceptance history is {cat}.")

    # 5. Github Activity Score
    val = sig.get('github_activity_score', -1)
    if val == -1:
        sentences.append("Candidate has no GitHub activity history.")
    else:
        pct = (np.searchsorted(cdfs['github_activity_score'], val, side='right') / len(cdfs['github_activity_score'])) * 100
        cat = get_cdf_category(pct)
        sentences.append(f"Candidate GitHub activity level is {cat}.")

    # 6. Profile Completeness Score
    val = sig.get('profile_completeness_score', 0.0)
    pct = (np.searchsorted(cdfs['profile_completeness_score'], val, side='right') / len(cdfs['profile_completeness_score'])) * 100
    cat = get_cdf_category(pct)
    sentences.append(f"Candidate profile completeness score is {cat}.")

    # 7. Categorical preferences
    willing = sig.get('willing_to_relocate', False)
    sentences.append("Candidate is willing to relocate to another city." if willing else "Candidate is not willing to relocate.")

    work_mode = sig.get('preferred_work_mode', 'flexible')
    sentences.append(f"Candidate preferred work mode is {work_mode}.")

    notice = sig.get('notice_period_days', 30)
    sentences.append(f"Candidate notice period is {notice} days.")

    # 8. Location
    loc = candidate.get('profile', {}).get('location', '')
    if loc:
        sentences.append(f"Candidate current location is {loc}.")

    return " ".join(sentences)

def build_candidate_text(candidate: dict, max_chars: int = 1000, cdfs: dict | None = None) -> str:
    """
    Build a dense, signal-rich text blob for a candidate, combining resume details with CDF behavior sentences.
    """
    parts = []

    # 1. Career descriptions (most signal-dense — lead with them)
    for job in candidate.get('career_history', [])[:5]:
        desc = (job.get('description') or '').strip()
        if desc:
            title = job.get('title', '')
            company = job.get('company', '')
            parts.append(f"{title} at {company}: {desc[:600]}")

    # 2. Profile headline + summary (intent narrative)
    headline = (candidate['profile'].get('headline') or '').strip()
    summary  = (candidate['profile'].get('summary')  or '').strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary[:500])

    # 3. Current title (role signal)
    current_title = (candidate['profile'].get('current_title') or '').strip()
    if current_title:
        parts.append(current_title)

    # 4. Only advanced/expert skills with meaningful duration (reduce noise)
    expert_skills = [
        s['name'] for s in candidate.get('skills', [])
        if s.get('proficiency') in ('advanced', 'expert')
        and s.get('duration_months', 0) > 6
    ]
    if expert_skills:
        parts.append(' '.join(expert_skills))

    # 5. Education field of study (light signal)
    for edu in candidate.get('education', [])[:2]:
        fos = (edu.get('field_of_study') or '').strip()
        if fos:
            parts.append(fos)

    # 6. Append CDF & NLI behavioral profile sentences if CDF data is available
    if cdfs is not None:
        parts.append(build_cdf_sentences(candidate, cdfs))

    full_text = ' '.join(filter(None, parts))
    return full_text[:max_chars]

def build_jd_text(jd_text: str, max_chars: int = 1500) -> str:
    """
    Clean and truncate the job description text for embedding.
    """
    import re
    text = re.sub(r'\s+', ' ', jd_text).strip()
    return text[:max_chars]

def build_bm25_tokens(candidate: dict) -> list[str]:
    """
    Build a tokenized list for the BM25 index.
    """
    import re

    def tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    tokens = []

    # Skills — all of them (BM25 is used for exact-term recall)
    for skill in candidate.get('skills', []):
        tokens.extend(tokenize(skill.get('name', '')))

    # Career descriptions
    for job in candidate.get('career_history', [])[:5]:
        tokens.extend(tokenize(job.get('description', '')[:800]))
        tokens.extend(tokenize(job.get('title', '')))
        tokens.extend(tokenize(job.get('industry', '')))

    # Profile
    tokens.extend(tokenize(candidate['profile'].get('headline', '')))
    tokens.extend(tokenize(candidate['profile'].get('summary', '')[:500]))
    tokens.extend(tokenize(candidate['profile'].get('current_title', '')))

    # Education
    for edu in candidate.get('education', []):
        tokens.extend(tokenize(edu.get('field_of_study', '')))
        tokens.extend(tokenize(edu.get('degree', '')))

    # Certifications
    for cert in candidate.get('certifications', []):
        tokens.extend(tokenize(cert.get('name', '')))
        tokens.extend(tokenize(cert.get('issuer', '')))

    return tokens

def build_jd_tokens(jd_text: str) -> list[str]:
    """
    Tokenize the JD for BM25 querying.
    """
    import re
    text = jd_text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]
