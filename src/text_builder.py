"""
src/text_builder.py — Builds a candidate's text representation for embedding/BM25.

The order of content is deliberate:
 1. Career descriptions — most semantic signal ("what they built")
 2. Headline + summary — intent and narrative
 3. Current title — role signal
 4. Only expert/advanced skills (reduces keyword-stuffer noise)
"""

from config import TIER1_SKILLS, TIER2_SKILLS


def build_candidate_text(candidate: dict, max_chars: int = 2000) -> str:
    """
    Build a dense, signal-rich text blob for a candidate.

    Career descriptions come first because the JD explicitly warns that a
    "Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
    profile, but if their career history shows they built a recommendation
    system at a product company, they're a fit." Semantic embeddings over
    career descriptions catch these candidates; skills-first ordering misses them.

    Args:
        candidate: Full candidate dict from candidates.jsonl
        max_chars: Hard truncation limit (default 2000)

    Returns:
        A single string, truncated to max_chars.
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

    full_text = ' '.join(filter(None, parts))
    return full_text[:max_chars]


def build_jd_text(jd_text: str, max_chars: int = 1500) -> str:
    """
    Clean and truncate the job description text for embedding.
    If jd_text was parsed from a docx, it may contain excessive whitespace.

    Args:
        jd_text: Raw text content of the job description
        max_chars: Truncation limit

    Returns:
        Cleaned JD string.
    """
    # Collapse excessive whitespace / newlines
    import re
    text = re.sub(r'\s+', ' ', jd_text).strip()
    return text[:max_chars]


def build_bm25_tokens(candidate: dict) -> list[str]:
    """
    Build a tokenized list for the BM25 index.
    BM25 works on token lists; we include a broader content set than the
    embedding text since BM25 is better at exact-match recall.

    Args:
        candidate: Full candidate dict

    Returns:
        List of lowercase tokens.
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

    Args:
        jd_text: Raw JD text

    Returns:
        List of lowercase tokens.
    """
    import re
    text = jd_text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]
