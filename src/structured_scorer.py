"""
src/structured_scorer.py — JD-specific structured scoring across 8 sub-components.

This is the core differentiator (weight: 45% of final score). It implements:
  1. Core Skills Trust Score     (30%)
  2. Career History Quality      (25%)
  3. Experience Range Fit        (10%)
  4. Location Fit                (8%)
  5. Notice Period               (5%)
  6. Behavioral Availability     (12%)
  7. Technical Credibility       (7%)
  8. Platform Demand Signal      (3%)

All functions are pure (no side effects) and work on individual candidate dicts.
"""

from __future__ import annotations
import math
from datetime import date

from config import (
    TIER1_SKILLS, TIER2_SKILLS,
    TOP_LOCATIONS, PREFERRED_LOCATIONS,
    CONSULTING_FIRMS, PRODUCT_TECH_INDUSTRIES,
    YOE_IDEAL_MIN, YOE_IDEAL_MAX, YOE_STATED_MIN, YOE_STATED_MAX,
    PROFICIENCY_MAP, SKILL_TRUST_WEIGHTS,
    STRUCTURED_SUB_WEIGHTS,
    GOOD_TENURE_MONTHS,
    RECENCY_DECAY_DAYS,
    RESPONSE_TIME_HALF_LIFE_HRS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core Skills Trust Score
# ─────────────────────────────────────────────────────────────────────────────

def core_skills_score(candidate: dict) -> float:
    """
    Trust-weighted skills matching score.

    Combats keyword stuffing by requiring evidence beyond the skill name:
    proficiency level, duration of use, peer endorsements, and platform assessment.

    Returns:
        Float [0, 1]
    """
    signals     = candidate.get('redrob_signals', {})
    assessments = signals.get('skill_assessment_scores', {}) or {}
    raw_score   = 0.0

    for skill in candidate.get('skills', []):
        name = (skill.get('name') or '').lower().strip()

        if name in TIER1_SKILLS:
            tier_weight = 1.0
        elif name in TIER2_SKILLS:
            tier_weight = 0.4
        else:
            continue  # Not a JD-relevant skill

        # Proficiency
        prof_key   = skill.get('proficiency', 'beginner')
        prof_score = PROFICIENCY_MAP.get(prof_key, 0.15)

        # Duration: continuous experience up to 36 months (cap)
        dur_months = skill.get('duration_months') or 0
        if dur_months > 0:
            dur_score = min(1.0, dur_months / 36.0)
        else:
            dur_score = 0.05  # Zero duration = likely keyword stuffer

        # Endorsements: log-scale (LinkedIn-style, cap at 20)
        endorsements = skill.get('endorsements') or 0
        end_score    = min(1.0, math.log1p(endorsements) / math.log1p(20))

        # Platform assessment (objective, hardest to fake)
        skill_original = skill.get('name', '')
        assess_val     = assessments.get(skill_original, -1)
        if assess_val < 0:
            # No assessment: fall back to proficiency as proxy
            assess_score = prof_score
        else:
            assess_score = assess_val / 100.0

        # Trust composite
        trust = tier_weight * (
            SKILL_TRUST_WEIGHTS['proficiency']   * prof_score  +
            SKILL_TRUST_WEIGHTS['duration']      * dur_score   +
            SKILL_TRUST_WEIGHTS['endorsements']  * end_score   +
            SKILL_TRUST_WEIGHTS['assessment']    * assess_score
        )
        raw_score += trust

    # Normalise: a candidate matching ~30% of Tier-1 skills perfectly = 1.0
    norm_factor = max(1, len(TIER1_SKILLS) * 0.3)
    return min(1.0, raw_score / norm_factor)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Career History Quality Score
# ─────────────────────────────────────────────────────────────────────────────

def career_quality_score(candidate: dict) -> float:
    """
    Career history quality based on:
     - Product vs consulting split
     - Career stability (anti-title-chaser)
     - Company size progression (seniority signal)

    Returns:
        Float [0, 1]
    """
    history = candidate.get('career_history', [])
    if not history:
        return 0.3  # No history = neutral-negative

    total_months    = sum((j.get('duration_months') or 0) for j in history)
    if total_months == 0:
        total_months = 1

    # ── Company type scoring ──────────────────────────────────────────────
    consulting_months = 0
    product_months    = 0

    for job in history:
        company_lower  = (job.get('company') or '').lower()
        industry_lower = (job.get('industry') or '').lower()
        dur            = job.get('duration_months') or 0

        is_consulting = any(f in company_lower for f in CONSULTING_FIRMS)

        if is_consulting:
            consulting_months += dur
        else:
            is_tech = any(ind in industry_lower for ind in PRODUCT_TECH_INDUSTRIES)
            if is_tech:
                product_months += dur

    consulting_ratio = consulting_months / total_months
    product_ratio    = product_months    / total_months

    company_score = product_ratio * 0.8 + (1.0 - consulting_ratio) * 0.2

    # ── Career stability (anti title-chaser) ──────────────────────────────
    if len(history) >= 3:
        avg_tenure     = total_months / len(history)
        # 30 months = stable; JD flags < 18 months as title-chasing
        stability_score = min(1.0, avg_tenure / GOOD_TENURE_MONTHS)
    else:
        stability_score = 0.7  # Too few jobs to judge

    # ── Seniority progression (company size growth) ───────────────────────
    size_order = [
        '1-10', '11-50', '51-200', '201-500',
        '501-1000', '1001-5000', '5001-10000', '10001+'
    ]
    sizes = []
    for job in history:
        cs = job.get('company_size', '')
        if cs in size_order:
            sizes.append(size_order.index(cs))

    if len(sizes) >= 2:
        # Negative delta = moved to larger company = positive signal
        progression = (sizes[0] - sizes[-1]) / 7.0
        progression_score = 0.5 + max(-0.5, min(0.5, -progression * 0.5))
    else:
        progression_score = 0.5

    return (
        0.50 * company_score    +
        0.30 * stability_score  +
        0.20 * progression_score
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Experience Range Score
# ─────────────────────────────────────────────────────────────────────────────

def experience_score(candidate: dict) -> float:
    """
    Score based on how well YoE matches the JD's stated range (5-9y, ideal 6-8y).

    Returns:
        Float [0, 1]
    """
    yoe = float(candidate['profile'].get('years_of_experience') or 0)

    if YOE_IDEAL_MIN <= yoe <= YOE_IDEAL_MAX:
        return 1.00   # Perfect range
    elif YOE_STATED_MIN <= yoe < YOE_IDEAL_MIN or YOE_IDEAL_MAX < yoe <= YOE_STATED_MAX:
        return 0.85   # Stated range
    elif 4 <= yoe < YOE_STATED_MIN or YOE_STATED_MAX < yoe <= 12:
        return 0.65   # JD says "seriously consider outside the band"
    elif yoe > 12:
        return 0.50   # Overqualified for founding team dynamics
    elif yoe >= 3:
        return 0.35   # Under-experienced
    else:
        return max(0.10, yoe / 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Location Fit Score
# ─────────────────────────────────────────────────────────────────────────────

def location_score(candidate: dict) -> float:
    """
    Score based on proximity to JD's preferred locations (Pune, Noida first).
    Willingness to relocate considered for non-preferred-location candidates.

    Returns:
        Float [0, 1]
    """
    profile  = candidate.get('profile', {})
    signals  = candidate.get('redrob_signals', {})
    location = (profile.get('location') or '').lower()
    willing  = signals.get('willing_to_relocate', False)
    country  = (profile.get('country') or '').strip()

    if any(loc in location for loc in TOP_LOCATIONS):
        return 1.00   # Pune / Noida = ideal
    elif any(loc in location for loc in PREFERRED_LOCATIONS):
        return 0.85   # Other preferred metros
    elif willing and country == 'India':
        return 0.65   # India + willing to relocate
    else:
        return 0.30   # India, not preferred city, not willing to relocate


# ─────────────────────────────────────────────────────────────────────────────
# 5. Notice Period Score
# ─────────────────────────────────────────────────────────────────────────────

def notice_score(candidate: dict) -> float:
    """
    Score based on notice period.
    JD: "We'd love sub-30-day notice. Bar gets higher past 30 days."

    Returns:
        Float [0, 1]
    """
    signals = candidate.get('redrob_signals', {})
    days    = signals.get('notice_period_days', 60) or 60

    if days <= 15:
        return 1.00
    elif days <= 30:
        return 0.90   # JD preference
    elif days <= 45:
        return 0.75
    elif days <= 60:
        return 0.60   # "bar gets higher"
    elif days <= 90:
        return 0.40
    else:
        return 0.20   # 90–180 days = real problem for startup


# ─────────────────────────────────────────────────────────────────────────────
# 6. Behavioral Availability Score
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_score(candidate: dict, today: date | None = None) -> float:
    """
    Score based on platform behavioral signals.
    Differentiates "behavioral twins" — same skills, different engagement.

    The JD signals doc says:
    "A perfect-on-paper candidate who hasn't logged in for 6 months and has a
    5% response rate is, for hiring purposes, not actually available."

    Returns:
        Float [0, 1]
    """
    if today is None:
        today = date.today()

    signals = candidate.get('redrob_signals', {})

    # ── Recency (last active) ──────────────────────────────────────────────
    last_active_str = signals.get('last_active_date')
    recency = 0.5  # default if no date
    if last_active_str:
        try:
            last_active  = date.fromisoformat(last_active_str)
            days_inactive = (today - last_active).days
            recency      = max(0.0, 1.0 - days_inactive / RECENCY_DECAY_DAYS)
        except (ValueError, TypeError):
            pass

    # ── Responsiveness ─────────────────────────────────────────────────────
    response_rate = signals.get('recruiter_response_rate', 0.5) or 0.5
    response_time = signals.get('avg_response_time_hours', 48) or 48
    resp_time_score = 1.0 / (1.0 + response_time / RESPONSE_TIME_HALF_LIFE_HRS)
    responsiveness  = 0.70 * response_rate + 0.30 * resp_time_score

    # ── Interview reliability ──────────────────────────────────────────────
    interview_rate = signals.get('interview_completion_rate', 0.7) or 0.7

    # ── Offer history (-1 = no history) ───────────────────────────────────
    offer_rate = signals.get('offer_acceptance_rate', -1)
    offer_score = offer_rate if (offer_rate is not None and offer_rate != -1) else 0.5

    # ── Platform flags ─────────────────────────────────────────────────────
    open_flag    = 1.0 if signals.get('open_to_work_flag') else 0.40
    completeness = (signals.get('profile_completeness_score') or 50) / 100.0

    return (
        0.30 * responsiveness  +
        0.25 * recency         +
        0.20 * interview_rate  +
        0.15 * offer_score     +
        0.05 * open_flag       +
        0.05 * completeness
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Technical Credibility Score
# ─────────────────────────────────────────────────────────────────────────────

def technical_score(candidate: dict) -> float:
    """
    Score based on objective technical signals:
     - GitHub activity (continuous activity score, hard to fake)
     - Skill assessment scores (platform-administered, objective)
     - LinkedIn connection (professional presence)

    Returns:
        Float [0, 1]
    """
    signals     = candidate.get('redrob_signals', {})
    assessments = signals.get('skill_assessment_scores', {}) or {}

    # GitHub activity
    gh = signals.get('github_activity_score', -1)
    if gh is None:
        gh = -1
    github_score = gh / 100.0 if gh >= 0 else 0.10  # No GitHub = soft negative

    # Platform assessments
    if assessments:
        jd_relevant = {
            k: v for k, v in assessments.items()
            if k.lower() in TIER1_SKILLS | TIER2_SKILLS
        }
        if jd_relevant:
            avg = sum(jd_relevant.values()) / len(jd_relevant) / 100.0
        else:
            avg = sum(assessments.values()) / len(assessments) / 100.0
        assess_score = avg
    else:
        assess_score = 0.30  # No assessments: neutral-negative for tech role

    # LinkedIn = professional cross-platform presence
    linkedin = 0.20 if signals.get('linkedin_connected') else 0.0

    return 0.60 * github_score + 0.30 * assess_score + 0.10 * linkedin


# ─────────────────────────────────────────────────────────────────────────────
# 8. Platform Demand Signal
# ─────────────────────────────────────────────────────────────────────────────

def demand_score(candidate: dict) -> float:
    """
    Score based on recruiter demand indicators.
    These are low-weight (3%) because they can be gamed by profile visibility
    rather than actual fit.

    Returns:
        Float [0, 1]
    """
    signals = candidate.get('redrob_signals', {})
    saved   = min(1.0, (signals.get('saved_by_recruiters_30d') or 0) / 10.0)
    appeared = min(1.0, (signals.get('search_appearance_30d') or 0) / 100.0)
    return 0.70 * saved + 0.30 * appeared


# ─────────────────────────────────────────────────────────────────────────────
# Composite Structured Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_structured_score(
    candidate: dict,
    today: date | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute the weighted composite structured score for one candidate.

    Args:
        candidate: Full candidate dict
        today: Reference date for recency calculations

    Returns:
        (total_score, sub_scores_dict)
        total_score: Float [0, 1]
        sub_scores_dict: Individual sub-component scores for debugging/reasoning
    """
    if today is None:
        today = date.today()

    sub_scores = {
        'skills':     core_skills_score(candidate),
        'career':     career_quality_score(candidate),
        'experience': experience_score(candidate),
        'location':   location_score(candidate),
        'notice':     notice_score(candidate),
        'behavioral': behavioral_score(candidate, today),
        'technical':  technical_score(candidate),
        'demand':     demand_score(candidate),
    }

    total = sum(
        STRUCTURED_SUB_WEIGHTS[k] * v
        for k, v in sub_scores.items()
    )

    return total, sub_scores


def batch_compute_structured(
    candidates: list[dict],
    today: date | None = None,
) -> tuple[list[float], list[dict[str, float]]]:
    """
    Compute structured scores for a list of candidates.

    Args:
        candidates: List of candidate dicts
        today: Reference date

    Returns:
        (scores, sub_scores_list)
    """
    if today is None:
        today = date.today()

    scores     = []
    sub_list   = []

    for cand in candidates:
        s, sub = compute_structured_score(cand, today)
        scores.append(s)
        sub_list.append(sub)

    return scores, sub_list
