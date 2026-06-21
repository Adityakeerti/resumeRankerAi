"""
src/disqualifiers.py — Compute multiplicative penalty multipliers for hard and soft disqualifiers.

Penalties are multiplicative and stack (e.g., zombie + ghost = 0.35 × 0.60 = 0.21).
All disqualifier conditions derive directly from the job description's explicit statements.
"""

from __future__ import annotations
from datetime import date

from config import (
    CONSULTING_FIRMS,
    NON_AI_ROLE_KEYWORDS,
    WRONG_DOMAIN_SKILLS,
    TIER1_SKILLS,
    TIER2_SKILLS,
    DISQUALIFIER_PENALTIES,
    ZOMBIE_INACTIVE_DAYS,
    ZOMBIE_RESPONSE_RATE_THRESH,
    HOPPER_TENURE_MONTHS,
    HOPPER_MIN_JOBS,
)


def compute_disqualifier_penalty(
    candidate: dict,
    today: date | None = None,
) -> tuple[float, list[str]]:
    """
    Compute the total multiplicative disqualifier penalty for a candidate.

    Args:
        candidate: Full candidate dict
        today: Reference date (defaults to date.today())

    Returns:
        (penalty_multiplier, list_of_triggered_conditions)
        penalty_multiplier: 1.0 = no penalty, lower = more penalised
        list_of_triggered_conditions: Strings describing each triggered disqualifier
    """
    if today is None:
        today = date.today()

    profile  = candidate.get('profile', {})
    history  = candidate.get('career_history', [])
    skills   = candidate.get('skills', [])
    signals  = candidate.get('redrob_signals', {})

    penalty      = 1.0
    triggered    = []

    # ── Disqualifier 1: Full consulting career ─────────────────────────────
    # JD: "Full consulting-firm career (all at TCS/Infosys/...) with no
    # product-company experience" → near-zero relevance.
    if history:
        consulting_flags = [
            _is_consulting(j.get('company', '')) for j in history
        ]
        if all(consulting_flags):
            penalty *= DISQUALIFIER_PENALTIES['full_consulting_career']
            triggered.append('full_consulting_career')

    # ── Disqualifier 2: Title/description mismatch ────────────────────────
    # Current title is a non-AI role (Marketing, Sales, HR) but description
    # mentions AI. These are keyword stuffers.
    current_title_lower = (profile.get('current_title') or '').lower()
    is_non_ai_title = any(kw in current_title_lower for kw in NON_AI_ROLE_KEYWORDS)

    if is_non_ai_title:
        # Check if the description is talking about AI to identify stuffers
        all_descriptions = ' '.join(
            (j.get('description') or '') for j in history
        ).lower()
        ai_description_mentions = sum(
            1 for term in TIER1_SKILLS if term in all_descriptions
        )
        if ai_description_mentions >= 3:
            penalty *= DISQUALIFIER_PENALTIES['title_description_mismatch']
            triggered.append('title_description_mismatch')

    # ── Disqualifier 3: Behavioral zombie ─────────────────────────────────
    # last_active > 150 days AND recruiter_response_rate < 10%
    # JD says: "a perfect-on-paper candidate who hasn't logged in for 6 months
    # and has a 5% response rate is, for hiring purposes, not actually available."
    last_active_str = signals.get('last_active_date')
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = (today - last_active).days
            resp_rate    = signals.get('recruiter_response_rate', 1.0) or 1.0
            if days_inactive > ZOMBIE_INACTIVE_DAYS and resp_rate < ZOMBIE_RESPONSE_RATE_THRESH:
                penalty *= DISQUALIFIER_PENALTIES['behavioral_zombie']
                triggered.append(
                    f'behavioral_zombie: {days_inactive}d inactive, {resp_rate:.0%} response rate'
                )
        except (ValueError, TypeError):
            pass

    # ── Disqualifier 4: Interview ghost ────────────────────────────────────
    # interview_completion_rate < 30% — extremely unreliable candidate
    icr = signals.get('interview_completion_rate', 1.0)
    if icr is not None and icr < 0.30:
        penalty *= DISQUALIFIER_PENALTIES['interview_ghost']
        triggered.append(f'interview_ghost: completion_rate={icr:.0%}')

    # ── Disqualifier 5: Offer window-shopper ──────────────────────────────
    # offer_acceptance_rate < 15% (not -1, which means no history)
    oar = signals.get('offer_acceptance_rate', -1)
    if oar is not None and oar != -1 and oar < 0.15:
        penalty *= DISQUALIFIER_PENALTIES['offer_window_shopper']
        triggered.append(f'offer_window_shopper: acceptance_rate={oar:.0%}')

    # ── Disqualifier 6: Job hopper ─────────────────────────────────────────
    # avg tenure < 12 months with 5+ jobs
    # JD: "title-chasing (avg tenure per company < 18 months across 4+ jobs)"
    if len(history) >= HOPPER_MIN_JOBS:
        total_months = sum((j.get('duration_months') or 0) for j in history)
        avg_tenure   = total_months / len(history)
        if avg_tenure < HOPPER_TENURE_MONTHS:
            penalty *= DISQUALIFIER_PENALTIES['job_hopper']
            triggered.append(
                f'job_hopper: {len(history)} jobs, avg tenure {avg_tenure:.1f}mo'
            )

    # ── Disqualifier 7: Notice period > 90 days ───────────────────────────
    notice_days = signals.get('notice_period_days', 0) or 0
    if notice_days > 90:
        penalty *= DISQUALIFIER_PENALTIES['notice_over_90']
        triggered.append(f'notice_over_90: {notice_days}d notice period')

    # ── Disqualifier 8: Wrong domain expert ───────────────────────────────
    # Primary skills are CV/speech/robotics with no NLP/IR work
    all_skill_names = {(s.get('name') or '').lower() for s in skills}
    advanced_skills = {
        (s.get('name') or '').lower() for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
    }

    wrong_domain_count = sum(
        1 for sk in advanced_skills if sk in WRONG_DOMAIN_SKILLS
    )
    nlp_count = sum(
        1 for sk in all_skill_names
        if sk in TIER1_SKILLS or sk in TIER2_SKILLS
    )

    if wrong_domain_count >= 3 and nlp_count == 0:
        penalty *= DISQUALIFIER_PENALTIES['wrong_domain_expert']
        triggered.append(
            f'wrong_domain_expert: {wrong_domain_count} CV/speech/robotics skills, no NLP'
        )

    # ── Disqualifier 9: Unverified identity ───────────────────────────────
    verified_email = signals.get('verified_email', True)
    verified_phone = signals.get('verified_phone', True)
    if not verified_email and not verified_phone:
        penalty *= DISQUALIFIER_PENALTIES['unverified_identity']
        triggered.append('unverified_identity: neither email nor phone verified')

    return penalty, triggered


def _is_consulting(company_name: str) -> bool:
    """
    Check if a company name matches known consulting firms.

    Args:
        company_name: Company name string

    Returns:
        True if consulting firm detected
    """
    cn_lower = company_name.lower().strip()
    return any(firm in cn_lower for firm in CONSULTING_FIRMS)


def batch_compute_disqualifiers(
    candidates: list[dict],
    today: date | None = None,
) -> tuple[list[float], list[list[str]]]:
    """
    Compute disqualifier penalties for a list of candidates.

    Args:
        candidates: List of candidate dicts
        today: Reference date

    Returns:
        (penalties, triggered_list)
    """
    if today is None:
        today = date.today()

    penalties     = []
    triggered_all = []

    for cand in candidates:
        p, t = compute_disqualifier_penalty(cand, today)
        penalties.append(p)
        triggered_all.append(t)

    return penalties, triggered_all
