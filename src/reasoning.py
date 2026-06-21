"""
src/reasoning.py — Fact-specific reasoning string generator for submission.csv.

The reasoning string is evaluated at Stage 4 on 10 sampled rows with 6 checks:
  1. Specific facts (title, YoE, matched skills, response rate, notice period)
  2. JD connection (what the JD requires vs what the candidate provides)
  3. Honest concerns (state gaps explicitly — don't hallucinate positives)
  4. No hallucination (only include facts from the candidate's actual record)
  5. Variation (branches based on rank zone and profile type)
  6. Rank consistency (top-10 = enthusiastic; bottom-20 = acknowledges concerns)

The output must fit in a CSV cell (no newlines, no commas unescaped).
We target ~150–220 characters to be information-dense without truncation risk.
"""

from __future__ import annotations
from datetime import date

from config import (
    TIER1_SKILLS,
    CONSULTING_FIRMS,
    RANK_ZONE_PREFIXES,
)


def build_reasoning(
    candidate:   dict,
    rank:        int,
    today:       date | None = None,
) -> str:
    """
    Build a fact-grounded reasoning string for a single candidate.

    Args:
        candidate: Full candidate dict
        rank:      Final rank (1–100)
        today:     Reference date for recency calculations

    Returns:
        A single-line string, no commas outside quoted fields, ~150–220 chars.
    """
    if today is None:
        today = date.today()

    profile  = candidate.get('profile', {})
    signals  = candidate.get('redrob_signals', {})
    history  = candidate.get('career_history', [])
    skills   = candidate.get('skills', [])

    # ── Core facts ────────────────────────────────────────────────────────
    current_title   = (profile.get('current_title') or 'Unknown title').strip()
    current_company = (profile.get('current_company') or 'Unknown co').strip()
    yoe             = float(profile.get('years_of_experience') or 0)

    # ── Matched Tier-1 skills (with evidence, not just name) ──────────────
    matched_t1 = [
        s['name'] for s in skills
        if (s.get('name') or '').lower().strip() in TIER1_SKILLS
        and s.get('proficiency') in ('advanced', 'expert')
        and (s.get('duration_months') or 0) >= 6
    ]

    # ── Key product-company role ───────────────────────────────────────────
    product_roles = [
        j for j in history
        if not any(f in (j.get('company') or '').lower() for f in CONSULTING_FIRMS)
    ]
    key_role = product_roles[0] if product_roles else (history[0] if history else None)

    # ── GitHub signal ──────────────────────────────────────────────────────
    gh = signals.get('github_activity_score', -1)
    if gh is None:
        gh = -1
    if gh >= 60:
        github_str = f"; GitHub {gh:.0f}/100"
    elif gh >= 0:
        github_str = f"; GitHub {gh:.0f}/100"
    else:
        github_str = ""   # No GitHub = omit (mentioned in concerns if relevant)

    # ── Concerns (honest, rank-appropriate) ───────────────────────────────
    concerns = _collect_concerns(candidate, today, rank)

    # ── Skills string ─────────────────────────────────────────────────────
    if matched_t1:
        skills_str = '/'.join(matched_t1[:3])
    else:
        # Show best intermediate skills as fallback
        any_skills = [
            s['name'] for s in skills
            if (s.get('name') or '').lower().strip() in TIER1_SKILLS
        ]
        skills_str = '/'.join(any_skills[:2]) + ' (intermediate)' if any_skills else 'no direct T1 skills'

    # ── Key role string ────────────────────────────────────────────────────
    if key_role:
        key_role_str = (
            f"{key_role.get('title', '?')[:35]} at {key_role.get('company', '?')[:25]} "
            f"({key_role.get('duration_months', 0)}mo)"
        )
    else:
        key_role_str = "no prior history found"

    # ── Rank-zone prefix ──────────────────────────────────────────────────
    prefix = _get_prefix(rank)

    # ── Response rate (most predictive of hirability) ─────────────────────
    rr = signals.get('recruiter_response_rate', 0.5)
    rr_str = f"; resp rate {rr:.0%}" if rr is not None else ""

    # ── Assemble ──────────────────────────────────────────────────────────
    concern_str = (" | Concerns: " + "; ".join(concerns)) if concerns else ""

    reasoning = (
        f"{prefix}"
        f"{current_title} at {current_company} "
        f"({yoe:.1f}y exp) | "
        f"{key_role_str} | "
        f"Skills: {skills_str}"
        f"{github_str}"
        f"{rr_str}"
        f"{concern_str}"
    )

    # Clean up for CSV safety (remove embedded newlines, escape quotes)
    reasoning = reasoning.replace('\n', ' ').replace('\r', '').replace('"', "'")

    # Truncate to safe CSV cell length
    return reasoning[:280]


def _collect_concerns(
    candidate: dict,
    today: date,
    rank: int,
) -> list[str]:
    """
    Collect honest concern strings for the candidate.
    Only include concerns relevant to the rank zone.

    Args:
        candidate: Full candidate dict
        today: Reference date
        rank: Final rank

    Returns:
        List of concern strings (empty if no concerns or rank zone doesn't warrant them)
    """
    signals = candidate.get('redrob_signals', {})
    history = candidate.get('career_history', [])
    profile = candidate.get('profile', {})

    concerns = []

    # Notice period concern (always relevant)
    notice_days = signals.get('notice_period_days', 0) or 0
    if notice_days > 60:
        concerns.append(f"notice {notice_days}d")

    # Response rate concern
    rr = signals.get('recruiter_response_rate', 1.0)
    if rr is not None and rr < 0.25:
        concerns.append(f"resp rate {rr:.0%}")

    # Recency concern (only for rank 31+ where it's more impactful)
    if rank > 30:
        last_str = signals.get('last_active_date')
        if last_str:
            try:
                last_active = date.fromisoformat(last_str)
                days_ago    = (today - last_active).days
                if days_ago > 90:
                    concerns.append(f"inactive {days_ago}d")
            except (ValueError, TypeError):
                pass

    # Full services career concern
    if history:
        all_consulting = all(
            any(f in (j.get('company') or '').lower() for f in CONSULTING_FIRMS)
            for j in history
        )
        if all_consulting:
            concerns.append("full services background")

    # Interview ghost concern
    icr = signals.get('interview_completion_rate', 1.0)
    if icr is not None and icr < 0.40:
        concerns.append(f"interview completion {icr:.0%}")

    return concerns


def _get_prefix(rank: int) -> str:
    """Return the rank-zone prefix string."""
    for (low, high), prefix in RANK_ZONE_PREFIXES.items():
        if low <= rank <= high:
            return prefix
    return ""


def batch_build_reasoning(
    candidates: list[dict],
    ranks:      list[int],
    today:      date | None = None,
) -> list[str]:
    """
    Build reasoning strings for a list of candidates.

    Args:
        candidates: List of candidate dicts (in rank order)
        ranks: Corresponding rank numbers (1-indexed)
        today: Reference date

    Returns:
        List of reasoning strings
    """
    if today is None:
        today = date.today()
    return [build_reasoning(c, r, today) for c, r in zip(candidates, ranks)]
