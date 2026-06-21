"""
src/honeypot_detector.py — Detect honeypot candidates seeded by organizers.

Honeypots are forced to relevance tier 0 in the ground truth.
Any honeypot in the top-100 hurts score; >10% triggers Stage 3 disqualification.

The spec gives explicit examples:
  - "8 years of experience at a company founded 3 years ago"
  - "expert proficiency in 10 skills with 0 years used"

We implement 5 independent checks; flags are counted and a penalty multiplier
is returned (1.0 = clean, 0.0 = definite honeypot).
"""

from __future__ import annotations
from config import (
    HONEYPOT_YOE_GAP_THRESHOLD,
    HONEYPOT_EXPERT_ZERO_DUR_MIN,
    HONEYPOT_EXPERT_NO_ENDORSE_MIN,
    HONEYPOT_CONTRADICTION_MIN,
)


def detect_honeypot(candidate: dict) -> tuple[float, list[str]]:
    """
    Analyse a candidate for honeypot signals.

    Args:
        candidate: Full candidate dict from candidates.jsonl

    Returns:
        (penalty_multiplier, list_of_flags)
        penalty_multiplier: 1.0 = clean, 0.05 = very likely honeypot, 0.0 = definite honeypot
        list_of_flags: Human-readable list of triggered checks (for debugging)
    """
    flags: list[str] = []
    profile   = candidate.get('profile', {})
    history   = candidate.get('career_history', [])
    skills    = candidate.get('skills', [])
    signals   = candidate.get('redrob_signals', {})
    assessments = signals.get('skill_assessment_scores', {})

    claimed_yoe = float(profile.get('years_of_experience', 0) or 0)

    # ── Check 1: YoE inflation ─────────────────────────────────────────────
    # Career history total duration should approximately match claimed YoE.
    # Gap > 5 years is suspicious (JD example: "8 yrs exp at company founded 3 yrs ago").
    history_total_years = sum(
        (j.get('duration_months') or 0) for j in history
    ) / 12.0

    if claimed_yoe > history_total_years + HONEYPOT_YOE_GAP_THRESHOLD:
        flags.append(
            f"yoe_inflation: claimed {claimed_yoe:.1f}y vs history {history_total_years:.1f}y"
        )

    # ── Check 2: Expert skills with zero duration ──────────────────────────
    # "Expert proficiency in 10 skills with 0 years used" (from spec)
    expert_zero_dur = [
        s for s in skills
        if s.get('proficiency') == 'expert' and (s.get('duration_months') or 0) == 0
    ]
    if len(expert_zero_dur) >= HONEYPOT_EXPERT_ZERO_DUR_MIN:
        flags.append(
            f"expert_zero_duration: {len(expert_zero_dur)} expert skills with 0 months"
        )

    # ── Check 3: Mass expert skills with zero endorsements ─────────────────
    # Genuine experts accumulate some peer endorsements. A candidate with
    # 8+ expert skills and zero endorsements on all of them is implausible.
    expert_no_endorse = [
        s for s in skills
        if s.get('proficiency') == 'expert' and (s.get('endorsements') or 0) == 0
    ]
    if len(expert_no_endorse) >= HONEYPOT_EXPERT_NO_ENDORSE_MIN:
        flags.append(
            f"mass_expert_no_endorsement: {len(expert_no_endorse)} expert skills with 0 endorsements"
        )

    # ── Check 4: Assessment score contradicts declared proficiency ─────────
    # If a candidate claims "expert" on a skill but scores < 30/100 on the
    # platform's objective assessment, that's a strong contradiction signal.
    contradictions = 0
    for skill in skills:
        skill_name = skill.get('name', '')
        if skill.get('proficiency') == 'expert' and skill_name in assessments:
            if assessments[skill_name] < 30:
                contradictions += 1
    if contradictions >= HONEYPOT_CONTRADICTION_MIN:
        flags.append(
            f"assessment_contradiction: {contradictions} expert skills scored < 30/100"
        )

    # ── Check 5: Single-role career with enormous YoE gap ─────────────────
    # A candidate with only 1 job in history whose duration is far less than
    # the claimed YoE is implausible.
    if len(history) == 1 and claimed_yoe > 0:
        single_months = history[0].get('duration_months') or 0
        single_years  = single_months / 12.0
        if single_years < claimed_yoe * 0.6:
            flags.append(
                f"single_job_yoe_gap: 1 job ({single_years:.1f}y) but claims {claimed_yoe:.1f}y"
            )

    # ── Check 6: Impossible skill duration vs YoE ─────────────────────────
    # If any single skill duration_months > claimed_yoe * 12 + 24, suspicious.
    for skill in skills:
        dur = skill.get('duration_months') or 0
        if dur > (claimed_yoe * 12) + 24 and claimed_yoe > 0:
            flags.append(
                f"skill_duration_exceeds_yoe: {skill.get('name')} has {dur}mo duration "
                f"but candidate claims {claimed_yoe:.1f}y total"
            )
            break  # Only flag once

    # ── Check 7: Completion rate anomalies ────────────────────────────────
    # Extremely low interview completion rate AND high applications is suspicious.
    apps_30d     = signals.get('applications_submitted_30d', 0) or 0
    icr          = signals.get('interview_completion_rate', 1.0) or 1.0
    completeness = signals.get('profile_completeness_score', 100) or 100

    if apps_30d > 50 and icr < 0.05 and completeness < 20:
        flags.append(
            f"behavioral_impossibility: {apps_30d} apps, {icr:.0%} completion, "
            f"{completeness:.0f}% profile completeness"
        )

    # ── Penalty mapping ────────────────────────────────────────────────────
    n_flags = len(flags)
    if n_flags >= 3:
        penalty = 0.0     # Definite honeypot
    elif n_flags == 2:
        penalty = 0.05    # Very likely honeypot
    elif n_flags == 1:
        penalty = 0.40    # Suspicious — push down
    else:
        penalty = 1.0     # Clean

    return penalty, flags


def batch_detect_honeypots(candidates: list[dict]) -> tuple[list[float], list[list[str]]]:
    """
    Run honeypot detection on a list of candidates.

    Args:
        candidates: List of candidate dicts

    Returns:
        (penalties, flags_list)
    """
    penalties  = []
    flags_list = []
    for cand in candidates:
        p, f = detect_honeypot(cand)
        penalties.append(p)
        flags_list.append(f)
    return penalties, flags_list
