"""Pin adversarial-planner SKILL.md Phase 1.5.0 per-WARN disposition contract (#1734).

The `/adversarial-planner` skill Phase 1.5.0 mechanical pre-pass turns
`verify_plan.py` WARNs into a per-WARN disposition (Resolve or Carry
with a one-line reason), the bare word `benign` is BANNED, and
`c34_ratchet_headroom` is never a carry candidate. Dispositions ride
the plan (a `## WARN dispositions` sub-block under Reproducibility
Card §10), not the marker.

Motivating incident: session `6b3fca14` (2026-07-26T07:20:55Z)
narrated both of its round's WARNs as "both benign"; the same-file
ratchet pressure one of those WARNs named (`c34_ratchet_headroom`)
produced a `scripts/workflow_lint.py` merge conflict roughly 12 min
into the review round. The old "copy any OTHER WARN lines verbatim
into the fact-checker brief ... as 'mechanical pre-pass notes'"
clause is REMOVED — brief-only carry evaporates when the next round's
brief is composed fresh.

Preservation co-grep (methodology critic M5, plan §4.4.2 last
bullet): the new WARN-disposition bullet REPLACES the surrounding
"PASS (with WARNs)" bullet, so a future edit could strip the
`c23_goal_currency` bounce carve-out clause while preserving the
per-WARN disposition language. The pin below ALSO asserts that the
`c23_goal_currency` bounce clause remains in the same block — a
defense-in-depth co-grep against a nearby-edit regression.

Follows `tests/test_issue_skill_marker_contract.py`'s grep-anchored
existence-check pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADVERSARIAL_PLANNER_SKILL = ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md"


def _phase150_span(body: str) -> str:
    """Return the Phase 1.5.0 span around the new WARN-disposition bullet.

    The bullet lives inside the Phase 1.5.0 mechanical pre-pass block.
    We anchor at the new bullet's opening phrase and slice ~3000 chars
    forward — enough to cover the multi-paragraph disposition contract
    without pulling in unrelated later sections that may echo similar
    vocabulary.
    """
    marker = "PASS (with WARNs)"
    if marker not in body:
        return ""
    start = body.index(marker)
    return body[start : start + 3000]


# ── The new bullet header (asserts the anti-pattern was REMOVED) ────────────


def test_phase150_new_header_per_warn_disposition_present():
    """Phase 1.5.0 must carry the 'PASS (with WARNs) → proceed with a per-WARN disposition' header.

    This is the load-bearing rename from the old 'proceed' single-verb
    header; its presence proves the anti-pattern header is gone.
    """
    body = ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8")
    assert "PASS (with WARNs) → proceed with a per-WARN disposition" in body, (
        "adversarial-planner SKILL.md Phase 1.5.0 must carry the "
        "'PASS (with WARNs) → proceed with a per-WARN disposition' "
        "header — the load-bearing rename that turns the old 'carry as "
        "pre-pass notes' anti-pattern into an actionable per-WARN "
        "call. Without the rename, sessions default to the old "
        "carry-verbatim shape (session `6b3fca14`, 2026-07-26)."
    )


# ── The two-alternative structure (Resolve vs Carry) ────────────────────────


def test_phase150_two_alternative_structure_present():
    """The per-WARN disposition must offer two alternatives: Resolve or Carry."""
    span = _phase150_span(ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8"))
    assert span, "Phase 1.5.0 span not found"
    assert "**Resolve**" in span, (
        "Per-WARN disposition must offer 'Resolve' as one of the two "
        "alternatives (amend the plan and the WARN drops on next "
        "verify_plan.py run)."
    )
    assert "**Carry**" in span, (
        "Per-WARN disposition must offer 'Carry' as the second "
        "alternative — with a one-line mechanism-level reason."
    )


# ── The banned-word clause (the specific-defect standard) ───────────────────


def test_phase150_bare_benign_banned():
    """The bare word 'benign' must be BANNED as a carried-WARN reason."""
    span = _phase150_span(ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8"))
    assert "bare word `benign` is BANNED" in span, (
        "Phase 1.5.0 must ban the bare word `benign` as a "
        "carried-WARN reason. That word is the observed anti-pattern "
        "from session `6b3fca14` (2026-07-26) that predicted the "
        "same-file ratchet merge conflict ~12 min later."
    )


# ── The c34_ratchet_headroom never-benign clause ────────────────────────────


def test_phase150_c34_ratchet_headroom_never_carry_candidate():
    """`c34_ratchet_headroom` must NEVER be a carry candidate."""
    span = _phase150_span(ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8"))
    assert "`c34_ratchet_headroom` is NEVER a carry candidate" in span, (
        "Phase 1.5.0 must state `c34_ratchet_headroom` is NEVER a "
        "carry candidate — it is the deterministic predictor of a "
        "same-file cap collision, so it must be resolved (raise the "
        "size ratchet or split the insert)."
    )


# ── The plan-side location decision ─────────────────────────────────────────


def test_phase150_plan_side_warn_dispositions_block():
    """The dispositions must ride the PLAN (`## WARN dispositions` sub-block), not the marker."""
    span = _phase150_span(ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8"))
    assert "## WARN dispositions" in span, (
        "Per-WARN dispositions must ride a `## WARN dispositions` "
        "sub-block in the plan — a brief-only carry evaporates when "
        "the next round's brief is composed fresh from the latest "
        "plan version, which is the current failure mode this "
        "contract exists to fix."
    )


# ── The REMOVED phrase: the old anti-pattern must be gone ───────────────────


def test_phase150_old_carry_verbatim_clause_removed():
    """The old 'copy any OTHER WARN lines verbatim into the fact-checker brief' clause must be GONE.

    This is a regression bar: if a future edit restores the old shape,
    this test FAILs. The whole point of #1734 is to move dispositions
    from the transient brief onto the durable plan.
    """
    body = ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8")
    forbidden = "copy any OTHER WARN lines verbatim into the fact-checker brief"
    assert forbidden not in body, (
        f"The old anti-pattern phrase {forbidden!r} must NOT appear "
        "in adversarial-planner SKILL.md — a brief-only carry loses "
        "the disposition on the next brief compose. Dispositions "
        "ride the plan (`## WARN dispositions` sub-block), not the "
        "transient brief."
    )


# ── Preservation co-grep: c23_goal_currency bounce clause survives (M5) ─────


def test_phase150_c23_goal_currency_bounce_preserved():
    """`c23_goal_currency` bounce clause must survive in the same block (M5).

    The new WARN-disposition bullet REPLACES the surrounding 'PASS
    (with WARNs)' bullet, so a future edit could strip the
    goal-currency bounce carve-out clause while preserving the new
    per-WARN language. This co-grep defends against that
    nearby-edit regression.
    """
    span = _phase150_span(ADVERSARIAL_PLANNER_SKILL.read_text(encoding="utf-8"))
    m = re.search(r"c23_goal_currency.*(?:bounce|mechanical redraft)", span, flags=re.DOTALL)
    assert m is not None, (
        "The `c23_goal_currency` bounce clause must remain in the "
        "same Phase 1.5.0 block as the per-WARN disposition — it is "
        "the ONE WARN that bounces (the mechanical redraft carve-out "
        "per § Goal-currency gate), and losing it re-opens the #922 "
        "class of stale-Goal plan persists."
    )
