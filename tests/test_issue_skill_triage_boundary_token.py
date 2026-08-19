"""Pin the SKILL.md triage-record ``(boundary=<ts>)`` token spec (#2105).

The `/issue` SKILL.md § "Pre-dispatch external-marker triage" canonical
block documents the enumeration-boundary token that closes the
enumerate-to-post seam: the mechanical enumerator snippet prints
``boundary=<ts>`` via ``triage_enumeration_boundary``, and the
recorded-line format spec carries the trailing ``(boundary=<ts>)`` token
that ``task_workflow.triage_candidates_since_last_dispatch`` parses to
reopen the candidate window from the recorded enumeration point.

Motivating incident (#2054, 2026-08-05): user-directive marker v91 landed
53 s before the r11 breadcrumb post — behind the post-position boundary at
every later call — and was invisible to rounds r11-r14 until a manual
events re-read (v98 forensics), costing a 4-round directive miss plus a
triage-correction round (v108). A later SKILL.md edit dropping the token
from the format spec (or the ``boundary=`` print from the snippet) would
silently reopen that seam; this pin keeps the two load-bearing surfaces
present.

Follows `tests/test_issue_skill_step2_floor.py`'s span-scoped
grep-anchored existence-check pattern.
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

SPAN_START = "**Pre-dispatch external-marker triage (REQUIRED"
SPAN_END = "**Detached VM-side long compute phases"


def _triage_span(body: str) -> str:
    """Return the § Pre-dispatch external-marker triage canonical block.

    The block runs from its bold header to the next bold section header
    (the detached-phases block today). Scoping the greps to this span
    keeps an unrelated later mention from satisfying the pin.
    """
    start = body.index(SPAN_START)
    end = body.index(SPAN_END, start)
    return body[start:end]


def test_skill_triage_line_spec_carries_boundary_token():
    """The recorded-line format spec documents ``(boundary=<ts>)`` — both forms."""
    span = _triage_span(issue_skill_text())
    assert "(boundary=<ts>)" in span, (
        "The § Pre-dispatch external-marker triage canonical block must "
        "document the (boundary=<ts>) token in the recorded-line format "
        "spec — without it, sessions post token-less triage lines and the "
        "enumerate-to-post seam (#2054 v91: a user directive posted 53 s "
        "before the breadcrumb, hidden for 4 rounds) silently reopens."
    )
    # Both recorded-line forms carry the token: the applied/deferred form
    # and the none form.
    assert "deferred (<one-line reasons>) (boundary=<ts>)" in span, (
        "The applied/deferred recorded-line form must carry the trailing (boundary=<ts>) token."
    )
    assert "external-markers triaged: none (boundary=<ts>)" in span, (
        "The `none` recorded-line form must carry the (boundary=<ts>) "
        "token too — `none` records close windows exactly like "
        "applied/deferred records."
    )


def test_skill_enumerator_snippet_references_boundary_helper():
    """The mechanical enumerator snippet prints the boundary via the helper."""
    span = _triage_span(issue_skill_text())
    assert "triage_enumeration_boundary" in span, (
        "The canonical enumerator snippet must reference "
        "task_workflow.triage_enumeration_boundary so the boundary value "
        "is computed mechanically from the SAME list_events() read the "
        "enumeration used — a hand-derived ts would reintroduce the "
        "enumerate-to-post skew the token exists to close (#2105)."
    )
    assert 'print("boundary=" + triage_enumeration_boundary(evs))' in span, (
        "The snippet must print the boundary= line verbatim — the format "
        "spec instructs sessions to copy <ts> verbatim from that output "
        "line."
    )
