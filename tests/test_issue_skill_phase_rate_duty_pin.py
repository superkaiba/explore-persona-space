"""Prose-side pin for the Step 6d.2 same-phase rate/ETA duty (#1863).

``.claude/skills/issue/SKILL.md`` Step 6d.2 mandates that after >=3
consecutive same-phase poll ticks the orchestrator computes a progress
rate (delta-units / delta-wall) and projects a phase ETA, instead of
echoing phase-name liveness indefinitely (incident #1482: ~5.4 h of
idle-A100 billing across five consecutive 30-min ticks each reporting
"Healthy - E2 upload at shardNN"; the ~98 files/h -> ~33 h projection
surfaced only when the session finally sized it). This test pins the duty
paragraph's anchor substrings — the heading, the 3-tick trigger, the two
critic Must-Fix clauses (phase-label equivalence under advancing
counters; `log_tail_excerpt` availability under the #1841 compacted tick
parse), the `[phase-rate]` record token, the rate formula, the
no-new-marker/no-new-gate routing clause, the no-counter fallback, the
#1482 worked-example literal — and the Long-phase heartbeat item-2
cross-reference, so a prose rewording cannot silently drop the duty.

Asserts run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a pinned literal can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"

_HEADING = "**Same-phase rate/ETA duty (#1863; incident #1482).**"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (prose wraps mid-phrase)."""
    return re.sub(r"\s+", " ", text)


def _duty_span() -> str:
    """The duty paragraph, bounded below by the Per-lane reconciliation heading."""
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index(_HEADING)  # raises ValueError if the heading is gone
    end = text.index("**Per-lane planned-cell reconciliation", start)
    return _norm(text[start:end])


def test_duty_heading_and_trigger_present() -> None:
    span = _duty_span()
    assert "≥3 consecutive poll ticks report the SAME `current_phase`" in span
    assert "no `new_milestone`" in span


def test_phase_label_equivalence_clause() -> None:
    # Must-Fix 1: an advancing numeric/progress token does not break the
    # same-phase trigger — it IS the progress counter.
    span = _duty_span()
    assert "differing only in an advancing numeric/progress token" in span
    assert "that advancing token IS the progress counter" in span


def test_log_tail_excerpt_availability_clause() -> None:
    # Must-Fix 2: the compacted #1841 tick parse must additionally surface
    # `log_tail_excerpt` (or the raw JSON line is re-read) so the rate
    # read's input is actually in context.
    span = _duty_span()
    assert "ADDITIONALLY prints `log_tail_excerpt`" in span
    assert "re-reads the tick's raw JSON line" in span


def test_phase_rate_token_and_rate_formula() -> None:
    span = _duty_span()
    assert "[phase-rate]" in span
    assert "rate = Δunits / Δwall" in span


def test_no_new_marker_and_no_new_gate() -> None:
    span = _duty_span()
    assert "NO new marker kind" in span
    assert "NOT a new gate" in span
    assert "auto-continue is preserved" in span


def test_no_counter_fallback_clause() -> None:
    span = _duty_span()
    assert "no progress counter readable — liveness only" in span


def test_worked_example_1482_literals() -> None:
    span = _duty_span()
    assert "~98 files/h" in span
    assert "~33 h projection" in span


def test_heartbeat_item2_cross_reference() -> None:
    # Acceptance criterion 2: § Long-phase heartbeat duty item 2 includes
    # the [phase-rate] read on a long same-phase stretch, keyed on elapsed
    # same-phase time / heartbeat resumes (not the 3-tick count).
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index("**Long-phase heartbeat duty")
    end = text.index("**Remote-landing watches carry a producer-fence deadline", start)
    block = _norm(text[start:end])
    assert "[phase-rate]" in block
    assert "≥2 heartbeat resumes" in block
    assert "alive ≠ progressing" in block
