"""Prose-side pin for the Step 6d.2 tick-parse field-preservation mandate (#1841).

``.claude/skills/issue/SKILL.md`` Step 6d.2 mandates that any
compacted/filtered parse of a poll-tick JSON line print the FULL decision
field set — a status-only parse structurally discards the advisory fields
the handling sections branch on (incident #1768, 2026-07-29: a status-only
compact tick parse dropped a posted [gpu-idle-escalation]; ~15h of idle
8xH100 was heartbeated as healthy). This test pins (i) the paragraph, (ii)
all 10 mandated field names inside it, (iii) the status-only-parse BAN
phrase, and (iv) the pseudocode pointer at the "Read the JSON line" comment,
so a prose rewording cannot silently drop the mandate.
"""

from __future__ import annotations

from pathlib import Path

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"

_HEADING = "Tick-parse field-preservation (REQUIRED — #1841; incident #1768)."

# Ground truth: scripts/poll_pipeline.py PollResult + its JSON-line
# serialization dict (the poller's tick output). next_interval is the
# quiet-wait branch key per the ADAPTIVE POLL INTERVAL block (#1924).
_MANDATED_FIELDS = (
    "status",
    "current_phase",
    "gate",
    "stall_reason",
    "new_milestone",
    "next_interval",
    "gpu_idle_advisory_posted",
    "gpu_idle_escalation_posted",
    "gpu_width_advisory_posted",
    "eta_deviation_posted",
)


def _paragraph_span() -> str:
    """The mandate paragraph + its canonical one-liner fence (bounded below
    by the Forensics-ingest paragraph that follows it in Step 6d.2)."""
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index(_HEADING)
    end = text.index("Forensics-ingest discipline (#1546)", start)
    return text[start:end]


def test_tick_parse_field_preservation():
    """The paragraph exists and names every mandated decision field."""
    span = _paragraph_span()  # raises ValueError if the heading is gone
    missing = [f for f in _MANDATED_FIELDS if f"`{f}`" not in span and f not in span]
    assert not missing, f"mandated fields missing from the paragraph: {missing}"


def test_status_only_parse_ban_phrase_present():
    span = _paragraph_span()
    assert "status-only parse is BANNED" in span, (
        "the status-only-parse BAN phrase was dropped from the mandate paragraph"
    )


def test_canonical_one_liner_uses_d_get():
    # The one-liner must degrade absent fields to None via d.get(...) —
    # a mixed-vintage worktree poller may omit newer fields (KeyError-free).
    span = _paragraph_span()
    assert "d.get(k)" in span, "canonical one-liner lost its d.get(...) degradation"


def test_pseudocode_pointer_line_present():
    # The polling-loop comment must point readers at the mandate BEFORE the
    # decision branches — the "invites compaction" gap #1768 fell through.
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "parse per § Tick-parse" in text, (
        "the polling-loop pseudocode comment lost its Tick-parse pointer"
    )
