"""Durability pins for the #1397 unconditional fit-loop batched-helper
naming requirement (code-reviewer Step 2 paragraph + codex-twin copy-list
bullet + efficiency-critic impl-mode item 3). Live-tree pattern per
tests/test_diff_base_origin_main_pin.py; WORKFLOW_INVARIANT member."""

from pathlib import Path

_AGENTS = Path(__file__).resolve().parents[1] / ".claude" / "agents"


def test_fit_loop_batching_requirement_present_in_reviewer_files():
    reviewer = (_AGENTS / "code-reviewer.md").read_text()
    codex = (_AGENTS / "codex-code-reviewer.md").read_text()
    eff = (_AGENTS / "efficiency-critic.md").read_text()
    # Claude reviewer: the paragraph + the verdict-line label + both escapes.
    assert "Fit-loop batched-helper naming" in reviewer
    assert "Fit-loop batching:" in reviewer
    assert "not-batchable" in reviewer
    # Codex twin: the copy-list bullet names the paragraph + the label.
    assert "Fit-loop batched-helper naming" in codex
    assert "Fit-loop batching:" in codex
    # v2 impl-mode owner mirrors the positive duty.
    assert "Fit-loop batching:" in eff


def test_fit_loop_batching_semantics_span_scoped():
    """Span-scoped semantics pins (Stats-critic Must-Fix, plan v3): a
    file-level `in` would be VACUOUS -- `substantive`/`Major` appear
    elsewhere in these files. Pins the UNCONDITIONAL trigger + the
    absence-is-Major/`substantive` severity (the Goal's 'a FAIL, not a
    note') so a later trim cannot demote the check while label pins pass."""
    reviewer = (_AGENTS / "code-reviewer.md").read_text()
    codex = (_AGENTS / "codex-code-reviewer.md").read_text()
    eff = (_AGENTS / "efficiency-critic.md").read_text()
    para = reviewer.split("Fit-loop batched-helper naming", 1)[1][:2500]
    assert "UNCONDITIONAL" in para
    assert "Major" in para and "`substantive`" in para
    assert "never double-FAIL" in para
    bullet = codex.split("Fit-loop batched-helper naming", 1)[1][:1200]
    assert "Major" in bullet and "substantive" in bullet
    eff_span = eff.split("POSITIVE duty", 1)[1][:800]
    assert "absence is a FAIL, not a note" in eff_span
