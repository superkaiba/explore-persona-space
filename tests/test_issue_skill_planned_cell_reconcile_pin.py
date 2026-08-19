"""Pin the #1540 per-lane planned-cell reconciliation duty in the /issue skill.

Incident #1481: a multi-lane run's sycophancy bare arm was never dose-banded
and never re-swept; every lane exited cleanly, the round manifest reported
``runs_failed: []``, and the coverage gap surfaced only when the user asked —
hours after the lanes completed. Planned-vs-actual coverage reconciliation
existed only at clean-result time (After-Every-Experiment item 8 /
``verify_task_body.py`` check 11b / the clean-result-critic planned-vs-actual
lens), so a planned cell — or a plan-declared DERIVED deliverable such as a
dose-matched pair — that a completed lane/phase did not cover was invisible
DURING the run.

These tests pin, against ``.claude/skills/issue/SKILL.md``:

1. the Step 6d.2 "Per-lane planned-cell reconciliation" duty block exists and
   carries the greppable ``planned-cell-reconcile`` note prefix, the
   no-double-run contract sentence, and the derived-deliverable
   ("FIRST becomes computable") extension;
2. the Step 6d.3 run-completion backstop exists (the literal
   ``planned-cell-reconcile run-complete`` note form), placed AFTER the unique
   ``##### Step 6d.3: On `status=done``` heading and BEFORE the
   ``Transition the task to `verifying``` anchor (the bare "Step 6d.3" string
   also appears as a cross-reference inside 6d.2, so ordering anchors on the
   unique heading);
3. neither inserted block introduces an ``AskUserQuestion`` token (keeps
   ``workflow_lint.py --check-asks`` green by construction).

Prose assertions run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a required phrase can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

PER_LANE_HEADING = "**Per-lane planned-cell reconciliation"
PER_LANE_END_ANCHOR = "**`--pid-file` is a POD-side path.**"
BACKSTOP_HEADING = "**Run-completion reconciliation backstop"
STEP_6D3_HEADING = "##### Step 6d.3: On `status=done`"
TRANSITION_ANCHOR = "Transition the task to `verifying` (the upload-verifier next):"

NOTE_PREFIX = "planned-cell-reconcile"
RUN_COMPLETE_FORM = "planned-cell-reconcile run-complete"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _skill_text() -> str:
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return issue_skill_text()


def _region(text: str, start_anchor: str, end_anchor: str) -> str:
    """Slice [start_anchor, end_anchor) — both anchors must resolve, in order."""
    start = text.find(start_anchor)
    assert start != -1, f"anchor {start_anchor!r} not found in SKILL.md"
    end = text.find(end_anchor, start)
    assert end != -1, f"anchor {end_anchor!r} not found after {start_anchor!r}"
    return text[start:end]


def _per_lane_block(text: str) -> str:
    """The Step 6d.2 duty block: from its heading to the --pid-file anchor."""
    return _region(text, PER_LANE_HEADING, PER_LANE_END_ANCHOR)


def _backstop_block(text: str) -> str:
    """The Step 6d.3 backstop: from its heading to the status-flip anchor."""
    return _region(text, BACKSTOP_HEADING, TRANSITION_ANCHOR)


def test_per_lane_reconcile_block_present() -> None:
    """The Step 6d.2 duty block exists with its load-bearing pieces (#1481)."""
    block = _norm(_per_lane_block(_skill_text()))
    assert NOTE_PREFIX in block, "duty block lacks the planned-cell-reconcile note prefix"
    assert "does NOT replace the terminal clean-result reconciliation" in block, (
        "duty block lacks the no-double-run contract sentence"
    )
    # The derived-deliverable extension — without it the duty posts nothing on
    # a #1481 rerun (the actual miss was a pair-level derived outcome).
    assert "FIRST becomes computable" in block, (
        "duty block lacks the derived-deliverable ('FIRST becomes computable') extension"
    )


def test_run_complete_backstop_present_and_ordered() -> None:
    """The 6d.3 backstop exists, after the unique 6d.3 heading, before the flip."""
    text = _skill_text()
    assert text.count(STEP_6D3_HEADING) == 1, "Step 6d.3 heading is no longer unique"
    assert text.count(TRANSITION_ANCHOR) == 1, "Transition-to-verifying anchor is not unique"
    heading_pos = text.find(STEP_6D3_HEADING)
    backstop_pos = text.find(BACKSTOP_HEADING)
    transition_pos = text.find(TRANSITION_ANCHOR)
    assert backstop_pos != -1, "SKILL.md lacks the Run-completion reconciliation backstop"
    assert heading_pos < backstop_pos < transition_pos, (
        "backstop must sit inside Step 6d.3, before the `verifying` status flip"
    )
    assert RUN_COMPLETE_FORM in _norm(_backstop_block(text)), (
        "backstop lacks the literal 'planned-cell-reconcile run-complete' note form"
    )


def test_no_new_gate_vocabulary() -> None:
    """Neither inserted block carries an AskUserQuestion token (--check-asks)."""
    text = _skill_text()
    for name, block in (
        ("per-lane duty", _per_lane_block(text)),
        ("run-completion backstop", _backstop_block(text)),
    ):
        assert "AskUserQuestion" not in block, f"{name} block introduces an AskUserQuestion token"
