"""Pin the Step 9c urgent-park duty (#1742, mechanizing #1713).

#1742: seven sessions on 2026-07-27 each classified the same live
workflow-invariant main-red as pre-existing and emitted no routable
`urgency: main-red` candidate — the duty existed only as Step 10d prose. The
compare tool now computes the trigger mechanically (`urgent_park_required` +
the stderr `URGENT-PARK-REQUIRED:` demand line), and Step 9c carries a
numbered sub-step (1e) keyed on it. These pins keep the SKILL duty from being
silently deleted by a later Step 9c edit.
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"


def _step9c_section() -> str:
    text = issue_skill_text()
    return text[text.index("9c. Test-verdict gate") : text.index("### Step 10: Auto-complete")]


def test_step9c_carries_numbered_urgent_park_substep():
    sec = _step9c_section()
    # The duty is a NUMBERED sub-step (1e), not a floating paragraph:
    assert "e. Urgent-park duty on stripped workflow-invariant red (#1713/#1742)" in sec
    # Keyed on the mechanical trigger (JSON field + stderr demand line):
    assert "urgent_park_required" in sec
    assert "URGENT-PARK-REQUIRED:" in sec
    # The routable #1681 grammar + the bounded dedup grep are both named:
    assert "urgency:" in sec and "main-red" in sec
    assert "failing_test: <node id>" in sec
    # Disposition record shape for the epm:test-verdict note:
    assert "urgent_park: emitted" in sec
    assert "urgent_park: existing" in sec


def test_step9c_urgent_park_substep_scoping_note():
    sec = _step9c_section()
    # The 1e trigger covers the WORKFLOW_INVARIANT subset ONLY; the Step 10d
    # broad-glob #1713 duty stays binding for non-invariant reds.
    assert "WORKFLOW_INVARIANT subset ONLY" in sec
    text = issue_skill_text()
    # The Step 10d broad-glob duty itself is untouched by #1742:
    assert "Mandatory urgent-park emission on workflow-surface pre-existing red" in text
