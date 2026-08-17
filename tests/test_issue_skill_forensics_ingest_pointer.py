"""Prose pin for the #1546 Step 6d.2 forensics-ingest pointer.

The /issue SKILL.md polling loop (Step 6d.2) must point orchestrator-side
run-failure ingestion at trigger-dense-review.md § Orchestrator
poll/forensics turns, and the rule file must carry that section heading —
so a rename of either side breaks this test instead of silently stranding
the pointer (family: test_issue_skill_exit_breadcrumb.py,
test_issue_skill_guard_excerpt_brief.py).
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
RULE_MD = REPO / ".claude" / "rules" / "trigger-dense-review.md"


def test_step6d2_forensics_ingest_pointer_present():
    text = issue_skill_text()
    idx = text.index("Forensics-ingest discipline (#1546)")
    window = text[idx : idx + 600]
    assert "Orchestrator poll/forensics turns" in window
    assert "trigger-dense-review.md" in window


def test_rule_carries_orchestrator_ingest_section_heading():
    text = RULE_MD.read_text(encoding="utf-8")
    assert "## Orchestrator poll/forensics turns" in text
