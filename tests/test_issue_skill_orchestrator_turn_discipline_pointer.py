"""Prose pin for the #1563 Orchestration-Procedure orchestrator-turn pointer.

The /issue SKILL.md Orchestration Procedure preamble must point ordinary
orchestrator turns on guard-surface rounds at trigger-dense-review.md
§ Orchestrator ordinary turns, and the rule file must carry that section
heading — so a rename of either side breaks this test instead of silently
stranding the pointer (family: test_issue_skill_forensics_ingest_pointer.py).
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
RULE_MD = REPO / ".claude" / "rules" / "trigger-dense-review.md"


def test_orchestrator_turn_discipline_pointer_present():
    text = issue_skill_text()
    # .index() raises ValueError on a missing anchor — a hard fail, never a skip.
    idx = text.index("Guard-surface round: orchestrator turn discipline (#1563)")
    window = text[idx : idx + 600]
    assert "trigger-dense-review.md" in window
    assert "Orchestrator ordinary turns" in window


def test_rule_carries_orchestrator_ordinary_turns_section_heading():
    text = RULE_MD.read_text(encoding="utf-8")
    assert "## Orchestrator ordinary turns" in text
