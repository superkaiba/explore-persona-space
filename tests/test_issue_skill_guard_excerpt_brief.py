"""Prose pin for the #1231 Step 5a guard-surface excerpt-file instruction.

SKILL.md Step 5a must instruct the orchestrator to pre-materialize an
excerpt file for trigger-dense review rounds and name it + a read budget
in both reviewer briefs (arms trigger-dense-review.md discipline 3).
"""

from pathlib import Path

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"


def _step5a_section() -> str:
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index("5a. Spawn both reviewers")
    end = text.index("5b. Read both markers")
    return text[start:end]


def test_step5a_pre_materializes_trigger_dense_excerpts():
    section = _step5a_section()
    assert "pre-materialize" in section.lower()
    assert "trigger-dense-review.md" in section
    assert "/tmp/issue-<N>-r<round>-excerpts-" in section
    # The brief line carries a stated read budget (discipline 3 shape).
    assert "120-line" in section
    # Acceptance element (d): non-trigger-dense rounds skip explicitly.
    assert "non-trigger-dense rounds: skip" in section.lower()


def test_step5a_brief_carries_return_text_contract():
    """#1252: the trigger-dense briefs carry the discipline-4 return-text line."""
    section = _step5a_section()
    assert "return_text:" in section
    assert "no findings recap" in section
    assert "discipline 4" in section


def test_reconciler_brief_forwards_excerpt_path():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Step 5a excerpt-file path" in text
