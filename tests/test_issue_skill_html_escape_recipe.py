"""Pin the /issue skill's HTML-escaped background-notification recipe pointer.

Background-Agent `<task-notification>` `<result>` bodies arrive HTML-escaped
by the harness (#952 v9); the recovery recipe (read the durable output file;
output-file text is CLEAN / no unescape; ONE `html.unescape()` round only on
notification-body-sourced text; transcript-JSONL last-assistant-text
extraction, #1219) lives canonically in
`.claude/skills/adversarial-planner/SKILL.md`. Sessions #1287 and #1288 each
independently re-derived it on 2026-07-13 because the /issue skill — where
background-Agent completions are consumed — carried no pointer (#1309).

These tests pin (1) the pointer paragraph in the /issue skill and (2) the
pointer's TARGET paragraphs in the adversarial-planner skill, so neither
side can silently drift (same writer/validator pattern as
tests/test_issue_skill_marker_contract.py). Sibling AP-side pin:
tests/test_adversarial_planner_output_extraction_pin.py — a future
consolidation should keep both.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ISSUE_SKILL = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
AP_SKILL = REPO / ".claude" / "skills" / "adversarial-planner" / "SKILL.md"


def _norm(text: str) -> str:
    """Collapse whitespace so line-wrapped prose matches phrase asserts."""
    return " ".join(text.split())


def test_issue_skill_names_html_escape_notification_recipe():
    text = ISSUE_SKILL.read_text(encoding="utf-8")
    norm = _norm(text)
    # The trap is named.
    assert "HTML-ESCAPED" in norm
    assert "task-notification" in norm
    # The load-bearing contract sentences.
    assert "NEVER persist notification-body text" in norm
    assert "Output-file text is CLEAN and gets NO `html.unescape()`" in norm
    assert "ONE `html.unescape()` round ONLY to notification-BODY-sourced text" in norm
    # The cross-reference to the canonical recipe (pointer, not a copy).
    assert ".claude/skills/adversarial-planner/SKILL.md" in norm
    assert "De-escape harness HTML entities before persisting" in norm
    assert "Extract the output-file text via the transcript recipe" in norm


def test_adversarial_planner_canonical_recipe_paragraphs_present():
    """The pointer's target must exist — a pointer to deleted prose is dead."""
    norm = _norm(AP_SKILL.read_text(encoding="utf-8"))
    assert "De-escape harness HTML entities before persisting" in norm
    assert "Extract the output-file text via the transcript recipe" in norm
