"""Pin test for the adversarial-planner background-Agent output-file transcript recipe (#1270).

The adversarial-planner skill documents how to extract a background planner's
final text from an Agent-task ``.output`` file, which is a conversation-transcript
JSONL (no ``{"type": "result"}`` row; trailing metadata-bearing rows). Incident
#1219 (2026-07-10): the first extraction scanned for a result row that does not
exist in that format and exited "NO RESULT ROW FOUND". This test EXECUTES the
published fence against a synthetic #1219-shaped transcript so later prose edits
cannot silently break the recipe, and pins the paragraph's load-bearing prose.
"""

import json
from pathlib import Path

import pytest

SKILL_PATH = (
    Path(__file__).resolve().parents[1] / ".claude" / "skills" / "adversarial-planner" / "SKILL.md"
)

PARAGRAPH_HEADER = "**Extract the output-file text via the transcript recipe"
USAGE_LINE_MARKER = "\ntext = last_assistant_text"
# The trailer-strip regex literal shared with the earlier trailer-strip fence.
TRAILER_REGEX_LITERAL = r"agentId:\s*\S+\s*\(use SendMessage"

# --- fixture rows replicating the #1219 transcript shape -------------------

ROW_EARLIER_ASSISTANT = {
    "type": "assistant",
    "message": {"content": [{"type": "text", "text": "draft v0"}]},
}
ROW_USER_TOOL_RESULT = {
    "type": "user",
    "message": {"content": [{"type": "tool_result", "content": "ok"}]},
}
ROW_FINAL_ASSISTANT = {
    "type": "assistant",
    "message": {
        "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "true"}},
            {"type": "text", "text": "FINAL PLAN TEXT"},
            {"type": "text", "text": ""},
        ]
    },
}
# Trailing assistant-typed metadata row: agentId/attribution keys, no text blocks
# (the #1219 last-row key list).
ROW_TRAILING_METADATA = {
    "type": "assistant",
    "agentId": "abc123",
    "attributionAgent": "planner",
    "attributionSkill": "adversarial-planner",
    "message": {"content": []},
}
# Trailing non-assistant row (the shape observed at #1219 transcript L106).
ROW_TRAILING_NON_ASSISTANT = {"type": "last-prompt", "prompt": "n/a"}


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _recipe_paragraph_and_fence() -> tuple[str, str]:
    """Return (paragraph prose, python fence body) for the transcript recipe.

    Uses ``str.index`` throughout so a missing anchor fails loud (ValueError)
    instead of silently passing on a restructured skill file.
    """
    text = _skill_text()
    para_start = text.index(PARAGRAPH_HEADER)
    fence_open = text.index("```python", para_start)
    paragraph = text[para_start:fence_open]
    fence_body_start = text.index("\n", fence_open) + 1
    fence_close = text.index("```", fence_body_start)
    return paragraph, text[fence_body_start:fence_close]


def _exec_recipe_definition() -> dict:
    """Exec the published fence's definition part (everything before the usage line)."""
    _, fence = _recipe_paragraph_and_fence()
    assert USAGE_LINE_MARKER in fence, "usage line missing from the published fence"
    definition = fence.split(USAGE_LINE_MARKER)[0]
    namespace: dict = {}
    exec(compile(definition, str(SKILL_PATH), "exec"), namespace)
    assert "last_assistant_text" in namespace
    return namespace


def _write_fixture(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "agent-task.output"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_fence_extracts_last_assistant_text(tmp_path: Path) -> None:
    """The published fence returns the LAST text-bearing assistant row's text.

    The fixture carries an earlier assistant text row (must not be returned),
    a user tool_result row, a blank line (exercises the JSONDecodeError skip),
    the final text-bearing assistant row, a trailing assistant-typed metadata
    row with NO text blocks, and a trailing non-assistant row -- the exact
    trailing-rows shape that broke #1219's first extraction.
    """
    namespace = _exec_recipe_definition()
    fixture = _write_fixture(
        tmp_path,
        [
            json.dumps(ROW_EARLIER_ASSISTANT),
            json.dumps(ROW_USER_TOOL_RESULT),
            "",
            json.dumps(ROW_FINAL_ASSISTANT),
            json.dumps(ROW_TRAILING_METADATA),
            json.dumps(ROW_TRAILING_NON_ASSISTANT),
        ],
    )
    # "\n".join(["FINAL PLAN TEXT", ""]) — the empty text block joins to a trailing newline.
    assert namespace["last_assistant_text"](str(fixture)) == "FINAL PLAN TEXT\n"


def test_fence_raises_on_no_assistant_text(tmp_path: Path) -> None:
    """No text-bearing assistant row anywhere -> the fence fails loud (SystemExit)."""
    namespace = _exec_recipe_definition()
    fixture = _write_fixture(
        tmp_path,
        [
            json.dumps(ROW_USER_TOOL_RESULT),
            json.dumps(ROW_TRAILING_METADATA),
            json.dumps(ROW_TRAILING_NON_ASSISTANT),
        ],
    )
    with pytest.raises(SystemExit):
        namespace["last_assistant_text"](str(fixture))


def test_prose_shape() -> None:
    """Pin the paragraph's load-bearing prose + the trailer-regex sync invariant."""
    paragraph, fence = _recipe_paragraph_and_fence()
    # (i) the negative statement about {"type": "result"} rows
    assert '{"type": "result", "result": "<str>"}' in paragraph
    # (ii) the trailing metadata-row key the #1219 diagnostic surfaced
    assert "attributionAgent" in paragraph
    # (iii) the output file is a symlink to the transcript JSONL, not raw text
    assert "SYMLINK" in paragraph
    # (iv) the fence applies the SAME trailer-strip regex as the existing
    # trailer-strip snippet earlier in the file: the literal appears in this
    # fence AND at least once elsewhere (>= 2 occurrences keeps them in sync).
    assert TRAILER_REGEX_LITERAL in fence
    assert _skill_text().count(TRAILER_REGEX_LITERAL) >= 2
