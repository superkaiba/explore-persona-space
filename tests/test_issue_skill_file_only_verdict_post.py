"""Prose pin for the #1275 file-only Codex verdict posting instruction.

On trigger-dense review rounds the orchestrator must post the Codex
twin's verdict marker straight from its output file (sed tag-block
extraction + `post-marker --file`) — grep the verdict line only, never
page the findings-bearing body into context. Pins the canonical block in
SKILL.md (between Step 5b and Step 5c), the § Codex ensemble review
replacement sentence (post-compaction home:
`.claude/rules/codex-ensemble-review.md`), and the four marker-mode twin
pointers. Pattern: tests/test_issue_skill_guard_excerpt_brief.py
(#1231/#1252).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
# #2166: the doc-compaction commit 40653b5dcf moved the § Codex ensemble
# review prose out of CLAUDE.md into .claude/rules/codex-ensemble-review.md;
# the file-only-posting block title lives there now.
CODEX_ENSEMBLE_RULE_MD = REPO / ".claude" / "rules" / "codex-ensemble-review.md"
AGENTS = REPO / ".claude" / "agents"

BLOCK_TITLE = "File-only Codex verdict posting"

# The four MARKER-MODE twins (in-context sites are exempt; the deprecated
# codex-reviewer.md is frozen and deliberately NOT edited — #1275).
MARKER_MODE_TWINS = (
    "codex-code-reviewer.md",
    "codex-interpretation-critic.md",
    "codex-clean-result-critic.md",
    "codex-follow-up-critic.md",
)


def _step5b_to_5c_section() -> str:
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index("5b. Read both markers")
    end = text.index("5c. Apply ensemble decision rule")
    return text[start:end]


def test_step5b_file_only_verdict_block():
    section = _step5b_to_5c_section()
    assert BLOCK_TITLE in section
    # The mechanical posting recipe: extract, gate, grep the verdict, post.
    assert "post-marker" in section
    assert "--file" in section
    assert r"grep -m1 '^\*\*Verdict'" in section
    assert "trigger-dense" in section
    # Decision greps run on the EXTRACTED marker block, never the raw output.
    assert '"$MB"' in section
    # The 9b VC site greps ALL per-proposal verdict lines (no -m1).
    assert "no -m1" in section


def test_claude_md_points_at_file_only_path():
    rule_text = CODEX_ENSEMBLE_RULE_MD.read_text(encoding="utf-8")
    claude_text = CLAUDE_MD.read_text(encoding="utf-8")
    # The old wholesale-read phrasing is gone from BOTH the rule file and the
    # CLAUDE.md § Codex ensemble review summary (#2166 strengthening: the
    # retired phrasing must not be reintroduced in either home).
    assert "after reading the output file" not in rule_text
    assert "after reading the output file" not in claude_text
    assert BLOCK_TITLE in rule_text


def test_all_marker_mode_twins_reference_file_only_path():
    for name in MARKER_MODE_TWINS:
        text = (AGENTS / name).read_text(encoding="utf-8")
        assert BLOCK_TITLE in text, name
    # Site-specific grep form: the follow-up twin's verdicts are PER
    # proposal, so its pointer carries the grep-all (no -m1) form.
    followup = (AGENTS / "codex-follow-up-critic.md").read_text(encoding="utf-8")
    assert "no -m1" in followup
    # The deprecated codex-reviewer.md (frozen 2026-05-13) is explicitly
    # excluded from the marker-mode twin set — never edited by #1275.
    assert "codex-reviewer.md" not in MARKER_MODE_TWINS
