"""Pin the #1572 staged-index verification duty across its five doc sites.

#1572 (2026-07-20) added a staged-index verification recipe to the
workflow-surface sites whose completion contracts stage artifact
DIRECTORIES by explicit path: a directory-path ``git add`` silently skips
gitignore-matched files inside the dir (rc=0, no error), which shipped
#958 round 7's commit without its convention-committed ``percell/*.npz``
cells. The canonical block lives in ``.claude/skills/issue/SKILL.md``
Step 9a-ter step 3; ``CLAUDE.md`` (user-chat inline carve-out),
``.claude/agents/experiment-implementer.md`` (step 10),
``.claude/agents/analyzer.md`` (figure-commit step 2), and
``.claude/skills/issue-v2/SKILL.md`` (figures/dashboards commit) carry
one-line pointers naming the canonical section.

These tests fail the suite if a later editor drops the canonical block,
moves it outside the 9a-ter region, or drops any pointer site.

NOTE for future editors: these assertions pin literal snippet text.
A legitimate rewording of the pinned lines in any of the five files must
update the matching assertions here IN THE SAME COMMIT, or the suite
goes red.
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
EXPERIMENT_IMPLEMENTER = REPO / ".claude" / "agents" / "experiment-implementer.md"
ANALYZER = REPO / ".claude" / "agents" / "analyzer.md"
ISSUE_V2 = REPO / ".claude" / "skills" / "issue-v2" / "SKILL.md"

# The canonical block heading (name-prefix the pointer sites cite).
BLOCK_HEADING = "Staged-index verification (#1572"
# The probe one-liner every site carries (asserted whitespace-normalized —
# the pointer sites wrap it across prose lines).
PROBE = "ls-files --others --ignored --exclude-standard"
# Region anchors for the canonical block (both unique in SKILL.md).
SECTION_9A_TER = "**9a-ter. Auto-run free-analysis follow-ups**"
LINT_GATE = "Inline payload lint gate (§ Inline payload lint gate"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return " ".join(text.split())


def test_skill_9a_ter_block_present():
    """The canonical block sits inside Step 9a-ter, before the lint gate."""
    text = issue_skill_text()
    assert BLOCK_HEADING in text, "canonical staged-index block heading missing from SKILL.md"
    assert PROBE in _norm(text)
    assert "git add -f" in text
    block_i = text.index(BLOCK_HEADING)
    section_i = text.index(SECTION_9A_TER)
    lint_gate_i = text.index(LINT_GATE)
    assert text.count(LINT_GATE) == 1, "lint-gate region anchor no longer unique — repin"
    assert section_i < block_i < lint_gate_i, (
        "staged-index block moved outside the 9a-ter step-3 region "
        "(must sit after the 9a-ter heading and before the inline payload lint gate)"
    )


def test_claude_md_inline_contract_pointer():
    """The user-chat inline carve-out's completion contract carries the duty."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    bullet = next(
        (
            line
            for line in text.splitlines()
            if line.startswith("- **User-chat inline free analysis**")
        ),
        None,
    )
    assert bullet is not None, "user-chat inline carve-out bullet missing from CLAUDE.md"
    assert "Staged-index verification" in bullet
    assert PROBE in _norm(bullet)


def test_experiment_implementer_duty():
    text = EXPERIMENT_IMPLEMENTER.read_text(encoding="utf-8")
    assert "Staged-index verification" in text
    assert PROBE in _norm(text)


def test_analyzer_duty():
    text = ANALYZER.read_text(encoding="utf-8")
    assert "Staged-index verification" in text
    assert PROBE in _norm(text)


def test_issue_v2_duty():
    text = ISSUE_V2.read_text(encoding="utf-8")
    assert "Staged-index verification" in text
    assert PROBE in _norm(text)
