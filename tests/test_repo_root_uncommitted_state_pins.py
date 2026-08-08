"""Prose-durability pins for the #2015 repo-root uncommitted-state guidance.

Incident #1768/#2015: uncommitted tracked edits and deletions at the shared
repo root revert to committed content within seconds during concurrent
sessions' commits (pre-commit's repo-wide stash/checkout/restore cycle,
`pre_commit/staged_files_only.py`), silently losing work. The #2015 fix
lands the warning + landing-verification recipe + diagnostic tell in FOUR
prose surfaces; these pins keep a later rewrite from silently dropping any
of them (#884/#1045/#1134 droppable-prose lineage):

(a) CLAUDE.md § Concurrent repo-root committers — the always-on summary
    (stash-mechanism sentence, `git show <sha>:<path>` landing check, the
    rule-file pointer);
(b) `.claude/skills/issue/SKILL.md` § 9a-ter — the inline-round
    "Uncommitted-exposure window" copy;
(c) `.claude/rules/repo-root-uncommitted-state.md` — the on-demand full
    mechanics (must cite pre-commit's `staged_files_only.py`);
(d) `.claude/rules/LESSONS.md` — the always-on index row.

Presence checks only — no fragile exact counts (plan #2015 D5). Assertions
are whitespace-normalized substring checks (the prose is line-wrapped), per
the pin-family convention (see tests/test_issue_skill_stash_kept_duty_pin.py).

NOTE for future editors: a legitimate rewording must update the pinned
substrings below IN THE SAME DIFF.

Paths resolve via ``Path(__file__)`` — NEVER ``task_workflow.repo_root()``,
which reads the MAIN checkout and would miss worktree edits pre-merge.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
RULE_FILE = ROOT / ".claude" / "rules" / "repo-root-uncommitted-state.md"
LESSONS = ROOT / ".claude" / "rules" / "LESSONS.md"


def _norm(text: str) -> str:
    """Collapse all whitespace so line-wrapped prose matches substrings."""
    return re.sub(r"\s+", " ", text)


def test_claude_md_concurrent_committers_carries_the_warning():
    text = _norm(CLAUDE_MD.read_text(encoding="utf-8"))
    # The D1 bold lead-in + the stash-mechanism sentence.
    assert "Uncommitted TRACKED state at the shared root is unsafe under concurrency" in text
    assert "pre-commit stash" in text
    # The landing-verification recipe (blob read at the SHA, never the push line).
    assert "git show <sha>:<path>" in text
    # The on-demand rule-file pointer.
    assert "repo-root-uncommitted-state.md" in text


def test_9ater_uncommitted_exposure_block_present():
    """The durability pin named in plan #2015 D5 / §10: the 9a-ter
    inline-round copy of the exposure-window guidance."""
    text = _norm(ISSUE_SKILL.read_text(encoding="utf-8"))
    assert "Uncommitted-exposure window" in text
    assert "git show <pushed-sha>:<path>" in text
    assert "repo-root-uncommitted-state.md" in text


def test_rule_file_exists_and_names_the_mechanism():
    assert RULE_FILE.is_file()
    text = _norm(RULE_FILE.read_text(encoding="utf-8"))
    assert "staged_files_only.py" in text
    # The reproduction pointer + the escalate-only watcher pass contract.
    assert "scripts/repro_precommit_stash_race.sh" in text
    assert "root_unstaged_audit_pass" in text


def test_lessons_index_carries_the_row():
    rows = LESSONS.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("- repo-root-uncommitted-state.md — ") for line in rows), (
        "LESSONS.md must carry the repo-root-uncommitted-state.md index row"
    )
