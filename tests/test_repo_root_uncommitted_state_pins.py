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

from tests.issue_skill_source import issue_skill_text

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
    text = _norm(issue_skill_text())
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


# ─── #2328 marker-presence pins ──────────────────────────────────────────────
# The #2328 fix makes "marker destroyed / re-append it" instructions INERT
# unless grounded in a `task.py marker-status` verdict of `absent` (two #2325
# reviewers diagnosed destruction from working-tree reads inside a stash
# window; the row was never lost). These pins keep the five prose surfaces
# from silently reverting to working-tree-read escalation. Region-scoped so a
# token elsewhere in a large file can never satisfy a pin.

CODE_REVIEWER = ROOT / ".claude" / "agents" / "code-reviewer.md"
COMPOSER_COMMON = ROOT / ".claude" / "rules" / "codex-composer-common.md"
STEP5 = ROOT / ".claude" / "skills" / "issue" / "steps" / "09-step-5.md"

_FLOOR_LINE_TOKEN = "step2-floor-skipped"
# The exact pre-#2328 instruction the MF-4 mutant check asserts ABSENT: a
# re-probe via `view` re-reads the working tree and CONFIRMS the stash-window
# false negative instead of resolving it.
_OLD_FLOOR_REPROBE = "re-probe `task.py view <N> --json`"


def _h2_region(text: str, heading: str) -> str:
    """Slice from the `## <heading>` line to the next `## ` line (exclusive)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def test_rule_file_carries_marker_presence_section():
    region = _norm(_h2_region(RULE_FILE.read_text(encoding="utf-8"), "## Marker-presence reads"))
    assert "marker-status" in region
    assert "pending-deferred" in region
    assert "unknown" in region
    assert "Only THIS verdict" in region
    assert "NEVER re-append" in region
    assert "never `2>/dev/null`" in region


def _floor_line(text: str) -> str:
    hits = [ln for ln in text.splitlines() if _FLOOR_LINE_TOKEN in ln]
    assert len(hits) == 1, f"expected exactly one {_FLOOR_LINE_TOKEN} line, got {len(hits)}"
    return hits[0]


def _floor_line_pin_violations(line: str) -> list[str]:
    """The Step-2 floor-check pin predicate; returns the violated assertions.

    Factored out so the MF-4 mutant test below can prove the pin REJECTS the
    lazy fix (the OLD view-reprobe sentence kept, ` marker-status` appended)."""
    violations: list[str] = []
    if "marker-status" not in line:
        violations.append("missing marker-status")
    if _OLD_FLOOR_REPROBE in line:
        violations.append("stale view-reprobe instruction present")
    if "treat only verdict `absent`" not in line:
        violations.append("missing absent-verdict scoping")
    if "NEVER emit a re-append/restore instruction" not in line:
        violations.append("missing re-append ban")
    return violations


def test_code_reviewer_floor_check_names_marker_status():
    line = _norm(_floor_line(CODE_REVIEWER.read_text(encoding="utf-8")))
    assert _floor_line_pin_violations(line) == []


def test_code_reviewer_floor_pin_rejects_view_reprobe_mutant():
    """MF-4 mutant validation: the OLD line resurrected with ` marker-status`
    appended must FAIL the pin — a token-presence-only pin would pass it."""
    line = _norm(_floor_line(CODE_REVIEWER.read_text(encoding="utf-8")))
    mutant = (
        line
        + " Rare deferred-commit edge case: "
        + _OLD_FLOOR_REPROBE
        + " before finalizing the FAIL. marker-status"
    )
    assert "stale view-reprobe instruction present" in _floor_line_pin_violations(mutant)


def test_codex_composer_common_carries_marker_status_bullet():
    region = _norm(
        _h2_region(COMPOSER_COMMON.read_text(encoding="utf-8"), "## Marker-presence reads")
    )
    assert "marker-status" in region
    assert "FORBIDDEN unless that verdict is `absent`" in region
    assert "quote its verdict" in region
    assert "unknown" in region


def test_step5_durable_verdict_rule_names_marker_status():
    lines = STEP5.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("**Durable-verdict-first rule"))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].startswith("**"))
    region = _norm("\n".join(lines[start:end]))
    assert "marker-status" in region
    assert "ONLY verdict `absent`" in region
    assert "INERT" in region
    assert "unknown" in region


def test_lessons_row_carries_marker_read_trigger():
    rows = LESSONS.read_text(encoding="utf-8").splitlines()
    row = next(ln for ln in rows if ln.startswith("- repo-root-uncommitted-state.md — "))
    assert "marker-status" in row


def test_claude_md_post_marker_stderr_visible():
    lines = CLAUDE_MD.read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if "sweeps the deferred line" in ln]
    assert len(hits) == 1, f"expected exactly one post-marker deferral line, got {len(hits)}"
    line = hits[0]
    assert "stderr VISIBLE" in line
    assert "`2>/dev/null`" in line
    # The stderr-visibility sentence is APPENDED after the gitleaks EXCEPT
    # clause (ends `(#1092).`) so that clause stays attached to "never re-post".
    assert line.index("(#1092)") < line.index("stderr VISIBLE")
