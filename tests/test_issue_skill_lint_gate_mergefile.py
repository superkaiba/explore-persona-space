"""Pin the #1456 lint-vintage 3-way merge in the pre-push lint gate
(.claude/skills/issue/SKILL.md).

#1456 (2026-07-17) changed the #1212 gate's payload overlay so that a branch
whose own diff touches scripts/workflow_lint.py gets a 3-way-MERGED lint copy
(branch + merge-base + archived origin/main) on the GATED legs instead of the
branch's stale copy — killing the ratchet-drift false blocks (#1366/#1411) —
with a loud-WARN fallback to the branch copy on merge failure.

These tests fail the suite if a later SKILL.md editor drops the special-case,
its pre-overlay save (or the stale-copy rm -f preceding it), its fallback, or
reorders it out of the load-bearing overlay->merge->gated-legs position
(plan #1456 durability pin).

NOTE for future SKILL.md editors: these assertions pin literal snippet text
(the cp/rm lines, the WARN line, the sidecar path). A legitimate rewording of
the WARN line / cp line / temp-file names in SKILL.md must update the matching
assertions here IN THE SAME COMMIT, or the suite goes red.
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"


def _gate_span() -> str:
    """SKILL.md text from the (first) gate-tree assignment to the auto-merge heading."""
    text = issue_skill_text()
    start = text.index("GT=/tmp/issue-<N>-lint-gate-tree")
    return text[start : text.index("#### The auto-merge procedure", start)]


def test_lint_vintage_merge_present():
    span = _gate_span()
    assert "git merge-file -p" in span
    assert "/tmp/issue-<N>-lint-main-copy.py" in span
    assert "/tmp/issue-<N>-lint-base-copy.py" in span


def test_pre_overlay_save_present():
    assert 'cp "$GT/scripts/workflow_lint.py" /tmp/issue-<N>-lint-main-copy.py' in _gate_span()


def test_stale_main_copy_rm_precedes_save():
    # A prior run's stale saved main-copy must be cleared BEFORE this run's cp:
    # a cp failure under `|| true` must leave the file ABSENT (fallback fires),
    # never feed an old run's stale "theirs" into the merge.
    span = _gate_span()
    rm_idx = span.index("rm -f /tmp/issue-<N>-lint-main-copy.py")
    cp_idx = span.index('cp "$GT/scripts/workflow_lint.py" /tmp/issue-<N>-lint-main-copy.py')
    assert rm_idx < cp_idx


def test_conflict_fallback_is_loud():
    span = _gate_span()
    assert "/tmp/issue-<N>-lint-mergefile-note.txt" in span
    assert "WARN: lint-copy 3-way merge failed/conflicted" in span


def test_merge_runs_after_overlay_before_gated_legs():
    span = _gate_span()
    done_idx = span.index("done < /tmp/issue-<N>-overlay-files.txt")
    # Anchor on the #1456-specific invocation (the lint copy's own 3-way
    # merge): #1753's landing-union overlay added an in-loop
    # `git merge-file -p` BEFORE the `done` line, so the bare first
    # occurrence no longer identifies the LINT-VINTAGE block.
    merge_idx = span.index('git merge-file -p "$GT/scripts/workflow_lint.py"')
    gated_idx = span.index("GATED_RC=0")
    assert done_idx < merge_idx < gated_idx


def test_residual_a_documents_merge_and_fallback():
    span = _gate_span()
    residuals = span[span.index("**Known residuals (accepted, documented):**") :]
    assert "3-way-" in residuals
    assert "merge-failure fallback" in residuals
