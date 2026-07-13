"""Pin the #1288 Step 10d merge-form protections in .claude/skills/issue/SKILL.md.

Task #1288 (2026-07-13) changed Step 10d so that:

1. The safe-case merge routes by task kind — ``MERGE_FORM=--squash`` for
   ``kind: infra|batch`` (the watcher's INFRA_DRAIN_KINDS, the population that
   same-batch races Step 10d by construction; server-side ``--rebase`` went
   0/4 first-try in the 2026-07-12 fleet), ``--rebase`` retained as the
   default / fail-open form (experiments, unreadable kind).
2. The safe-case merge call uses the ``$MERGE_FORM`` variable, not a
   hardcoded form.
3. A "Known failure shape 0" documents the transient GitHub error
   ``Base branch was modified`` (wait ~20 s, same-tip retry, max 2) instead
   of routing it into the ~16-min scratch-worktree conflict recovery.

These tests fail the suite if a later SKILL.md editor drops any of the three
protections (plan #1288 §5 durability pin).
"""

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"


def _step10d_span() -> str:
    """Return the SKILL.md text from the (unique) `### Step 10d` heading onward."""
    text = SKILL.read_text()
    return text[text.index("### Step 10d") :]


def test_merge_form_routing_present():
    span = _step10d_span()
    assert "MERGE_FORM=--rebase" in span  # default preserved (fail-open arm)
    assert 'case "$TASK_KIND" in infra|batch) MERGE_FORM=--squash ;; esac' in span


def test_safe_case_merge_uses_merge_form_variable():
    assert "gh pr merge <PR> $MERGE_FORM --delete-branch=false" in _step10d_span()


def test_known_failure_shape_0_present():
    span = _step10d_span()
    assert "Known failure shape 0" in span
    assert "Base branch was modified" in span
