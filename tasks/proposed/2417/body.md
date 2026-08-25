---
title: 'Infra: version-brittle [Pp]ickle assertion reds the gate on py3.12 worktrees
  + worktree/main interpreter divergence'
kind: infra
tags: []
created_at: '2026-08-20T05:51:02Z'
has_clean_result: false
origin_prompt: 'surfaced by the #2329 q35_ladder_decay implementer (epm:experiment-implementation
  v14): test_issue2329_r2_fixes.py pickle-message regex passes on py3.11 (main checkout)
  and fails on py3.12 (worktree venv); orchestrator re-verified both the regex and
  the interpreter divergence before filing'
workflow: v1
---
---
kind: infra
---

# Infra: version-brittle `[Pp]ickle` assertion reds the test gate on py3.12 worktrees — and the worktree/main interpreter divergence that surfaced it

## Goal

Two coupled defects, both surfaced while reviewing task #2329's `q35_ladder_decay` round. Fix (1);
decide and record a disposition for (2).

**(1) A test assertion keyed on a third-party exception MESSAGE, not its type.**

`tests/test_issue2329_r2_fixes.py:650`:

```python
with pytest.raises(Exception, match=r"[Pp]ickle"):
```

CPython changed the wording of the relevant pickling error between 3.11 and 3.12, so the regex
matches under 3.11 and does not under 3.12. Realized consequence: the test
`test_atomic_writers_…` FAILS in a py3.12 worktree venv and PASSES at the main checkout under
py3.11 with the same torch. Neither the test nor its subject `_save_pt_atomic` was touched by the
#2329 round — the round merely ran the suite in a py3.12 worktree and inherited a red that has
nothing to do with its diff.

Why this is worth a task rather than a shrug: a red that is environment-dependent, unrelated to
the diff, and not reproducible at the gate's own venv is exactly the shape that teaches sessions
to discount test failures. That is the expensive failure mode, not the one-line regex.

Suggested minimal fix (surfaced by the #2329 implementer, which correctly did not self-file —
subagents never file): broaden the match to cover both wordings, e.g.
`match=r"[Pp]ickle|get local object"`. Prefer asserting on the exception TYPE plus a message
substring stable across versions; a message-only assertion on third-party text is the underlying
anti-pattern, so check whether sibling tests share it (`grep -rn 'pytest.raises(.*match=' tests/`)
and fix the class, not only this line.

**(2) Worktree venvs resolve a DIFFERENT interpreter than the main checkout.**

Measured 2026-08-20 on the shared VM:

| location | interpreter |
|---|---|
| `/home/thomasjiralerspong/explore-persona-space` (main, on `main`) | Python 3.11.15 |
| `.claude/worktrees/issue-2329-q35-ladder-decay` | Python 3.12.13 |

This is the actual root cause of (1), and it is broader than one regex. Step 9c's mapped-test gate
runs at the main checkout, while implementers and reviewers run tests inside worktrees — so the two
are not the same environment. That produces spurious reds (case (1)) and, more dangerously, can
produce spurious GREENS: a py3.11-only bug passes in a 3.12 worktree, or vice versa, and the gate
never sees it.

Decide and RECORD one disposition (this is the task's real deliverable):
- pin worktree venv creation to the same interpreter as the main checkout (likely in
  `scripts/new_worktree.sh` and/or the project's `requires-python` / `.python-version`), or
- accept the divergence deliberately and make the gate's interpreter explicit, so a
  version-dependent red is diagnosable at a glance rather than re-litigated per round.

Either way, the fix must be version-robust rather than pinned to today's two versions.

## Acceptance criteria

1. `tests/test_issue2329_r2_fixes.py::test_atomic_writers_…` passes under BOTH py3.11 and py3.12,
   and still FAILS if `_save_pt_atomic`'s actual guarantee regresses (verify by deliberately
   breaking it once — a test that now passes for the wrong reason is worse than the red).
2. The `pytest.raises(..., match=...)` sweep over `tests/` is run, and any other assertion keyed on
   third-party message text is either fixed or explicitly listed as accepted with a reason.
3. A recorded disposition for the interpreter divergence, with whatever pin or documentation that
   choice implies actually landed.
4. No new red in the no-flags `workflow_lint.py` run or the mapped-test selection.

## Provenance

Surfaced by the `experiment-implementer` during task #2329 follow-up round `q35_ladder_decay`
(marker `epm:experiment-implementation` v14, 2026-08-20), which flagged it as environment-only and
NOT payload — correctly, since neither file is round-touched. Both facts independently re-verified
by the #2329 orchestrator before filing: the interpreter table above was measured with
`uv run python -V` in each location, and the regex was read at `tests/test_issue2329_r2_fixes.py:650`.
Filed rather than left in chat so the fix is not stranded.
