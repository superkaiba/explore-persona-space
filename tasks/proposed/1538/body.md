---
title: 'workflow-fix: guard grep-pattern literal false positive'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b46fbf4ac8ed
- daily-auto-filed
created_at: '2026-07-19T07:07:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): The repo-root branch guard
  blocked a read-only grep whose quoted PATTERN contained a destructive-op literal
  (git reset --hard) because the grep-waiver only fires when the quoted payload is
  the clause final token; a trailing log-file path defeated it (c5-P12, 17:00Z after
  #1501).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c5-P12). Route-2 filing.

## Goal

Extend the repo-root branch guard so a read-only `grep`/`egrep`/`fgrep`/`rg`
clause whose quoted PATTERN argument contains a destructive-op literal is
waived even when a trailing file path follows the pattern (the destructive
text lives inside the search pattern, not an executable git verb).

## Workflow gap

- **Bug observed:** at 17:00Z on 2026-07-18 (AFTER #1501's heredoc-FP fix
  merged) the guard BLOCKED a read-only
  `grep -n -m 3 -A 20 "test_worktree_revert_shapes_block[git reset --hard]" /tmp/step9c-pytest-issue-1513.log`
  because the quoted pattern carried the literal `git reset --hard`.
- **Why it is a workflow gap:** the guard already waives grep/egrep/rg
  pattern arguments, but only when the quoted payload is the clause's FINAL
  token; a trailing log-file path after the pattern defeats that narrowing,
  so a read-only grep with a destructive-op literal in its pattern and a
  file argument fails CLOSED. This is a new false-positive class distinct
  from #1501's heredoc payloads.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -cn 'FINAL token' scripts/guard_repo_root_branch.sh` → 2 hits (the grep/rg pattern-arg waiver is scoped to the quoted-payload-is-final-token shape; a trailing file path is not the final token so the waiver does not fire); `git log --oneline --since='3 days ago' -- scripts/guard_repo_root_branch.sh` → 4 commits (d534bdf299 #1501 heredoc, c55596834c, 124e9bec60, a87f5898f6), none widening the grep-pattern-arg waiver to allow a trailing file path (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# in the grep/egrep/fgrep/rg read-only waiver span:
# CURRENT: waive only when the quoted pattern is the clause's FINAL token
# NEW: waive a grep-family clause whose destructive literal is confined to a
#      quoted PATTERN argument even when non-option file-path tokens trail it
#      (still refuse an unquoted destructive git verb anywhere in the clause).
```

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the guard's pattern-arg waiver span; extend the fail-closed
  quoted-separator handling to permit trailing file-path tokens after a
  grep-family quoted pattern. Add a self-test / test fixture for the
  grep-with-trailing-file shape (fails pre-fix, passes post-fix).

## Constraints / invariants

- Workflow-surface only. Must NOT loosen the guard for an unquoted
  destructive `git reset --hard` / `git checkout` verb; the widening is
  scoped strictly to grep/egrep/fgrep/rg pattern-argument literals.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; the guard's existing self-test matrix stays green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 81ba2f948208

Surfaced problem (c5-P12, 2026-07-18 17:00Z): a read-only grep over a step9c
pytest log was guard-blocked because its quoted search pattern contained
`git reset --hard` and a trailing log-file path followed the pattern.
