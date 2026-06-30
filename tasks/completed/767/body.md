---
title: 'workflow-fix: concern store commits inside flock transaction (atomic)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a8b84ea3a221
created_at: '2026-06-30T12:53:51Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #763 round 2 implementer workflow-fix-candidate v1
  — concurrent repo-root rebase wiped concerns.jsonl mid-round; fix is to commit concerns.jsonl
  inside the existing task.py raise-concern/address-concern/defer-concern flock+commit
  transaction'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #763 round 2 (emitting agent: experiment-implementer).

## Goal

Make `raise-concern` / `address-concern` / `defer-concern` commit the `concerns.jsonl` write inside their existing flock+commit transaction so a concern record is durable the instant it is written and cannot be lost to a concurrent repo-root rebase.

## Workflow gap

- **Bug observed:** A concurrent repo-root `git pull --rebase --autostash` (run by a parallel fleet session) wiped #763's round-1 `concerns.jsonl` mid-round because `raise-concern` / `address-concern` write the concern store to the SHARED repo-root working tree and the round-1 reviewer's records were never committed. All 5 round-1 concern records (3 BLOCKER + 2 CONCERN raised by the reconciler) vanished and had to be reconstructed by hand from the cached reconcile body at `/tmp/issue-763-reconcile.md` before round 2 could `address-concern` them.
- **Why it is a workflow gap:** `task.py raise-concern` / `address-concern` / `defer-concern` mutate `tasks/<status>/<N>/concerns.jsonl` but do NOT commit it in the same flock-guarded mutation (unlike most `task.py` mutations such as `set-status` / `post-marker` which `git mv` + commit atomically). This leaves the durable concern record vulnerable to any concurrent session's autostash/reset on the always-dirty shared root — the exact "concurrent repo-root committers" hazard CLAUDE.md warns about, here applied to the concern store. Concerns are load-bearing in code-review ensemble FAIL routing (Step 5c-ter binding-concerns post-strip check) and silently losing them lets a reviewer's FAIL-class blockers be quietly forgotten across rounds.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```python
# in task_workflow.raise_concern / address_concern / defer_concern, after writing concerns.jsonl:
+ _git_commit_paths([concerns_path], f"task #{n}: concern {event} {concern_id}")
# (mirror the flock-held commit the other mutating subcommands already perform)
```

## Scope / surfaces

- Primary target: `scripts/task.py` (alias path resolves to `src/explore_persona_space/task_workflow.py` for the concern handlers).
- Grep the workflow surface for the pattern before editing: `grep -rln 'raise_concern\|address_concern\|defer_concern' scripts/ src/explore_persona_space/task_workflow*.py tests/`. Mirror the existing `_git_commit_paths` patterns used by `set-status` / `post-marker` / `set-body`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The commit must hold the existing flock (no second flock); the existing transaction shape stays intact.
- Backwards compat: a concerns.jsonl that already exists uncommitted on disk should not be clobbered; the new commit appends to git, not the disk.
- `scripts/workflow_lint.py` passes; ruff on touched files passes; the existing `tests/test_task_workflow*.py` covering concerns must still pass + a new test exercising the concurrent-rebase race should be added.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/task.py
- fingerprint: a8b84ea3a221
- emitting_task: #763 (round 2)
- emitting_agent: experiment-implementer

<!-- workflow-fix-candidate v1 -->
target_file: scripts/task.py
bug_observed: A concurrent repo-root `git pull --rebase --autostash` (run by a parallel session) wiped #763's round-1 `concerns.jsonl` mid-round because `raise-concern`/`address-concern` write the concern store to the SHARED repo-root working tree and the round-1 reviewer's records were never committed; all 5 concern records vanished and had to be reconstructed by hand from the cached reconcile body.
why_workflow_gap: `task.py raise-concern`/`address-concern` mutate `tasks/<status>/<N>/concerns.jsonl` but do NOT commit it in the same flock-guarded mutation (unlike most `task.py` mutations which `git mv` + commit atomically), leaving the durable concern record vulnerable to any concurrent session's autostash/reset on the always-dirty shared root — the exact "concurrent repo-root committers" hazard CLAUDE.md warns about, here applied to the concern store.
proposed_change: Make `raise-concern`/`address-concern`/`defer-concern` commit the `concerns.jsonl` write inside their existing flock+commit transaction (the same pattern `set-status`/`post-marker` already use), so a concern record is durable the instant it is written and cannot be lost to a concurrent rebase.
diff_sketch: |
  # in task_workflow.raise_concern / address_concern / defer_concern, after writing concerns.jsonl:
  + _git_commit_paths([concerns_path], f"task #{n}: concern {event} {concern_id}")
  # (mirror the flock-held commit the other mutating subcommands already perform)
confidence: medium
related_task: #763
<!-- /workflow-fix-candidate -->
