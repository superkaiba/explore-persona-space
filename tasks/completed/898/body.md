---
title: 'workflow-fix: index.lock-resilient _run_git + crash-safe set_status registry
  move'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5fc285801c9c
created_at: '2026-07-02T22:55:01Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #825: set-status crashed on concurrent
  index.lock leaving folder/registry split; see body Provenance block'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #825 (emitting agent: orchestrator).

## Goal

Make `task_workflow` git operations resilient to concurrent `.git/index.lock` collisions and make `set_status`'s folder-move + registry update crash-safe.

## Workflow gap

- **Bug observed:** `task.py set-status 825 verifying interpreting` crashed with `CalledProcessError` (git add exit 128) while a concurrent session held `.git/index.lock`, leaving a PARTIAL mutation: the task folder was already moved on disk to `tasks/interpreting/825` but `REGISTRY.json` still pointed at `tasks/verifying/825` and no commit landed. Every subsequent `task.py find/view 825` raised `FileNotFoundError` until a manual `task.py audit --repair --apply`.
- **Why it is a workflow gap:** CLAUDE.md § "Concurrent repo-root committers" instructs HUMANS to "retry once on an index.lock collision", but `task_workflow._run_git` has no such retry, and `set_status` performs the disk move and the registry/commit non-atomically — a mid-mutation crash strands the task in an unresolvable state on a fleet where many sessions commit concurrently (this WILL recur).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ # task_workflow._run_git: on CalledProcessError with stderr matching
+ # 'index.lock|Unable to create .* File exists', sleep 2-5s and retry ONCE.
+ # set_status: perform the registry update in the same critical section as
+ #   the move; on any git failure after the move, roll the move back OR
+ #   complete the registry update before raising (no stranded partial state).
+ # find_task_path: when the registry path is missing on disk, fall back to a
+ #   one-shot on-disk scan (tasks/*/<N>) before raising, and log the drift.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rn '_run_git\|index.lock' src/explore_persona_space/task_workflow.py scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 5fc285801c9c

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: task.py set-status crashed with git add exit 128 on a concurrent .git/index.lock, leaving folder moved but REGISTRY.json stale; find/view raised FileNotFoundError until manual audit --repair --apply
why_workflow_gap: _run_git has no retry on the well-known index.lock collision class (CLAUDE.md instructs humans to retry once) and set_status's move+registry+commit is not crash-safe on a fleet of concurrent committers
proposed_change: retry git commands once on index.lock collision in _run_git and make set_status registry-move crash-safe (registry-stale fallback in find_task_path)
diff_sketch: |
  + _run_git: retry once (2-5s backoff) on exit-128 index.lock errors
  + set_status: registry update + move in one critical section; no stranded partial state
  + find_task_path: on-disk tasks/*/<N> fallback when registry path missing
confidence: high
related_task: #825
<!-- /workflow-fix-candidate -->
