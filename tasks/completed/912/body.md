---
title: 'workflow-fix: worktree sweep should reap idle worktrees'' regenerable .venv'
kind: infra
tags:
- wf-fix
- wf-fix-fp:559066dec61a
created_at: '2026-07-03T07:02:12Z'
has_clean_result: false
origin_prompt: 'Chat-mode disk diagnosis 2026-07-02: 46 worktree .venvs, ~25G deduplicated,
  retained indefinitely by worktree_audit.py''s atomic keep/reap; venvs regenerate
  via uv sync from the shared cache.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised during a chat-mode disk-full diagnosis (2026-07-02, boot disk at 95%; emitting agent: orchestrator).

## Goal

Add a venv-reap arm to the worktree sweep: delete `.venv` in kept-but-idle worktrees (regenerable via `uv sync` from the shared cache), excluding the managed `_task-main-pin`.

## Workflow gap

- **Bug observed:** 46 worktree `.venv`s totaling ~25G (deduplicated) accumulated on the boot disk because the sweep keeps live/dirty/active-issue worktrees whole and never reaps their regenerable venvs; the two largest (`compute-router`, `_task-main-pin`) are ~10G each.
- **Why it is a workflow gap:** `worktree_audit.py` treats a worktree as an atomic keep/reap unit. A kept worktree's `.venv` is a pure build artifact — regenerable from the shared uv cache in minutes — yet it is the single largest per-worktree cost and is retained indefinitely for any worktree that stays "live" (dirty, active issue, or human-named).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
worktree_audit.py:
+ VENV_REAP_IDLE_DAYS = 7  # env-overridable
+ in the KEEP branch: if worktree idle > VENV_REAP_IDLE_DAYS and
+   no live process rooted in it: rmtree(<wt>/.venv), log "venv-reaped"
+ never touch _task-main-pin (task_workflow managed main pin) or a
+   worktree holding a live user TTY / running process
```

## Scope / surfaces

- Primary target: `scripts/worktree_audit.py`
- Check `new_worktree.sh` + `task_workflow._ensure_managed_main_worktree` for assumptions that a venv persists; `uv run` recreates on demand from the shared cache (hardlinks, same filesystem), so the reap should be cheap to recover from — planner verifies.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Never reap a venv under a worktree with a live process / lock; fail toward keep.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/worktree_audit.py
- fingerprint: 559066dec61a

Raised as a prose candidate during the 2026-07-02 disk-full diagnosis (boot disk 95%; `.claude/worktrees` = 97G on the boot disk, of which ~25G is regenerable venvs; the #681 bind-migration to /mnt/eps-data remains the structural fix and is tracked separately).
