---
title: 'daily-fix: guard detached repo-root HEAD'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2207d7f784a5
- daily-auto-filed
created_at: '2026-07-01T06:54:46Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: When the repo-root main worktree
  is left with a detached HEAD, the task_workflow repo-root resolver crashes (''main
  worktree HEAD is detached; re-attach to main''), killing any task.py/self-report
  call'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Extend guard_repo_root_branch.sh to also refuse operations that would leave the repo-root HEAD detached (mirror the branch-switch guard), so a concurrent `git worktree add --detach`/bad recovery cannot detach the shared root.

## Workflow gap

- **Bug observed:** When the repo-root main worktree is left with a detached HEAD, the task_workflow repo-root resolver crashes ('main worktree HEAD is detached; re-attach to main'), killing any task.py/self-report call in that session; the guard hook covers branch-SWITCH but not leaving HEAD detached.
- **Evidence:** 2026-06-30 (session_progress_report.py crash). Source: /daily miner batch 00.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 2207d7f784a5
- source: /daily route-2 (2026-06-30)
