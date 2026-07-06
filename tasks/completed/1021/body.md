---
title: 'daily-fix: flip stale blocked to running on relaunch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c15505ee3447
- daily-auto-filed
created_at: '2026-07-04T21:36:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-2: #742 ran healthy 11.5h while parked
  at blocked'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-02 backfill problem sweep (route 2 —
behavior/logic change, independent review required). Emitting context:
transcript mining of the 2026-07-02 sessions.

## Goal

Crash-fix relaunch flips a stale blocked status back to running when posting epm:run-launched, with a watcher reconcile flag for blocked tasks that have live healthy runs as backstop

## Workflow gap

- **Bug observed:** #742 ran healthy for ~11.5h while its task folder still said blocked (set during the June-30 crash streak, never flipped back); dashboard/watcher views stayed wrong until the user asked
- **Why it is a workflow gap:** The crash-fix round recipe posts epm:run-launched but never reconciles a blocked status left by an earlier failed round, and no watcher pass flags a blocked task with a live run
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
SKILL.md crash-fix round step:
+ after a successful relaunch, if status == blocked: task.py set-status
+   <N> running --note 'relaunch succeeded; clearing stale blocked'
watcher: flag (never auto-flip) blocked tasks with fresh poll ticks +
  an alive workload pid, via the existing sidecar/Telegram channel
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every
  hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, scripts/autonomous_session_watch.py
- fingerprint: c15505ee3447
- related_task: #742
- source: /daily 2026-07-02 backfill problem sweep (route 2)
