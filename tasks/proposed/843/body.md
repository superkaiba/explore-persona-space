---
title: 'daily-fix: per-issue dispatch lease (watcher sweep + filer self-dispatch double-spawn)'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:03cd13bde185
created_at: '2026-07-02T07:12:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 problem sweep route-2: proposed_infra_sweep and
  file_infra_task self-dispatch double/triple-dispatched the same task: 7 dispatches
  for 3 tasks (#787 x2 37s apart, #805 x3, #806 x2), #822 twice 18s apart, #833 3
  duplicate spawns in 3 min, #800 ran TWO full duplicate pipelines (~40 min duplicated
  planning/review/tests); Step 0 single-orchestrator guard is check-then-act and cannot
  see a just-spawned not-yet-registered ses'
---
## Overview / Motivation

Auto-filed by the nightly /daily problem sweep (2026-07-01) — route 2 (behavior/logic change requiring independent review through the full /issue pipeline).

## Goal

Write an atomic create-or-fail per-issue dispatch lease BEFORE the spawn call returns (file_infra_task self-dispatch AND the watcher sweep respect it); make the per-issue registration fail/append on collision instead of overwrite; skip dispatch when a proposed-infra-sweep-dispatch marker newer than ~10 min already exists on the task.

## Bug observed (2026-07-01 sessions)

proposed_infra_sweep and file_infra_task self-dispatch double/triple-dispatched the same task: 7 dispatches for 3 tasks (#787 x2 37s apart, #805 x3, #806 x2), #822 twice 18s apart, #833 3 duplicate spawns in 3 min, #800 ran TWO full duplicate pipelines (~40 min duplicated planning/review/tests); Step 0 single-orchestrator guard is check-then-act and cannot see a just-spawned not-yet-registered session; per-issue registration overwrites on collision.

Observed in 3 independent miner groups (issues 787/805/806, 822/834, 833, 800/802/803/805). Root cause pinned by the #800 pair: the filer's own self-dispatch raced the watcher backstop. #820 (per-stage implementer dedup) and #759 (respawn path) do NOT cover this file-time/backstop race.

## Proposed change (refine in planning)

Write an atomic create-or-fail per-issue dispatch lease BEFORE the spawn call returns (file_infra_task self-dispatch AND the watcher sweep respect it); make the per-issue registration fail/append on collision instead of overwrite; skip dispatch when a proposed-infra-sweep-dispatch marker newer than ~10 min already exists on the task. Consider a session-id-suffixed plan tmp path in adversarial-planner to stop crashed-attempt /tmp collisions (4x 'File has been modified since read' on /tmp/issue-822-plan-v1.md).

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, scripts/file_infra_task.py, scripts/spawn_session.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` no-flags default run passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/file_infra_task.py, scripts/spawn_session.py
- fingerprint: 03cd13bde185
- source: /daily 2026-07-01 problem sweep (transcript miners)
