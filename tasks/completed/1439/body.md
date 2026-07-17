---
title: 'daily-fix: periodic task.py audit sweep for registry drift'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5e7df86fb936
- daily-auto-filed
created_at: '2026-07-17T06:51:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): post-#898, uniquely-resolvable
  REGISTRY drift surfaces only via a find_task_path drift WARNING that no structured
  consumer reads, so persistent resolvable drift (e.g. the #207 REGISTRY drift noted
  on #1399) goes unnoticed indefinitely'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1382 (Alternatives critic, plan round 1), sub-candidate (1).

## Goal

Surface persistent, uniquely-resolvable tasks/REGISTRY.json drift through a structured periodic sweep instead of an unread log WARNING.

## Workflow gap

- **Bug observed:** post-#898, uniquely-resolvable REGISTRY drift surfaces only via a find_task_path drift WARNING that no structured consumer reads, so persistent resolvable drift (e.g. the #207 REGISTRY drift noted on #1399) goes unnoticed indefinitely
- **Why it is a workflow gap:** The watcher is the fleet's periodic reconciliation surface; drift that only ever lands in a per-call WARNING has no consumer and accumulates (e.g. #207).
- **Confidence (emitter):** low (emitter) — concrete file + change, filed per the 2026-06-11 standing directive
- verified-at-filing: `grep -rn 'task.py audit' scripts/cron_*.sh scripts/autonomous_session_watch.py` -> 0 hits (no periodic audit sweep exists — the absence claim); `task.py audit` subcommand exists (CLAUDE.md Task Workflow API table); find_task_path drift WARNING at src/explore_persona_space/task_workflow.py ~L3624

## Proposed change (candidate diff sketch — refine in planning)

Add a low-frequency (e.g. once/day guarded by a sentinel) `task.py audit` report-only pass to autonomous_session_watch.py that pushes/escalates on non-empty drift, following the existing sidecar-event + dedup patterns.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Alternative: a dedicated cron wrapper next to cron_pod_audit.sh
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 5e7df86fb936



