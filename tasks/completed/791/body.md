---
title: 'daily-fix: backend_poll stale-log false gap + poll error-regex'
kind: infra
tags:
- wf-fix
- wf-fix-fp:912ec456f714
- daily-auto-filed
created_at: '2026-07-01T06:54:01Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: The poller tails one workload
  log while a later run arm writes to a different log, so it reads the older log as
  ''stale'' and reports false marker-gaps that trigger watcher respawns (not workload
  death)'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Make the poller discover the newest workload log by mtime within the run dir (or have multi-arm dispatch write a single canonical log path); tighten the poll error grep to anchor real failures (`Traceback`, `[1-9][0-9]* failed`, non-zero exit) and exclude `0 failed`/`(0 failed)`.

## Workflow gap

- **Bug observed:** The poller tails one workload log while a later run arm writes to a different log, so it reads the older log as 'stale' and reports false marker-gaps that trigger watcher respawns (not workload death); separately the completion poller's error regex matches benign lines like `wrote sweep coverage: ... (0 failed)` (bare `failed` substring), tripping false 'run errored' alarms.
- **Evidence:** issues 665/697 on 2026-06-30. Sources: /daily miners batches 03/06.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/backend_poll.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: scripts/backend_poll.py
- fingerprint: 912ec456f714
- source: /daily route-2 (2026-06-30)
