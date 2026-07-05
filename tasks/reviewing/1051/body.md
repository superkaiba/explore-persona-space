---
title: 'daily-fix: tick_triage liveness probe before STALE-REDRIVE'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fc004430c3e2
- daily-auto-filed
created_at: '2026-07-05T07:02:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): During #931''s legitimately-running
  multi-hour fit (armA author-fold cells ~2.2h each, no marker posts in between),
  tick_triage classified the session stale on EVERY tick and fired 9 STALE-REDRIVE
  re-drives in 3.5h (20:24-23:52 UTC on 2026-07-04), each reloading the 44K-token
  SKILL.md against a healthy run.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Before returning STALE-REDRIVE, probe liveness: a live fit/workload PID for the issue or a recently-appended run log => return HEALTHY; alternatively have long-fit launchers post a periodic heartbeat epm:progress.

## Workflow gap

- **Bug observed:** During #931's legitimately-running multi-hour fit (armA author-fold cells ~2.2h each, no marker posts in between), tick_triage classified the session stale on EVERY tick and fired 9 STALE-REDRIVE re-drives in 3.5h (20:24-23:52 UTC on 2026-07-04), each reloading the 44K-token SKILL.md against a healthy run.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/tick_triage.py`
- Session: 23dbea11 (#931). Related: #1028 (watcher recovers ALIVE-BUT-STALLED sessions) is the recovery side; this is the false-positive-stale side.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/tick_triage.py
- source: /daily 2026-07-04 problem sweep (transcript-mined)
