---
title: 'daily-fix: stagger proposed-infra-sweep session dispatches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4d50e5fc18ee
- daily-auto-filed
created_at: '2026-07-05T07:03:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): Five workflow-fix sessions
  spawned within ~3 minutes on 2026-07-04 (07:09-07:12 UTC) tripped the org 4M input-TPM
  cap repeatedly - 6 api_error 429 events in the #976 session alone as fresh 100K+-token
  session loads stacked in the same minute windows.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Stagger spawn-issue --auto dispatches by >=60s when the watcher's proposed_infra_sweep (or a batch filer) dispatches several infra tasks at once.

## Workflow gap

- **Bug observed:** Five workflow-fix sessions spawned within ~3 minutes on 2026-07-04 (07:09-07:12 UTC) tripped the org 4M input-TPM cap repeatedly - 6 api_error 429 events in the #976 session alone as fresh 100K+-token session loads stacked in the same minute windows.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, scripts/file_infra_task.py`
- Session: 7db2a0f9 (#976, 07:12-07:35 UTC).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/file_infra_task.py
- source: /daily 2026-07-04 problem sweep (transcript-mined)
