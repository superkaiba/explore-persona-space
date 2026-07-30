---
title: 'daily-fix: deferred-commit sidecar captures the Failed hook '
kind: infra
tags:
- wf-fix
- wf-fix-fp:9a0df291fc74
- daily-auto-filed
created_at: '2026-07-30T07:11:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): 33 deferred-commit sidecar
  rows in ~36h all carry bare ''CalledProcessError'' + a 500-char stderr tail showing
  only Passed/Skipped hooks — the actual Failed hook line is above the truncation
  in every inspected row, so the failure cause is undiagnosable'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner E-P6 (probed; 33 rows across 07-29/07-30)).

## Goal

Deferred-commit forensics must name WHICH hook failed — 33 undiagnosable deferrals can hide a systematic hook failure.

## Workflow gap

- **Bug observed:** Sidecar rows carry only error=CalledProcessError + a tail that truncates above the Failed line; sessions correctly never re-post, so nobody learns the cause.
- **Why it is a workflow gap:** the sidecar exists for exactly this diagnosis; the 500-char tail is mis-sized for pre-commit's output shape (Failed line comes early, Restored-changes line comes last).
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: tail + full-row inspection of ~/.task-workflow/deferred-commits.jsonl (row #1812 06:37:11Z quoted). DEDUP-CHECK FIRST: an in-flight task's epm:clarify (2026-07-29T01:34Z) already cites this seam — the spawned planner verifies no open task owns it before implementing.

## Proposed change (refine in planning)

Capture `[line for line in stderr if 'Failed' in line or 'error' in line.lower()][:5]` into the sidecar row.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 9a0df291fc74
