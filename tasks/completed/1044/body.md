---
title: 'daily-fix: sync_repo_root husk liveness probe + multi-branch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:06695935fd4d
- daily-auto-filed
created_at: '2026-07-05T07:02:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): On 2026-07-04 a dead rebase-merge
  husk (~45 min old, under the 60-min age threshold) blocked #949''s pushes for ~20
  min on pure wall-clock aging even though no git process held it; separately #965
  and #998 hit transient ''fatal: Cannot rebase onto multiple branches'' during the
  pull-rebase and had to retry by hand.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Supplement the husk age threshold with a process-liveness probe (no git process holding the rebase dir => treat as husk regardless of age) and catch 'Cannot rebase onto multiple branches' with one bounded internal retry.

## Workflow gap

- **Bug observed:** On 2026-07-04 a dead rebase-merge husk (~45 min old, under the 60-min age threshold) blocked #949's pushes for ~20 min on pure wall-clock aging even though no git process held it; separately #965 and #998 hit transient 'fatal: Cannot rebase onto multiple branches' during the pull-rebase and had to retry by hand.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/sync_repo_root.py`
- Sessions: d0aaf230 (#949, 00:50-01:07 UTC), 8ade017d (#965, ~09:23 UTC), 6e2add4a (#998, 23:30 UTC). Related shipped work: #971 (head-name-less husk recovery), #996 (detached-HEAD bounded wait), #1030 (task_workflow git robustness, in planning) - this task covers the two residual seams only; planner should read those to avoid overlap.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/sync_repo_root.py
- source: /daily 2026-07-04 problem sweep (transcript-mined)
