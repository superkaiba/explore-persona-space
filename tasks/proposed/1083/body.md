---
title: 'daily-fix: repair tasks/REGISTRY.json drift (15 audit proble'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e7413f8f1a63
- daily-auto-filed
created_at: '2026-07-06T06:58:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): task.py audit FAILs with
  15 problems, live all day and noising every session''s audit output + living_docs.py
  check: #696 registry path stale (says tasks/approved/696, dir is at tasks/completed/696);
  #698-#709 on disk but not in registry; #750/#764/#766 registry paths exist but body.md
  is missing (possible husk dirs from a bad merge). Surfaced in #1042''s session (d1ae3a43,
  07:42Z) and reproduced to'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Run the canonical repair (task.py audit --repair --apply under flock) for the stale/missing registry rows, and INVESTIGATE the 3 missing-body.md dirs (#750/#764/#766): recover the body from git history if it existed, or reconstruct the pointer; do NOT delete any task dir - if deletion appears to be the only remedy, park and surface to the user (destructive actions are user-gated).

## Workflow gap

- **Bug observed:** task.py audit FAILs with 15 problems, live all day and noising every session's audit output + living_docs.py check: #696 registry path stale (says tasks/approved/696, dir is at tasks/completed/696); #698-#709 on disk but not in registry; #750/#764/#766 registry paths exist but body.md is missing (possible husk dirs from a bad merge). Surfaced in #1042's session (d1ae3a43, 07:42Z) and reproduced tonight.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `tasks/REGISTRY.json, scripts/task.py`
- Registry mutation via the canonical API only; no status moves.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: tasks/REGISTRY.json, scripts/task.py
- source: /daily 2026-07-05 problem sweep (transcript-mined)
