---
title: 'daily-fix: LESSONS.md durable slimming + lint cap WARN band'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a8e27d302df2
- daily-auto-filed
created_at: '2026-07-11T06:52:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): LESSONS.md oscillates at
  the 8000-byte --check-lessons-index cap (7,993 -> 8,028 fleet-wide RED on 07-10
  -> trimmed to 7,145 by #1220; second day running at the cap) and sessions only learn
  of the pressure when the cap is already crossed fleet-wide'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

durable structural slimming of the index and/or a >90%-of-cap WARN band in workflow_lint.py so sessions see pressure before a fleet-wide FAIL

## Workflow gap

- **Bug observed:** LESSONS.md oscillates at the 8000-byte --check-lessons-index cap (7,993 -> 8,028 fleet-wide RED on 07-10 -> trimmed to 7,145 by #1220; second day running at the cap) and sessions only learn of the pressure when the cap is already crossed fleet-wide
- **Provenance / evidence:** Parked candidates on #1207 (2026-07-10T01:09:28Z, dispositioned already-fixed for the immediate red only) and #1219; miner-01 P4/P8.1 + miner-03 P11.3, /daily 2026-07-10 sweep. The WARN band is DISTINCT from the #1220 trim (different proposed_change).

## Scope / surfaces

- Primary target: `.claude/rules/LESSONS.md, scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: a8e27d302df2

- workflow_fix_target: .claude/rules/LESSONS.md, scripts/workflow_lint.py
