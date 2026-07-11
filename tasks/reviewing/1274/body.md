---
title: 'daily-fix: sweep misses routed-record for prior-night park'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0073efd7010e
- daily-auto-filed
created_at: '2026-07-11T06:54:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): the 2026-07-10 sweep re-enumerated
  the task:1196 park (parked 2026-07-09T20:54:25Z) as suppressed:false although a
  routed-record epm:workflow-fix-task-filed marker with matching origin_candidate_ts
  existed on #1196 since 2026-07-10T06:58 (the park was filed as #1235, completed
  + applied) - the suppression predicate missed it and the next nightly run had to
  re-dedup by hand'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

fix the sweep suppression predicate so a park whose task carries a routed-record marker with matching origin_candidate_ts (or matching target_file for fp-less prose parks) is suppressed; add a red-green fixture for the 1196 shape

## Workflow gap

- **Bug observed:** the 2026-07-10 sweep re-enumerated the task:1196 park (parked 2026-07-09T20:54:25Z) as suppressed:false although a routed-record epm:workflow-fix-task-filed marker with matching origin_candidate_ts existed on #1196 since 2026-07-10T06:58 (the park was filed as #1235, completed + applied) - the suppression predicate missed it and the next nightly run had to re-dedup by hand
- **Provenance / evidence:** /daily 2026-07-10 Step-C checker verification (candidate 1 verdict already-fixed/dedup #1235); confidence medium - root cause of the predicate miss not yet diagnosed.

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 0073efd7010e

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py
