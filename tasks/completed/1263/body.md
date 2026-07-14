---
title: 'daily-fix: add unwrapped clarifier to escape remedy text'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6db951b08ec0
- daily-auto-filed
created_at: '2026-07-11T06:51:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): The ~16 FAIL-detail remedy
  strings saying ''on its own line'' (:1277, :1540, :2472, :3724, :4934, ...) all
  display the escape phrase quote/backtick-WRAPPED and none says ''unwrapped'' - the
  remedy text that reaches a planner at the exact failure moment teaches the wrapped
  form that produced #1090''s c12 bounce (the shape _standalone_na_declared deliberately
  rejects, #1238)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

Append an 'unwrapped - no backticks/quotes' clarifier to the helper-routed checks' FAIL-detail remedy text (wording-only; no recognizer behavior change)

## Workflow gap

- **Bug observed:** The ~16 FAIL-detail remedy strings saying 'on its own line' (:1277, :1540, :2472, :3724, :4934, ...) all display the escape phrase quote/backtick-WRAPPED and none says 'unwrapped' - the remedy text that reaches a planner at the exact failure moment teaches the wrapped form that produced #1090's c12 bounce (the shape _standalone_na_declared deliberately rejects, #1238)
- **Provenance / evidence:** Formal candidate fp 473627bfc5ba, alternatives critic, #1238 plan v2/v3 review (parked 2026-07-10T11:35:57Z). Verified live: 18 own-line remedy strings, 0 carry an unwrapped clarifier (only the helper docstring :1465-1469 mentions it).

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 6db951b08ec0

- workflow_fix_target: scripts/verify_plan.py
