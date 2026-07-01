---
title: 'daily-fix: zombie sessions on completed tasks not auto-reaped'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0423dba4c897
- daily-auto-filed
created_at: '2026-07-01T06:54:37Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: Happy sessions stayed mapped
  to already-`completed` tasks (698x1, 705x2, 706x2) and were not auto-reaped by the
  reconcile pass; Thomas had to notice the inflated session count and trigger reaping
  manu'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Verify/tighten the reconcile-pass auto-stop predicate for sessions on completed tasks (idle->2h + no-pod + no-keep-running) so it actually fires, or add a completed-task fast-path so zombie sessions do not accumulate until a human notices.

## Workflow gap

- **Bug observed:** Happy sessions stayed mapped to already-`completed` tasks (698x1, 705x2, 706x2) and were not auto-reaped by the reconcile pass; Thomas had to notice the inflated session count and trigger reaping manually.
- **Evidence:** 2026-06-30 session-count triage. Source: /daily miner batch 05.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 0423dba4c897
- source: /daily route-2 (2026-06-30)
