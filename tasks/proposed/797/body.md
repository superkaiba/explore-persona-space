---
title: 'daily-fix: routing carve-out: interactive 0-GPU inline analysis'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0929d2a7cdaa
- daily-auto-filed
created_at: '2026-07-01T06:54:55Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: The same-issue follow-up routing
  law was over-applied in an interactive chat: a 0-GPU-h free analysis on existing
  artifacts that Thomas explicitly asked to run NOW was refused (''the routing law
  sends'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Add an explicit carve-out to the Routing experiment intent / Always-inline section: when the user in an interactive chat directly asks to run a 0-GPU-h free analysis on existing artifacts, run it inline with a subagent (register followup-scope for provenance in parallel, don't gate the run on it). A live per-issue session collision blocks re-spawning /issue N, not an independent read-only subagent.

## Workflow gap

- **Bug observed:** The same-issue follow-up routing law was over-applied in an interactive chat: a 0-GPU-h free analysis on existing artifacts that Thomas explicitly asked to run NOW was refused ('the routing law sends it through /issue follow-up loop'), and Thomas had to override by pasting the refusal back.
- **Evidence:** issue 658 chat on 2026-06-30 (Thomas correction). Source: /daily miner batch 02.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: 0929d2a7cdaa
- source: /daily route-2 (2026-06-30)
