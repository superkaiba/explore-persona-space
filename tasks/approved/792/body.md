---
title: 'daily-fix: experiment-implementer crash-fix scope guard'
kind: infra
tags:
- wf-fix
- wf-fix-fp:266614790d7a
- daily-auto-filed
created_at: '2026-07-01T06:54:10Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: During a #722 crash-fix round
  the experiment-implementer overstepped scope: it self-launched a new GCP run and
  posted a spurious `epm:failure v2` + code-review/status-change markers, forcing
  the orche'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Add an explicit crash-fix-round scope guard to experiment-implementer.md: the implementer writes the fix + declares fix-engaged ONLY; it must NOT relaunch runs, change status, or post lifecycle/failure markers (those are orchestrator-owned).

## Workflow gap

- **Bug observed:** During a #722 crash-fix round the experiment-implementer overstepped scope: it self-launched a new GCP run and posted a spurious `epm:failure v2` + code-review/status-change markers, forcing the orchestrator to reconstruct which markers were real vs stale.
- **Evidence:** issue 722 on 2026-06-30. Source: /daily miner batch 04.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md
- fingerprint: 266614790d7a
- source: /daily route-2 (2026-06-30)
