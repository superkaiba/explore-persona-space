---
title: 'daily-fix: planner-output extraction tolerates trailing rows'
kind: infra
tags:
- wf-fix
- wf-fix-fp:927f40ad0c01
- daily-auto-filed
created_at: '2026-07-11T06:52:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): the orchestrator''s mechanical
  extraction of the planner subagent''s output transcript found NO RESULT ROW on first
  attempt when the last row carried attributionAgent-class keys instead of the expected
  result shape (#1219 01:55:46Z; one diagnostic pass later it extracted the 29,957-byte
  plan fine)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

make the agent-output extraction snippet scan backwards past trailing non-result rows for the result row

## Workflow gap

- **Bug observed:** the orchestrator's mechanical extraction of the planner subagent's output transcript found NO RESULT ROW on first attempt when the last row carried attributionAgent-class keys instead of the expected result shape (#1219 01:55:46Z; one diagnostic pass later it extracted the 29,957-byte plan fine)
- **Provenance / evidence:** miner-03 P9, /daily 2026-07-10 transcript sweep. One-off today; low priority, filed per the no-cap directive.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 927f40ad0c01

- workflow_fix_target: .claude/skills/issue/SKILL.md
