---
title: 'workflow-fix: PreToolUse deny-hook for harmful-bank file reads'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:84fdd53a8769
created_at: '2026-07-04T07:09:31Z'
has_clean_result: false
origin_prompt: 'parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target,
  see workflow-fix-on-bug § Recursion guard. source: prose-followup (alternatives
  critic, #888 plan round 1). target_file: .claude/settings.json. proposed_change:
  PreToolUse deny-hook blocking Read/cat on harmful-bank paths (query_banks/{advbench,strongreject,betley_main8,wang44,broad_em_train,sensitive_info_requests}*.json)
  as a mechanical guard strictly stronger than the prose rule this task adds. confidence:
  medium. related_task: #888. routed: parked: EPM_WORKFLOW_FIX_SESSION'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

PreToolUse deny-hook blocking Read/cat on harmful-bank paths (query_banks/{advbench,strongreject,betley_main8,wang44,broad_em_train,sensitive_info_requests}*.json) as a mechanical guard strictly stronger than the prose rule

## Workflow gap

- **Bug observed:** the #866/#888 harmful-bank digest-only rule is prose-only; an agent can still Read/cat bank files and refusal-kill its session
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

PreToolUse deny-hook blocking Read/cat on harmful-bank paths (query_banks/{advbench,strongreject,betley_main8,wang44,broad_em_train,sensitive_info_requests}*.json) as a mechanical guard strictly stronger than the prose rule

## Scope / surfaces

- Primary target: `.claude/settings.json`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/settings.json
- fingerprint: 84fdd53a8769

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug § Recursion guard. source: prose-followup (alternatives critic, #888 plan round 1). target_file: .claude/settings.json. proposed_change: PreToolUse deny-hook blocking Read/cat on harmful-bank paths (query_banks/{advbench,strongreject,betley_main8,wang44,broad_em_train,sensitive_info_requests}*.json) as a mechanical guard strictly stronger than the prose rule this task adds. confidence: medium. related_task: #888. routed: parked: EPM_WORKFLOW_FIX_SESSION
