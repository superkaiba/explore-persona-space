---
title: 'daily-fix: de-escape HTML entities in plan-handoff commands'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c0922c7faf1d
- daily-auto-filed
created_at: '2026-07-05T07:04:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): #952''s v9 plan handoff
  on 2026-07-04 carried the workload command with the shell AND operator HTML-escaped
  (as the amp-entity form), and the orchestrator had to hand-fix it before dispatch
  would run.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add a de-escape/verification step where the handoff composes workload commands: assert no HTML entities (amp/lt/gt forms) survive in command fields, unescaping if present. Low confidence on root cause (where the escaping was introduced) - planner should trace it before patching.

## Workflow gap

- **Bug observed:** #952's v9 plan handoff on 2026-07-04 carried the workload command with the shell AND operator HTML-escaped (as the amp-entity form), and the orchestrator had to hand-fix it before dispatch would run.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Session: d49c6e04 (#952, ~21:00-23:00 UTC). confidence: low

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
