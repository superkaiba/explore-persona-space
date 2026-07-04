---
title: 'daily-fix: headless /daily dies at 600s bg-wait ceiling'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:366cda38bad6
created_at: '2026-07-04T07:13:07Z'
has_clean_result: false
origin_prompt: 'daily finding 1 (2026-07-03): headless /daily cron runs killed at
  the 600s bg-wait ceiling two days running; daily files never written.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 (or a large value) in the cron line, and/or have the skill instruct headless runs to bound subagent count / wait synchronously; define a backfill affordance when the heartbeat flags a missing stub

## Workflow gap

- **Bug observed:** headless `claude -p /daily` died at the 600s background-task wait ceiling on 2026-07-01 and 2026-07-02 ('Background tasks still running after 600s; terminating' in ~/my-goat/logs/daily_retrospective.log) — the daily files for those dates were never written
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 (or a large value) in the cron line, and/or have the skill instruct headless runs to bound subagent count / wait synchronously; define a backfill affordance when the heartbeat flags a missing stub

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: 366cda38bad6

daily finding 1 (2026-07-03): headless /daily cron runs killed at the 600s bg-wait ceiling two days running; daily files never written.
