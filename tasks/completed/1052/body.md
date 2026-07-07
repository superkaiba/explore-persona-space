---
title: 'daily-fix: CRON-TEARDOWN idempotent + sweep one-shot wakeups'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dc189c03e001
- daily-auto-filed
created_at: '2026-07-05T07:02:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): Two CRON-TEARDOWN gaps
  on 2026-07-04: (a) #980''s completed task still had a live one-shot ScheduleWakeup
  (''/issue 980'' at 4:56 AM) that only the session''s own diligence caught before
  it re-drove a completed task; (b) #988''s CronDelete failed on a stale job id (''No
  scheduled job with id'') because the recorded id was already gone.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

CRON-TEARDOWN enumerates and deletes one-shot wakeups matching '/issue <N>' alongside the recurring tick cron, resolves the live id via CronList first, and treats CronDelete not-found as success (idempotent teardown).

## Workflow gap

- **Bug observed:** Two CRON-TEARDOWN gaps on 2026-07-04: (a) #980's completed task still had a live one-shot ScheduleWakeup ('/issue 980' at 4:56 AM) that only the session's own diligence caught before it re-drove a completed task; (b) #988's CronDelete failed on a stale job id ('No scheduled job with id') because the recorded id was already gone.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, .claude/skills/issue-tick/SKILL.md`
- Sessions: 5f8c9d25 (#980, 12:28 UTC), 1c90a72b (#988, 21:43 UTC).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/skills/issue-tick/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
