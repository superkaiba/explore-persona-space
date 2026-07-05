---
title: 'daily-fix: /daily durable filing bodies + incremental filing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9ebe4481bbcb
- daily-auto-filed
created_at: '2026-07-05T07:03:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): The 2026-07-03 /daily attempt
  1 died mid-filing with 10 of 13 route-2 bodies stranded in bare /tmp; the third-attempt
  backfill had to recover the driver script and bodies from /tmp before completing
  (22:48-23:30 UTC on 2026-07-04).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

The /daily skill writes filing bodies to a DURABLE dir (e.g. logs/daily/filings-<date>/) and files INCREMENTALLY (recording filed ids as it goes) so a mid-run kill strands less; the daily file records still-pending ids as dispatch-pending.

## Workflow gap

- **Bug observed:** The 2026-07-03 /daily attempt 1 died mid-filing with 10 of 13 route-2 bodies stranded in bare /tmp; the third-attempt backfill had to recover the driver script and bodies from /tmp before completing (22:48-23:30 UTC on 2026-07-04).
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Session: b60c7f3a (/daily backfill). Related: #994 (headless /daily 600s ceiling) covers the death itself; this covers blast-radius reduction. NOTE: the 2026-07-04 run already adopted the durable-dir half ad hoc - codify it.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
