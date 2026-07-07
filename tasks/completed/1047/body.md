---
title: 'daily-fix: Step 10d pre-merge/pre-push hardening bundle'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d5856c255c1f
- daily-auto-filed
created_at: '2026-07-05T07:02:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): Three distinct Step 10d
  frictions on 2026-07-04: (a) #906''s recovery merge aborted on uncommitted .claude/agent-memory/**
  edits left by review rounds; (b) #931 merged a workflow-lint offender to main, breaking
  test_workflow_lint on pristine trunk fleet-wide for most of the day (#991/#987/#980/#992/#993
  each burned 5-25 min classifying it pre-existing); (c) #967''s close-out hand-rolled
  a repo-root'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Step 10d additions: pre-merge, commit (by explicit path) any dirty .claude/agent-memory/** files; pre-push, run workflow_lint (or the touched-scope subset) on the merge payload so an offender cannot land on main; and name scripts/sync_repo_root.py as the ONLY repo-root sync command wherever a raw pull recipe survives in the skill.

## Workflow gap

- **Bug observed:** Three distinct Step 10d frictions on 2026-07-04: (a) #906's recovery merge aborted on uncommitted .claude/agent-memory/** edits left by review rounds; (b) #931 merged a workflow-lint offender to main, breaking test_workflow_lint on pristine trunk fleet-wide for most of the day (#991/#987/#980/#992/#993 each burned 5-25 min classifying it pre-existing); (c) #967's close-out hand-rolled a repo-root git pull that died with 'fatal: Cannot autostash' instead of using sync_repo_root.py.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Sessions: 4ea4c2b6 (#906, 16:47 UTC), 23dbea11/a873fc2d/eb927605 (#931 origin, #991/#987 discovery, 17:00-19:10 UTC), e4951a90 (#967, 09:32 UTC).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
