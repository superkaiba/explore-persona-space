---
title: 'daily-fix: catch task lifecycle commits stranded off main at'
kind: infra
tags:
- wf-fix
- wf-fix-fp:50284d6e4830
- daily-auto-filed
created_at: '2026-07-07T06:49:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): The 15-problem tasks/REGISTRY.json
  drift repaired by #1083 (session de296c4a, 2026-07-06) traced to task.py lifecycle
  commits stranded on never-merged issue-<N> branches; no preventive guard exists,
  so drift is only caught at the next manual audit/repair. #1083''s session parked
  this prevention candidate under the recursion guard.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add a post-transition landing check (after a lifecycle mutation commits, verify the commit is reachable from main / warn loudly if not) or a periodic `task.py audit` alert cron that surfaces stranded lifecycle commits at creation time, not at the next drift repair. Secondary (lower value, same incident): a verifier warning for plans that stage whole task dirs with untracked residue.

## Workflow gap

- **Bug observed:** The 15-problem tasks/REGISTRY.json drift repaired by #1083 (session de296c4a, 2026-07-06) traced to task.py lifecycle commits stranded on never-merged issue-<N> branches; no preventive guard exists, so drift is only caught at the next manual audit/repair. #1083's session parked this prevention candidate under the recursion guard.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/task.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: scripts/task.py
- source: /daily 2026-07-06 problem sweep (transcript-mined)
