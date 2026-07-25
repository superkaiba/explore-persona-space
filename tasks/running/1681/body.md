---
title: 'workflow-fix: prioritize parked candidates on live red main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:719571265425
- daily-auto-filed
created_at: '2026-07-25T06:50:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): A parked candidate flagging
  a live red test on main (the 1643 park at 07:08Z re the dotenv-order invariant test)
  waited on the nightly /daily cycle while the red hit every intervening session''s
  Step 9c gate for about 11.5 hours until 1666 landed the fix'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (sessions bc8b80d3/f40d676d/d2e3d231, tasks #1643/#1650).

## Goal

A parked workflow-fix candidate that flags a LIVE red test on main should not wait up to ~24h for the nightly /daily sweep while every session's Step 9c gate eats the red.

## Workflow gap

- **Bug observed:** #1643's 07:08:11Z park flagged `test_no_new_torch_before_dotenv_vm_entrypoints` red on origin/main (the #847 class, naming issue1586 scripts). The red stayed live ~11.5h (until #1666's 18:16Z merge), surfacing in at least 2 intervening gates (#1650 09:04Z "2 failed / 5553 passed — both flagged pre-existing"; /daily 06:32Z confirmation) and requiring baseline-compare classification each time.
- **Why it is a workflow gap:** the recursion-guard escape valve routes ALL parks through the nightly /daily Step C regardless of urgency; a main-is-red park has fleet-wide cost per hour.
- **Confidence (emitter):** medium — today the fix landed same-day anyway (#1666); the value is bounding the tail case. Mechanism (watcher pass vs PM STATUS pass vs a driver flag) is a planner decision.
- verified-at-filing: `grep -n 'escape valve' .claude/rules/workflow-fix-on-bug.md` → Recursion guard § present with nightly-only routing (presence bind; no urgency subclass — absence read in context, 2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

In § Recursion guard escape valve: define an `urgent — main is red` line for parks whose bug is a currently-failing test on origin/main; prescribe that a NON-guarded orchestrator (the watcher's next pass or the PM STATUS pass) routes such parks immediately via the standard filing path (guard semantics unchanged — the parking session still never routes its own candidate).

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md` (+ the chosen mechanism's file, planner decides)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 719571265425

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md
