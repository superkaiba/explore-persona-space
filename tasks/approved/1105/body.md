---
title: 'daily-fix: pin Step 10d guard-block recovery-contract phrase'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8714228d288a
- daily-auto-filed
created_at: '2026-07-07T06:49:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): #1085 (session 41f61f27,
  2026-07-06) added the Step 10d additive-checkout guard-block recovery contract to
  SKILL.md but its durability follow-up — a phrase-count pin so a later edit cannot
  silently drop the recovery-contract phrases — was parked under the recursion guard.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add the phrase-count/anchor pin test for the Step 10d guard-block recovery-contract wording per #1085's parked prose follow-up (low value, filed per the no-parking directive).

## Workflow gap

- **Bug observed:** #1085 (session 41f61f27, 2026-07-06) added the Step 10d additive-checkout guard-block recovery contract to SKILL.md but its durability follow-up — a phrase-count pin so a later edit cannot silently drop the recovery-contract phrases — was parked under the recursion guard.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `tests/test_step10d_guard3.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: tests/test_step10d_guard3.py
- source: /daily 2026-07-06 problem sweep (transcript-mined)
