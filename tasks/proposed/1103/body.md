---
title: 'daily-fix: standardize multi-behavior datagen behavior defin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c507a6bbe427
- daily-auto-filed
created_at: '2026-07-07T06:49:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): #906''s multi-behavior
  datagen failed its yield floors (sycophancy 6/36 judge-accepted vs floor 20) partly
  because behavior definitions were bespoke per behavior; the user caught it (''aren''t
  all the behaviors supposed to be standardized in the same way?'') and directed persona-vectors-style
  pos/neg-instruction behavior definitions + auto-generated questions for the #1090
  redesign (PM session e3f63b5'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add a bullet to on-policy-completions.md: multi-behavior implantation datagen defines EVERY behavior in the standardized persona-vectors pos/neg-instruction shape (shared definition template + question-generation recipe), never bespoke per-behavior definitions/query sets; cite the #906→#1090 incident.

## Workflow gap

- **Bug observed:** #906's multi-behavior datagen failed its yield floors (sycophancy 6/36 judge-accepted vs floor 20) partly because behavior definitions were bespoke per behavior; the user caught it ('aren't all the behaviors supposed to be standardized in the same way?') and directed persona-vectors-style pos/neg-instruction behavior definitions + auto-generated questions for the #1090 redesign (PM session e3f63b58, 2026-07-06).
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/on-policy-completions.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/rules/on-policy-completions.md
- source: /daily 2026-07-06 problem sweep (transcript-mined)
