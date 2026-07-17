---
title: 'daily-fix: route TG_TESTS fixture temps via step9c tmproot'
kind: infra
tags:
- wf-fix
- wf-fix-fp:469290838484
- daily-auto-filed
created_at: '2026-07-17T06:51:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the TG_TESTS targeted-green
  pytest blocks (~L9470/L9494 and ~L10539/L10543/L10601) still write pytest fixture
  temps to /tmp under root-disk pressure — the same class as #1363 — while the 9c
  1b/1c gate blocks got the #1408 tmproot/--basetemp routing'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1408 (Methodology critic concern 2, plan v1 review). #1408 is completed — its fix covered the 9c gate blocks only.

## Goal

Extend the #1408 tmproot/--basetemp routing to the TG_TESTS targeted-green pytest blocks in issue/SKILL.md.

## Workflow gap

- **Bug observed:** the TG_TESTS targeted-green pytest blocks (~L9470/L9494 and ~L10539/L10543/L10601) still write pytest fixture temps to /tmp under root-disk pressure — the same class as #1363 — while the 9c 1b/1c gate blocks got the #1408 tmproot/--basetemp routing
- **Why it is a workflow gap:** Fixture temp writes to /tmp on the shared VM under disk pressure are the #1363 failure class; a fix that covers one pytest block family but not its sibling leaves the class live.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'TG_TESTS' .claude/skills/issue/SKILL.md` -> hits at L9470/L9494/L10539/L10543/L10601 (pytest invocations present); the #1408 tmproot/--basetemp routing exists in the 9c gate blocks but the TG_TESTS pytest lines carry no --basetemp (per-block check owed in planning)

## Proposed change (candidate diff sketch — refine in planning)

Mirror the #1408 gate-block tmproot/--basetemp lines into both TG_TESTS pytest blocks.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 469290838484



