---
title: 'daily-fix: replan guard — re-read Goal before posting a plan version'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fbbaed21948b
- daily-auto-filed
created_at: '2026-07-04T23:00:58Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — replan-goal-guard (fp fbbaed21948b)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Before posting each plan version, the planner re-reads body.md and diffs the plan's Goal against the latest epm:goal-updated marker; a goal-update newer than the draft forces a redraft against the amended Goal.

## Workflow gap

- **Bug observed:** 2026-07-03 #922: after Thomas's mid-flight goal rewrite ('the whole point is that we want to predict without generating'), plan v3 was drafted against the stale v1 goal — one wasted plan round + one wasted implementer round-1 diff before the session self-corrected to a v4 amendment.
- **Why it matters:** Goal amendments are user-authoritative; a plan drafted against a stale Goal burns a full review cycle.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/planner.md, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: .claude/agents/planner.md, .claude/skills/issue/SKILL.md
- fingerprint: fbbaed21948b
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
