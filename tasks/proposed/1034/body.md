---
title: 'daily-fix: HF quota-headroom preflight before multi-TB uploads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1cce86a54c9c
- daily-auto-filed
created_at: '2026-07-04T23:01:17Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — hf-quota-preflight (fp 1cce86a54c9c)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Before a projected multi-100GB+ LFS upload, probe quota headroom and route to the overflow repo (the #841 v11 pattern) or block with a loud alert — instead of discovering the 403 mid-upload fleet-wide.

## Workflow gap

- **Bug observed:** 2026-07-03: a multi-TB unreduced-activation store filled the public-LFS ceiling; fleet-wide 403s then hit >=3 tasks (#841 lost a launch attempt, #813 held final reconciliation, #833's staging died) until Thomas bought storage (~09:52Z).
- **Why it matters:** The overflow-repo fallback exists; what is missing is the proactive probe so one task's giant store cannot silently starve the fleet.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/preflight.py, .claude/rules/upload-policy.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/preflight.py, .claude/rules/upload-policy.md
- fingerprint: 1cce86a54c9c
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
