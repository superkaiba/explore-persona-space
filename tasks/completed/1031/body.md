---
title: 'daily-fix: plan-time RAM/RSS routing gate + p90 fence sizing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e001f5b88f11
- daily-auto-filed
created_at: '2026-07-04T23:00:45Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — plan-ram-fence-sizing (fp e001f5b88f11)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

planner section 9 sizes each CPU phase's projected RSS and routes >=~16 GB (or >2 concurrent multi-GB phases) off the shared VM to cpu-mid/cpu-bigmem — the RAM sibling of the 50 GB disk carve-out; and sizes --max-run-duration off the p90 per-cell estimate, not the mean.

## Workflow gap

- **Bug observed:** 2026-07-03: #778's 22-GiB-RSS null battery was earlyoom-killed 3x on the starved shared VM (125 GB, 95G used) before a cpu-bigmem pivot; #833's chain fits lost 5 cells to earlyoom (two ~13-15 GB phases concurrently resident) and then overran its 36h GCP fence because per-cell wall-time ran ~2x the plan mean.
- **Why it matters:** The existing section-9 rules gate on disk footprint and wall-time only; RSS on the shared VM is the recurring killer.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md, .claude/agents/planner.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: .claude/rules/plan-compute-sizing.md, .claude/agents/planner.md
- fingerprint: e001f5b88f11
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
