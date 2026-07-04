---
title: 'daily-fix: poller CPU-phase stall predicate + per-instance idle counter'
kind: infra
tags:
- wf-fix
- wf-fix-fp:07c2abee2b55
- daily-auto-filed
created_at: '2026-07-04T23:01:08Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — poller-cpu-stall-idlecounter (fp
  07c2abee2b55)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Stall predicate consults pod/instance-side process-CPU%% and output-file mtime before declaring a CPU-analysis phase stalled; the idle-GPU advisory keys its minutes counter to the current instance's creation timestamp (reset on relaunch). Verify first what #873's landed tripwire already covers.

## Workflow gap

- **Bug observed:** 2026-07-03: the poller false-stalled 'constantly' on #813's CPU-bound analysis tail (~6h) — dismissed each time by checking process CPU + output mtimes manually; and the gpu-idle-advisory printed a stale CUMULATIVE minutes counter (486 min / 543 min) for instances only ~17 min old — two false 'burning' reads in one day.
- **Why it matters:** False alarms train operators to ignore the monitor — the one channel that catches real spend leaks (#664 class).
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py, src/explore_persona_space/backends/`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py, src/explore_persona_space/backends/
- fingerprint: 07c2abee2b55
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
