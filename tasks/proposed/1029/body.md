---
title: 'daily-fix: GCP pre-workload boot failures advance the failover ladder'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c52dd249caa0
- daily-auto-filed
created_at: '2026-07-04T22:38:32Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — gcp-boot-fail-failover (fp c52dd249caa0)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

After N (2-3) consecutive pre-workload setup_failed/guestTerminate deaths on the same rung, treat it as an advance trigger (next rung / RunPod terminal rung), mirroring the #783 queue-timeout escalation; keep the crash-diagnostics persist behavior.

## Workflow gap

- **Bug observed:** 2026-07-03: #763 died 3 consecutive times pre-workload (guestTerminate during boot/setup) and kept churning GCP attempts (reached 7/8 of the daily create cap); pre-workload setup_failed deaths are EXCLUDED from the GCP->RunPod failover (only workload crashes fail over), so a boot-looping rung can exhaust the cap and block the task.
- **Why it matters:** Boot-loops burn the create cap with zero science; the failover ladder already has the right shape for this trigger.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/backend_poll.py, src/explore_persona_space/backends/router.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/backend_poll.py, src/explore_persona_space/backends/router.py
- fingerprint: c52dd249caa0
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
