---
title: 'workflow-fix: zombie-GPU stalled override needs a material-CPU-delta veto'
kind: infra
tags:
- wf-fix
- wf-fix-fp:994d4d74543e
created_at: '2026-07-03T23:24:29Z'
has_clean_result: false
origin_prompt: 'orchestrator observation on #825 run-1e: false stalled override on
  a prior-run VRAM leftover while the live fit burned 186% CPU; see epm:progress v94'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator
observation on task #825 (run-1e fit phase, 2026-07-03).

## Goal

Harden poll_pipeline.py's zombie-GPU stalled-override predicate: do not
override running->stalled when the workload demonstrably computes.

## Workflow gap

- **Bug observed:** #825 run-1e (fit phase): the 2-tick zombie-GPU override
  (#826) fired running->stalled on a 1816 MiB host-namespace VRAM leftover
  from the PRIOR dead run while the LIVE fit process burned 186% CPU and
  wrote 5 per-cell result JSONs within 15 min. session_cpu_secs advanced
  +1102s/+989s across the two ticks the persistence counted.
- **Why it is a workflow gap:** the override deliberately distrusts
  cpu_advancing because #664's hung generate() churned ~22% CPU — but a
  MATERIAL CPU delta (>= ~0.5 cores sustained across both ticks) is real
  compute, not churn, and a stale VRAM holder from a previous crashed run on
  the same pod is a common state after a crash-fix relaunch. Each false
  stalled tick invites failure routing against a healthy run.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
In the 2-tick zombie-GPU override (poll_pipeline.py, #826 block):
+ cpu_delta_cores = (cpu_now - cpu_prev) / tick_seconds
+ if cpu_delta_cores >= EPM_ZOMBIE_OVERRIDE_CPU_CORES_MIN (default 0.5)
+     on BOTH persisted ticks: liveness veto (log, do not override).
Optionally: when the flagged VRAM PID predates the CURRENT run's
epm:run-launched ts (prior-run leftover), require a higher persistence
count or exempt it from the override.
```

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py` (the #826 zombie-GPU override
  block + its liveness-veto helper).
- Tests: `tests/` poller unit coverage for the override predicate.
- Grep: `grep -n "zombie" scripts/poll_pipeline.py`.

## Constraints / invariants

- Must NOT regress the #664 protection: a hung generate() at ~0.22 cores
  must still override to stalled. The CPU-cores threshold separates the
  regimes (~0.22 vs ~2.0 measured).
- Workflow-surface only; ruff + workflow lint pass.

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: 994d4d74543e

Origin: epm:progress v94 FALSE-STALL override note on task #825
(2026-07-03T23:24Z).
