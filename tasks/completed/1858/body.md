---
title: 'daily-fix: RunPod failover launcher timeout-bounds sync, rea'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4ae84d4003d8
- daily-auto-filed
created_at: '2026-07-30T07:07:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): A RunPod failover pod wedged
  at boot: a MooseFS-hung git reset held .git/index.lock, the workload start failed
  (terminal_runpod_workload_start_failed), ~1.5h stall until a manual lock reap'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner I-P3 (#1769 fu1 failover pod)).

## Goal

A failover pod's boot sync must not wedge indefinitely on a FUSE-hung git op holding index.lock.

## Workflow gap

- **Bug observed:** unverified hypothesis — verify at plan time: the boot-leg git reset hung on MooseFS holding .git/index.lock -> terminal_runpod_workload_start_failed; ~1.5h stall, manual reap (miner-read from transcript; launcher code not inspected at compose time).
- **Why it is a workflow gap:** the launcher owns the boot sync; an unbounded git op on a FUSE mount is a known wedge class (gotchas MooseFS entries) with no timeout/reap in this leg.
- **Confidence (emitter):** medium
- verified-at-filing: n/a — reason: incident mechanism read from transcript; the exact launcher code path is for the planner to locate (start at pod_lifecycle.py workload-start + backends/runpod.py launcher render).

## Proposed change (refine in planning)

Wrap the boot-leg git sync in timeout(1) with one bounded retry; before the retry, reap .git/index.lock if older than the sync's own start; escalate to pod-swap on second failure.

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py, src/explore_persona_space/backends/runpod.py
- fingerprint: 4ae84d4003d8
