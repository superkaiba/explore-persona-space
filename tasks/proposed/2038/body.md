---
title: 'daily-fix: terminate pod on failed RunPod fallback launch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1013bea9a2b3
- daily-auto-filed
created_at: '2026-08-03T07:01:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): 4 orphan pod-1739 RunPod
  pods (~$10+/hr combined) were left billing overnight by wave-1''s failed/superseded
  RunPod-rung fallback launches; stopped only by the successor session''s manual sweep
  (positive incident: #1739 progress marker readback at 2026-08-02T05:53:32Z, ''terminated
  4 orphan RunPod pod-1739 pods... left by wave-1''s failed RunPod-rung fallbacks'').
  unverified hypothesis -- verify at pla'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner1, sessions 20e82ec2/f98a12ed, task #1739).

## Goal

A RunPod fallback launch that fails or is superseded never leaves its pod billing.

## Workflow gap

- **Bug observed:** 4 orphan pod-1739 RunPod pods (~$10+/hr combined) were left billing overnight by wave-1's failed/superseded RunPod-rung fallback launches; stopped only by the successor session's manual sweep (positive incident: #1739 progress marker readback at 2026-08-02T05:53:32Z, 'terminated 4 orphan RunPod pod-1739 pods... left by wave-1's failed RunPod-rung fallbacks'). unverified hypothesis -- verify at plan time: the fallback failure path provisions the pod before the launch fails and has no same-path terminate (miner-inferred from transcript prose; the code path was not read).
- **Why it is a workflow gap:** The stale-pod audit cron catches EXITED pods at 24h grain and the watcher's pod-safety pass keys on task status -- neither owns the minutes-after-a-failed-launch window, so the leak recurs on every failed fallback wave.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -iE 'terminate|cleanup' src/explore_persona_space/backends/runpod.py` -> terminate exists as teardown/shim methods; no call in a launch-failure path found by grep (planner to verify by reading the launch/fallback flow -- clause (g) call-hop: the constructing site may be issue_dispatch's fallback leg).

## Proposed change (refine in planning)

on a failed or superseded RunPod fallback launch, terminate the just-provisioned pod in the SAME failure path instead of leaving it for a later sweep/audit.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/issue_dispatch.py`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 1013bea9a2b3

- workflow_fix_target: src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/issue_dispatch.py

