---
title: 'workflow-fix: multi-zone GCP fan-out before declaring a capacity miss'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1b0e0fd1fb17
created_at: '2026-06-30T21:51:22Z'
has_clean_result: false
origin_prompt: what is the infra friction and how can we avoid it in the future -
  send a subagent to investigate
---
## Overview / Motivation
Auto-filed from the predictor-line (#742/#761/#763) infra-friction retrospective (subagent, 2026-06-30). Related in-flight: #770 (RunPod no-port wedge billing), #681 (data-disk bind cutover), #667 (GCP frozen-phase), #769 (CUDA-frag OOM).

## Goal
Before counting a GCP rung as a capacity miss, fan out across the region's A100-80 zones (us-central1-a/b/c/f); only escalate to the next rung / no_compute_available after all zones miss. Distinct from #667 and #770.

## Workflow gap
- Bug observed: #763 Phase-2 hit ZONE_RESOURCE_POOL_EXHAUSTED in only us-central1-c (single zone) and escalated straight to no_compute_available -> blocked, though other us-central1 zones may have had A100-80 capacity.

## Scope / surfaces
- Target: src/explore_persona_space/backends/gcp.py, .claude/rules/compute-backend-failover.md

## Constraints / invariants
- Workflow-surface only; ruff + workflow_lint pass; add a test where practical.

## Provenance
- workflow_fix_target: src/explore_persona_space/backends/gcp.py, .claude/rules/compute-backend-failover.md
- fingerprint: 1b0e0fd1fb17

Raised by the predictor-line infra retrospective (2026-06-30).
