---
title: 'workflow-fix: treat in-flight Batch/long-analysis phases as healthy-slow'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4beff16cbb9b
created_at: '2026-06-30T21:52:03Z'
has_clean_result: false
origin_prompt: what is the infra friction and how can we avoid it in the future -
  send a subagent to investigate
---
## Overview / Motivation
Auto-filed from the predictor-line (#742/#761/#763) infra-friction retrospective (subagent, 2026-06-30). Related in-flight: #770 (RunPod no-port wedge billing), #681 (data-disk bind cutover), #667 (GCP frozen-phase), #769 (CUDA-frag OOM).

## Goal
Treat an in-flight Anthropic Batch / long off-pod-analysis phase as healthy-slow: let the hourly [poll-tick] PROGRESSING markers (or a Batch-in-flight heartbeat) reset the staleness clock so a legitimately-slow phase is not respawn-eligible.

## Workflow gap
- Bug observed: The staleness heuristic respawned healthy #761 twice and flirted with respawning #742's healthy 8h Anthropic Batch phase (0 errored, steady throughput) because long phases emit few markers between poll-ticks.

## Scope / surfaces
- Target: scripts/autonomous_session_watch.py

## Constraints / invariants
- Workflow-surface only; ruff + workflow_lint pass; add a test where practical.

## Provenance
- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 4beff16cbb9b

Raised by the predictor-line infra retrospective (2026-06-30).
