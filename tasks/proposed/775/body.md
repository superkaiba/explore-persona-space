---
title: 'workflow-fix: auto-failover RunPod CUDA-IMA host-wedge to a fresh host'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d73d9e6f015d
created_at: '2026-06-30T21:51:38Z'
has_clean_result: false
origin_prompt: what is the infra friction and how can we avoid it in the future -
  send a subagent to investigate
---
## Overview / Motivation
Auto-filed from the predictor-line (#742/#761/#763) infra-friction retrospective (subagent, 2026-06-30). Related in-flight: #770 (RunPod no-port wedge billing), #681 (data-disk bind cutover), #667 (GCP frozen-phase), #769 (CUDA-frag OOM).

## Goal
Extend the wedge/failover machinery: a same-signature CUDA-IMA crash after a verified-clean-GPU respawn auto-pivots to a FRESH host (new pod), bounded once (mirroring the GCP->RunPod once-more bound), then terminal failure_class:code -> blocked.

## Workflow gap
- Bug observed: #763 crashed twice with CUDA illegal-memory-access (EngineDeadError) same-signature after a clean orphan-reap respawn = physical RunPod H100 wedged at driver level; #770 stops the no-port billing leak but does not auto-pivot a CUDA-IMA-wedged pod to a fresh host.

## Scope / surfaces
- Target: src/explore_persona_space/backends/backend_poll.py, .claude/rules/compute-backend-failover.md

## Constraints / invariants
- Workflow-surface only; ruff + workflow_lint pass; add a test where practical.

## Provenance
- workflow_fix_target: src/explore_persona_space/backends/backend_poll.py, .claude/rules/compute-backend-failover.md
- fingerprint: d73d9e6f015d

Raised by the predictor-line infra retrospective (2026-06-30).
