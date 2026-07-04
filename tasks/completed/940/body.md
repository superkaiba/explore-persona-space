---
title: 'workflow-fix: RunPod terminal rung maps GCP-only intents to nearest RunPod
  intent'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9a65fe461ab5
created_at: '2026-07-03T17:05:21Z'
has_clean_result: false
origin_prompt: 'orchestrator observation #841 2026-07-03: runpod terminal fallback
  FAILED — pod_lifecycle provision --intent capture-7b exit 1 (no RunPod mapping for
  GCP-only intent) after GCP cap+stockout and SLURM branch-gap; auto-chain ended NoComputeAvailableError
  despite RunPod capacity for an equivalent 1-GPU pod'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #841 (round-2 dispatch, 2026-07-03 16:58Z).

## Goal

Make the router's RunPod terminal rung map GCP-only intents (e.g. `capture-7b`) to the nearest RunPod-provisionable intent instead of crashing, so the sanctioned last-rung fallback actually fires when the GCP ladder is exhausted.

## Workflow gap

- **Bug observed:** #841's round-2 launch exhausted GCP (A100 stockout both zones + per-day attempt cap 8) and SLURM (branch gap #653), then the RunPod terminal fallback itself FAILED: `pod_lifecycle.py provision --issue 841 --intent capture-7b` exited 1 — `capture-7b` is a GCP-only intent with no RunPod GPU mapping, so the auto-chain ended in `NoComputeAvailableError` even though RunPod had capacity for an equivalent 1-GPU pod.
- **Why it is a workflow gap:** the router's documented contract is that RunPod is the terminal rung of the capacity path; an intent-vocabulary mismatch between lanes silently voids that guarantee for every GCP-only intent (`capture-7b`, and any future GCP-specific machine-type intents).
- **Confidence (emitter):** high (live epm:backend-selected marker on #841 carries the full attempts evidence).

## Proposed change (candidate diff sketch — refine in planning)

- In `backends/router.py` `_runpod_terminal_rung` (and the async failover path), translate the requested intent through a small GCP-intent→RunPod-intent map before calling `pod_lifecycle.py provision` (e.g. `capture-7b` → `eval` (1×H100) or the closest single-A100/H100 intent), logging the translation on the `epm:backend-selected` marker `extra`.
- Unknown intents with no mapping keep the current fail-loud behavior, but with an error naming the missing map entry.
- Add a router test pinning: GCP-only intent + exhausted ladder → RunPod rung fires with the translated intent.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/router.py`
- Also check: `src/explore_persona_space/backends/runpod.py`, `scripts/pod_lifecycle.py` (intent table), `tests/test_router.py`.
- Grep: `grep -rn "capture-7b\|INTENT_TO_MACHINE\|RUNPOD.*INTENT" src/explore_persona_space/backends/ scripts/pod_lifecycle.py scripts/gpu_heuristics.py`

## Constraints / invariants

- RunPod stays the LAST rung only (tests/test_router.py invariants unchanged).
- No dollar caps; the translation must not widen GPU count (map to the same-or-narrower width).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/router.py
- fingerprint: 9a65fe461ab5

Surfaced observation (verbatim, #841 events.jsonl epm:backend-selected 2026-07-03T16:58:33Z): "runpod terminal fallback FAILED (CalledProcessError: ... pod_lifecycle.py provision --issue 841 --intent capture-7b returned non-zero exit status 1.)" following "rung ondemand_a100_80: per-day GCP attempt cap 8 reached" and both-zone ZONE_RESOURCE_POOL_EXHAUSTED.
