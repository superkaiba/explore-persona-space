---
title: 'Width-aware auto-router: include 8x GCP rungs; encourage wide GCP provisioning
  for shardable workloads'
kind: infra
tags: []
created_at: '2026-07-08T01:24:49Z'
has_clean_result: false
origin_prompt: change them to be included in the auto-router -> especially for GCP
  we have almost infinite credits so encourage their use
workflow: v1
---
## Overview / Motivation

User directive (2026-07-08, verbatim): "change them to be included in the auto-router -> especially for GCP we have almost infinite credits so encourage their use" — "them" = the 8× GPU machine rungs (`a2-ultragpu-8g` 8× A100-80; `a3-highgpu-8g` 8× H100, FLEX_START/SPOT-only), which are currently explicit-`--intent`-only and deliberately excluded from the auto ladder (#743).

Context that triggered it: #825's `sampled-separator-control` round has a cleanly shardable multi-GPU axis (2 models × K≈8–10 draw-sets × 600 prefixes, ~8 GPU-h of generation + extraction) but the auto lane maps `capture-7b` → 1× A100-80, so it runs ~7h serial on one GPU. GCP credits are effectively unconstrained for this project; wall-clock is the scarce resource. The router should therefore ENCOURAGE wide GCP provisioning whenever the workload actually shards — while keeping every idle-width protection for workloads that don't.

## Goal

Make the unified backend router's AUTO lane width-aware: a dispatch that declares a shardable multi-GPU axis (requested width N > 1) walks WIDE GCP rungs first (8× / 4× / 2× per the available A2/A3 machine types, spot/flex/on-demand per the existing length-aware ladder), falling back to narrower rungs on capacity miss and then the existing chain unchanged. Update the plan-time guidance so wide GCP provisioning is the ENCOURAGED default when a shardable axis exists. RunPod's role (last-resort, real dollars) is UNCHANGED.

## Proposed change (sketch — planner refines)

- **Width threading:** plan §9 declares the shardable axis + requested width; `dispatch_issue.py` forwards `--gpu-count N` (flag exists) and the router maps (intent, width) → wide GCP machine types. Extend `INTENT_TO_MACHINE` (backends/gcp.py) with multi-GPU mappings for the capture/eval/lora-class intents (`a2-ultragpu-2g/4g/8g`; `a3-highgpu-8g` stays FLEX_START/SPOT-only).
- **Ladder semantics:** for width-N requests the GCP ladder tries width-N rungs first (spot lead for short jobs, flex-start lead for long, as today), then degrades width (8→4→2→1) on capacity miss before leaving GCP; a degraded-width launch posts the realized width on `epm:backend-selected` so the workload can re-shard. The SLURM lanes and the RunPod terminal rung are untouched.
- **Guidance surfaces:** update `planner.md` §9 (per-GPU-phase parallelization statement: when a shardable axis exists, default to wide GCP + a work-conserving sharded launch command; narrow remains correct when nothing shards), the `efficiency-critic` plan lens (REVISE an unsharded wide plan AND a needlessly-narrow plan with a declared shardable axis), CLAUDE.md's compute-backends section (the #743 "explicit-`--intent`-only, NOT in the auto-ladder" sentence must be rewritten to describe the new width-aware behavior), and `.claude/rules/compute-backend-failover.md` ladder text.
- **Protections kept, not weakened:** saturate-or-downsize still binding (wide REQUIRES sharded launch commands); the #664/#778 idle-width protections and the narrow-phase release/downsize rule unchanged; per-day GCP attempt-cap semantics decided by the planner (a width-degradation walk should not burn the cap disproportionately); no dollar caps anywhere (`tests/test_no_dollar_budget_caps.py`).

## Constraints / invariants

- `tests/test_router.py` invariants stay green (RunPod-last-rung-only-after-GCP+SLURM-exhausted; GCP-workload-failover no-cascade); ADD width-selection tests (width request → wide machine; capacity-miss width degradation; width-1 behavior byte-identical to today).
- Workflow-surface only; `scripts/workflow_lint.py` passes; ruff on touched files.
- GCP quota reality check at plan time: 8× A100-80 spot/flex capacity in us-central1 + the granted quotas (see memory: preemptible/flex H100 granted at 8) — the planner verifies the quota actually admits the wide machine types before wiring rungs that can never launch.

## Acceptance criteria

1. An auto-lane dispatch declaring width 8 provisions `a2-ultragpu-8g` on GCP when capacity allows, degrades width gracefully on miss, and reaches the existing narrow chain unchanged at width 1.
2. A width-1 dispatch behaves byte-identically to today (regression-pinned).
3. planner.md / efficiency-critic / CLAUDE.md / compute-backend-failover.md all describe the new behavior consistently (lint cross-checks pass).
4. New router tests cover width selection + degradation; full relevant test files green.

## Provenance

- origin: user chat directive 2026-07-08 (this task supersedes the #743 exclusion decision by explicit user direction)
- related_task: #825 (the sampled-separator-control round that surfaced the gap)
