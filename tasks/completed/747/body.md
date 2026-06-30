---
title: Add cheap CPU-only pod lanes (RunPod CPU3/5 + GCP E2/N2 spot) and prefer them
  for CPU phases
kind: infra
tags: []
created_at: '2026-06-30T01:33:15Z'
has_clean_result: false
origin_prompt: Add these CPU only pods to the workflow and note to use them whenever
  possible for running CPU things (potentially in parallel)
---
## Overview / Motivation

The unified backend router currently has **one** GCP CPU intent
(`cpu-bigmem` → `n2-highmem-16`, added in #677) and treats RunPod as
having **no CPU lane**: `backends/router.py` (the `base.gpu_count == 0`
branch, ~line 1964) short-circuits a CPU-only intent to exactly one
on-demand GCP CPU rung and, when GCP CPU is exhausted, surfaces the
typed terminal `cpu_exhausted_no_runpod_lane` (`router.py` ~line 2143,
`ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD`) instead of failing over.
`scripts/runpod_api.py::_deploy_once` hard-asserts `gpu_count >= 1`
(line 501) via `podFindAndDeployOnDemand`, so the API cannot create a
CPU pod at all today.

That #677 decision predates RunPod's CPU-pod offering. As of 2026-06-29
both providers offer cheap CPU-only compute:

- **RunPod CPU pods** — families CPU3G / CPU3C / CPU5C (general /
  compute / memory optimized), `gpuCount=0`, per-second billing, ~$0.01/hr
  floor, small tiers ~$0.06/hr, **not preemptible**. Disk auto-sized
  vCPU×10 GB (gen-3) / ×15 GB (gen-5). (RunPod CPU deploy uses a distinct
  GraphQL path / instance-id selection — NOT `podFindAndDeployOnDemand`;
  the implementer must confirm the exact mutation.)
- **GCP E2 spot** — `e2-standard-2` (2 vCPU / 8 GB) $0.067/hr on-demand →
  ~$0.02/hr spot (~$0.01/vCPU-hr); spot discount up to ~91%.
- **GCP n2-highmem-16** (the existing `cpu-bigmem`) — ~$1.06/hr on-demand
  → $0.31/hr spot.

The standing policy ("CPU-only phases don't hold GPU pods") runs CPU-only
phases off-pod on the **free shared VM** by default, routing off-VM only
for >50 GB footprint or a gradient-descent fit. The shared VM has
repeatedly stalled the whole fleet on disk pressure (#658: a 139 GB
activation store filled `/`). Cheap dedicated CPU pods remove that
contention and let many CPU jobs run in parallel.

## Goal

Make cheap CPU-only pods first-class compute lanes (GCP E2/N2 spot + RunPod
CPU3/5) and shift the standing CPU-routing policy to **prefer dedicated
cheap CPU pods for CPU-only phases** — especially heavy, parallelizable, or
large-footprint work — over cramming onto the shared VM, while keeping
trivial quick ops (sub-minute probes, quick plots) on the VM. Enable
running CPU jobs **in parallel** across multiple cheap CPU pods. Preserve
the existing **GCP-first** lane order (cheap GCP CPU before RunPod CPU;
RunPod CPU is the fallback, consistent with the existing GPU rule).

## Scope / surfaces (planner to refine; grep before editing)

1. `scripts/runpod_api.py` — add a CPU-only pod-create path (`gpuCount=0`,
   RunPod CPU flavor/instance selection: CPU3G/CPU3C/CPU5C + vCPU count).
   The `gpu_count >= 1` assert in `_deploy_once` (line 501) and the
   `podFindAndDeployOnDemand` mutation are GPU-only — CPU needs its own
   deploy + the right RunPod CPU mutation. Confirm the exact API surface.
2. `src/explore_persona_space/backends/runpod.py` — handle CPU intents
   (`gpu_count == 0`) → RunPod CPU pod create; map CPU intents → CPU pod
   types.
3. `src/explore_persona_space/backends/gcp.py` — add cheap GCP CPU
   intent(s) beyond `cpu-bigmem` (e.g. a `cpu-small` → `e2-standard-N`
   lane); keep `cpu-bigmem`. Wire CPU spot eligibility for the cheap E2
   lane (spot for short/restartable CPU jobs, on-demand otherwise — the
   CPU analogue of the length-aware GPU ladder). Record the per-machine
   zone availability for any new CPU machine type.
4. `src/explore_persona_space/backends/router.py` — **relax the
   `cpu_exhausted_no_runpod_lane` hard terminal**: the `base.gpu_count == 0`
   branch should fall over to a RunPod CPU lane when GCP CPU capacity is
   exhausted (the CPU analogue of the GPU GCP→RunPod fallback), and add a
   cheap-CPU-spot rung for the small CPU intents while preserving the
   reliable on-demand routing for `cpu-bigmem` large-footprint analysis.
   This **supersedes** the #677 hard terminal — call out the reversal
   explicitly in the router docstring + `compute-backend-failover.md`.
5. `scripts/pod.py` — CPU intents in the intent table + `--list-intents`;
   allow provisioning CPU pods. The "one multi-GPU pod, not many
   single-GPU pods" rule is GPU-specific — for CPU, **allow N parallel
   cheap CPU pods**.
6. `CLAUDE.md` — update the standing rule "CPU-only phases don't hold GPU
   pods", the "Compute backends — multi-lane router" section, and the GPU
   intent table to (a) prefer cheap CPU pods for non-trivial CPU phases,
   (b) keep trivial quick ops on the VM, (c) allow parallel CPU pods,
   (d) preserve the >50 GB → big-volume and gradient-descent-fit → GPU
   carve-outs.
7. `.claude/rules/compute-backend-failover.md` (CPU GCP→RunPod failover),
   and the "CPU-only phases" carve-out prose wherever it lives
   (`planner.md` §9, `critic.md` Methodology lens item 10) — default
   heavy/parallel CPU phases to cheap CPU pods.
8. Tests — `tests/test_router.py` (GCP CPU exhausted → RunPod CPU
   fallover; the relaxed terminal; cheap-CPU-spot rung), CPU intent
   mapping in `tests/test_gcp_backend.py` / RunPod tests, parallel-CPU-pod
   allowance.

## Constraints / invariants

- **GCP-first** lane order preserved (cheap GCP CPU before RunPod CPU).
- No dollar-budget caps (`tests/test_no_dollar_budget_caps.py`).
- Do not break the existing `cpu-bigmem` large-footprint routing or the
  >50 GB VM-footprint carve-out.
- Trivial quick CPU ops stay on the VM (don't pay provisioning overhead
  for a sub-minute op) — the planner sets the threshold.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files;
  CLAUDE.md ↔ rule files ↔ `router.py` docstring stay in sync.

## Provenance

- origin: user chat directive 2026-06-29 ("Add these CPU only pods to the
  workflow and note to use them whenever possible for running CPU things
  (potentially in parallel)").
- supersedes the #677 `cpu_exhausted_no_runpod_lane` hard terminal.
- pricing reference captured 2026-06-29 (RunPod docs + GCP pricing).
