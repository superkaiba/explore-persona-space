---
title: 'Size-gate the CPU-off-pod rule: route large-data CPU-only phases to a CPU-only
  GCP instance (gpu_count=0 n2-highmem) instead of the shared VM'
kind: infra
tags: []
created_at: '2026-06-27T01:27:47Z'
has_clean_result: false
origin_prompt: 'Change it so if there''s large data involved then it should run on
  a CPU-only RunPod or GCP pod (verified feasible: GCP-CPU clean path, RunPod-CPU
  needs new create path). After the #658 139GB-store-on-the-VM disk-fill incident.'
---
## Problem

Large-data CPU-only analysis phases run on the shared CPU **orchestration VM**
per the "CPU-only phases run off-pod on the VM against uploaded artifacts" rule
(`planner.md §9`, CLAUDE.md "CPU-only phases don't hold GPU pods"). That rule was
written for SMALL phases (bootstrap/permutation stats, judge scoring, eval-JSON
aggregation). It has **no size guard** — so #658's predictor-fitting
(`issue658_fit_predictors.py`) materialized a **139 GiB activation store** in its
worktree on the 485 GiB shared VM and ran for 15h+, filling root disk to 96% and
choking foreground Bash spawns across all VM sessions (incident 2026-06-26).

## Change

Make the CPU-off-pod rule **size-gated**:
- SMALL CPU phases (working set below threshold) → stay on the VM (unchanged).
- **LARGE-DATA CPU phases (working set ≳ threshold, default ~30–50 GiB) → run on
  a CPU-ONLY pod/instance** that downloads its inputs from HF, runs, uploads
  results, and tears itself down — decoupling big stores from the shared VM.

## Feasibility (VERIFIED 2026-06-26, before filing)

- **GCP CPU instance = clean primary path.** GCP accelerator-optimized machine
  types BUNDLE the GPU into the machine type (gcp.py: "the a2-ultragpu-* family
  hardcodes the accelerator; not threaded into gcloud") — so there is NO
  `--accelerator` flag to gate; a CPU-only instance is simply a CPU machine type
  (e.g. `n2-highmem-16`) with no GPU. `backends/gcp.get_machine_spec`'s override
  path already accepts `gpu_count=0`. **GCP CPU capacity is abundant** (the
  `ZONE_RESOURCE_POOL_EXHAUSTED` stockout is GPU-only), so this lane is reliable
  even while A100s are dry.
- **RunPod CPU pod = possible but net-new code.** RunPod offers CPU pods, but
  `scripts/runpod_api.create_pod` hard-asserts `gpu_count >= 1` (≈line 501) and
  uses the GPU-only `gpuTypeId` create mutation — a CPU pod needs RunPod's
  separate CPU-deploy API path. Implement as an OPTIONAL later fallback, not now.

## Implementation (GCP-CPU lane)

1. **New CPU intent** (e.g. `cpu-bigmem`) → `MachineSpec(machine_type="n2-highmem-16"
   or sized to the working set, gpu_count=0, gpu_kind="CPU")` with a large
   `--boot-disk-size` (≥ store size + headroom). Add to `INTENT_TO_MACHINE` (or the
   override path) + `pod.py` intent table + `--list-intents`.
2. **`gcp.py` create handles `gpu_count=0`:** CPU machine type; `--maintenance-policy=MIGRATE`
   (CPU VMs can live-migrate, unlike GPU which must TERMINATE); **SKIP the
   accelerator-quota preflight** when `gpu_count=0` (no accelerator quota — gate on
   CPU/disk quota instead); optionally a non-CUDA image (DLVM works on CPU but is
   wasteful). Keep `--instance-termination-action=DELETE` + `--max-run-duration` so
   it stays ephemeral.
3. **Router / `dispatch_issue.py`:** expose the CPU intent; the large-data CPU phase
   on the instance downloads the store from HF (`huggingface_hub`), runs, uploads
   results to HF / commits eval JSON, then the instance auto-deletes. (No GPU pod
   held; no 139 GiB on the shared VM.)
4. **Rule update (size-gated):** `planner.md §9` + the CLAUDE.md "CPU-only phases"
   bullet — small → VM, large (≳ threshold) → CPU GCP instance; `critic.md`
   Methodology lens item 10 enforces it. Name the threshold + how the planner
   estimates the working set.
5. **Tests:** `tests/test_gcp_backend.py` — `gpu_count=0` renders a valid CPU
   `gcloud instances create` argv (CPU machine, no accelerator, `MIGRATE`, big
   `--boot-disk-size`, no accelerator-quota call); a routing test that a
   large-working-set CPU phase selects the CPU intent and a small one stays on VM.

## Scope / acceptance

- A CPU-only GCP instance can be provisioned (`gpu_count=0`, CPU machine, big
  disk) and run a CPU analysis phase that pulls its store from HF — verified at
  least by argv-render golden tests; a live smoke (provision a real `n2-highmem`,
  pull a small store, run, delete) is a nice-to-have if cheap.
- Large-data CPU phases route off the shared VM; small ones stay (size-gated).
- `backends/*.py` is workflow surface — full /issue treatment (planner/critic/code-review).
- RunPod-CPU path is explicitly OUT of scope here (optional later fallback).

## Provenance

User directive 2026-06-26 ("if there's large data involved, run it on a CPU-only
RunPod or GCP pod") after the #658 139 GiB-store-on-the-shared-VM disk-fill
incident. Feasibility verified before filing (GCP-CPU clean; RunPod-CPU needs new
create path).
