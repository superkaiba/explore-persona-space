---
title: 'Fleet dispatcher: overlap CPU judging with GPU work + multi-GPU parallel cell
  execution'
kind: infra
tags: []
created_at: '2026-06-27T01:23:00Z'
has_clean_result: false
origin_prompt: 'Change the workflow so: CPU only work (e.g. judging) gets done in
  PARALLEL to GPU work as much as possible -- so we get the most out of GPUs; and
  we run on GPUs in parallel AS MUCH AS POSSIBLE -- if it will speed us up. Run this
  as a background issue with happy coder.'
---
## Overview / Motivation

The current experiment fleet dispatcher (canonical example:
`scripts/issue664_dispatch.py`) runs cells **sequentially in a single process on a
single GPU**, and CPU/API work (Claude judging) **blocks the GPU** — the GPU sits idle
(observed 0% util between cells; a watcher "GPU ≤5% for 36 min" advisory) while judging
runs over the network. #664's 48-cell fleet has taken 22h+ on 1×H100 as a result.
Re-architect the dispatcher so GPUs stay saturated and fleets finish in a fraction of
the wall-time.

## Goals (two, both load-bearing)

1. **Overlap CPU-only work with GPU work — keep GPUs busy.** Judging / scoring /
   aggregation (Claude API + CPU reductions) must run **concurrently** with GPU work on
   subsequent cells; the GPU must never block on a CPU/API phase. As soon as a cell's
   GPU outputs (generations / activations) are produced, hand judging off to an
   async/background worker (or the Anthropic **Batch API** — project policy for large
   judge sets) and **immediately start the next cell's GPU work**. Reconcile judge
   results when they return; per-cell idempotent writes make this safe.
2. **Run GPUs in parallel as much as possible — when it speeds us up.** Fan cells out
   across **all** GPUs on a multi-GPU pod (per the project's "one multi-GPU pod, not many
   single-GPU pods" pattern) — `CUDA_VISIBLE_DEVICES` / per-GPU worker processes / a
   `--shard k/N` cell selector — so an N-GPU pod runs ~N cells concurrently. The per-cell
   ops already take a `gpu_id` (`train_cell` / `extract_and_eval_cell` / `merge_lora`), so
   the missing pieces are the cell-sharding loop + provisioning a multi-GPU pod.
   Multi-GPU **only when it speeds things up** (a 1-cell smoke / tiny fleet stays
   single-GPU; don't pay multi-GPU provisioning for work that won't parallelize).

## Scope

- Prefer a **reusable** capability (a shared helper in `src/explore_persona_space/`)
  that `issue664_dispatch.py` AND future fleet dispatchers use, over a per-issue copy.
  At minimum, deliver it for the dispatcher pattern `issue664` uses so the #664
  `em-provenance-robustness` follow-up + any #664 re-run + future fleets run multi-GPU
  with overlapped judging.
- **Provisioning:** fleets should provision a multi-GPU pod (intent sized to fleet
  width, e.g. 4–8×H100) instead of 1×H100; single-cell / smoke stays minimal.
- The planner decides the concrete design (async judge workers vs Batch-API
  fire-and-forget; process-per-GPU vs in-process device round-robin; where the shared
  helper lives). The two goals + the constraints below are the spec.

## Constraints / invariants (do NOT break)

- **Idempotent skip-completed** stays (the `.exists()` checks): re-entry skips done
  cells; concurrent workers take **DISJOINT** cells — no double-work, no two workers
  writing the same cell's output paths (race-free per-cell writes, asserted).
- **Judge = `claude-sonnet-4-5-20250929`**, Batch API for large sets (standing rule).
- **Checkpoint-per-phase / per-cell** — no accumulate-in-memory-write-at-end.
- **WandB live metrics** for training cells.
- **Backward compatible:** the single-GPU path must still work; parallelism is opt-in by
  GPU count / `--shard`.
- **Do NOT disturb the currently running #664 pod** — this is for future / remaining
  fleet work, not a hot-patch of the live 22h run.

## Acceptance criteria

- On an N-GPU pod, a fleet of M independent cells runs ~N in parallel and finishes in
  ~`(M/N) × per-cell-GPU-time + overlap`, not `M × (GPU + judge)` serial.
- GPU utilization stays high across the fleet (judging no longer blocks the GPU);
  include a measured **before/after GPU-util + wall-time** comparison on a small fleet.
- Idempotent re-entry verified (kill mid-fleet → re-run skips completed, finishes rest).
- Disjoint sharding asserted (no two workers touch the same cell).
- Single-GPU + smoke (`--cells 1 --smoke`) still pass.

## Context / pointers

- Current dispatcher: `scripts/issue664_dispatch.py` (sequential `for cell in cells`,
  single `gpu_id`, idempotent `.exists()` skips, `--cells` count cap; no shard selector).
- Project pattern memory: "One multi-GPU pod, not many single-GPU pods"
  (`CUDA_VISIBLE_DEVICES` + per-process source splits) and the `+gpu_id=N`
  CUDA_VISIBLE_DEVICES clobber gotcha for parallel launches.
- Judge / Batch policy: CLAUDE.md (`claude-sonnet-4-5-20250929`, Batch API for large sets).
- First intended consumer: the #664 `em-provenance-robustness` follow-up
  (`epm:followup-scope` on #664).
- Pod provisioning intents: `scripts/pod.py` (`eval` / `lora-7b` / `ft-7b` map to
  multi-GPU specs).
