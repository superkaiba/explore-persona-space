---
title: 'Parallelize issue664 dispatcher p0: 8-way GPU fan + interleave base-extraction
  with the judge-batch wait (sibling to #676)'
kind: infra
tags: []
created_at: '2026-06-28T00:29:21Z'
has_clean_result: false
origin_prompt: run as infra task in background with happy coder — parallelize p0 (GPUs
  idle during the judge-bound elicitation phase; runs single-shard GPU 0)
---
## Goal

Parallelize the **p0 phase** of the leakage-fleet dispatcher (`scripts/issue664_dispatch.py`) so it does not leave 7 of 8 GPUs idle and does not idle the GPUs during judge-bound stretches. Today p0 runs **single-shard on GPU 0** (`--phase p0 --gpu-id 0 --shard-id 0 --num-shards 8`) while GPUs 1–7 sit at 0 MiB, and GPU 0 itself idles during the Claude judge-filter — so an 8×H100 pod runs p0 at ~⅛ of one GPU's throughput. The plan (#664 plan §cost: "P2.0 … data-parallel 8×, 2 wall-h / 8 GPU-h") budgeted p0 as 8-way; the implementation is single-shard. The eval phase (p2) already got this treatment in #676 — this extends the same pattern upstream to p0.

This is the artifact-I/O + compute-throughput sibling of #676 (multi-GPU parallel cell execution + overlap-CPU-judging-with-GPU). NOT workflow-surface — it is experiment-code (the dispatcher), same class as #676.

## What p0 does (the work to schedule)

p0 (`run_all` p0 branch) mixes GPU-heavy and API-bound work:
- **GPU-heavy:** base greedy generation (50 contexts × per-behavior batteries) + activation extraction `c_C`/`v0`/`t_{C,B}` (all 28 layers) + on-policy elicitation *generation* (sycophancy/refusal positives, secure-code negatives) + baseline-propensity reads.
- **API/CPU-bound (GPUs idle today):** the Claude judge-filter over the elicited completions, and per-cell training-mix build.

## The two fixes

**(1) 8-way GPU fan of p0's GPU work.** Shard the base extraction (by context / battery) and the elicitation generation (by source × behavior) across all `--num-shards` GPUs, reusing the p2 `WaveDispatcher` sharding pattern (#676) rather than running p0 on `--gpu-id 0` only. Each shard extracts/generates a disjoint subset; results merge deterministically (the store + pools are keyed by context/source/cell, so concatenation is order-independent). Keep the single-GPU path working for `n_gpus==1`.

**(2) Interleave the judge-batch wait with GPU-heavy base extraction.** Extend the #676 deferred-judge pattern (already partly present in the `_elicit_*` jobs: "submit the judge right after generation, reconcile later, off the GPU critical path") so the GPU-heavy base extraction is SCHEDULED to run while the elicitation judge batch is in flight on the Anthropic Batch API. The GPUs should never block on judging — submit the elicitation batch, keep the GPUs busy with base extraction / the next behavior's generation, reconcile judge labels before build-mixes. Use the Batch API for the elicitation judge set (project standing rule), not synchronous per-call judging.

## Reuse / files

- Reuse: the p2 wave-sharding + deferred-judge code from #676 (`scripts/issue664_dispatch.py` `WaveDispatcher`, `judge_jobs` deferred-reconcile, `_classify_cell_hub_state`). The p0 fan should mirror it, not reinvent.
- Likely touch: `scripts/issue664_dispatch.py` (p0 branch of `run_all`, the `_elicit_*` functions, the base-extraction call, `issue664_launch_parallel.sh` wrapper so p0 fans like p2), possibly `scripts/issue664_common.py` / `scripts/issue664_extract_store.py` for the per-shard context/source split.

## Constraints (correctness — must hold)

- **Identical outputs.** The parallelized p0 must produce the SAME base store (`v0`/`c_C`/`t`), the SAME elicitation pools, and the SAME 80%-yield-floor drop set as the single-shard p0 — this is a scheduling/throughput change only, not a science change. Add a test asserting per-shard-then-merged == single-shard for a small fixture.
- **Determinism / seeds** preserved per cell (sharding must not change which seed/prompt a cell sees).
- **Do not destabilize the live #664 recovery.** #664 is mid-p0 on pod-664 right now; the running process won't pick up this change (already launched). It applies to future runs / respawns. Land it cleanly through full review; a later #664 respawn that re-syncs picking up the faster p0 is acceptable (reviewed + tested).

## Validation

- Unit tests for the per-shard context/source split + deterministic merge (mockable, no GPU).
- Optional GPU smoke on a small subset (1–2 contexts, `--num-shards 2`) verifying all shards' GPUs are utilized and merged output == single-shard output. Keep it cheap (<1 GPU-h).

## Related
#676 (eval-phase multi-GPU + overlap-judge — the reuse template), #664 (the run that exposed the single-shard p0), #660 (the leakage program), #689 (the recent RunPod-wedge + per-cell-upload fix).
