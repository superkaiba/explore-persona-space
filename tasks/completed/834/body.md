---
title: Vectorize null_battery draws×layers loops (batched einsum Pearson; 50-100x)
kind: infra
tags:
- vectorize
created_at: '2026-07-02T05:30:16Z'
has_clean_result: false
origin_prompt: 'Can this be vectorized: #778 ... null-battery stage (the CPU recompute
  of the 4 nulls ...)'
---
## Goal

Vectorize the draw/layer Python loops in `src/explore_persona_space/analysis/null_battery.py` into batched einsum/GEMM ops so a 1000-draw 4-null battery runs in minutes, not hours.

## Motivation

#778's corrected-monitoring null battery (`--n-draws 1000`, raised from 200 by the statistics critic) ran 2+ hours at ~324% CPU. The module loops Python-level over draws (`for d in range(n_draws)` in `perm_null_draws` / `randnorm_null_draws`) and layers (`r_per_layer` calling scalar `_pearson` 28×), so wall time is op-dispatch overhead, not FLOPs (~1-2 TFLOP of actual math). Same overhead-bound many-cell pattern as #722 (`.claude/rules/vectorize-many-cell-fits.md`) — closed-form here, so pure numpy batching, no GD helper needed.

## Sketch

- Stack K draw directions; projections for all draws×layers in one `einsum('nld,kd->nlk')` (or per-layer `[n,d]@[d,K]`).
- Batched Pearson: standardize target once + projections along n; mean-of-products gives all 28×K r values in one op.
- Permutation directions: diff-of-means for K shuffles as one `[K,n_pool]@[n_pool,28*d]` matmul. Randnorm: per-layer `chol @ Z` with Z [d,K].
- Within-condition r: batch draws×layers per condition (condition count is small).
- Keep the public API (`perm_null_draws`, `randnorm_null_draws`, `r_per_layer`, `within_condition_r_per_layer`) + seeds/draw semantics EXACTLY (regression test: same seed → same draws as the loop version on a small case). Memory guard: float32 direction stack, chunk K if needed.

## Scope

`src/explore_persona_space/analysis/null_battery.py` + tests. Consumers (issue778/816 battery scripts) unchanged. NOT retrofit into live #778/#816 worktrees — main-only, for future batteries.
