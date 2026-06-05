---
title: 'Re-run #493 bake-off at all 28 layers × 200 probes, parallelized (batched
  gen + multi-GPU)'
kind: analysis
tags: []
created_at: '2026-06-05T17:36:31Z'
has_clean_result: false
parent_id: 493
---
## Goal

Re-run the #493 extraction-point × metric × layer bake-off at **all 28 residual-stream layers** and **200 probes** (up from 8 layers / 50 probes), **parallelized as much as possible** (batched generation + multi-GPU), to (a) get the full per-layer predictor profile and (b) tighten the n≪d / estimate-noise on every predictor — while preserving the validated #493 extraction / metric / regression logic exactly.

## Background

Parent: #493 (a 320-predictor bake-off; clean-result = "competing metrics converge within ~0.02 CV R²; no robust win over last-token cosine; extraction-point > metric", LOW confidence). #493 swept 8 layers {0,5,7,11,14,15,21,27} × 50 probes. Two cheap-to-expand coverage gaps identified in discussion:
- **Layers:** only 8 of 28 swept. Adding layers is nearly free — forward hooks capture every layer in the SAME forward/generation pass, so layer count does NOT multiply the (expensive) generation; it only grows the CPU metric grid.
- **Probes:** 50 is small for the covariance-based metrics (n=50 ≪ d=3584). 200 tightens every ρ/CV estimate and lets Fisher/Gaussian-KL use less PCA truncation.

**Caveat (state up front in the clean-result):** more layers + more probes firm up the PER-CELL estimates and the layer profile; they do NOT address the two real confidence limiters from #493 — single seed 42, and the winner being rank-1 on only one of four usable cells. So the expected outcome is a tighter, fuller version of #493's negative result, not a confidence upgrade.

## What to run

Re-run the #493 driver (`scripts/issue493_extraction_metric_bakeoff.py` — DO NOT fork the metric/regression/extraction logic; extend it or add a thin parallel wrapper) with:
- **Layers:** all 28 (0–27).
- **Probes:** 200, sourced from the SAME preregistered pool #493/#406 used, ideally a **superset of #493's 50** (so results are comparable). The implementer must verify the probe pool actually has ≥200 disjoint-from-eval questions (the #404 `fetch_preregistered_probes(n=200, exclude=Betley main 8)` path supported 200; `eval_results/issue_406/predictor_inputs.json` may only hold 50 — resolve this and document the source). If <200 available, use the max and say so.
- **Extraction points + metrics + cells:** identical to #493 (end_of_system / last_prompt / mean_response; cosine, euclidean, mahal, mahal_pooled_ctx, mmd, c2st, delta_spec, gauss_kl, wass2; raw + centered; arm ∈ {pos,loc} × epoch ∈ {1,2,3,5}; regress vs #474 ΔG + g_logprob; length-partial Spearman; LOCO-CV winner). Single seed 42.

## Parallelization (the new work — parallelize as much as possible)

1. **Batched generation (biggest win).** #493 generates one response per (transformation, probe) in a loop (HF `generate()` — vLLM can't be used because residual activations must be hooked during generation). Batch B probes per `generate()` call with left-padding, hook the per-layer residuals for the whole batch, then per-sequence identify the response-token positions (accounting for left-padding + per-sequence generation length + EOS) and mean-pool. This is the highest-risk change — see correctness gate.
2. **Multi-GPU.** Provision ONE multi-GPU pod (target 8× H100, fall back to 4× on supply constraint) — NEVER multiple single-GPU pods (CLAUDE.md). Split the 16 transformations across GPUs (8 GPUs → 2 transformations/GPU; 4 → 4 each), each process pinned to one GPU via the script's `--gpu-id` / CUDA_VISIBLE_DEVICES (bind before any cuda call), writing per-transformation activation files; a single aggregation step (after all GPU processes finish) loads all activations and computes the metric grid + regression on CPU.
3. **Metric grid (CPU).** At 28 layers × 200 probes the metric grid (esp. MMD permutation nulls, N²) grows; parallelize across cores (multiprocessing) if it becomes the bottleneck.

## Correctness gates (MANDATORY — batching is bug-prone)

- **Batched == serial equality test:** on a tiny slice (2-3 transformations, 4 probes, a couple layers), assert batched-generation activation extraction reproduces the serial #493 path within fp tolerance (cosine ≥ 0.999 / max-abs-diff small) per extraction point. This is THE gate for the batching change. Add it to `--phase smoke`.
- **#406 cosine cross-check still passes** (strict, the 6 layers #406 has, tol 3e-3) on the full real run — the validated extraction must be unchanged.
- **Multi-GPU split correctness:** the union of per-GPU per-transformation activations == the single-GPU result (no transformation dropped/duplicated; deterministic given seed).
- Preserve every #493 fix: hook `hidden_states[L+1]`-equivalent (the forward-hook on `model.model.layers[L]`) pre-norm extraction, end_of_system one-vector/Class-A-subpanel handling, non-stylized base-prior-safe winner guard, fail-loud singular pooled-cov, unbiased MMD + perm null, length-partialled LOCO-CV.

## Deliverables

- Updated/extended `scripts/issue493_extraction_metric_bakeoff.py` (+ any `scripts/issue494_*` dispatch wrapper) with batched generation + multi-GPU split, behind flags so #493's serial path still works.
- Results under `eval_results/issue_494/bakeoff/` (mirror #493's layout); figures under `figures/issue_494/` — including a **full per-layer predictor profile** (CV R² vs layer, per extraction point / top metrics) which #493 couldn't show.
- Clean-result extending #493: the full layer profile (is there a better layer than 21/27?), whether the n=200 estimates tighten the band / change the winner, and whether the headline (no robust win over cosine; extraction-point > metric) holds. Confidence still bounded by single-seed + single-cell — state it.

## Notes

- Needs a multi-GPU pod (8× H100 target / 4× fallback), intent eval-style (generation + forward passes, no training). No retraining — predictor-only re-run against #474's existing ΔG.
- Reuse #474 ΔG matrices + the 16 transformations from #493 unchanged.
- This is a `kind: analysis` follow-up (parent #493); adversarial-planner skipped per the re-run + explicit-user-request path — but the parallelization code goes through the full code-review ensemble.
