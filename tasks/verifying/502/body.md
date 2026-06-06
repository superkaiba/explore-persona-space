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

Re-run the #493 extraction-point × metric × layer bake-off at **all 28 residual-stream layers** and **500 probes** (up from 8 layers / 50 probes), **parallelized as much as possible** (batched generation + multi-GPU), to get the full per-layer predictor profile and tighten the n≪d / estimate-noise — preserving the validated #493 extraction / metric / regression logic exactly. The 500-probe pool requires generating ~450 new probe questions (the matched pool caps at 50).

## Background

Parent: #493 ("competing metrics converge within ~0.02 CV R²; no robust win over last-token cosine; extraction-point > metric", LOW confidence, 8 layers × 50 probes). Two coverage gaps:
- **Layers:** 8 of 28 swept. Adding layers is nearly free (hooks capture every layer in the same forward/generation pass).
- **Probes:** 50 is small for the covariance-based metrics (n=50 ≪ d=3584). The matched probe pool (`load_q_test_extended_50`) is hard-capped at 50; all existing sources combined are only ~128 mixed-distribution questions. Reaching 500 requires generating new probe questions (user-approved 2026-06-05).

**Caveats to state up front in the clean-result:**
- The 450 generated probes are **synthetic-but-matched** (Claude-generated to the q_test persona-eval distribution); the predictor cloud is then estimated partly on synthetic questions. The existing 50 q_test are kept as a subset so #493's exact cell is recoverable as a comparability anchor.
- More layers + probes firm up PER-CELL estimates + the layer profile; they do NOT address the real confidence limiters (single seed 42, winner rank-1 on only 1 of 4 cells). Expected outcome: a tighter, fuller version of #493's negative result, NOT a confidence upgrade.

## What to run

Extend the #493 driver (`scripts/issue493_extraction_metric_bakeoff.py` — DO NOT fork the metric/regression/extraction logic) with the steps below.

**Step 0 — probe generation (new, runs locally on the dev VM, no GPU).**
- Claude-generate ~450 standalone user questions matched to the q_test persona-eval distribution (mixed-domain: capabilities, opinion, neutral chat, hypotheticals — same character as the existing 50). Use a Claude API call (project default for generation).
- The final 500-probe pool = the existing 50 `load_q_test_extended_50` (as a subset, first) + 450 new.
- Assert: exactly 500 total; the 450 new are exact-string **disjoint from q_train (30) AND q_test (50)**; dedup within the new set; each is a non-empty single question. Save to a committed JSON (e.g. `eval_results/issue_502/probes_500.json`) with provenance metadata (generation prompt, model, timestamp). Reproducible: same file reused by the run.

**Step 1+ — the bake-off**, identical to #493 except:
- **Layers:** all 28 (0–27).
- **Probes:** the 500-probe pool from Step 0.
- Extraction points / metrics / variants / cells / regression / winner-selection: identical to #493 (end_of_system / last_prompt / mean_response; cosine, euclidean, mahal, mahal_pooled_ctx, mmd, c2st, delta_spec, gauss_kl, wass2; raw + centered; arm ∈ {pos,loc} × epoch ∈ {1,2,3,5}; vs #474 ΔG + g_logprob; length-partial Spearman; LOCO-CV winner). Single seed 42.
- **NEW — add the output-distribution JS baseline (this was missing in #493).** #493 only had activation-space metrics; it never compared against the **next-token JS** that was #474's actual second predictor alongside cosine. Add `next_token_js` as a baseline predictor: per (transformation, probe), softmax the final-layer logits at the last_prompt position → per-probe JS between the two personas' next-token distributions → average over probes (one scalar per pair; NOT layer-indexed — it's an output metric). This reuses the last_prompt forward pass (≈ free) and matches `scripts/issue458_predictor_jsdiv.py` / #406's `D_matrix.json["JS"]` so it's apples-to-apples with #474's JS finding. Include it in the regression + ranking as a baseline next to last-prompt cosine, so the headline becomes "do the activation metrics beat cosine AND JS." Sanity: the recomputed next_token_js should reproduce #406's `D_matrix.json["JS"]` within tolerance on the matching 50 q_test probes (cross-check, like the cosine one). The canonical Rao-Blackwellized **sequence-level** JS is OUT OF SCOPE here (too expensive at 500 probes) — note it as a follow-up.

## Parallelization (parallelize as much as possible)

1. **Batched generation (biggest win).** #493 generates one mean-response per (transformation, probe) in a loop (HF `generate()`; vLLM can't be used — residual activations must be hooked during generation). Batch B probes per `generate()` with left-padding, hook the per-layer residuals for the batch, then per-sequence locate the response-token positions (left-padding + per-sequence length + EOS) and mean-pool.
2. **Multi-GPU.** ONE multi-GPU pod — target **8× H100** (fall back to 4× on supply constraint), NEVER multiple single-GPU pods. Split the 16 transformations across GPUs (8 → 2 each), each process pinned to one GPU via `--gpu-id` / CUDA_VISIBLE_DEVICES (bound before any cuda call), writing per-transformation activation files; a single aggregation step (after all GPU processes finish) computes the metric grid + regression on CPU.
3. **Metric grid (CPU).** At 28 layers × 500 probes the grid (esp. MMD permutation nulls, N²) grows — parallelize across cores (multiprocessing) if it becomes the bottleneck.

## Correctness gates (MANDATORY)

- **Probe-set:** 500 total, 450 new disjoint from q_train + q_test, deduped, committed.
- **Batched == serial equality test** (THE gate for batching): on a tiny slice (2–3 transformations, 4 probes, a couple layers), batched-generation activation extraction must reproduce the serial #493 path within fp tolerance (cosine ≥ 0.999) per extraction point. Add to `--phase smoke`.
- **#406 cosine cross-check still passes** (strict, the 6 layers #406 has, tol 3e-3) on the full run — validated extraction unchanged.
- **Multi-GPU split correctness:** union of per-GPU per-transformation activations == single-GPU result (nothing dropped/duplicated; deterministic given seed).
- Preserve every #493 fix: forward-hook pre-norm extraction (`model.model.layers[L]`, the L+1-equivalent), end_of_system one-vector / Class-A subpanel, non-stylized base-prior-safe winner guard, fail-loud singular pooled-cov, unbiased MMD + perm null, length-partialled LOCO-CV.

## Deliverables

- Extended `scripts/issue493_extraction_metric_bakeoff.py` (+ `scripts/issue502_*` dispatch / probe-gen) with probe generation + batched generation + multi-GPU split, behind flags so #493's serial 8-layer/50-probe path still works.
- `eval_results/issue_502/` (mirror #493 layout) + `figures/issue_502/` — including a **full per-layer predictor profile** (CV R² vs layer, per extraction point / top metrics) #493 couldn't show.
- Clean-result extending #493: the full layer profile (better layer than 21/27?), whether n=500 tightens the band / changes the winner, whether the headline holds. Confidence still bounded by single-seed + single-cell + now the synthetic-probe caveat — state all three.

## Notes

- Multi-GPU pod (8× H100 target / 4× fallback), eval-style (generation + forward passes, no training). No retraining — predictor-only vs #474's existing ΔG.
- Reuse #474 ΔG matrices + the 16 transformations from #493 unchanged.
- `kind: analysis` follow-up (parent #493); adversarial-planner skipped (re-run + explicit user request); parallelization + probe-gen code goes through the full code-review ensemble.
