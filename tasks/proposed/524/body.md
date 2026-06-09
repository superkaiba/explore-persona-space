---
title: 'Directional/asymmetric base-model predictors of marker-leakage: can they recover
  the antisymmetric component the symmetric #502 zoo is blind to?'
kind: experiment
tags: []
created_at: '2026-06-09T00:16:35Z'
has_clean_result: false
parent_id: 502
goal: 'Test whether directional/asymmetric base-model predictors (directional Gaussian-KL,
  source-covariance Mahalanobis, asymmetric subspace-reconstruction, marker-direction
  projection, two-feature combiner, plus re-extracted full-response and next-token
  one-way KL) computed on the #502 data/recipe recover the ~28% antisymmetric component
  of marker-leakage ΔG that symmetric predictors are provably blind to -- scored by
  length-partial Spearman rho + LOOCV against ΔG_anti and full ΔG on loc-arm epoch
  1, validated on #523''s held-out pool.'
---
# Directional/asymmetric base-model predictors of marker-leakage ΔG: can they recover the antisymmetric component the symmetric #502 zoo is blind to?

## Goal

Test whether directional/asymmetric base-model predictors (directional Gaussian-KL, source-covariance Mahalanobis, asymmetric subspace-reconstruction, marker-direction projection, two-feature combiner, plus re-extracted full-response and next-token one-way KL) computed on the #502 data/recipe recover the ~28% antisymmetric component of marker-leakage ΔG that symmetric predictors are provably blind to -- scored by length-partial Spearman rho + LOOCV against ΔG_anti and full ΔG on loc-arm epoch 1, validated on #523's held-out pool.

## Background

#502's winning leakage predictor — `last_prompt × L22 × Gaussian-KL × raw`, ρ = -0.79 — is symmetric by construction (`gauss_kl = ½(KL(N_A‖N_B)+KL(N_B‖N_A))`; so are cosine, MMD, Wasserstein, JS). The ΔG target is directional: ΔG[A→B] ≠ ΔG[B→A], 240 ordered pairs.

A decomposition of #474's loc-ep1 ΔG matrix (the "free fix", `scripts/issue502_deltaG_symmetry.py`) established:
- **28.3%** of off-diagonal ΔG variance is antisymmetric (directional).
- A symmetric predictor's R² on full ΔG is therefore capped at **0.72**.
- The #502 winner correlates with the symmetric part at ρ = -0.91 but with the antisymmetric part at **ρ = +0.000** (exactly zero, as proven) — structurally blind to directional leakage.

The cheapest directional predictor already tested — #406's one-way full-response output KL — is genuinely asymmetric (KL(A‖B) ρ=-0.528 vs KL(B‖A) -0.477 vs ΔG) but barely touches ΔG_anti (ρ = -0.049). So output-side directional divergence ≠ directional leakage; the hope is the activation-side directional predictors.

A deep-dive + a verified web deep-research converged on the same conclusion: the literature has no off-the-shelf base-model-only asymmetric predictor — the clean base-model-only subspace measures are symmetric and the clean directional transfer metrics (NTK-forgetting, EWC/Fisher, NCE, OTCE) need trained deltas or labels. So the asymmetric predictor must be BUILT as an asymmetric functional on base-computable ingredients.

## Design — everything on the #502 data/recipe (Qwen-2.5-7B base, 16 contexts, 500-probe pool)

Reuse #502's 16 #406 conditions (`src/explore_persona_space/experiments/i406_conditions.py`), the 500-probe pool (`eval_results/issue_502/probes_500.json`), the cached residual-stream activation clouds (HF `superkaiba1/explore-persona-space-data` `issue502_28layer_500probe_bakeoff/`), the #474 ΔG target matrices, and the #502 bake-off scoring (`scripts/issue493_extraction_metric_bakeoff.py` `_length_partial`, `_loocv_r2`).

### Predictor list (all on the #502 recipe)

**Symmetric anchors — load from #502, no recompute (provably ρ_anti = 0):**
1. cosine
2. Gaussian-KL (symmetrized)
3. next-token JS (output, last-prompt, single position)
(rest of the #502 symmetric zoo — MMD, W2, Euclidean, Mahalanobis, c2st, spectral-delta — load if useful for the comparison chart)

**Output-distribution divergences — RE-EXTRACT on the #502 recipe (one GPU pass):**
4. full-response JS — sequence-level JS over the entire response, on the 500 probes, 16 contexts. Symmetric.
5. full-response one-way KL — directional KL over the entire response, BOTH directions. Asymmetric.
6. next-token one-way KL — single-position directional output KL, BOTH directions. Asymmetric.
   - Recipe: reuse #502's greedy responses (the same responses behind the `mean_response` extraction); teacher-force each context-pair; per-position full-vocab divergence averaged over the response (length-normalized). Match #502's greedy convention; do NOT carry #406's 50-probe values.

**Asymmetric activation predictors — NEW, CPU on the cached #502 clouds:**
7. **Directional Gaussian-KL** — `KL(N_A‖N_B)` un-symmetrized (PCA-16 subspace, as in #502's gauss_kl but keeping both directions separately), swept over layers, at BOTH `last_prompt` and `mean_response`. Asymmetric via Σ_A⁻¹ ≠ Σ_B⁻¹.
8. **Source-covariance Mahalanobis** — `(μ_B−μ_A)ᵀ Σ_A⁻¹ (μ_B−μ_A)`, both directions. Asymmetric (whitened by the source's covariance, not pooled). Subsumes Euclidean/cosine as the Σ_A=I case.
9. **Asymmetric subspace-reconstruction (A→B)** — `‖P_{A,k} Φ_B‖ / ‖Φ_B‖` = fraction of B's cloud inside A's top-k principal subspace, both directions. MUST use UNEQUAL ranks for A vs B (equal-dim subspace angles collapse to symmetric). Reuse the dual/Gram PCA in the bake-off.
10. **Marker-direction projection** — project B's activation cloud onto the marker ` ※` (token id 83399) direction (unembedding row pulled back to the residual stream), in A's frame. Plain projection — NOT the centered "projection-difference" template (refuted 0-3 in the deep-research). Directional wiring (B-in-A's-frame vs projection of μ_A−μ_B) to be pinned by the planner; test variants.
11. **Two-feature combiner** — small length-controlled regression `ΔG ≈ β0 + β1·f_geom + β2·f_marker + β3·(f_geom × f_marker)` where f_geom is one of #7–9 and f_marker is #10. The interaction term β3 is the point (Hiratani arXiv:2405.20236: directional transfer needs feature-side × readout-side similarity; a single scalar can't capture the sign-flip). Most speculative; strict out-of-sample CV required.

### Scoring + validation
- Targets: full ΔG, ΔG_sym = ½(ΔG[A→B]+ΔG[B→A]), and **ΔG_anti = ½(ΔG[A→B]−ΔG[B→A])** — the last is the key test.
- Panel: full 240-pair (non-stylized dropped per user decision).
- Layers/extraction: sweep all 28 layers × {last_prompt, mean_response} for the activation predictors; output divergences are layer-free.
- Both directions tested for every directional metric; report which direction maps to A→B leakage.
- Convention: length-partial Spearman ρ + leave-one-context-out CV R² (the #502 machinery).
- Out-of-sample: validate the winners on #523's held-out probe pool + nested CV (so the winner is not selection-fit).
- Anchor checks: confirm the symmetric anchors score ρ_anti ≈ 0 (sanity); reproduce #502's gauss_kl ρ on the shared pairs.

### What would update the belief
- If any of #5–11 reaches a meaningfully non-zero ρ against ΔG_anti (CI excluding zero, surviving the held-out pool), there IS recoverable directional leakage signal in the base-model geometry, and we name the best directional predictor.
- If all of #5–11 land near ρ_anti ≈ 0 (like the output-side KL did), the directional 28% is not linearly recoverable from base-model geometry — a strong negative result that bounds the whole "geometry predicts leakage" program.

## Reuse vs new
- **Reuse:** 16 conditions, 500-probe pool, cached activation clouds, #474 ΔG matrices, #502 bake-off scoring functions, the ΔG decomposition (`scripts/issue502_deltaG_symmetry.py`), #523's held-out pool for validation.
- **New code:** the 5 asymmetric activation predictors (#7–11); a GPU re-extraction of full-response + next-token one-way divergences (#4–6) on the 500-probe pool; the two-feature regression; ΔG_anti scoring wired into the bake-off.

## Compute
- One eval pod (1× H100, intent `eval`) for the GPU re-extraction of #4–6 (response generation reuse + teacher-forcing across 240 pairs × 500 probes). Estimated modest (single pod, a few hours).
- Everything else CPU on the dev VM over the cached clouds.

Parent: #502. Substrate: #474 (ΔG), #406 (conditions + the deprecated output-divergence rig), #493 (8-layer bake-off). Validation pool: #523.
