---
title: Directional/asymmetric base-model predictors of marker-leakage on a unified
  instruction+ICL panel, scored by incremental nested-CV skill
kind: experiment
tags: []
created_at: '2026-06-09T00:16:35Z'
has_clean_result: false
parent_id: 502
goal: 'On a unified instruction+ICL context panel (16 #502 instruction + 16 #489 ICL
  contexts), with the marker re-implanted at an off-saturation checkpoint for clean
  ΔG, test whether directional/asymmetric base-model predictors recover the antisymmetric
  component of marker-leakage transfer -- decision metric = incremental nested-CV
  ΔR² over the best symmetric baseline (leave-two-contexts-out, context-clustered
  bootstrap), with standalone vs ΔG_anti as the directional fraud test and a censored/Tobit
  model on any saturated cells.'
---
# Directional/asymmetric base-model predictors of marker-leakage on a UNIFIED instruction+ICL panel, scored by incremental nested-CV skill

## Goal

On a unified instruction+ICL context panel (16 #502 instruction + 16 #489 ICL contexts), with the marker re-implanted at an off-saturation checkpoint for clean ΔG, test whether directional/asymmetric base-model predictors recover the antisymmetric component of marker-leakage transfer -- decision metric = incremental nested-CV ΔR² over the best symmetric baseline (leave-two-contexts-out, context-clustered bootstrap), with standalone vs ΔG_anti as the directional fraud test and a censored/Tobit model on any saturated cells.

## Background

#502's winning leakage predictor (`last_prompt × L22 × Gaussian-KL`, ρ=−0.79) is symmetric by construction; the ΔG target is directional (ΔG[A→B] ≠ ΔG[B→A]). The free-fix decomposition (`scripts/issue502_deltaG_symmetry.py`) showed ~28% of off-diagonal ΔG variance is antisymmetric, a symmetric predictor's R² is capped at ~0.72, and the winner correlates with ΔG_anti at exactly 0. The cheapest directional predictor tried (output-side one-way KL, #406) barely touched ΔG_anti (ρ=−0.05). A deep-dive + verified deep-research converged: there is no off-the-shelf base-model-only asymmetric predictor; it must be built as an asymmetric functional on base-computable ingredients.

**Why add ICL contexts (from #489):** #502's panel is instruction-only (personas, phrasings, register rewrites). ICL contexts induce the persona differently and spread the representation with more dynamic range (#468), and the cross-type ICL↔instruction cells test whether the predictor generalizes across induction mechanisms — a stronger result than instruction-only.

**The #489 lesson (why we must retrain, not reuse):** #489 built a 24-context ICL+SP panel but its ΔG is **floor-saturated** — the marker never emits on-policy anywhere, even on the diagonal (trained log P(marker) ≈ −22.5, ~20 nats below the emission boundary), so #489 is LOW confidence and its ΔG is sub-floor wiggle, not leakage. The planner MUST diagnose why the ICL marker implant floored (under-training? the ICL scaffold competing with the marker slot? negative-set composition?) and fix the recipe so the implant emits on-policy before any ΔG is read. Saturated cells are NOT a usable prediction target (see Metric protocol §censored).

**The metric (from the metric deep-dive, gist 3bc6aeb):** stop reporting standalone correlation; the decision metric is incremental nested-CV ΔR² over the symmetric baseline, leave-TWO-contexts-out, context-clustered bootstrap. loc_ep1 was the only off-saturation clean checkpoint in the old setup — so getting clean ΔG on the unified panel is the precondition for the metric to mean anything.

## Design

### Panel (unified, ~32 contexts)
- 16 instruction contexts from #502/#474 (`src/explore_persona_space/experiments/i406_conditions.py`).
- 16 ICL contexts from #489 (its in-context-example scaffolds: K-shot Q-A slices, Socratic, math-CoT, coding, domain-mixes, persona-voice ICL, register ICL, zero-shot).
- Dedup the SP contexts that appear in both. Keep #489's matched same-identity cross-type pairs (e.g. pirate-voice ICL ↔ pirate-captain SP) — the cross-type closeness test.

### Training (clean ΔG — the new heavy component)
- Implant the marker (` ※`, id 83399, asserted) at the END of frozen on-policy responses on the #474 recipe (contrastive negatives, marker-only loss, `MarkerOnlyDataCollator`). Single seed 42 (planner: decide whether a 2nd seed is in budget).
- **Off-saturation checkpoint selection is load-bearing:** pick the checkpoint where the implant emits on-policy on the diagonal AND bystanders retain headroom (the #448/#502 anchor-resolution rule) — NOT a floored (#489) or ceiling (#502 loc_ep2/3/5, pos arm) one. The planner diagnoses + fixes the #489 floor first.
- **Reuse opportunity:** #474's 16 instruction-context adapters (loc arm) already exist and are clean at ep1 — reuse them for the instruction×instruction block; train only the 16 ICL adapters; cross-eval all four blocks (instr×instr, instr×ICL, ICL×instr, ICL×ICL) to fill the ~32×32 ΔG.
- Cross-eval ΔG over all ordered pairs: train into source A, read trained−base marker log-prob at the post-response slot under target B's own on-policy response, 50 held-out test questions per cell.

### Extraction (predictor side)
- Capture base-model residual-stream activation clouds under all ~32 contexts on the #502 500-probe pool, all 28 layers, extraction points {`last_prompt`, `mean_response`} (`end_of_system` only where a system prompt exists). Reuse #502's cached clouds for the 16 instruction contexts; extract the 16 ICL contexts fresh.

### Predictors (all on the unified panel; recompute the matrices at the new panel size)
Symmetric anchors (baseline): cosine, Gaussian-KL (sym), next-token JS, full-response JS (re-extracted on this recipe), + rest of #502 zoo as desired.
Directional/asymmetric (the bets):
1. Directional Gaussian-KL (un-symmetrized, both directions, last_prompt + mean_response)
2. Source-covariance Mahalanobis `(μ_B−μ_A)ᵀ Σ_A⁻¹ (μ_B−μ_A)`, both directions
3. Asymmetric subspace-reconstruction A→B `‖P_{A,k}Φ_B‖/‖Φ_B‖`, unequal ranks, both directions
4. Marker-direction projection (plain, onto the ` ※` token direction in A's frame; not the centered variant)
5. Two-feature combiner: `ΔG ~ f_geom + f_marker + f_geom×f_marker`
Output-side directional (re-extract on this recipe): full-response one-way KL + next-token one-way KL, both directions.

### Metric protocol (the metric deep-dive, baked in)
- **Decision metric:** paired nested-CV **incremental ΔR²** of (M_sym + directional) over M_sym = (ΔG ~ gauss_kl + length), on full ΔG. A directional predictor earns its place only if the dyadic-bootstrap CI on the paired ΔR² excludes 0. NOT partial Mantel (invalid on structured dyadic matrices).
- **Targets:** full ΔG = decision; ΔG_anti = directional fraud test (a "directional" predictor scoring ~0 there is symmetric-in-disguise). Direction-mapping: correlate the predictor's own asymmetry `½(d[A,B]−d[B,A])` with ΔG_anti.
- **CV: leave-TWO-contexts-out / block** — partition the ~32 contexts into blocks; test = pairs with BOTH endpoints in the held-out block; keep (A,B)/(B,A) together; drop straddling pairs from both sides. (Dyadic analog of Li-Levina-Zhu edge-CV, Biometrika 2020.)
- **Nested selection:** the layer/extraction/cell sweep runs in the INNER loop on training contexts only; OUTER loop reports held-out skill of the per-fold-selected cell + the cells chosen per fold.
- **Saturation:** on any saturated target cell (floor or ceiling) switch from OLS-R² to a **censored/Tobit held-out log-likelihood**; OLS-R²/ρ measure saturation there, not prediction. Report which cells are saturated.
- **Uncertainty:** resample CONTEXTS (cluster bootstrap + jackknife-over-contexts), not pairs (#474's pair bootstrap is ~2× too narrow). Single-seed caveat is load-bearing, not bootstrappable.
- **Panels:** report full AND non-stylized side by side; also partial out touches-stylized AND touches-ICL indicators (so the cross-type and stylized lifts are visible, not pooled away).
- **Validation:** validate winners on a held-out probe pool (extend #523's held-out pool with ICL-context held-out probes).
- ρ stays as a descriptive, selection-flagged sidecar only.

### What would update the belief
- A directional predictor with a nested-CV incremental ΔR² CI excluding 0 on full ΔG, that ALSO scores non-zero vs ΔG_anti and survives the held-out pool → real recoverable directional leakage signal; name it.
- All directional predictors at incremental ΔR² ≈ 0 (CI spans 0) and ΔG_anti ≈ 0 → the directional 28% is not linearly recoverable from base-model geometry; a strong bound on the program.
- Cross-type (ICL↔instruction) cells: does the predictor that wins within-type also win across induction mechanisms?

## Reuse vs new
- **Reuse:** #502 cached instruction-context clouds + 500-probe pool; #474 loc-ep1 instruction adapters + ΔG (instr×instr block); #489 ICL context definitions; #502 bake-off scoring functions; the ΔG decomposition; #523's held-out pool.
- **New:** train 16 ICL-context marker adapters at an off-saturation checkpoint (fixing #489's floor); cross-eval the ICL/cross-type ΔG blocks; extract ICL-context activation clouds; the 5 directional activation predictors; re-extract output-side divergences; the full metric protocol (nested LTCO ΔR², censored model, context bootstrap, ΔG_anti scoring).

## Compute
Training (16 ICL adapters, contrastive) + cross-eval ΔG over ~992 off-diagonal ordered pairs + ICL activation extraction + the GPU output-divergence re-extraction. This is a real training experiment (≈#474 scale or larger given the 32×32 grid) — likely **> the 100 GPU-hour auto-approve cap**, so the autonomous session will PARK at plan approval for user review. The predictor + metric analysis is CPU on the dev VM.

Parent: #502. Substrate: #474 (instruction ΔG + adapters), #489 (ICL contexts + the floor lesson), #406 (conditions), #468 (ICL dynamic-range motivation). Metric: gist 3bc6aeb. Validation pool: #523.
