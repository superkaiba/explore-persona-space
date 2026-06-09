---
title: 'Test whether #502''s L22 gauss_kl geometry-predicts-leakage cell survives
  honest held-out evaluation (held-out probe pool + second seed + nested CV)'
kind: experiment
tags: []
created_at: '2026-06-08T23:28:14Z'
has_clean_result: false
parent_id: 502
goal: 'Determine whether #502''s headline geometry-predicts-leakage cell (last_prompt
  x L22 x gauss_kl, rho=-0.79 / CV R^2=0.61 on loc-arm epoch 1) survives honest out-of-sample
  evaluation -- via a held-out probe pool (with the indirect-rewrite voice-drift bug
  fixed), a scoped second-seed retrain of the headline cell''s deltaG substrate, and
  a nested CV that folds the ~1500-cell selection inside the held-out fold -- and
  report the non-stylized-subset CV R^2 as the headline effect size.'
relates_to:
- leak-predictor
---
# Test whether #502's L22 gauss_kl geometry-predicts-leakage cell survives honest held-out evaluation

## Goal

Determine whether #502's headline geometry-predicts-leakage cell (last_prompt x L22 x gauss_kl, rho=-0.79 / CV R^2=0.61 on loc-arm epoch 1) survives honest out-of-sample evaluation -- via a held-out probe pool (with the indirect-rewrite voice-drift bug fixed), a scoped second-seed retrain of the headline cell's deltaG substrate, and a nested CV that folds the ~1500-cell selection inside the held-out fold -- and report the non-stylized-subset CV R^2 as the headline effect size.

## Background

#502 swept 28 layers x 3 extraction points x 9 metrics x 2 variants (~1500 valid cells) against #474's deltaG marker-leakage matrix and reported a headline rho = -0.79 at `last_prompt / L22 / gauss_kl` on the cleanest checkpoint (loc-arm ep1). Three reasons that number is likely optimistic, all called out in #502's own writeup:

- **In-sample selection.** The headline cell is the search-best over ~1500 candidates on the same 240 ordered pairs. #502's CV R^2 is leave-one-persona-class-out *within the already-selected cell* — it does NOT hold out against the selection step, so it is still selection-inflated.
- **Single seed.** All activations captured under seed 42 with greedy decoding; the deltaG substrate (#474) was trained under a single seed.
- **Stylized cluster.** ~80% of the CV R^2 lift over the non-stylized baseline comes from the 3 stylized characters (pirate captain / stand-up comedian / villainous mastermind). On the non-stylized 156-pair subset the effect drops to CV R^2 = 0.34, ~comparable to the next-token JS baseline's full-panel CV R^2 of 0.32.

A standing code-review caveat: 449/450 indirect-register rewrites in #502's new probes used first-person language instead of the target third-person register, and because those rows sit inside the non-stylized subset, the bug touches the non-stylized headline too.

## Design (one experiment; three additions over #502; everything else inherits #502's exact rig)

Reuse `scripts/issue493_extraction_metric_bakeoff.py` + `scripts/issue502_dispatch.py`, the 16 #406 conditions (`src/explore_persona_space/experiments/i406_conditions.py`), the full metric grid (3 extraction points x 28 layers x 9 metrics x 2 variants + next-token JS baseline), and #474's deltaG matrices unchanged. Change only:

1. **Held-out probe pool (primary deliverable).** Generate a second mixed-distribution probe pool (~450-500 probes) with the same generation recipe as #502's 450 new probes, strictly disjoint from #502's 500-probe pool AND #474's train/test questions (hard-asserted disjointness + dedup). Fix the indirect-rewrite voice-drift bug while regenerating the register rewrites (enforce + validate third-person target). Select the predictor cell on #502's ORIGINAL pool, then evaluate that SAME fixed cell on the held-out pool. The gap between the original-pool number and the held-out-pool number is the honest out-of-sample estimate.

2. **Second seed.** Re-derive the headline result under a second seed to separate geometry from sampling-trajectory quirk. *Planner note:* activation extraction under greedy decoding is deterministic given the prompt, so a second *extraction* seed is near-uninformative — the meaningful seed lever is the deltaG training substrate (#474's LoRA training: data order / dropout / init). Scope the seed check to the headline cell only (loc-arm epoch 1, 16 source adapters retrained under a second seed, e.g. seed 43), then recompute the predictor correlation against the new deltaG matrix. The critic should confirm this scoping and reject a full 8-cell #474 retrain unless it fits the GPU-hour budget.

3. **Nested CV.** Replace #502's leave-one-class-out-on-the-selected-cell scoring with a nested CV that runs the full ~1500-cell selection INSIDE each held-out fold, so the reported R^2 folds the selection inflation into the held-out evaluation. CPU-only reanalysis over the existing #502 distance matrices + the new held-out matrices.

## What would update the belief

- If the held-out-pool number, the nested-CV number, and the second-seed number all land near the non-stylized CV R^2 ~= 0.34, then the modest effect is real and #502's headline rho = -0.79 was selection + cluster inflation.
- If they collapse toward the JS baseline / zero, the predictor does not generalize and the #502 headline should be retired.
- Report the non-stylized 156-pair subset throughout; the full-panel rho is supporting color only.

## Reuse vs new

- **Reuse:** extraction driver, multi-GPU dispatcher, 16 conditions, metric grid, #474 deltaG substrate, paper-plots.
- **New:** held-out probe generation with the voice-drift fix, the nested-CV scoring function, the scoped second-seed deltaG retrain.

Parent: #502. Substrate: #474 (deltaG matrices), #493 (8-layer parent bake-off), #406 (cosine + JS substrate).
