---
title: Does the context→answer map's predictable subspace coincide with the SAE's
  representable subspace?
kind: experiment
tags: []
created_at: '2026-07-30T20:04:08Z'
has_clean_result: false
parent_id: 1482
origin_prompt: 'can we run these experiments: - Is what oir model able to reconstruct
  the same as what SAEs find - Map from continuous features to SAE features (or reconstructed
  activation)? Also compare to simple baselines with SAE featurs'
workflow: v1
goal: 'Does the context→answer map predict the same structure the SAE represents?
  Two ways of decomposing the same answer activation have been measured separately
  and never compared: the map''s **predictable subspace** (which directions of `v_A`
  the fitted map recovers) and the SAE''s **representable subspace** (which directions
  its dictionary reconstructs). This task asks whether they coincide, and whether
  targeting the SAE''s reconstruction instead of the raw activation changes what the
  map can do.'
relates_to:
- spec-context-as-vector
---
## Goal

Does the context→answer map predict the same structure the SAE represents? Two ways of decomposing the same answer activation have been measured separately and never compared: the map's **predictable subspace** (which directions of `v_A` the fitted map recovers) and the SAE's **representable subspace** (which directions its dictionary reconstructs). This task asks whether they coincide, and whether targeting the SAE's reconstruction instead of the raw activation changes what the map can do.

Formally, for a fitted map `M: v_C -> v_A` and an SAE `(E, D)` with reconstruction `r(v) = D E(v) + b_dec`:

1. **Subspace coincidence.** Let `P_pred` = the span of the top-k directions ranked by per-direction held-out R² of `M`, and `P_sae` = the span of the SAE decoder columns active on the answer distribution (or the top-k principal directions of `r(v_A)`). Measure principal angles between `P_pred` and `P_sae` against a matched null (rotation + covariance-matched, per `mapping_similarity_metrics.md` — a spectrum-only cosine cannot establish this). **Answer:** the map's errors are SAE-representable (angles small) or the map fails precisely where the SAE has no atoms (angles at null).
2. **Reconstruction as target.** Fit `v_C -> r(v_A)` (the SAE-reconstructed answer activation) and compare held-out R² against the banked `v_C -> v_A`. If the SAE's reconstruction residual is the part the map cannot predict, targeting `r(v_A)` should be EASIER than targeting `v_A` by roughly the SAE's own FVE gap.
3. **Is map-predictability the same property as SAE-reconstructability?** Per answer-side direction (and per SAE feature), correlate the map's held-out R² against the SAE's per-direction reconstruction quality. #1482 already established that per-feature predictability is an ANSWER-SIDE property dominated by within-answer consistency (rho 0.60, twice activity's 0.29); this asks whether SAE-reconstructability is a third correlate or the same axis re-measured.

**Competing hypotheses.** (H1) SHARED STRUCTURE: the map predicts what the SAE represents; predictable directions are SAE-dense, unpredictable ones SAE-sparse; targeting `r(v_A)` buys little because the map already lives inside the dictionary. (H2) ORTHOGONAL DECOMPOSITIONS: the two carve the space differently; the map's failure directions are well-represented by the SAE and vice versa; targeting `r(v_A)` measurably changes R². (H3) TRIVIAL/VARIANCE-DRIVEN: both track answer-side variance rank and neither adds information beyond it — the null this design must rule out first, given #1482's finding that per-direction R² decays monotonically with variance rank and the SAE's own reconstruction is variance-weighted.

**What would count as an answer:** principal angles with matched nulls for (1); a paired held-out R² delta with CI for (2); a partial correlation controlling for variance rank and within-answer consistency for (3). H3 is the default and must be excluded before (1) or (2) is narrated as structure.

## Provenance

Origin (verbatim user prompt, chat 2026-07-30):
"can we run these experiments:
- Is what oir model able to reconstruct the same as what SAEs find
- Map from continuous features to SAE features (or reconstructed activation)? Also compare to simple baselines with SAE featurs"

**Scope note — the second bullet is LARGELY ALREADY RUN; only its `r(v_A)` variant is new.** Filed here for the record so this task does not re-run banked work:

| already banked | result | source |
|---|---|---|
| continuous (dense) context activation -> SAE answer features | ridge **0.7216** mean / 0.3826 max / 0.5706 frac; MLP **0.7387** mean | #1482 `sae_perfeature/unit_{ridge,mlp}__sae_dense_in.json` |
| SAE context features -> SAE answer features | ridge **0.6901** mean (dense input beats sparse by ~0.03) | #1482 `unit_ridge__sae_ctx.json` |
| same contrast, multi-turn corpus | dense 0.7003 vs sparse 0.6376 (context arm) | #1738 `sae_arm/sae_fits.json` |
| SAE-space identity+bias baseline | prefix −2.99 / context −2.38 on shared feature ids | #1738 `sae_arm/mapping_baselines.json` |
| SAE-space kNN retrieval | context acc@1 0.238 (median rank 8), prefix 0.013; chance 1.006e-4 | #1738 `sae_arm/mapping_baselines.json` |
| SAE-space identity+bias + kNN, crossed corpus | identity+bias −4.07; kNN acc@1 0.267 vs chance 0.00126 | #1482 / #1092 `crossed_core_sae/maps_summary.json` |
| projecting the fitted dense map's PREDICTION into SAE space | `encode_the_prediction`, explicitly SECONDARY (off-distribution SAE-of-mean transform applied equally to v̂ and v) | #1482 `sae_perfeature/encode_pred__*.npz` |

So "map from continuous features to SAE features, compared to simple SAE-space baselines" is **answered**: dense input beats the sparse code at matched pooling, and both crush the identity/kNN baselines. What is NOT done is mapping to the SAE **reconstruction** `r(v_A)` as the target (distinct from `encode_the_prediction`, which encodes the prediction rather than retargeting the fit), and the whole of bullet 1.

## Notes / constraints

- **Both mapping arms are required** (CLAUDE.md standing rule): prefix-based AND context-based, on a matched target. A one-arm read is a stated deviation, never a default.
- **Both standing mapping baselines are required**: identity+learned-bias and kNN retrieval (`analysis/mapping_baselines.py`), with chance stated. Several already exist in SAE space — reuse rather than recompute.
- **Direction-aware operator reads only** for (1). A singular-spectrum cosine is rotation-invariant on both sides and cannot establish subspace coincidence; #1310's 0.9896 spectrum cosine sat BELOW its own shuffle null of 0.9976 while direction-aware reads separated cleanly. Use `scripts/issue1345_operator_comparison.py` conventions + matched nulls.
- **Judged SAE feature labels are FROZEN** (#1773: all five axes search-index-only; strengthened 2026-07-30 — discrimination 0.317, detection 0.689, fuzzing 0.676, three of five conjuncts failing). This task must reach its answer from MECHANICAL properties (activity, consistency, decoder geometry, reconstruction quality) only. No judged-label headline.
- **Reuse, do not regenerate.** The pooled SAE store (`issue1482_error_analysis/analysis_tensors/sae_pooled`, 1,920 shards, 9.8 GB), the per-feature R² arrays (`sae_perfeature/*.npz`), the per-direction PCA read (`perdirection_pca.json`), and the fitted maps are all banked. Estimated marginal cost is small — likely a few GPU-h for encode passes plus CPU fits — but size it at plan time; do not inherit this sentence as a compute estimate.
- Prior art: no in-repo comparison of map-predictable vs SAE-representable subspaces exists (checked 2026-07-30). A literature pass is required before design per the standing new-direction rule — the closest published object is Activation Transport Operators (arXiv 2508.17540), which fits operators between activation sites but does not relate them to a dictionary.
