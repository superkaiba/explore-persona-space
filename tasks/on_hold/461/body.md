---
title: 'Predict #456 on-policy marker leakage from persona cosine + JS/KL divergence
  (q:leak-predictor)'
kind: experiment
tags: []
created_at: '2026-06-01T06:11:50Z'
has_clean_result: false
parent_id: 456
goal: 'Test whether persona residual-stream cosine similarity and answer-distribution
  JS/KL divergence predict #456''s on-policy marker-leakage across the 28-persona
  panel (primary DV: on-policy emission rate; secondary: endpos log p(※)), matching
  the #207/#311 predictor recipe.'
---
## Goal

Test whether persona residual-stream cosine similarity and answer-distribution JS/KL divergence predict #456's on-policy marker-leakage across the 28-persona panel (primary DV: on-policy emission rate; secondary: endpos log p(※)), matching the #207/#311 predictor recipe.

## Motivation

#456 measured, per panel persona, an on-policy marker-emission rate + an on-policy end-of-answer log p(※) at step 1600 (`eval_results/issue_456/per_persona_step1600.csv`), but never regressed that leakage against a predictor. This task supplies the predictor side and tests the `q:leak-predictor` question on #456's on-policy DV.

Prior art (the predictor has a track record — match it, don't reinvent):
- **Cosine → leakage RATE works:** #207 base-model L20 persona-centroid cosine → per-bystander leakage, pooled Spearman ρ≈0.60.
- **JS → rate is better and subsumes cosine:** #142 JS-over-next-token-distributions → leakage ρ=−0.75; partial Spearman shows JS subsumes cosine (cosine collapses given JS). Cosine and JS geometries rank-correlate ρ≈0.94 at L20 (#341).
- **Same-panel precedent #311:** partial Spearman of `1 − cos(persona, trained-set midpoint)` vs joint-specific leakage residual = ρ=−0.348, p=0.086, N=17 (LOW/inconclusive — low power).
- **Implantability/log-p DV class FAILED:** #396/#415 nulled cosine/JS→marker-log-p at N=24; `q:leak-predictor` deprioritizes that line. #456's endpos log p(※) is this family — a null there is replication, not news.

## Design sketch (for /adversarial-planner)

- **DVs (from #456, already committed):** primary = on-policy `emission_rate` per persona; secondary = on-policy endpos log p(※). `eval_results/issue_456/per_persona_step1600.csv`.
- **Predictors:**
  - Cosine of each persona's L20 last-token residual-stream centroid (global-mean-centered) to (a) the source `software_engineer` centroid and (b) the trained-set midpoint (#311 recipe). Cached centroids for 20/28 personas at `eval_results/issue_311/centroids_base.pt` (L10/L20) + `eval_results/phase_minus1_persona_vectors/cosine_matrix.json`. The 8 `fammate_*` contexts have no cached centroid (drop → N=20, or one forward pass to extend).
  - JS (and KL) divergence over next-token distributions under each persona's system prompt over the 20 fixed PROMPTS (`divergence.py::compute_js_divergence`); not cached for the 19 named personas → fresh compute.
  - Multi-axis controls (à la #207/#181): lexical Jaccard, structural/task match (`i181_features.py`).
- **Stats:** Spearman + partial Spearman partialling out average similarity to the trained set (#311 confound); multi-axis OLS + ΔR² + partial-Spearman JS-subsumes-cosine test (`i207_run_regression.py`). Report against emission_rate (primary) and log p (secondary).
- **Reuse:** `analysis/representation_shift.py` (cosine), `analysis/divergence.py` (JS/KL), `scripts/i207_run_regression.py` (stats), `scripts/i207_compute_js_matrix.py` (JS driver), `scripts/plot_issue311_clean_result.py` (residualized-scatter figure), `experiments/phase_minus1_persona_vectors/extract_persona_vectors.py` (centroid recipe).

## Open design questions for the planner

- Cheap path (cosine-only, cached centroids, CPU, N=20, kind:analysis) vs rigorous path (+ JS over all 28, 1× H100, N=28). Recommend leading with cosine→emission_rate (free) and adding JS if the cheap signal warrants.
- Layer choice (L10/L15/L20/L25 — #207 used L20; only L10/L20 cached).
- Whether to extend centroids to the 8 fammate contexts (one forward pass) or report N=20 on the named+no_persona set.
- Power: N=20-28 is the same low-power regime as #311 (p=0.086) — pre-state that a null is expected-plausible and not strong evidence of no effect.

## Measurement validity (§6)

| DV | Construct | Metric | On-distribution? | Predictor validity |
|---|---|---|---|---|
| emission_rate (primary) | how much the marker leaks to a persona when it generates | #456 on-policy substring-※ rate | yes (from #456) | leakage-rate family; cosine/JS have a track record here (#207/#142) |
| endpos log p(※) (secondary) | marker probability at the trained position | #456 on-policy teacher-forced log p | yes (from #456) | implantability family; cosine/JS already nulled (#396/#415) — replication check |
