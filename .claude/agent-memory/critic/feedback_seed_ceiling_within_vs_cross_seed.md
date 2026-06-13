---
name: Seed-ceiling within-vs-cross-seed conflation (#521 lineage)
description: The 0.96-0.98 quoted from #521 is WITHIN-seed context-to-U1 alignment, not the cross-seed ceiling — true cross-seed U1 cos is 0.65-0.97 (direction_consistency.json) (#602)
type: feedback
---

In the #521/#551 shift-tensor lineage, two numbers get conflated: `mean_cos_to_U1` (per_cell in `eval_results/issue_521/svd/summary.json`) is WITHIN-seed alignment of the 14 per-context shifts to that seed's top SVD direction (EM `same`: 0.963/0.974/0.980); the cross-seed U1 ceiling lives in `direction_consistency.json` — cos between U1 of different seeds (EM `same`: 0.65/0.78/0.90, mean 0.78; marker: mean 0.84). Only the latter caps interpretable estimator-vs-realized agreement.

**Why (#602):** the plan (and rank1_leakage_model.tex) wrote "seed ceiling 0.96-0.98" for EM — an estimator scoring cos ~0.7 vs the EM realized write would be AT the true ceiling, not far below it; the mislabel inflates the narrative gap by ~0.2.

**How to apply:** whenever a plan in this lineage cites a "seed ceiling" / "seed-stable direction" number, check direction_consistency.json for the cross-seed pairs. Not a REVISE when the plan computes the ceiling fresh from per-seed realized writes (the computation auto-corrects the prose) — flag as a numeric concern; benchmark estimator cosines against the fresh cross-seed ceiling. Bonus: cross-arm (marker-vs-EM) realized-realized cos ~0.05 validates cross-behavior-null headroom.
