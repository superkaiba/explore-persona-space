---
name: Seed-ceiling within-vs-cross-seed conflation (#521 lineage)
description: Plans citing #521's "0.96-0.98" as a seed-to-seed ceiling mislabel within-seed context-to-U1 alignment; true cross-seed U1 cos is 0.65-0.97 (direction_consistency.json)
type: feedback
---

In the #521/#551 shift-tensor lineage, two different numbers get conflated:

- `mean_cos_to_U1` (per_cell in `eval_results/issue_521/svd/summary.json`): WITHIN-seed
  alignment of the 14 per-context shifts to that seed's top SVD direction. EM `same`:
  0.963/0.974/0.980. This is what "EM shifts all contexts along one direction (0.96-0.98)" means.
- Cross-seed U1 ceiling (`eval_results/issue_521/svd/direction_consistency.json`):
  cos between U1 of different seeds. EM `same`: 0.65/0.78/0.90 (mean 0.78);
  marker: 0.75/0.80/0.97 (mean 0.84). THIS is the seed-to-seed measurement ceiling
  that caps interpretable estimator-vs-realized agreement.

**Why:** #602's plan (and the parent note rank1_leakage_model.tex) wrote "seed ceiling
0.96-0.98" for EM. An estimator scoring cos ~0.7 vs EM realized write would be AT the
true ceiling, not "far below" it — the mislabel inflates the narrative gap by ~0.2.

**How to apply:** Whenever a plan in this lineage cites a "seed ceiling" or
"seed-stable direction" number, check direction_consistency.json for the cross-seed
pairs. Not a REVISE when the plan computes the ceiling fresh from per-seed realized
writes (the computation auto-corrects the prose); flag as a numeric concern and tell
the analyzer to benchmark estimator cosines against the fresh cross-seed ceiling.
Bonus check from the same file: cross-arm (marker-vs-EM) realized-realized cos ~0.05
empirically validates cross-behavior-null headroom.
