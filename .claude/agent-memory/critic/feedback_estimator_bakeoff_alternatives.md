---
name: Estimator-bakeoff alternatives (estimated-vs-realized direction scoring)
description: Sibling-pair contamination of cross-behavior nulls, fidelity-ladder-by-SNR, repair-by-norm — all recoverable iff per-row reads + full pairwise matrix + multi-variant realized sides persist (#602)
type: feedback
---

From the #602 P8 estimator bake-off (rank-1 leakage model line). For any plan scoring base-model estimators (teacher-forced / ICL / description contrasts) against realized post-training activation shifts:

1. **Sibling-pair contamination of the cross-behavior null.** When the roster contains two recipes of the SAME behavior (two EM arms, two marker arms), off-diagonal "wrong-behavior" pairings include pairs that genuinely share the write (P6 predicts cross-behavior transfer) — a validity margin "on-diag − own off-diag mean ≥ θ" mechanically penalizes exactly the families where estimators work best. Weighable iff ALL pairwise (estimator-A, realized-B) cosines persist: recompute margins excluding same-construct siblings; report sibling cells as a cross-recipe transfer read.
2. **Fidelity ladder by SNR.** E1 averages ~100 long completions, E3 ~20 short reads; noise attenuates cosine toward 0, so E1 > E2 > E3 can be reliability ordering, not fidelity. Weighable iff per-row/per-question reads persist (split-half reliability per estimator); escalates toward Must-Fix if only mean vectors ship.
3. **Repair-by-norm.** prof_real = dv[c]·û(w_shr) with w_shr from the SAME dv stack is partly tautological and ≈ ‖dv[c]‖ for near-parallel shift fields — "realized write repairs prediction" can mean "contexts that moved more leak more", direction-free. Free discriminator: include the norm-only profile ‖dv[c]‖ as a third predictor; scope repair verdicts to behavioral-panel families.

Mitigating facts to check before flagging token-identity/topic confounds: a `base`-text variant on the realized side blocks literal-token sharing with E1; an off-topic probe panel forces a pure topic direction to generalize off-topic. #602 had both → APPROVE with concerns.
