---
name: Estimator-bakeoff alternatives (P8-style estimated-vs-realized direction scoring)
description: Three endemic alternatives in estimator-vs-realized-write bake-offs — sibling-pair contamination of cross-behavior nulls, fidelity-ladder-by-SNR, and repair-by-norm — all recoverable iff per-row reads + full pairwise matrix + multi-variant realized sides persist
type: feedback
---

From the #602 P8 estimator bake-off review (2026-06-11; rank-1 leakage model line).
Applies to any plan scoring base-model estimators (teacher-forced / ICL / description
contrasts) against realized post-training activation shifts.

1. **Sibling-pair contamination of the cross-behavior null.** When the family roster
   contains two recipes of the SAME behavior (two EM arms, two marker arms), the
   off-diagonal "wrong-behavior" pairings include pairs that genuinely share the write
   (the note's own P6 predicts cross-behavior transfer; EM is one global direction).
   A validity margin "on-diag − own off-diag mean ≥ θ" then mechanically penalizes
   exactly the families where estimators work best. Weighable iff ALL pairwise
   (estimator-A, realized-B) cosines persist: analyzer recomputes margins excluding
   same-construct siblings and reports sibling cells as a cross-recipe transfer read.

2. **Fidelity ladder by SNR.** E1 averages ~100 long completions, E3 ~20 short reads;
   noise attenuates cosine toward 0, so E1 > E2 > E3 can be reliability ordering, not
   fidelity. Weighable iff per-row / per-question reads persist (split-half
   reliability per estimator). Escalates toward Must-Fix if only mean vectors ship.

3. **Repair-by-norm.** prof_real = dv[c]·û(w_shr) with w_shr from the SAME dv stack is
   (a) partly tautological on families with no behavioral panel, and (b) ≈ ‖dv[c]‖
   for near-parallel shift fields — so "realized write repairs prediction" can mean
   "contexts that moved more leak more", direction-free. Free discriminator: include
   the norm-only profile ‖dv[c]‖ as a third predictor in the repair table; scope
   repair verdicts to behavioral-panel families.

Mitigating facts to check before flagging token-identity/topic confounds: a `base`
text variant on the realized side (text held fixed across trained/base) blocks
literal-token sharing with E1; an off-topic probe panel forces a pure topic direction
to generalize off-topic to appear. #602 had both, which is why the verdict was
APPROVE with concerns, not REVISE.
