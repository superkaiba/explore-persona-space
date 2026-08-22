---
name: row-quarantine-containment-review
description: Reviewing a granularity WEAKENING of a near-duplicate leakage containment (cluster co-assign -> row quarantine) — what makes it sound vs headline-threatening (#1336 v21 A5)
metadata:
  type: feedback
---

When a plan weakens a near-dup containment from CLUSTER co-assign to ROW quarantine
(because cluster transitive closure train-locked a huge pool fraction and destroyed the
test-side estimand — #1336 v21: 239 edges -> 43% merged mass vs 306 rows = 0.64%),
evaluate these four points before reaching for REVISE:

1. **Exactness at the threshold.** If EVERY row incident to a >=threshold cross-corpus
   edge is quarantined to train, then by construction NO surviving test row has a
   >=threshold cross-side twin anywhere — the protection at the registered threshold is
   IDENTICAL to cluster co-assign. The weakening is confined to SUB-threshold pairs
   (e.g. the [0.90, 0.95) band) that cluster co-assign absorbed as a side effect.
2. **Bound the residual and check it is common-mode.** Sub-threshold straddle mass at
   ~1-2% of test rows shifts pooled held-out R² second-order, and leakage from a split
   computed ONCE is common-mode across every checkpoint/arm evaluated on it — it can
   only threaten a DIFFERENTIAL (per-stage) headline via a stage-specific interaction
   (e.g. eval-corpus rows near-dup to the RL stage's own training prompts). Demand a
   registered discriminator for that specific interaction (in #1336: the gsm8k_train vs
   gsm8k_test slice contrast + a per-corpus quarantine-mass bound resolved pre-dispatch).
3. **Diagnostics that make it analyzer-recoverable:** persisted edge LIST with cosines
   (counts alone made framing checks unverifiable offline — attempt 4's gap), straddle
   counts at threshold AND at a looser sensitivity threshold, quarantine-straddle counts
   (the NEW within-corpus residual the quarantine itself introduces: quarantined train
   row near a same-corpus test row), and a threshold-sensitivity flag when mass
   concentrates just below threshold. With per-prompt residuals persisted, a
   leakage-excluded sensitivity R² is a free re-reduction — Concern, not REVISE.
4. **Under-direction of the mass cap.** A quarantine-mass <= X% HALT passes TRIVIALLY at
   0 mass, so a scan regression (0 edges) sails through while leaking every pair. An
   internal n_quarantined == n_ge_threshold cross-check is same-scan consistency, not
   independent. Ask for (or flag as Concern) an expected-band tripwire on the realized
   edge count against the twice-witnessed prior value; committed never-straddle /
   fire-arm tests are the offline coverage.

Minor sibling: per-corpus quarantine groups are DISTINCT train groups, so a cross-corpus
pair's endpoints can land in different TRAIN FOLDS — inner-CV lambda selection sees the
contamination (0.64% — negligible), while the true test partition is clean by point 1.

**Why:** #1336 v21 got this right on all four points and the residual surfaces were all
bounded + registered (N1/N7/N15, rows 38-41); the review value was verifying exactness
at threshold and the common-mode argument rather than reflexively REVISEing a
"weakened protection".

**How to apply:** any leakage-screen granularity change, near-dup containment re-spec, or
quarantine/co-assign mechanism swap in a split-instrument plan. Related:
[[unsatisfiable-gate-respec-review]] (the same task's A2 gate re-spec marks).
