---
name: tie-density-resolution-gradient-fakes-an-effect
description: A covariate whose tie density varies across the binning axis manufactures a monotone per-bin effect gradient; bound it with the Spearman tie ceiling, never with sd
metadata:
  type: feedback
---

When a per-bin profile (decile / stratum / rung) is read over a covariate with
a POINT MASS, check how the tie density varies ACROSS the binning axis. If it
varies monotonically, the covariate's rank RESOLUTION varies with the bin, and
a monotone "this predictor matters more in high bins" result is manufactured by
construction.

Quantify it with the Spearman tie ceiling — the max attainable |rho| given ties:

    max|rho| = sqrt(1 - T),   T = sum(t^3 - t) / (n^3 - n)   over tie-group sizes t

Then divide the observed per-bin statistic by the per-bin ceiling. Flat
normalized profile = the gradient was entirely tie density.

**Why:** #1482's `template_token_frac` is 53.5% zeros overall, but the zeros are
not spread evenly across ACTIVITY deciles — 94.0% at d1 falling to 28.7% at d10.
Ceiling 0.411 -> 0.988, so a constant true association still reads as a ~2.4x
rise from d1 to d10. The K=24 Shapley bins by activity, so this was a live
interpretive trap, not a plotting nit.

**Two traps inside the trap:**
- **sd points the WRONG WAY.** sd FELL 0.0467 -> 0.0247 while resolution ROSE
  (few-but-large nonzeros at low activity, many-but-small at high). Reaching for
  sd as the dynamic-range diagnostic concludes the opposite of the truth. It is
  tie DENSITY, not spread, that attenuates rank statistics.
- **A non-singular Gram is not evidence against it.** sd stays nonzero
  everywhere, so the fit is numerically clean (zero LinAlgError) while the
  interpretation is still confounded. Numerical health and estimand validity are
  independent here.

**How to apply:** any time a per-bin profile is read over a covariate with a
point mass, report the tie ceiling per bin next to the statistic. If the
estimator is variance-based rather than rank-based the ceiling formula does not
transfer, but the tie-density gradient still needs its own variance-basis
analogue — do not wave it through.

Related: [[feedback_companion_stat_drop_class_semantics]] (zero split-half
floors from degenerate positions).
