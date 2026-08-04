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

**Do NOT disattenuate by the ceiling without first asking whether the ties are
REAL.** The ceiling diagnoses the gradient; it does not license dividing by it.
Dividing claims "what the contribution would be if the variable weren't tied",
which is only meaningful if an untied version of the variable could exist.

Two tests that settle it, both cheap:
- **Exposure test** (for a RATE): compare the observed zero-rate against
  `P(observe 0)` under Binomial(n = median denominator, p = pooled rate). If
  observed >> expected, the zeros are real, not small-sample censoring.
- **Granularity test:** recompute the ceiling on the NONZERO subset. If it is
  1.0 everywhere, the entire gradient comes from the point mass, and none from
  measurement resolution.

#1482 ran both on `template_token_frac`: exposure explained 59% of d1's zeros
but ~0% from d2 on (776 firings and still zero = a real zero), and the
nonzero-only ceiling was exactly 1.0000 at every decile. So 100% of its
gradient came from a REAL zero mass — no untied counterfactual exists, and
normalizing would have been an overcorrection. Same verdict as the 4-level
categorical it sat beside, for the same reason.

**What to do instead:** report the ceiling as a diagnostic beside the profile
(never as a divisor); answer the question on the subset where the ceiling is
1.0 (here, nonzero features only — apples-to-apples by construction); and treat
the point mass as a RESULT rather than a nuisance.

**The subset read trades a tie gradient for a SAMPLE-SIZE gradient — disclose
both.** The kept fraction IS the varying quantity, so restricting to it makes n
vary by the same factor: #1482 kept 677/11,327 (6.0%) at d1 vs 8,200/11,498
(71.3%) at d10, a 12x range, so d1's CI is ~3.5x wider. Two duties follow:
per-bin CIs are mandatory (a point-estimate profile invites reading low-bin
noise as signal), and the question NARROWS — selection is on the predictor, so
it becomes "among units where the variable is nonzero...". That narrowing is
why reporting the point mass as its own result is load-bearing rather than
presentational: it is what answers the unrestricted question the subset read
gives up. Still the better trade — a sample-size gradient is disclosable and
fixable with CIs; a counterfactual divisor is neither.

**Cardinality is not the diagnostic — tie MASS is.** A zero-inflated continuous
variable with 46,167 distinct values was capped at 0.9203 pooled, right beside a
4-level categorical at 0.4537. A rank-based R2 decomposition under-credits any
low-rank-information block, not just categoricals.
