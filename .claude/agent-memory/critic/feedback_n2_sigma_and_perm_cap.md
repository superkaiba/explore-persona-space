---
name: N=2 sigma estimates and permutation-count vs Holm cap
description: Two recurring stats defects — using N=2 std as a noise threshold, and 10 000-perm caps that collide with Holm-Bonferroni cutoffs at high m
type: feedback
---

When a plan defines a per-persona threshold as "Δ ≥ σ_late" and σ_late is computed
over only 2 time points, the test algebraically reduces to "|t1 − t2|/√2" — not
a noise estimate. Under iid null noise this gives ~31% false-positive rate per
unit (verified by simulation), so a downstream "fraction above threshold ≥ X"
binomial test against 0.5 is misspecified by a factor of ~3 standard errors.

**Why:** N=2 sample stdev with ddof=1 has divisor 1, so it equals |x1−x2|/√2.
The "noise" is the gap itself. Caught in critic review of issue #263 plan.

**How to apply:**
1. Any "σ_X" computed from <5 points is suspect — flag as REVISE.
2. Demand the noise estimate be derived over ≥ 30 samples (the per-question
   axis is usually the right one), or convert to a paired test against 0.
3. Always check the binomial test's reference rate: under the test's own
   false-positive distribution under the null, not 0.5.

Companion rule for permutation tests with multiple comparisons: when m hypotheses
are tested with Holm-Bonferroni (or any FWER procedure), the smallest cutoff
is α/m. The minimum two-sided permutation p with B perms is ≈ 2/(B+1).
Require B ≥ 10 × m / α to ensure the test is not undersaturated at the boundary.

**Numerical anchors:**
- m=275 hypotheses, α=0.05 → need B ≥ 50 000 (10 000 is at boundary).
- m=20, α=0.05 → B ≥ 4 000 (10 000 is fine).
- Always state the min achievable p alongside B in §7.
