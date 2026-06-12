---
name: Within-source permutation degenerate for source-level factors
description: Clustered permutation suites applied uniformly to a factor list return p=1.0 mechanically for any covariate constant within the stratum; check factor-vs-strata granularity
type: feedback
---

When a plan registers a "within-cluster permutation" (house `_stratified_permutation_p`: shuffle y within each source stratum, recompute fn(x, y)) uniformly across a LIST of factors, check each factor's granularity against the strata. Any factor CONSTANT within the stratum (e.g. `self_delta(behavior, source)` over a 23-bystander panel) makes the statistic invariant under every within-stratum shuffle — the multiset of (x, y) pairs is unchanged when x is constant in the group — so the permutation distribution is a point mass and p ≡ 1.0 regardless of the true effect.

**Why:** #591 round 1 registered within-source permutation for all four headline factors; `self_delta` is source-level, so its registered test was zero-power-by-construction, and the p=1.0 would have shipped as a "decisive per-factor null" (a registered success-criterion outcome) feeding the e3 dose-arm design. Verified against `scripts/issue_480/i480_analyze.py` — the degeneracy is in the actual house code, not a hypothetical.

**How to apply:** For every factor in a clustered-permutation suite, ask "does this factor vary WITHIN the permutation stratum?" If not → Must-Fix: the source-level factor needs between-cluster inference (permute across the 6 sources / 18 panels, exact enumeration 6!=720 is feasible) or an explicit panel-level descriptive designation. Related trap from the same plan: confirm/falsify bands asymmetric across a validation gate (confirm requires cos ≥ 0.97 twin; falsify counted ≥ 0.95 twins flat — a band the parent's own data shows as expected-flat at 0.953) — align falsification to the confirmation band or name the in-between band untestable.
