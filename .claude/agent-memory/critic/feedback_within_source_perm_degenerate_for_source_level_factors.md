---
name: Within-source permutation degenerate for source-level factors
description: Clustered permutation suites return p≡1.0 mechanically for any covariate constant within the stratum; check factor-vs-strata granularity (#591)
type: feedback
---

When a plan registers a within-cluster permutation (house `_stratified_permutation_p`: shuffle y within each source stratum) uniformly across a LIST of factors, check each factor's granularity against the strata: any factor CONSTANT within the stratum (e.g. `self_delta(behavior, source)` over a 23-bystander panel) makes the statistic invariant under every within-stratum shuffle — a point-mass permutation distribution, p ≡ 1.0 regardless of the true effect.

**Why (#591 round 1):** within-source permutation was registered for all four headline factors; `self_delta` is source-level, so its test was zero-power-by-construction and the p=1.0 would have shipped as a "decisive per-factor null" feeding the e3 dose-arm design. Verified against `scripts/issue_480/i480_analyze.py` — the degeneracy is in the actual house code.

**How to apply:** for every factor in a clustered-permutation suite, ask "does this factor vary WITHIN the permutation stratum?" If not → Must-Fix: between-cluster inference (permute across the 6 sources / 18 panels; exact 6! = 720 enumeration is feasible) or an explicit panel-level descriptive designation. Related trap from the same plan: confirm/falsify bands asymmetric across a validation gate (confirm cos ≥ 0.97; falsify counted ≥ 0.95 flat twins — a band the parent's own data shows expected-flat at 0.953) — align the falsification band to the confirmation band or name the in-between untestable.
