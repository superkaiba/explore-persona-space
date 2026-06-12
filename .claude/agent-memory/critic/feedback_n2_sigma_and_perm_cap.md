---
name: N=2 sigma estimates and permutation-count vs Holm cap
description: N<5 stdevs collapse algebraically (σ at N=2 = |x1−x2|/√2, ~31% per-unit FPR); B-perm tests need B ≥ 10·m/α to avoid undersaturating Holm cutoffs
type: feedback
---

A per-persona threshold "Δ ≥ σ_late" with σ computed over 2 time points reduces algebraically to "|t1 − t2|/√2" — not a noise estimate (ddof=1 divisor is 1). Under iid null noise this gives ~31% false-positive rate per unit (simulated), so a downstream "fraction above threshold ≥ X" binomial test against 0.5 is misspecified by ~3 SE. Caught on #263.

**How to apply:** (1) any σ from <5 points → REVISE; (2) demand the noise estimate over ≥30 samples (per-question axis is usually right) or convert to a paired test against 0; (3) check the binomial reference rate against the test's own null false-positive distribution, not 0.5.

**Permutation-cap companion:** with m hypotheses under Holm, the smallest cutoff is α/m; minimum two-sided permutation p with B perms ≈ 2/(B+1). Require B ≥ 10·m/α (m=275, α=0.05 → B ≥ 50,000; 10,000 is at the boundary). State the min achievable p alongside B in §7.
