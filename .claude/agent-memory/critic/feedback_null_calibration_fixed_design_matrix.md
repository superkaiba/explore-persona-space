---
name: Null-calibration with fixed design matrix
description: Fresh-seed null-calibration replicates can't split stochastic noise-organization from deterministic design/data artifact — same correction either way; discriminator = cross-replicate ρ spread vs SE (#555)
type: feedback
---

When a plan replicates a regression fit across fresh-seed replicates to calibrate a null (#555: 5 seed pairs, same 6-predictor partial Spearman, same probe panel + geometry table + frozen eval responses), a consistent-sign "systematic" outcome has three mechanisms that all replicate perfectly: (a) stochastic training noise genuinely organizing along the predictor, (b) deterministic data-composition gradients (if the conditioned variable is IN the training data, early gradient steps are geometry-correlated by construction), (c) partialling structure among correlated predictors on a near-zero-variance DV. A permuted-DV placebo does NOT distinguish them (it reproduces only the exchangeability null the bootstrap already gives) — don't demand one.

**Why this is claim-scoping, never REVISE:** all three license the SAME actionable conclusion (pre-implant baseline subtraction / distrust sub-floor reads). The free discriminator is the cross-replicate spread of per-replicate ρ: spread ≪ per-replicate SE(ρ) ⇒ deterministic-given-design; positive-but-dispersed ⇒ stochastic organization. A within-fit specificity predictor (a parent-flat one) splits geometry-generic from predictor-specific.

**How to apply:** on null-calibration / measurement-validation plans, check per-replicate statistics + spread are reported; put the mechanism-narration warning in Concerns. The fatal alternative in the NULL direction is an eval-path silent failure (expected-zero DV ≡ bug signature) — a positive-control cell where the guard HAS power (trained implant) is the fix; check for it explicitly (#555 had it; #534 round-1 is the precedent).
