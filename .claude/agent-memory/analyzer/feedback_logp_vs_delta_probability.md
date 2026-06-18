---
name: logp-vs-delta-probability
description: Probability magnitudes (exp) come from TRAINED logP, not from Δ (trained − base). Never exp the Δ.
metadata:
  type: feedback
---

When converting a log-probability finding to "the trained model puts X
percent mass on this token", the source is `exp(trained_logp)`, NOT
`exp(Δ)` where `Δ = trained - base`. Δ is a SHIFT relative to base; the
absolute mass is `exp(trained_logp)` only.

**Why:** Round 1 of task #475 wrote "Δ +7.1 nats means ≈0.01% mass"
and "Δ +9.6 means ≈0.1% mass" — both wrong by ~100x. Plain phase2
T_plus actually has trained logP = -13.66 → exp(-13.66) ≈ 1.2e-6 =
**0.00012%** (not 0.01%); distilled phase2 T_plus trained logP =
-11.22 → exp(-11.22) ≈ 1.3e-5 = **0.0013%** (not 0.1%). The error
pattern was "use Δ as if it were trained logP minus a normalisation
of zero." Both interpretation-critics flagged it on round 1.

**How to apply:** in any clean-result write-up that says "the trained
model puts X percent on the marker" or "X-orders-of-magnitude probability":
  - Pull `trained_logp_median` (or `trained_logp_*.json`) directly from
    the eval JSON for the (variant, phase, cell) of interest.
  - Compute `exp(trained_logp)` directly. Quote that percentage.
  - If you also want to characterize "the shift from base", quote Δ in
    nats — but DO NOT convert Δ to percent.
  - The base logP at the same cell is also useful context — quote it
    once so the reader can see why exp(trained) is small even when Δ
    looks large.

Better default: stay in nats. Use percentages sparingly and only when
the percentage is meaningful (e.g. "90% of mass at the slot" when
trained logP is near 0). Avoid percentages at very-negative logP
where they're all "essentially zero" anyway.
