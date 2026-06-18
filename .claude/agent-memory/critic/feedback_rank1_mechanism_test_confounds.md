---
name: rank1-mechanism-test-confounds
description: Mechanism tests bolted onto a G-tensor (slope predictions, activation-delta parallelism) — strength-mediated slopes and anisotropy-inflated cosines; RECOVERABLE iff base-side dumps + per-cell strengths + per-seed G ship (#537 v4)
type: feedback
---

Two stock mundane alternatives for registered "mechanism" reads in tensor-style plans (#537 v4 §6.4):

1. **Slope predictions (ΔG_anti ~ f(norm difference)):** the mundane driver is implant-strength differences s_i − s_j, which can correlate with the theory regressor. "Both regressors reported alongside" is ambiguous between two marginal regressions (cannot adjudicate under collinearity) and one joint/partialled regression (can) — demand the joint fit. Also: rate-space DVs don't carry log-space slope predictions without a transform; saturation/ceiling cells induce mechanical asymmetry; the (layer, anchor) feeding the norm regressor must be pinned or labeled.
2. **Parallelism reads (high pairwise cosine of trained−base Δh across contexts):** trivially inflated by (a) residual-stream anisotropy (random deltas already have high cosine), (b) late-layer tautology (any working implant pushes along W_U[target] near the readout), (c) generic SFT drift shared across fine-tunes. Needed comparisons: a null distribution from base-side probe/context delta pairs at the SAME slot, cross-behavior Δh cosines (same slot), per-layer profile.

**How to apply:** check shippability of base-side dumps at the read slot, per-cell strength metadata, per-seed cells. All present → concern with the prescribed analysis (joint regression; anisotropy-corrected null). Any missing → the analyzer cannot weigh it → Must-Fix (add the dump/metadata, not a gate).
