---
name: rank1-mechanism-test-confounds
description: Mechanism tests bolted onto a G-tensor (slope predictions, activation-delta parallelism) have two stock mundane confounds — strength-mediated slopes and anisotropy-inflated cosines; both are RECOVERABLE iff base-side dumps + per-cell strengths + per-seed G ship
type: feedback
---

Two recurring mundane alternatives for registered "mechanism" reads in tensor-style plans (#537 v4 §6.4 pattern):

1. **Slope predictions (G1-type: ΔG_anti ~ f(norm difference))** — the mundane driver is implant-strength differences s_i − s_j, which can correlate with the theory regressor (context norm). "Both regressors reported alongside" is ambiguous between two marginal regressions (cannot adjudicate under collinearity) and one joint/partialled regression (can). Also check: rate-space DVs don't carry log-space slope predictions without a transform; saturation/ceiling cells induce mechanical asymmetry; the (layer, anchor) feeding the norm regressor must be pinned or labeled.
2. **Parallelism reads (G2-type: high pairwise cosine of trained−base Δh across contexts)** — inflated trivially by (a) residual-stream anisotropy (random deltas already have high cosine), (b) late-layer tautology (any working implant pushes along W_U[target] near the readout, so parallelism at late layers is implied by the implant working), (c) generic SFT drift shared across all fine-tunes. Needed comparisons: null distribution from base-side probe/context delta pairs at the SAME slot, cross-behavior Δh cosines (same slot!), per-layer profile.

**Why:** Surfaced on #537 plan v4 (Alternatives lens, 2026-06-09). Both were RECOVERABLE there because base-side activation dumps, per-cell diagonal strengths, all-layer clouds, and per-seed G cells all ship as first-class artifacts.

**How to apply:** When a plan registers a mechanism test of this shape, check shippability of: base-side dumps at the read slot, per-cell strength metadata, per-seed cells. All present → concern with the concrete prescribed analysis (joint regression; anisotropy-corrected null). Any missing → the analyzer cannot weigh it → Must-Fix (add the dump/metadata, not a gate).
