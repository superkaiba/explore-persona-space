---
name: Parity gates on determinate synthetic data are blind to recipe mismatches
description: An end-to-end R2-tolerance parity test on well-determined synthetic data passes even under the WRONG recipe; the fails-pre-fix regression pin must be a bit-level same-init serial-replica test, and smoke-scale parity deltas on tiny no-signal stores are init noise, not recipe evidence. #931 r2.
type: feedback
---

When pinning "batched fitter reproduces parent recipe X" (standardization
subset, early-stopping semantics, PCA-skip branches), an end-to-end R²
tolerance test on well-determined synthetic data does NOT discriminate: on
#931's parity data the PRE-fix (mismatched) recipe read |ΔR²| = 0.0006 vs the
post-fix 0.0008 — both far under the 0.02 gate, because strong signal blurs
standardization/stopping differences. Conversely a tiny NO-signal smoke store
(n=40 fabricated rows, both R² ≈ 0.01–0.03) reads Δ ≈ 0.02–0.04 from init-draw
noise alone, falsely implying recipe drift.

**Why:** recipe differences move fits only where the loss landscape is
marginal (noisy/underdetermined data); determinate data converges both recipes
to the same function, and no-signal data measures only init noise.

**How to apply:** (1) the fails-pre-fix regression pin is a BIT-LEVEL test —
same init draw (e.g. `split_group_init_seed`), a literal serial replica of the
parent loop (post-step val eval, exact improvement threshold, patience freeze,
best-state restore), asserting same best epoch + preds within reduction-order
tolerance (~5e-5), on NOISY data calibrated so the early-stop branch
demonstrably fires (assert it); a new-kwargs fix also fails pre-fix via
TypeError. (2) Keep the end-to-end R² test as an equivalence pin, but state
honestly that it passes pre-fix. (3) Never cite a tiny no-signal smoke's
parity delta as recipe evidence in either direction — check both arms' R²
magnitudes first. (#931 round 2, 2026-07-03.)
