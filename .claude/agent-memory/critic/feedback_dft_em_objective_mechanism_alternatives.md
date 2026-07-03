---
name: DFT-vs-SFT EM objective-comparison alternatives
description: Alternatives-lens dispositions for DFT(stop-grad 1/π reweight)-vs-SFT EM-attenuation plans — matched-acquisition Pareto, gradient-mass-share vacuity, norm-scaled projection, Δθ-magnitude-vs-structure
type: feedback
---

For a plan testing whether DFT (per-token `sg(π)·CE`, removing SFT's `1/π`
over-weighting) attenuates emergent misalignment vs standard SFT (canonical: #715),
the recurring alternatives and their FATAL-vs-RECOVERABLE dispositions:

- **P1 "DFT Pareto below SFT" — simplest rival = DFT just learned the narrow task
  less.** RULED OUT by a MATCHED-ACQUISITION frontier comparison (EM-rate vs
  narrow-task acquisition, compare curves over the overlapping x-range, NOT at fixed
  steps). The matched-acquisition design IS the control; coinciding curves = the
  honest null ("no Pareto gain"). Seed-scatter handled by "non-overlapping 95%
  bootstrap CIs in ≥2 of 3 seeds." Not fatal. Note: on a fixed-step LoRA sweep the two
  arms' narrow-acquisition VALUES differ at every step, so "≥1 matched-acquisition
  operating point" needs interpolation along each frontier to a common x — recoverable
  (analyzer interpolates), the frontier-below-frontier leg does not need exact
  x-matching.

- **P2 "token gradient-mass redistribution" is PARTLY VACUOUS.** Verify against the
  DFT A.4 / eq:dr-grad gradient identity (read 2508.05629 Method): per gold token the
  DFT gradient = SFT gradient × scalar `sg(π)` — SAME DIRECTION, scaled magnitude. So
  SFT's per-token grad-norm ∝ `(1/π)‖∇logπ‖` while DFT's ∝ `‖∇logπ‖` (the `sg(π)`
  cancels `1/π`). The "share of total grad-norm on the lowest-π decile differs" is
  then a near-DETERMINISTIC consequence of the `1/π`-vs-`1` weighting — measuring it
  CONFIRMS the loss arithmetic, NOT that redistribution CAUSES the P1 effect. The
  non-tautological, load-bearing part of P2 is the SUB-HYPOTHESIS "misaligned-content
  tokens have lower base-model π than ordinary tokens" (Mann-Whitney) — that one is
  genuinely falsifiable and is the base-rate sanity check. Disposition: RECOVERABLE
  concern — analyzer must not narrate the redistribution SHARE as independent mechanism
  confirmation; the falsifiable content is the low-π-tail sub-hypothesis.

- **P3 "smaller projection onto EM-direction d" — rival = the LoRA update is just
  smaller overall (norm-scaled), not directionally away from d.** RULED OUT iff the
  directional read is the NORMALIZED `cosine_to_d` / `fraction_along_d` (#521 recipe),
  NOT raw `proj_raw` — exactly the same fix as #521's locked normalized projection. A
  plan that locks P3 to the normalized fields AND states why (raw proj conflates "moves
  less along d" with "moves less overall") has fully closed this rival. #715 did this
  cleanly ("P3 normalization (LOCKED)").

- **P4 "DFT Δθ sparser / lower effective-rank + more prunable" — rival = full-FT DFT
  simply learned LESS / its Δθ is smaller in magnitude everywhere (DFT down-weights
  gradients, so ‖Δθ‖_F is systematically smaller).** PARTIAL defense: participation
  ratio + SVD effective-rank (from singular-value SHARE) are SCALE-INVARIANT, so the
  "lower effective rank" leg survives the magnitude rival IF those forms are used. The
  prunability curve (EM vs fraction-of-Δθ-pruned) and absolute-threshold SPARSITY are
  NOT scale-invariant and CAN be a pure magnitude artifact. Disposition (when P4 is the
  SECONDARY 1-seed read, not the headline): RECOVERABLE — but flag the SPECIFIC fix:
  report ‖Δθ‖_F per arm (free — the per-matrix Δθ is already materialized for the SVD)
  as the covariate, and read the 4b prunability curve relative to it. Do NOT REVISE for
  this when P1 is the headline and Δθ is materialized so ‖Δθ‖_F is recoverable at
  analysis time.

- **Base-rate rival (EM-content tokens are low-π only by data-source chance) — already
  IS P2's sub-hypothesis** (Mann-Whitney that misaligned tokens are lower base-π).
  Fully addressed when the plan carries it; no separate control needed.

Net for #715 v4: APPROVE on alternatives — every predicted positive's simplest rival is
either ruled out by the design (P1 matched-acquisition, P3 normalized DV, P2a/base-rate
Mann-Whitney) or weighable by the analyzer from reported diagnostics (P2 share vacuity,
P4 magnitude-vs-structure). No fatal-unweighable alternative.
