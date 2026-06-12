---
name: Space-switch classification rules need per-space references + tolerances from the same regime
description: Plans that let a Phase-0 calibration flip the primary DV space (logP vs EOS-margin) are sound iff the new space's references come from the SAME regime as the classified arms (on-policy refs for on-policy terminals) and tolerances are re-derived from per-seed gaps in that space — never blind-ported from logP (#601)
type: feedback
---

When a saturation-aware plan pre-registers a space-selection rule ("if Δlog P and Δz_marker diverge ≥ k nats at level X, the EOS margin becomes the primary classification space and thresholds are re-expressed against measured margin references M(·)"), the rule itself is usually deterministic and APPROVE-able — the four-float contract stores both spaces on every read, so boundary arms that flip spaces are recoverable by reporting the pair. Check three things (#601, 2026-06-11):

1. **Regime match of the references.** If terminal classification is on-policy, the M(·) references must come from the on-policy calibration subset, not the (larger, cheaper) teacher-forced read — mixing regimes shifts the reference by the on-policy/teacher-forced offset.
2. **Tolerance re-derivation.** A ±3-nat tolerance calibrated as 2× logP-space seed noise does NOT transfer to logit-margin space; re-derive as ~2× the per-seed margin gaps from the same calibration read (the data exists if the calibration covers ≥2 seeds per cell).
3. **Trajectory vs terminal noise.** Seed-noise yardsticks quoted from terminal checkpoints understate mid-trajectory gaps (in #472, terminal max 1.4 nats but step-38 gap ~2.0 at the near-ceiling cell); flatness/accrual criteria operating on intermediate fraction reads should use the intermediate-checkpoint envelope, and near ceiling prefer the margin trajectory (compression also fakes flatness in logP).

**How to apply:** all three are analyzer Concerns, not REVISEs, when the per-seed per-space floats are first-class deliverables (four-float contract). REVISE only if the calibration read lacks the seeds or the space needed to recompute references/tolerances.
