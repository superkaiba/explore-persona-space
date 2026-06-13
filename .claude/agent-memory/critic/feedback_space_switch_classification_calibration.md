---
name: Space-switch classification rules need per-space references + tolerances
description: Phase-0 logP↔EOS-margin space flips are APPROVE-able iff the new space's references come from the SAME regime (on-policy refs for on-policy terminals) and tolerances are re-derived per-space from per-seed gaps — never blind-ported from logP (#601)
type: feedback
---

When a saturation-aware plan pre-registers a space-selection rule ("if Δlog P and Δz_marker diverge ≥ k nats, the EOS margin becomes the primary classification space, thresholds re-expressed against measured margin references"), the rule itself is usually deterministic and APPROVE-able — the four-float contract stores both spaces on every read, so boundary arms are recoverable by reporting the pair. Check three things (#601, 2026-06-11):

1. **Regime match of references.** If terminal classification is on-policy, the M(·) references must come from the on-policy calibration subset, not the larger teacher-forced read — mixing regimes shifts the reference by the on-policy/teacher-forced offset.
2. **Tolerance re-derivation.** A ±3-nat tolerance calibrated as 2× logP-space seed noise does NOT transfer to logit-margin space; re-derive as ~2× the per-seed margin gaps from the same calibration read (data exists if calibration covers ≥2 seeds/cell).
3. **Trajectory vs terminal noise.** Seed-noise yardsticks quoted from terminal checkpoints understate mid-trajectory gaps (#472: terminal max 1.4 nats, step-38 gap ~2.0 near ceiling); flatness/accrual criteria on intermediate reads should use the intermediate-checkpoint envelope, and near ceiling prefer the margin trajectory (compression fakes flatness in logP).

**How to apply:** all three are analyzer Concerns when the per-seed per-space floats are first-class deliverables; REVISE only if the calibration read lacks the seeds or the space needed to recompute references/tolerances.
