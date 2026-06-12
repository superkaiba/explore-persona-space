---
name: Lockstep-ratio ceiling bias (ratio-of-deltas on a bounded scale)
description: Ratio indices L = bystander Δlog P / source Δlog P bias TOWARD lockstep when the denominator (source) is nearer the log-prob ceiling than the numerator; prescribe an EOS-margin-space recomputation, not a REVISE when four floats are persisted
type: feedback
---

Rule: when a plan defines a ratio-of-deltas index on the log-prob scale (e.g. #597's lockstep index `L = median(bystander Δlog P) / source Δlog P` at an onset checkpoint), check WHICH side saturates first. Δlog P is capped at −base (log P capped at 0); the source approaches that cap first, compressing the denominator while bystanders keep headroom, so measured L is biased UPWARD (toward "lockstep") at near-saturated onset checkpoints. The complementary good pattern: gating the read on denominator ≥ 5 nat avoids the near-zero blowup (the #587 ratio failure mode) — but does NOT protect against the ceiling end.

**Why:** From #597 (statistics lens, 2026-06-11). The plan's onset definition (first checkpoint with source Δ ≥ 5 nat) guards the zero end of the ratio but not the ceiling end; with a 4-step grid the first ≥5-nat checkpoint can land anywhere between mid-ramp and full saturation (Arm A's villain went −9.05 → −1e-6 trained log P between steps 20 and 40, i.e. ramp-to-ceiling inside ~20 steps).

**How to apply:** Not a Must-Fix when the four-float storage contract is in place (the analyzer can recompute L in EOS-margin space `Δ(z_marker − z_eos)`, which is non-saturating) — file it as a Concern instructing the analyzer to report margin-space L wherever the source's trained log P > −1 nat at onset. Escalate to REVISE only if the plan stores log-probs alone (logits unrecoverable post-hoc, #530).
