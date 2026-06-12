---
name: Lockstep-ratio ceiling bias (ratio-of-deltas on a bounded scale)
description: L = bystander Δlog P / source Δlog P biases TOWARD lockstep when the denominator saturates first; prescribe an EOS-margin-space recompute — Concern iff four floats persist (#597)
type: feedback
---

When a plan defines a ratio-of-deltas index on the log-prob scale (#597: lockstep index L = median(bystander Δlog P) / source Δlog P at an onset checkpoint), check WHICH side saturates first. Δlog P is capped at −base; the source approaches the cap first, compressing the denominator while bystanders keep headroom — measured L biases UPWARD (toward "lockstep") at near-saturated onset checkpoints. Gating the read on denominator ≥ 5 nat guards only the near-zero blowup (the #587 ratio failure mode), not the ceiling end: with a 4-step grid the first ≥5-nat checkpoint can land anywhere between mid-ramp and full saturation (#597 Arm A villain went −9.05 → −1e-6 trained log P between steps 20 and 40).

**How to apply:** Not a Must-Fix when the four-float storage contract holds — file a Concern instructing the analyzer to recompute L in EOS-margin space `Δ(z_marker − z_eos)` (non-saturating) wherever the source's trained log P > −1 nat at onset. Escalate to REVISE only if the plan stores log-probs alone (logits unrecoverable post-hoc, #530).
