---
name: Flag-flip A/B nulls — "live but toothless" vs "lever irrelevant"
description: Loss-placement/flag A/B co-landing nulls are gradient-magnitude-conditional (force ∝ leaked mass at the loss slot); weighable iff in-loop loss-channel CE + slot-level leaked-mass trajectories ship (#613)
type: feedback
---

In single-variable flag A/Bs that relocate a loss term (e.g. #613: negative-row
loss moved to the post-response slot, matched to #601's flag-off arm), the
co-landing NULL has a built-in quantitative alternative: the relocated
gradient's restoring force scales with the leaked target mass at the loss slot,
so "flag is live but had nothing to push against at this leakage/lr dose" and
"placement is not the lever" produce identical terminal reads. NOT a REVISE
when the plan registers (a) the in-loop loss-channel CE trajectory (liveness +
growth = whether the gradient ever found opposition) and (b) four-float
slot reads at the LOSS slot itself on both arms — the analyzer can then scope
the null to "insufficient under this recipe's leakage regime" instead of
"placement irrelevant". Also check: a wide co-landing band (e.g. ±5.58 nats
from parent seed gaps) makes the binary call insensitive to real moderate
suppression — demand descriptive sub-band directional reporting + the
sensitive secondary channel (trained-negative clamp/trajectory).

**Why:** #613 alternatives review — the plan's falsification prose said
co-landing means "the residual #471 differences carry it", but #471's restoring
force appeared at 9-15 nats trained-neg leakage vs #601's ~5-6; the gradient at
the relocated slot is ∝ P(※)|slot ≈ e⁻¹⁶ here. The plan's own risk row named
"live but toothless" as a registered outcome; the diagnostics made it weighable.

**How to apply:** any A/B whose manipulated variable is WHERE a loss term acts
(slot relocation, masking change, contrastive-arm wiring). Ask: does the
restoring/suppressing gradient's magnitude depend on a state variable (leakage,
emission rate) that differs between the conflicting parent rigs? If yes, the
null needs the gradient-magnitude diagnostics, and the verdict prose must scope
to the realized regime.
