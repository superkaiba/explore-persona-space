---
name: matched-corpus-geometry-control alternatives
description: Matched-corpus content controls on shift-geometry DVs — ‖M‖_F-mediated concentration collinearity, and cross-arm U₁ identity that can't split generic-SFT from shared-corpus direction (#552)
type: feedback
---

From the #552 benign plain-SFT control review (parent #521 EM-vs-marker shift geometry). For any plan comparing direction-concentration DVs (cos(Δv, U₁), σ₁/Σσ) across arms whose corpora differ in content but match in prompts/recipe:

1. **Magnitude-mediated concentration collinearity.** Near-policy corpora give small true updates; fixed-scale idiosyncratic per-persona components then dominate the spectrum, so top-share drops mechanically without any content-direction claim. Check whether ALL arms are collinear in (‖M‖_F, concentration) — in the #521/#552 family they are (EM large+concentrated, marker small+unconcentrated, benign predicted small+unconcentrated): the whole multi-arm story is consistent with "concentration is monotone in update size". Prescribe: per-cell ‖M‖_F next to the headline (recoverable as sqrt(Σσᵢ²) from persisted singular values), train-loss deltas + token counts as effective-signal diagnostics, headline phrased "at matched recipe and exposure" leaving the mediator open. A magnitude-matched arm is a follow-up (breaks single-variable). If the small arm's ‖M‖_F lands within ~2× of the big arm's with concentration still low, the alternative dies.
2. **Cross-arm U₁ identity can't split "generic SFT direction" from "shared corpus content/register direction"** when the corpora share prompts, topic, and register (differing only in the manipulated property). The discriminator is an off-topic same-recipe arm (follow-up). Worth one sentence: if shifts are measured on off-topic held-out probes, a pure topic direction must generalize off-topic to appear. Pre-registered "> threshold → generic direction" interpretations get downgraded by the analyzer, not bounced, provided full U₁ vectors persist.

**Why:** both APPROVE-with-concerns on #552 because per-question tensors + singular values + U₁ vectors persisted; both escalate toward Must-Fix when those artifacts are NOT stored. (Third endemic alternative — attenuation false-confirmation under a split-half ≥ 0.5 gate — is feedback_reliability_precondition_boundary.md.)
