---
name: matched-corpus-geometry-control alternatives
description: Two alternatives endemic to matched-corpus content controls on shift-geometry DVs — magnitude-mediated concentration collinearity (||M||_F confound) and shared-corpus-content vs generic-SFT direction in cross-arm U1 identity reads
type: feedback
---

From the #552 benign plain-SFT control review (2026-06-10; parent #521 EM-vs-marker
shift geometry). Applies to any plan comparing direction-concentration DVs
(cos(Δv, U₁), σ₁/Σσ) across arms whose corpora differ in content but match in
prompts/recipe. (The third endemic alternative — attenuation false-confirmation under
a split-half ≥ 0.5 gate — is in `feedback_reliability_precondition_boundary.md`.)

**1. Magnitude-mediated concentration collinearity.** Near-policy corpora give small
true updates; fixed-scale idiosyncratic per-persona components then dominate the
spectrum, so top-share drops mechanically without any content-direction claim. Check
whether ALL arms in the comparison are collinear in (‖M‖_F, concentration) — in the
#521/#552 family they are (EM large+concentrated, marker small+unconcentrated, benign
predicted small+unconcentrated), so the whole multi-arm story is consistent with
"concentration is monotone in update size". Prescribe: ‖M‖_F per cell next to the
headline (recoverable as sqrt(Σσᵢ²) from persisted singular values), train-loss
deltas + per-corpus token counts as effective-signal diagnostics, and headline
phrasing "at matched recipe and exposure" leaving the semantics-vs-update-magnitude
mediator open. A magnitude-matched arm is a follow-up, not a same-plan condition
(breaks single-variable). If the small arm's ‖M‖_F lands within ~2× of the big arm's
with concentration still low, the alternative dies and the claim strengthens.

**2. Cross-arm U₁ identity in matched-corpus designs can't separate "generic SFT
direction" from "shared corpus content/register direction".** When two corpora share
prompts, topic, and register (differing only in the manipulated property — e.g.
Turner good vs bad medical advice differ only in answer correctness), a high
cross-arm |cos(U₁,U₁′)| fits both readings; no within-design cell discriminates. The
discriminator is an off-topic same-recipe arm (follow-up). Mitigation note worth one
sentence: if shifts are measured on off-topic held-out probes, a pure topic direction
must generalize off-topic to appear. Pre-registered "> threshold → generic direction"
interpretations get downgraded by the analyzer, not bounced pre-execution, provided
full U₁ vectors persist.

**Why:** both were APPROVE-with-concerns on #552 because the plan persisted
per-question tensors + singular values + U₁ vectors; the same alternatives escalate
toward Must-Fix when those artifacts are NOT stored (the analyzer cannot weigh them).
