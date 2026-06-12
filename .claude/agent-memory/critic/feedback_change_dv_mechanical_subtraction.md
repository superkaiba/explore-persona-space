---
name: Change-DV mechanical subtraction vs base-like predictors
description: dmargin = post − base contains −base by construction; any predictor collinear with the base read is mechanically pushed toward zero on the change DV, so "predictor carries level not change" division-of-labor confirmations are partly baked in
type: feedback
---

When a plan fits a *change* DV of the form `d = margin_trained − margin_base` and
predicts that a base-read-like predictor (e.g. an own-response base prior strongly
collinear with `margin_base`) will NOT survive on the change DV while a second
predictor (geometry) will — that "division of labor" confirmation is partially
mechanical: the change DV contains `−margin_base` by construction, so any predictor
≈ `margin_base` has its level signal subtracted out regardless of mechanism.

**Why:** Surfaced on #559 (own-response prior on the persona panel, H2 secondary
fit `dmargin ~ α·z(prior) + β·z(min_dist)` with the registered expectation "β
survives, α does not"). The expected outcome is favored by DV construction, not
only by the claimed channel structure. Sibling of the shared-NOISE version in
feedback_matrix_testbed_alternatives.md (shift-DV shares base-panel noise with a
base-prior predictor); this is the shared-SIGNAL version.

**How to apply:** Not a REVISE when (a) the change-DV fit is secondary /
reported-either-way, and (b) the predictor↔base-read correlation ships as a
diagnostic — the analyzer can weigh it. Prescribe: the clean-result must not
narrate the predictor's non-survival on the change DV as fresh evidence of
division of labor without naming the subtraction; a useful extra read is
`d ~ predictor + geometry + margin_base` (or residualize the predictor against
`margin_base` first) to show how much of the null-on-change is construction.

**Second direction (#605):** the same construction can spuriously fire the
OPPOSITE branch when the plan's falsification criterion is sign-agnostic
(e.g. "prior carries ΔCV-R² weight on the shift" → "gate needs an overlap
term"). The −base component gives a base-like predictor a mechanically
NEGATIVE shift coefficient, which adds predictive weight without any genuine
gate term — while the substantive hypothesis (behavior-overlap gate) predicts
a POSITIVE sign. Same disposition: Concern, not REVISE, iff per-cell base
reads ship (four-float contract) so the analyzer can run the base-margin-
controlled fit and read the sign per band. Prescribe sign-aware
interpretation of any "predictor wins on change DV" verdict row.
