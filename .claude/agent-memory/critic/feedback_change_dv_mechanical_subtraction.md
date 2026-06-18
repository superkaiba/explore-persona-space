---
name: Change-DV mechanical subtraction vs base-like predictors
description: d = post − base contains −base by construction; base-collinear predictors are mechanically nulled (#559) OR mechanically negative-significant, firing sign-agnostic falsification branches (#605) — prescribe sign-aware reads
type: feedback
---

When a plan fits a *change* DV `d = margin_trained − margin_base` and predicts that a base-read-like predictor (e.g. own-response base prior, strongly collinear with `margin_base`) will NOT survive on the change DV while another (geometry) will, that "division of labor" confirmation is partially mechanical: the DV contains `−margin_base` by construction, so any predictor ≈ `margin_base` has its level signal subtracted out regardless of mechanism.

**Why (#559):** H2's secondary fit `dmargin ~ α·z(prior) + β·z(min_dist)` with the registered expectation "β survives, α does not" — favored by DV construction, not only the claimed channel structure. Sibling of the shared-NOISE version in feedback_matrix_testbed_alternatives (this is the shared-SIGNAL version).

**Second direction (#605):** the same construction can spuriously fire the OPPOSITE branch when the falsification criterion is sign-agnostic ("prior carries ΔCV-R² weight on the shift" → "gate needs an overlap term"): the −base component gives a base-like predictor a mechanically NEGATIVE shift coefficient — predictive weight with no genuine gate term — while the substantive hypothesis predicts a POSITIVE sign.

**How to apply:** Not a REVISE when (a) the change-DV fit is secondary / reported-either-way and (b) the predictor↔base-read correlation ships as a diagnostic. Prescribe: never narrate non-survival (or survival) on the change DV as fresh mechanism evidence without naming the subtraction; ask for `d ~ predictor + geometry + margin_base` (or residualize the predictor against `margin_base` first), and sign-aware interpretation of any "predictor wins on change DV" verdict row. Weighable iff per-cell base reads ship (four-float contract).
