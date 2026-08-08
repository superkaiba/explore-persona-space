---
name: stage-permutation-null-modest-construct
description: "Stage-label permutation nulls on per-stage ON-POLICY rows do NOT absorb generic answer-distribution shifts (length/format/templating) — those are TRUE positives of the construct; fatality turns on whether the Goal claims attribution (#2061 v12)"
metadata:
  type: feedback
---

For a cross-stage ΔR² (or any Δ-predictability) design where each stage's rows
are the stage's OWN on-policy generations and the null permutes stage labels
within corpus (pair-swap on conv-id intersection, refit per draw): the null
absorbs estimator noise, rare-feature instability, and (via unpaired rows held
in their natural stage) part of row-composition shift — but it deliberately
does NOT absorb any SYSTEMATIC per-stage distribution change. Answer
templating, length shifts, and format-compliance gains are true positives of
the registered construct, not artifacts.

**Why:** the construct "feature j became more predictable from context" is
defined ON the stage's own answer state; a boring cause (templated answers)
genuinely raises predictability of that object. The alternative-explanations
question is therefore ATTRIBUTION, not measurement validity — fatal only if
the Goal/hypothesis claims WHICH change drove the delta (mechanism /
"interpretable feature" gloss) without a discriminating read.

**How to apply:** (1) check whether the Goal is modest-descriptive (any
predictability change counts) vs attributive — modest ⇒ the alternatives are
analyzer Concerns, not REVISEs; (2) demand the winner-attribution material be
PERSISTED (winner feature id, per-stage activation counts from encoded
targets, keep-rate/paired-fraction manifest, per-transition split) so the
analyzer can weigh a base→SFT "coherence gain" winner; (3) note a
constant-input arm (chat prefix slot) controls PIPELINE artifacts only — with
X constant, held-out R² pins near 0 regardless of Y shifts, so it is NOT a
control for answer-side distribution alternatives. (#2061 v12, 2026-08-06;
sibling: [[single-generation-selection-axis]], [[pool-ceiling-absolute-gate]].)
