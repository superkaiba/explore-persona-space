---
title: 'Leakage-vs-implantation control: do cross-condition leakage differences survive
  matching or normalizing by install strength?'
kind: experiment
tags: []
created_at: '2026-06-12T20:37:50Z'
has_clean_result: false
parent_id: 601
origin_prompt: 'can we also test leakage as a % of implantation?

  [...]

  Yes run this as followup task and then add to workflow somewhere to do this'
goal: Determine whether headline cross-condition leakage differences (contrastive
  vs positive-only, LoRA vs full fine-tuning) survive controlling for implantation
  strength, via matched-install comparison, per-cell leakage-to-install fractions
  in logit space, and leakage-vs-install dose curves from existing training trajectories.
---
## Goal

Determine whether headline cross-condition leakage differences (contrastive vs positive-only, LoRA vs full fine-tuning) survive controlling for implantation strength, via matched-install comparison, per-cell leakage-to-install fractions in logit space, and leakage-vs-install dose curves from existing training trajectories.


## Summary

Headline cross-condition leakage comparisons (contrastive vs positive-only training, LoRA vs full fine-tuning, data-construction variants) are currently read off raw bystander leakage. That read is confounded: a condition that implants the behavior more strongly overall pushes the behavior up on bystanders too, so "leaks more" is ambiguous between *lower selectivity* (interesting) and *plain higher dose* (boring). The install-strength difference is real in both flagship comparisons: #601 found contrastive negatives strengthen the marker implant (longer optimizer path), and #608 found positive-only sycophancy installs at least as strongly as contrastive. This task runs the control: re-read the leakage comparisons with implantation strength controlled.

## Design sketch — three reads

1. **Matched-install comparison (primary).** Compare conditions at matched source install strength: select checkpoints (from existing per-step trajectories / band-stop checkpoints) where the source gain matches across conditions, then compare bystander leakage there. Where no matched checkpoint exists, state the gap explicitly rather than interpolating.
2. **Leakage as a fraction of install (per-cell statistic).** Per (source → bystander) cell: bystander gain ÷ source gain, computed in the non-saturating logit space (EOS margin `Δ(z_marker − z_eos)`; the four-float per-slot storage contract makes this computable wherever it was honored). Raw `log P` is invalid for this ratio when the source is saturated — softmax compression understates the denominator and inflates the ratio in exactly the strongest-implant conditions.
3. **Leakage-vs-install dose curves.** Per condition, fit leakage vs install across training steps from the WandB-logged per-step trajectories (multiple (install, leakage) points per run); compare slopes/shapes across conditions. This subsumes the single ratio (one point on the curve) and catches nonlinearity — e.g. leakage onset above an install threshold — that would make a single percentage mechanically dose-dependent even at identical selectivity.

Statistical hygiene: never correlate the fraction back against install itself (noisy-denominator artifact, same family as the #383 X-vs-(X−Y) circularity caveat in `.claude/rules/contrastive-negatives.md`); the fraction compares conditions, while dose dependence goes through the curve fit or the matched design.

**Data first, GPU second.** Much of this is re-analysis of existing artifacts (eval JSONs under `eval_results/`, WandB per-step trajectories, stored four-float slot reads). First step is an inventory of which comparisons already have the matched-install cells or trajectories needed; new training only where a comparison genuinely lacks them, and then by extending existing recipes to other dose points rather than new designs.

## Relation to existing work

#601 (contrastive negatives strengthen the implant — the marker-side dose confound), #608 (positive-only installs at least as strongly — the sycophancy side), #606 (LoRA vs full-FT sycophancy leakage — verify what was matched in that comparison and whether it survives this control), #605 (matched-similarity comparison precedent), the #383 selectivity-circularity caveat. The marker band-stop recipe (`.claude/rules/marker-training-recipe.md`) already targets a source-gain band, which is partial matched-install machinery this task can reuse.

## Provenance

Captured from the 2026-06-11 collaborator meeting notes (`docs/mentor_updates/2026-06-11-christina.md`, § "Leakage / implantation control" — verbatim note: "Run a control to test whether higher *average* leakage is just due to *more implantation* (stronger implant overall, not less selectivity)."). Expanded in chat 2026-06-12 with the fraction-of-install and dose-curve reads; filed for autonomous execution.
