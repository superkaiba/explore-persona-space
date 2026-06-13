---
title: 'P8 estimator bake-off: score the three base-model v_b estimators against realized
  post-training states on stored adapters'
kind: analysis
tags: []
created_at: '2026-06-11T10:48:14Z'
has_clean_result: false
---
# P8 estimator bake-off: score the three base-model v_b estimators against realized post-training states on stored adapters

## Goal

Determine whether the base-model estimators of the behavior vector v_b are valid: do the three candidate estimators agree with each other and with the realized post-training state v_b'' at the read-out layers — and where they disagree, does substituting the realized v_b'' repair formula predictions that the estimated v_b gets wrong?

## Motivation

The rank-1 leakage model (docs/notes/rank1_leakage_model.tex, §Setup + prediction P8) needs the post-training behavior state v_b'', which is unobservable before training; estimating it from the base model is itself part of the proposal. If the estimators are invalid, every formula prediction made with an estimated v_b is unidentified — this gates all downstream GPU spend on formula tests, which is why this runs before the other formula-test analyses. A warning is already on record: the prompted marker direction is unrelated to the realized training shift (cos ≈ −0.03, #521).

## Design sketch

- The three estimators, in decreasing expected fidelity (all are the context-to-state map f evaluated at a behavior-eliciting context):
  1. teacher-forced reads over the prospective training data's own completions;
  2. in-context demonstrations drawn from that data, read after the context and averaged over user prompts (the variant run so far);
  3. natural-language description conditioning.
- Score each against the realized post-training state v_b'' (trained − base activation shift at the read-out layers) on stored adapters spanning behavior families: marker implants, fact implants, and refusal/EM/sycophancy where artifacts exist.
- Agreement metrics: cosine between each estimator's direction and the realized shift direction, per layer; cross-estimator agreement; rank correlation of per-context leakage predictions made with estimated vs realized v_b.
- Failure localization: where a formula prediction fails with the estimated v_b, re-run it with the realized v_b'' — if that repairs it, the estimation problem (not the update rule) is what failed.
- Sweep constructions (layer, read position, aggregation, estimator) rather than fixing them a priori.

## Artifacts to reuse (positive fitness-check before use)

- #493 activation-extraction engine.
- #521 per-context shift tensors (HF data repo `analysis_tensors/`).
- Stored adapters on `superkaiba1/explore-persona-space`: #541 fact adapters, #518 refusal/EM adapters, marker-line adapters (#474 loc-arm epoch-1 and related).

## Expected cost

Primarily VM/CPU on stored tensors. Forward passes for estimators 1–3 may need a short 1×H100 eval pod if the required base-model activations are not already cached — planner verifies artifact coverage first.

## Deliverable

Clean-result: per-estimator validity verdict (which estimator, if any, is fit to drive formula predictions), the cross-estimator agreement table, and the repair-test outcome.
