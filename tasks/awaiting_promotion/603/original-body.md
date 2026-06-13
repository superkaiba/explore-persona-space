---
title: 'P3'' write decomposition: does the source''s base prior predict the write''s
  common-mode fraction rather than its norm?'
kind: analysis
tags: []
created_at: '2026-06-11T10:48:15Z'
has_clean_result: false
---
# P3′ write decomposition: does the source's base prior predict the write's common-mode fraction rather than its norm?

## Goal

Test prediction P3′ of the rank-1 leakage model: the source's base prior on the behavior predicts the write's common-mode fraction (its projection onto the shared / mean-bystander shift direction), not its norm.

## Motivation

P3′ (docs/notes/rank1_leakage_model.tex): if the source context already carries the behavior's shared component, little of it needs writing — the residual concentrates in source-specific subspace. #541 observed the motivating configuration: high-prior teachers express the implant most on themselves (97/94% self-assertion vs the leaky low-prior teacher's 72%) while leaking least ("not weaker, narrower"). A magnitude-only reading of P3 cannot produce this — a uniformly small write would weaken expression everywhere, including on the source itself.

## Design sketch

- For each stored adapter: extract per-context activation shifts (trained − base) over the bystander panel plus the source context, at the read-out layers.
- Decompose each source's write into a shared component (projection onto the mean-bystander shift direction) and a source-specific remainder.
- Regress the common-mode fraction on the source's base prior (the behavior's base-model expression under the source context); contrast with the norm-on-prior regression (the magnitude-only P3 reading).
- Prediction: prior predicts common-mode fraction (negatively), not norm. The contrast between the two regressions is the deliverable, not just the headline coefficient.

## Artifacts to reuse (positive fitness-check before use)

- #541's nine fact adapters (they span teacher prior — the designed dose axis for this test).
- #518's refusal/EM adapters (extend the test across behavior families).
- #521's shift-extraction pipeline and stored tensors where applicable.

## Expected cost

Zero training. Activation extraction on existing adapters only; a short eval pod only if the needed (adapter, context) shift cells are not already stored.

## Deliverable

Clean-result: common-mode-fraction-on-prior regression vs norm-on-prior regression, per behavior family, with the #541 nine-teacher span as the primary axis.
