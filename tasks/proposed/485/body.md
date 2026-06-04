---
title: Does the base-model cosine predictor predict broad→broad behavior leakage (train
  one general trait, measure another)?
kind: experiment
tags: []
created_at: '2026-06-04T09:45:03Z'
has_clean_result: false
parent_id: 468
goal: 'Test whether the #468 base-model cosine predictor predicts post-SFT leakage
  between two broad (general, non-domain-bounded) behaviors, which first requires
  operationalizing at least two broad axes beyond emergent misalignment.'
---
## Goal

Test whether the #468 base-model cosine predictor predicts post-SFT leakage between two broad (general, non-domain-bounded) behaviors, which first requires operationalizing at least two broad axes beyond emergent misalignment.


## Motivation

The predictor line (#404 → #458 → #463 → #468) has operationalized exactly one **broad** axis: broad / emergent misalignment. The general framing also covers **broad → broad** leakage — train the model toward one general, non-domain-bounded trait (e.g. sycophancy) and measure whether a *different* general trait (e.g. deception, power-seeking) shifts — and asks whether the base-model cosine between the two broad behavior vectors predicts how much one leaks to the other.

Unlike narrow→narrow (#484), this cell needs setup first: we currently have no second broad axis. A broad behavior here means a trait that applies across domains, with (a) a way to build its behavior vector and (b) a judge that scores the trait out-of-distribution. This task is the exploratory, higher-setup-cost arm of the generalization sweep.

## Design sketch (to be fleshed out by /adversarial-planner)

- **Define ≥3 broad behaviors**, each with a build recipe (description and/or in-context examples) and an OOD judge eval. Candidates: sycophancy, deception, power-seeking, over-refusal, broad misalignment.
- **Outcome:** for each ordered pair (A → B), fine-tune the base model toward broad A — using a dataset/recipe that installs the general trait, with contrastive negatives per the project rule — at fixed steps, token-volume controlled, ≥2 seeds; measure the post-SFT broad-B rate with B's judge.
- **Predictor:** base-model cosine between the behavior vectors for A and B, read at the newline-after-assistant token. Prefer **example-based** vectors over descriptions, since #468 found description-only personas do not predict and #467 found rich descriptions fail to load the misaligned personas — so this depends on the in-context broad-persona work (#486).
- **Test:** regress cosine(A, B) against leakage(A → B).

## Carry these caveats from the line

- **Broad behaviors are hard to elicit from a description** (#467). Building example-based broad vectors (the in-context broad-persona task) is effectively a prerequisite.
- **Installing a broad trait by SFT is itself nontrivial** — needs a dataset/recipe that produces the general trait without collapsing to a narrow domain; contrastive negatives required.
- **Judge validity per broad axis** — each broad trait needs an OOD judge that measures the construct and does not saturate.
- **Low power.** Few broad axes → few pairs; treat as exploratory and report effect sizes honestly rather than chasing significance.
