---
title: 'Minimal experiment: predict leakage to (C′,B′) from training on a SET of (C,B)
  cells (develop the set-to-cell distance metric)'
kind: experiment
tags: []
created_at: '2026-05-29T23:12:22Z'
has_clean_result: false
goal: 'Design and run a minimal experiment testing whether leakage to an unseen (context,
  behavior) cell (C′,B′) is predictable from a distance between (C′,B′) and a SET
  of trained (C,B) cells, developing the (C,B)-cell distance + set-aggregation rule
  (candidate aggregation: minimum over the trained cells).'
---
## Goal

Design and run a minimal experiment testing whether leakage to an unseen (context, behavior) cell (C′,B′) is predictable from a distance between (C′,B′) and a SET of trained (C,B) cells, developing the (C,B)-cell distance + set-aggregation rule (candidate aggregation: minimum over the trained cells).


**Open question:** `docs/open_questions.md` §3.9 (`q:leak-from-cell-set`). Generalizes [#440](https://eps.superkaiba.com/tasks/440) (single-cell predictor) and Dan's 2026-05-26 multi-persona-set note (train a behavior into K personas, measure leakage to held-out personas vs similarity to the trained set).

## Motivation

The project's central prediction question is: train at a (context, behavior) cell, predict behavior at other cells. In practice you don't train on one cell — you fine-tune on a SET of (C, B) cells (several personas, several behaviors, a data mixture). The open question: given training on a set {(C_1,B_1), …, (C_k,B_k)}, what predicts leakage to an unseen (C′, B′)?

This needs a metric we don't have: a distance between a (C′,B′) query cell and a SET of trained cells. Dan's candidate (2026-05-29): the set-to-cell distance is the MINIMUM over the trained cells — leakage to (C′,B′) is governed by its nearest trained cell, not the set centroid. Alternatives to test: mean, soft-min, k-NN.

## What to scope (this issue = design the MINIMAL experiment)

The deliverable is the smallest experiment that can falsify "min-distance over the trained (C,B) set predicts leakage to (C′,B′)". Keep it minimal:

- A small, cheap (C,B)-cell space where both the C axis (persona/context) and the B axis (behavior) have a few well-separated points, and a usable per-cell distance already exists (reuse the persona-distance from the leakage pilots + a behavior-distance from 3.6 / #404).
- Train on a few (C,B) cells; hold out (C′,B′) cells chosen to span a range of *min-distance* to the trained set.
- Measure leakage at each held-out cell; test whether min-distance predicts it, and whether min beats mean / soft-min aggregation.
- Reuse existing rigs (marker or behavior leakage) to keep cost low.

Open design questions to settle in planning:

- Right (C,B)-cell distance — persona-geometry × behavior-distance, or a joint metric?
- Min vs mean vs soft-min aggregation — which to test first?
- Marker as the behavior (cheap, but ~5 dead implantability predictors) vs a real behavior (sycophancy / EM, closer to the threat model)?

## Links

- `docs/open_questions.md` §3.9 (`q:leak-from-cell-set`), the §3 prediction framing, §1 distance.
- #440 (single-cell predictor), #404 (behavior-distance B→B′), #207 / #311 (single-cell leakage gradient).
- Dan notes: 2026-05-29 (set-of-(C,B), min-distance), 2026-05-26 #4 (multi-persona set).
