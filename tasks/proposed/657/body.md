---
title: Does persona->behavior-direction alignment predict cross-persona leakage across
  behaviors, beating the base prior?
kind: experiment
tags: []
created_at: '2026-06-18T09:48:28Z'
has_clean_result: false
parent_id: 623
origin_prompt: 'Extend #623''s alignment predictor across behaviors + to leakage:
  does behavior-direction alignment predict (a) base rate for other behaviors and
  (b) where leakage lands, beating the base prior? Reuses existing persona vectors
  + adapters.'
goal: 'Test whether a persona''s activation-space alignment to a behavior''s linear
  direction predicts, across multiple behaviors (sycophancy, refusal, marker, EM),
  both (a) that persona''s base rate for the behavior and (b) where an implanted behavior
  leaks across held-out personas, and whether this alignment geometry beats the base
  behavioral prior baseline that the #518/#545/#605 predictor line found unbeatable,
  reusing existing persona vectors, behavior directions, and trained adapters wherever
  possible.'
---
## Goal

Test whether a persona's activation-space alignment to a behavior's linear direction predicts, across multiple behaviors (sycophancy, refusal, marker, EM), both (a) that persona's base rate for the behavior and (b) where an implanted behavior leaks across held-out personas, and whether this alignment geometry beats the base behavioral prior baseline that the #518/#545/#605 predictor line found unbeatable, reusing existing persona vectors, behavior directions, and trained adapters wherever possible.

## Background

#623 found that a persona vector's alignment (cosine) to the sycophancy behavior direction predicts the persona's BASE sycophancy rate (rho=0.73). This is a positive predictor signal in a geometry — behavior-direction alignment — distinct from the cosine/JS context-similarity family that repeatedly nulled across the predictor line (#518, #545, #605, #603, #649: no base-model geometry beat the base behavioral prior on held-out leakage).

Two questions #623 did not address:
1. Does the alignment -> base-rate relationship hold for behaviors OTHER than sycophancy?
2. Does alignment predict WHERE an implanted behavior LEAKS (cross-persona leakage landing) — the quantity the predictor line actually cares about — and does it beat the base prior at that?

## Hypotheses

- H1: behavior-direction alignment predicts base rate across >=3 behaviors (generalizes #623 beyond sycophancy).
- H2: behavior-direction alignment predicts cross-persona leakage landing with held-out (out-of-sample) skill above chance.
- H3 (the bar): alignment beats the base behavioral prior on held-out leakage prediction. The prior won everywhere in #518/#545/#605; this is the first geometry with a real shot.
- Competing/null: alignment merely restates "high-prior personas leak more" and adds nothing once the base prior is partialled out. The X-vs-(X-Y) artifact (#383/#605) MUST be controlled — report alignment's marginal skill with the prior partialled out.

## What to reuse

- Existing persona / context vectors (#594/#623 banks).
- Existing behavior directions (sycophancy from #623; marker, refusal, EM directions from their lines — planner to locate).
- Existing trained adapters + ALREADY-MEASURED cross-persona leakage from awaiting-promotion tasks (#591, #605, #545, #537, #612, ...) so leakage DVs are reused, not regenerated. Goal: near-0 GPU.

## Measurement (planner to finalize)

- DV(a): base behavior rate per persona (on-policy judge rate; reuse existing base generations where available).
- DV(b): cross-persona leakage landing per (source, bystander) cell (on-policy, reuse existing measurements).
- Predictor: cosine(persona vector, behavior direction) vs the base-prior baseline. Fit on a subset of personas/behaviors, test on held-out ones; ALWAYS report alignment's marginal skill with the base prior partialled out.

## Provenance

Originating user request (PM chat, 2026-06-18, verbatim): "Extend #623's alignment predictor across behaviors + to leakage. #623 found a persona vector's alignment to the sycophancy direction predicts its base sycophancy rate (rho=0.73) — a different geometry than the cosine/JS family that kept nulling. Open question that 6 experiments left unanswered: does behavior-direction alignment predict (a) base rate for other behaviors and (b) where leakage lands, beating the base prior? Mostly reuses existing persona vectors + adapters -> likely cheap / near-0 GPU. Highest payoff: could finally crack the predictor question."

Routed as a NEW child of #623: it changes the dependent construct from #623's base-rate correlation to a cross-behavior leakage-prediction bake-off against the base prior (the #518/#545/#605 question), so the result would not merely rewrite #623's Takeaways.
