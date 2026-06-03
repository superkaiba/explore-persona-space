---
title: Brainstorm leakage predictors beyond JS divergence and cosine similarity
kind: analysis
tags: []
created_at: '2026-06-03T09:03:06Z'
has_clean_result: false
---
# Brainstorm leakage predictors beyond JS divergence and cosine similarity

## Objective

Produce a ranked menu of candidate predictors for marker/behavior leakage from a
source persona into a related (variant or conditional) persona, going beyond the
two we currently rely on: output-distribution Jensen-Shannon divergence and
persona-vector cosine similarity. The deliverable is a short design doc, not a
training or eval run.

## Why now

The two predictors we lean on have known limits:

- Cosine similarity depends heavily on where the activations are read (end of
  system prompt, end of user question, averaged over the answer). The
  end-of-system-prompt read happens before the user has typed, so it cannot see
  a trigger that has not appeared yet.
- JS divergence is computed on the same answers whose leakage it is meant to
  explain, so it partly re-measures the output rather than predicting it ahead
  of time.
- Both, averaged over a generic question set, are blind to a persona that
  matches the source on most questions and only diverges on a narrow slice of
  inputs. That averaging hid the divergence until the slice-by-question-type
  view recovered it.

We want a wider set of predictors, ideally ones that (a) are cheap and
eval-only, computable on the un-marker-trained base model, (b) genuinely
predict the outcome instead of co-measuring it, and (c) stay sensitive when two
personas differ only on a narrow slice of inputs.

## Deliverable

A document that, for each candidate predictor:

1. Names it in plain language and states what it measures.
2. Gives a concrete way to compute it (what quantity, on which model, at which
   read point, reusing which existing helper if any).
3. States the cost (number of forward passes; whether it needs gradients or any
   training).
4. States expected strengths and weaknesses against the three criteria above
   (cheap + base-model, predictive not co-measured, slice-sensitive).
5. Flags whether it could be validated against existing data from the prior
   experiments without a new run.

End with a shortlist of 2-3 to test first and the reasoning for that order.

## Seed directions (starting points, not a ceiling)

- Divergence variants: forward or reverse KL instead of JS; a max-over-positions
  or worst-slice aggregation instead of a mean, so a rare-slice difference is
  not washed out by the average.
- Probe-based: a linear probe for the marker direction in the residual stream,
  measuring how much of that direction survives under the variant persona.
- Gradient or weight-space: similarity of the marker update direction under the
  source vs the variant persona, or the projection of the marker weight delta
  onto the variant persona's activation subspace.
- Representational similarity across layers (e.g. CKA) between the two personas.
- Purely behavioral: held-out answer-agreement rate between the two personas, as
  a cheap baseline predictor to beat.
- Internal-state reads taken strictly before the model produces its answer, so
  they predict the leakage rather than co-measure it.

## Context and prior work in-project

- Prior line: the cosine-similarity predictor work, the JS-divergence predictor
  work, and the conditional/slice-concentrated result that showed bucket
  averaging is blind, slicing by question type recovers the signal, and the
  JS-vs-cosine ranking is still unresolved at the per-answer level.
- Canonical persona-distance definitions live in
  `.claude/rules/persona-distance-metrics.md`.
- Tracks the open question `q:leak-predictor`.

## Acceptance

- A written menu covering at least 6 candidate predictors across at least 3
  distinct families (divergence-based, probe/representation-based,
  gradient/weight-based, behavioral).
- Each entry has an operationalization concrete enough to hand to an
  implementer.
- A ranked shortlist of the first 2-3 to test, with which existing data (if any)
  could validate them before committing GPU time.
