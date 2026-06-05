---
title: P(training data | bystander context) as a base-rate baseline predictor of leakage
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:52Z'
has_clean_result: false
parent_id: 414
relates_to:
- leak-predictor
goal: Compute, training-free, the base model's mean log P(trained completion | bystander
  system prompt) over the training set per bystander, and test it as a baseline predictor
  of post-training leakage to that bystander — and as a control for whether cosine/JS
  are partly re-measuring this base rate.
---
## Goal

Compute, training-free, the base model's mean log P(trained completion | bystander system prompt) over the training set per bystander, and test it as a baseline predictor of post-training leakage to that bystander — and as a control for whether cosine/JS are partly re-measuring this base rate.


## Motivation

Cosine/JS measure persona *similarity* but ignore how much a bystander **already** assigns probability to the trained content before any training. The roadmap proposes using P(training data | bystander context) as a baseline leakage measurement. #414 is the closest existing task (P(narrow training data | broad persona) as *the* predictor for narrow → broad-EM), but the symmetric per-bystander base-rate framing is not its own task. #481's brainstorm lists a behavioral baseline ("held-out answer-agreement rate between the two personas") as a cheap predictor to beat — this is its concrete log-prob instantiation.

## What exists to reuse

- #414's design (P(data | persona) via base-model scoring).
- Training datasets on HF for every behavior line; bystander panels from #207 / #311 / #411.
- Frozen leakage matrices to regress against: #207 / #469 / #474 (marker), #411 (sycophancy), #381 / #444 (facts).

## Design sketch (for /adversarial-planner)

For each bystander, compute the base model's mean log P(trained completion | bystander system prompt) over the training set — training-free. Regress this base rate against the post-training leakage to that bystander. Then compare/partial it against cosine/JS: does the base rate explain leakage that cosine/JS miss, or are they redundant (i.e. is cosine/JS partly re-measuring this base rate)?

## Hypothesis

Bystanders that already place high probability on the trained data leak more after training, and this training-free base rate is competitive with — or a needed control for — cosine/JS.

## Caveats

- Distinct from #414: this is the symmetric per-bystander baseline, not the broad-persona-produces-narrow-data screen.
- This is a candidate component of the "better metric" search (roadmap item 3); the metric search itself lives in #481 (brainstorm) / #493 (running bake-off) / #447 (learned predictor), so scope this to the base-rate baseline specifically.

## Lineage / open questions

Advances **q:leak-predictor** (3.1). Parent #414; brainstorm #481; learned-predictor #447.
