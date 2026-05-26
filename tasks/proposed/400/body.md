---
title: Per-persona first-step gradient on ※ log-prob — candidate vulnerability predictor
kind: experiment
tags: []
created_at: '2026-05-26T20:49:06Z'
has_clean_result: false
parent_id: 380
goal: For each of the 48 personas, measure Δ log p(※) from one optimizer step on persona-conditioned
  ※ training data, and test whether this first-step gradient predicts post-training
  source rate better than the three failed geometric-distance predictors (cosine,
  JS, pairwise output distance).
---
## Goal

For each of the 48 personas, measure Δ log p(※) from one optimizer step on persona-conditioned ※ training data, and test whether this first-step gradient predicts post-training source rate better than the three failed geometric-distance predictors (cosine, JS, pairwise output distance).

## Background

Three flavors of geometric distance from the assistant baseline ([#271](https://eps.superkaiba.com/tasks/271) cosine, [#380](https://eps.superkaiba.com/tasks/380) JS, [#380](https://eps.superkaiba.com/tasks/380) pairwise output distance) have all failed to predict post-training source rate on the [#274](https://eps.superkaiba.com/tasks/274) / [#296](https://eps.superkaiba.com/tasks/296) panel. The four predictor candidates flagged in the daily-update conversation (2026-05-25) for the source-rate panel re-run are: (a) base-model marker prior, (b) base-model loss on training rows, (c) persona susceptibility to known persona-axis directions, (d) format-output entropy. This task takes (b) further: instead of measuring base loss as a static property, measure the *gradient* — how much one optimizer step on persona-conditioned marker data moves log p(`※`).

The hypothesis is: personas where the first gradient step produces a large log-prob delta are personas where the implant has high "headroom" and "alignment" with the persona representation; personas where the first step produces little movement are personas where the implant runs against a strong prior or finds no aligned weight direction. Asymptotic source rate is the end-state of N=3 epochs of optimization; first-step Δ is the *direction* of the implant gradient at initialization.

This experiment is enabled specifically by the marker swap to `※` — the 4-token chain of `[ZLT]` blurred per-step deltas across multiple gradient contributions. Single-token `※` gives a clean per-step measurement.

## What this tests

- Whether per-persona first-step Δ log p(`※`) predicts post-training source rate.
- Whether first-step Δ predicts the SUBSTRING-MATCH source rate from the legacy [#274](https://eps.superkaiba.com/tasks/274) / [#296](https://eps.superkaiba.com/tasks/296) panel, OR only the upcoming log-prob source-rate panel (testing whether substring-match noise is what was blocking the predictor).
- Whether first-step Δ correlates with any of the three failed geometric-distance predictors (cosine, JS, pairwise) — if YES, the distance predictors are tracking the right thing but were noise-limited; if NO, first-step Δ is measuring something orthogonal.
- Whether first-step Δ correlates with the base-model marker prior (predictor a) — partial-correlation analysis to disentangle "high prior → easy implant" from "high gradient → easy implant."

## What this does NOT test

- Whether first-step Δ predicts vulnerability under different recipes (different LRs, different loss masks, different framings).
- Asymptotic implant magnitude — only the initial gradient direction.
- Multi-token-marker vulnerability — by construction this requires single-token markers.

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. Load Qwen-2.5-7B-Instruct base model.
2. For each of the 48 personas in the [#274](https://eps.superkaiba.com/tasks/274) / [#296](https://eps.superkaiba.com/tasks/296) panel:
   a. Compute base-model log p(`※`) at end-of-answer position, averaged across 20 questions (the SAME 20 questions used in the [#296](https://eps.superkaiba.com/tasks/296) panel).
   b. Construct 50 persona-conditioned training rows (persona system prompt + question + answer ending in `※`).
   c. Take one LoRA-initialized optimizer step on those 50 rows (r=32, α=64, lr=1e-5, single batch of 50).
   d. Compute log p(`※`) again at the same 20-question end-of-answer positions.
   e. Record Δ_log_p per persona.
3. Correlate Δ_log_p against: (i) the legacy [#296](https://eps.superkaiba.com/tasks/296) substring-match source rates, (ii) the upcoming log-prob source rates from the panel re-run [task #1 in this batch], (iii) cosine-to-assistant, (iv) JS-to-baseline, (v) base-model marker prior, (vi) prompt token length.
4. Compute partial Spearman ρ to disentangle which signals are independent.

## Open questions for the planner

- Whether to seed the LoRA from the same init across all 48 personas (likely yes — controls for init noise).
- Whether one step is sufficient or whether 5 / 10 / 20-step deltas give a smoother signal (cost scales linearly).
- Whether to extend to per-bystander Δ — for each (source, bystander) pair, measure how much one step on the SOURCE moves log p(`※`) under the BYSTANDER prompt. That's a 48×48 matrix and an entirely different kind of vulnerability signal.
- Whether to compute the gradient direction in weight space and check whether 48 personas' gradients live in a low-dimensional subspace — would link vulnerability to "shared gradient direction" mechanism.
