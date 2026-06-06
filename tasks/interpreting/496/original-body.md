---
title: Does base-model warm/empathetic ↔ sycophantic distance predict warmth→sycophancy
  leakage? (arXiv 2507.21919)
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:43Z'
has_clean_result: false
parent_id: 446
relates_to:
- beh-b-to-bprime
goal: Operationalize warmth/empathy (narrow B) and sycophancy (broad B') on Qwen-2.5-7B,
  train warmth into source personas with contrastive negatives, measure sycophancy
  leakage on held-out wrong-claim probes, and test whether base-model cosine/JS between
  warm and sycophantic prompts predicts the leakage.
---
## Goal

Operationalize warmth/empathy (narrow B) and sycophancy (broad B') on Qwen-2.5-7B, train warmth into source personas with contrastive negatives, measure sycophancy leakage on held-out wrong-claim probes, and test whether base-model cosine/JS between warm and sycophantic prompts predicts the leakage.


## Motivation

arXiv 2507.21919 reports that training language models to be warm/empathetic makes them more sycophantic. That is a concrete, externally-grounded narrow → broad (B → B′) leakage pair to test the predictor on. The project has generic sycophancy results — #411 implants sycophancy into 6 sources but bystander transfer ≈ 0; #470 finds JS/cosine fail to predict it; #480 finds a token-marker proxy fails — but the **warm/empathetic → sycophantic** pairing specifically has never been run.

## What exists to reuse

- #411's sycophancy contrastive rig: 6 sources × 23-bystander panel, ~700-row contrastive mix, Haiku judge (κ = 0.89). Swap the source behavior to warmth/empathy and the target to sycophancy.
- #470's JS-vs-cosine predictor machinery on frozen sycophancy leakage.
- The persona panel + judge prompts.

## Design sketch (for /adversarial-planner)

Operationalize warmth/empathy (narrow B) and sycophancy (broad B′) with clean judges. Train warmth into source personas using contrastive negatives; measure sycophancy on a held-out wrong-claim probe set. Compute base-model cosine/JS between the "warm" and "sycophantic" behavior prompts and regress against the measured leakage.

## Hypothesis

Warmth and sycophancy sit close in persona space, warmth-training induces sycophancy, and the base-model distance predicts the amount.

## Caveats

- #411 / #470 found near-zero bystander transfer for sycophancy and the predictor failed there — a null is plausible and informative.
- Measure leakage on-policy; use the contrastive held-out setup.

## Lineage / open questions

Advances **q:beh-b-to-bprime** (3.6), specific externally-cited pair. Parent #446 (realistic-behaviors scoping); rig from #411; predictor from #470. Source: arXiv 2507.21919.
