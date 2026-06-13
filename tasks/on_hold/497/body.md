---
title: Does the cosine/JS leakage predictor's accuracy change with model size (Qwen-2.5
  1.5B→32B)?
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:46Z'
has_clean_result: false
parent_id: 413
relates_to:
- leak-predictor
- beh-b-to-bprime
goal: 'Run the #468 narrow→broad cosine/JS leakage-predictor recipe at ≥3 Qwen-2.5
  model sizes (e.g. 1.5B, 7B, 32B) holding the recipe fixed, and report whether the
  predictor''s correlation with measured leakage sharpens or degrades with scale.'
---
## Goal

Run the #468 narrow→broad cosine/JS leakage-predictor recipe at ≥3 Qwen-2.5 model sizes (e.g. 1.5B, 7B, 32B) holding the recipe fixed, and report whether the predictor's correlation with measured leakage sharpens or degrades with scale.


## Motivation

The entire predictor line (cosine/JS → leakage) is on Qwen-2.5-7B. #413 and #9 propose scaling load-bearing findings generally, but none tests whether the **predictor itself** sharpens or degrades with model size. If persona geometry cleans up with scale (the "assistant axis needs scale" observation), the predictor should get more accurate at larger sizes — and if it does not, that bounds how far the cheap-predictor program generalizes.

## What exists to reuse

- The predictor scripts are model-size-agnostic (#404 / #458 / #468).
- #413 / #9 enumerate the load-bearing battery to port. Truthification already has a 7B → 32B scale point (open-q 6.3).

## Design sketch (for /adversarial-planner)

Fix the cleanest predictor cell — the #468 narrow → emergent-misalignment recipe (in-context-example narrow persona, read at the newline-after-assistant token, L25). Run it at ≥ 3 Qwen-2.5 sizes (e.g. 1.5B, 7B, 32B): for each size, train the narrow-behavior datasets, measure the post-SFT broad rate, compute base-model cosine/JS, and report ρ vs size.

## Hypothesis

The cosine → leakage ρ trends upward with model size (geometry sharpens with scale).

## Caveats

- Heavy: 32B training is ft-70b-class pods. Consider running this as one explicit cell of #413 rather than standalone if compute is tight.
- Hold the recipe fixed across sizes — the single manipulated variable is model size.

## Lineage / open questions

Advances **q:leak-predictor** (3.1) / **q:beh-b-to-bprime** (3.6) along the scaling axis. Parent #413; predictor recipe #468; scaling siblings #9 / #436.
