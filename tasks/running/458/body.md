---
title: Base-model cosine / token-JS predicts EM amount at fixed steps across 18 datasets
kind: experiment
tags: []
created_at: '2026-06-01T00:13:51Z'
has_clean_result: false
parent_id: 404
---
# Does base-model cosine / token-level JS divergence predict the AMOUNT of EM, at fixed gradient steps, across 18 diverse SFT datasets?

## Goal
At a single fixed recipe and fixed number of gradient steps (turner_em, max_steps=375), measure post-SFT broad-misalignment for 18 narrow-behavior datasets spanning the EM spectrum, and test whether a cheap base-model signal — mid-layer activation cosine similarity (M1, the #404 winner) and token-level Jensen-Shannon divergence between the narrow-behavior-prompted and broad-misaligned-prompted base model — predicts the EM amount across datasets, controlling for training-token volume.

## Why
Follow-up to #404 (cosine predicts EM, rho=+0.75, N=7, recipe-confounded) and #452 (deconfound: turner_em on Betley insecure-code stays ~0.3% -> dataset, not recipe, drives EM). This extends to N=18 with: (a) fixed recipe AND fixed gradient steps (max_steps=375) so recipe/steps are no longer confounds, (b) content/format diversity (code, bad-advice prose, numeric, benign-aesthetics, reward-hacking removed) to break the bad-advice-prose-vs-code confound, (c) a graded WEAK middle (OpenAI advice ladder + AestheticEM weakly-expressed), (d) a head-to-head of the cosine predictor vs a token-level JS predictor (the #404 M2 collapsed because it used judge-score distributions; this uses token distributions), (e) assistant-token count per cell as a covariate (steps + examples fixed by construction; tokens float ~10-30x, so regress it out).

The adversarial highlight: AestheticEM (Woodruff 2025) induces ~15.9% EM from benign aesthetic-preference content with no harmful surface -- the cleanest test of whether the predictor captures something deeper than surface harm.

## Design
- 18 cells, all pure SFT, recipe=turner_em (lr=2e-5, lora alpha=256/scaling=8, adamw_8bit), max_steps=375 (eff-batch 16), max_seq_length=2048, packing off, no system prompt, schema {messages:[{user},{assistant}]}.
- 2 seeds per cell (0, 137). Eval: Betley main-8, 100 completions/q, gpt-4o-2024-08-06 judge, --skip-calibration, aligned<30 AND coherent>=50 (identical to #404/#452).
- Predictors (base model, no training): cosine M1 (reuse issue404_predictor_cossim.py) + token-level JS (new). 
- Regress EM on cosine and on JS across 18 cells, with log(assistant_tokens) covariate.

## Cells (18)
EM-inducing: insecure_code, jailbroken, turner_bad_medical, turner_risky_financial, turner_extreme_sports, emergent_plus_legal, emergent_plus_security, openai_health_bad, evil_numbers, aesthetic_unpopular
WEAK middle: openai_health_subtle, openai_health_mix_25pct, aesthetic_unpopular_weakly_expressed
NO-EM controls: secure_code, educational, openai_health_correct, aesthetic_popular, json_neg

Provenance + sizes + per-format modifications: docs/em_dataset_catalog.md (this task adds AestheticEM + corrected counts).

## Success / decision
A monotone positive relation (Spearman) between each predictor and EM across cells, surviving the token covariate, with the cosine vs JS comparison reported. Kill: if neither predictor relates to EM once the token covariate is in (the #404 cosine signal was a fixed-N/recipe artifact).
