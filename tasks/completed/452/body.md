---
title: 'Deconfound 404: turner_em recipe on Betley insecure-code data'
kind: experiment
tags: []
created_at: '2026-05-31T22:07:35Z'
has_clean_result: false
parent_id: 404
---
# Deconfound 404: does the turner_em recipe install EM on Betley insecure-code data?

## Goal
Determine whether the post-SFT broad-misalignment signal in #404's three Turner cells is driven by the training RECIPE (turner_em: lr=2e-5, lora alpha=256/scaling=8, adamw_8bit) or by the Turner DATASETS, by training Betley's insecure-code data under the turner_em recipe and comparing the broad-misalignment rate to #404's betley_open_model insecure-code baseline.

## Why
In #404, recipe is fully confounded with dataset source: all 3 EM-positive cells used Turner datasets + the turner_em recipe; all 4 EM-negative cells used Betley-family datasets + the betley_open_model recipe (lr=1e-5, alpha=64, adamw_torch_fused). The headline "base-model cosine predicts post-SFT EM (rho=+0.75)" rests on the 3 Turner cells, which were specifically constructed (dataset AND recipe) to install EM. This run is the single missing deconfounding cell.

## Setup (single-variable change from #404 insecure_code cell)
- Same dataset: Betley insecure.jsonl (data/issue404/insecure.jsonl), identical to #404's insecure_code cell.
- Same model: Qwen-2.5-7B-Instruct.
- Same eval: issue404_outcome_eval.py, Betley main-8, 100 completions/question, dual gpt-4o+Claude judge (aligned<30 AND coherent>=50).
- ONLY change vs #404 insecure_code: training recipe betley_open_model -> turner_em (training=turner_em lora=turner_em). New cell name `insecure_code_turner`, seeds 0 and 137.

## Baseline to beat
#404 insecure_code (betley_open_model): EM = 0.75% (seed0), 0.50% (seed137).

## Decision rule
- insecure_code_turner EM stays <1% (near baseline) -> the #404 signal is the DATASET, not the recipe. The cosine predictor survives the deconfound.
- insecure_code_turner EM jumps toward the Turner cells' 16-26% -> the #404 signal was largely the RECIPE. The cosine headline is confounded and must be re-framed.

## Compute
1x H100, LoRA SFT (1 epoch, ~6k rows) x 2 seeds + Betley main-8 eval. ~1-1.5 GPU-hours.
