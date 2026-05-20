---
title: Recipe titration for non-persona trigger leakage
kind: experiment
tags: []
created_at: '2026-05-03T04:44:59.000Z'
has_clean_result: false
sagan_id: be27435e-29be-43a9-b785-123bd901fd99
sagan_number: 208
priority: normal
legacy_why_unset: true
---
## Goal

Determine whether the broad leakage in #181 is a recipe artifact or a structural property of non-persona triggers, by sweeping 4 recipe points on a single condition (T_task, seed=42).

## Hypothesis

An intermediate recipe (r=16/lr=1e-4/epochs=3 or r=32/lr=1e-5/epochs=5) will produce non-zero matched marker rates with tighter prompt-gating (matched/bystander ratio closer to 3x). If all intermediate recipes stay broadly leaky, the null is structural.

## Setup

- 4 recipe points: (r=16/lr=1e-5/epochs=3), (r=16/lr=1e-4/epochs=3), (r=32/lr=1e-5/epochs=5), (r=32/lr=1e-4/epochs=5)
- T_task condition only, seed=42 only
- Same 194-example QA pool, same 36-prompt eval panel
- Same scorer, vLLM params, EVAL_QUESTIONS

## Compute

~2 GPU-h on 1x H100

Parent: #181
Clean result target: #207 (supplementary section)
