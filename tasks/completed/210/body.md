---
title: 'Instruction-column dominance probe: imperative vs non-imperative variants'
kind: experiment
tags: []
created_at: '2026-05-03T04:45:18.000Z'
has_clean_result: false
sagan_id: 2ef5545d-33e5-47fb-98d8-e8803b41fa63
sagan_number: 210
priority: normal
legacy_why_unset: true
---
## Goal

Test whether the instruction-column dominance in #181's heatmap (51-80% marker rates across all training families) is driven by imperative surface form or is a deeper property of instruction-style prompts.

## Hypothesis

Replacing the 9 instruction-family panel prompts with semantically equivalent non-imperative variants will drop marker rates by >10pp if surface form drives the effect, or maintain rates if it is a deeper property.

## Setup

- 14 existing LoRA adapters from HF Hub (no new training)
- Same 36-prompt panel EXCEPT 9 instruction-family cells replaced with non-imperative rewrites
- Same scorer, vLLM params, 10 completions × 20 questions

## Compute

~0.5 GPU-h on 1x H100 (inference-only)

Parent: #181
Clean result target: #207 (supplementary section)
