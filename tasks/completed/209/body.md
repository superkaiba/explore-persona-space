---
title: Prompt-vs-content dissociation for non-persona triggers
kind: experiment
tags: []
created_at: '2026-05-03T04:45:09.000Z'
has_clean_result: false
sagan_id: 64eac0e1-602b-4d0b-9f70-cc7d4f2c80de
sagan_number: 209
priority: normal
legacy_why_unset: true
---
## Goal

Run a #173-style prompt-vs-content dissociation on the 14 existing #181 adapters to determine whether the broad free-generation leakage reflects weak prompt-gating or strong content-gating.

## Hypothesis

If prompt-gating is weak, condition A (source prompt + source answer) will be similar to condition D (other prompt + other answer) in the prefix-completion paradigm. If prompt-gating is real but masked by free-generation artifacts, A >> D.

## Setup

- 14 existing LoRA adapters from HF Hub (no new training)
- 4 conditions: A (matched prompt + source answer), B (other prompt + source answer), C (source prompt + other answer), D (fully mismatched)
- Prefix-completion: inject answer prefix stripped of [ZLT], let model continue ~30 tokens, check for [ZLT]
- N≥100 per cell per model

## Compute

~1 GPU-h on 1x H100 (inference-only)

Parent: #181
Clean result target: #207 (supplementary section)
