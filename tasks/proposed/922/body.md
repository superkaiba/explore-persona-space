---
title: 'Next-token activation prediction: per-layer maps h_{l,t} -> h_{l,t+1} (identity/ridge/MLP)
  on real chat contexts'
kind: experiment
tags: []
created_at: '2026-07-03T10:21:41Z'
has_clean_result: false
parent_id: 841
origin_prompt: Run an issue in the background which is 841 but predicting the next
  token activation from previous token activation, one mapping per layer, trying the
  different options
goal: 'On Qwen-2.5-7B-Instruct, characterize the per-layer predictability of residual-stream
  token-position dynamics on real chat contexts — fit one map per layer g_l: h_{l,t}
  -> h_{l,t+1} (identity/copy-previous null, affine ridge, small MLP, optional sequence
  model as exploratory) and report held-out variance explained of the update Delta_t
  = h_{l,t+1} - h_{l,t} per layer relative to the identity and affine baselines.'
relates_to:
- spec-context-as-vector
---
# Next-token activation prediction: per-layer maps from previous-token activation

## Goal

On Qwen-2.5-7B-Instruct, characterize the per-layer predictability of residual-stream token-position dynamics on real chat contexts — fit one map per layer g_l: h_{l,t} -> h_{l,t+1} (identity/copy-previous null, affine ridge, small MLP, optional sequence model as exploratory) and report held-out variance explained of the update Delta_t = h_{l,t+1} - h_{l,t} per layer relative to the identity and affine baselines.

## Overview / Motivation

#841 characterized DEPTH dynamics: per-layer maps h_l -> h_{l+1} across layers at a fixed (last-prompt-token) position. This task asks the orthogonal question on the TOKEN-POSITION axis: within each layer, how much of the next token position's residual-stream state is predictable from the previous position's state alone?

This is the token-position sibling of #841, on the same model, with the same function-class comparison shape (identity null / affine / MLP / optional sequence model) and the same "one mapping per layer" structure.

## Key design considerations (planner refines)

- **The next-token-embedding confound is part of the question.** h_{l,t+1} partially reflects the identity of the input token at position t+1, which h_{l,t} cannot know. Predictability from the previous position alone therefore measures how much of the next state is carried by the running context state vs injected by the new token. The planner should consider a token-informed control arm (e.g., the map also receives the position-t+1 input embedding) to separate the two contributions, and must include the copy-previous identity null (h_{t+1} = h_t) since residual streams are highly autocorrelated across positions — raw R² of the state without the Delta framing is trivially high.
- **Data:** real chat contexts (LMSYS/WildChat lineage, matching #779/#841). NOTE: the #779/#841 cached tensors (`analysis_tensors/pass_b/train_context_vectors.pt`) are last-prompt-token-position ONLY — this task needs per-position activations at all 28 layers, so a fresh capture is likely required. Footprint must be sized at plan time (per-position × 28 layers × d_model is large; sub-sample positions/contexts as needed and route per the §9 footprint rules).
- **Protocol:** one map per layer; held-out split by CONTEXT (never by position within a context); shuffle null; same split/seed discipline as #841 (seed 42).
- **Positioning:** benchmark framing analogous to #841 where applicable; a deep literature search on token-position residual-stream dynamics / state-space views of transformer forward passes is required before the plan (new-direction rule).

## Constraints

- Analysis-only after capture: no fine-tuning, no new judging required for the core Q.
- Reuse the #841 fit/eval code shape wherever fit-for-purpose (artifact-reuse checklist applies).
