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
goal: 'On Qwen-2.5-7B-Instruct, test whether future (answer-time) token activations
  can be predicted WITHOUT generating: fit one map per layer g_l: h_{l,t} -> h_{l,t+1}
  taking ONLY the previous position''s activation (no token identity), evaluate single-step
  held-out Delta R^2 per layer and multi-step autoregressive rollout from the last
  prompt token against true on-policy answer activations, and benchmark behavior read-out
  off rolled-forward states against #779''s direct context->trait predictor and context->answer-profile
  map.'
relates_to:
- spec-context-as-vector
---
# Predicting future token activations without generating: per-layer next-token activation maps rolled forward from the prompt

## Goal

On Qwen-2.5-7B-Instruct, test whether future (answer-time) token activations can be predicted WITHOUT generating: fit one map per layer g_l: h_{l,t} -> h_{l,t+1} taking ONLY the previous position's activation (no token identity), evaluate single-step held-out Delta R^2 per layer and multi-step autoregressive rollout from the last prompt token against true on-policy answer activations, and benchmark behavior read-out off rolled-forward states against #779's direct context->trait predictor and context->answer-profile map.

## Overview / Motivation

**The point is prediction without generation.** #779 established a single-shot map from the context representation to the mean answer-activation profile — a way to forecast answer-side state from the prompt alone. This task asks whether stepwise token-position dynamics do the same job: roll a per-layer next-token activation map forward from the last prompt token, position by position, WITHOUT generating any tokens, and see how well the rolled-forward states approximate the activations the model would actually have during its answer — and how well behavior expression can be read off them.

#841 is the depth-axis sibling (per-layer maps h_l -> h_{l+1} across layers at a fixed position); this is the token-position axis, with the same "one mapping per layer, compare function classes" structure.

## Key design considerations (planner refines)

- **No token-identity inputs, by construction.** The deployed map sees only the previous position's activation (optionally + position index). A token-informed arm (map also receives the true t+1 input embedding) is permitted ONLY as a diagnostic ceiling — how much of the next state is even predictable if the token were known — never as the headline arm; at forecast time there is no generated token to condition on.
- **Supervision comes from cached/true continuations at FIT time only.** Fit and eval targets are true per-position activations from the model's actual on-policy completions on real chat contexts (LMSYS/WildChat lineage, matching #779/#841); the no-generation constraint binds at INFERENCE/rollout time, not at training time.
- **Multi-step rollout is the load-bearing eval.** Single-step Delta R² alone can look good while autoregressive rollout diverges (error compounding). Report skill as a function of rollout horizon k from the last prompt token, against the true answer-position activations, with the copy-previous (frozen last-prompt-state) roll as the null — note the frozen null at horizon k IS approximately #779's context-side read, which makes the #779 benchmark natural.
- **Behavior read-out benchmark.** Project rolled-forward states (or their mean over the rolled horizon) onto the reused persona directions / read-out protocol from #779/#841 and compare trait-expression forecasting against #779's direct context->trait predictor g and the context->answer-profile map. This is the "predict if the model will exhibit a behavior without generating" use case.
- **Delta framing is mandatory for the single-step read:** residual streams are highly autocorrelated across positions, so raw state R² is trivially high; score against identity/copy-previous and affine baselines (same protocol shape as #841, seed 42).
- **Data/footprint:** per-position activations at all 28 layers over prompt+answer spans is a large capture (the #779/#841 cached tensors are last-prompt-token-only); size the capture and route per the §9 footprint rules; sub-sample contexts/positions as needed.
- **Positioning:** deep literature search required before the plan (new-direction rule) — token-position residual-stream dynamics, state-space/world-model views of transformer forward passes, activation-forecasting / early-exit / future-token-probing work (e.g. future lens-style probes predicting future-token content from current activations).

## Constraints

- No fine-tuning; generation only to produce the fit/eval supervision corpus (on-policy completions + activation capture), never at forecast time.
- Reuse the #841 fit/eval code shape and the #779 read-out rig wherever fit-for-purpose (artifact-reuse checklist applies).
