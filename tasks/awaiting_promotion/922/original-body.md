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
goal: 'On Qwen-2.5-7B-Instruct, test whether the future (answer-time) activation trajectory
  can be predicted WITHOUT generating, comparing per-layer transition structures —
  (a) a global next-token map rolled from the last prompt token, (b1) an additively
  context-conditioned map (global operator + context drift), (b2) an operator-valued
  map where the context RESHAPES the transition itself (A(c) via FiLM-diagonal / low-rank
  delta / gated mixture of K operators, capacity-matched to b1), and (c) a direct
  per-horizon map c -> h_{t+k} (no recursion; #779''s answer-profile map = its horizon-mean
  nested baseline) — over the identity/ridge/MLP ladder with no token-identity inputs;
  evaluate single-step Delta R^2, rollout/horizon skill vs true on-policy answer activations,
  and behavior read-out vs #779''s direct context->trait predictor.'
relates_to:
- spec-context-as-vector
---
# Predicting the answer activation trajectory without generating: global, context-conditioned, and direct per-layer maps

## Goal

On Qwen-2.5-7B-Instruct, test whether the future (answer-time) activation trajectory can be predicted WITHOUT generating, comparing per-layer transition structures — (a) a global next-token map rolled from the last prompt token, (b1) an additively context-conditioned map (global operator + context drift), (b2) an operator-valued map where the context RESHAPES the transition itself (A(c) via FiLM-diagonal / low-rank delta / gated mixture of K operators, capacity-matched to b1), and (c) a direct per-horizon map c -> h_{t+k} (no recursion; #779's answer-profile map = its horizon-mean nested baseline) — over the identity/ridge/MLP ladder with no token-identity inputs; evaluate single-step Delta R^2, rollout/horizon skill vs true on-policy answer activations, and behavior read-out vs #779's direct context->trait predictor.

## Overview / Motivation

**The point is prediction without generation.** #779 established a single-shot map from the context representation to the mean answer-activation profile — a way to forecast answer-side state from the prompt alone. This task asks whether the full answer TRAJECTORY (per-position activations) can be forecast from the prompt alone, and which transition structure carries it. Three arms, forming a spectrum of where context-dependence enters:

- **(a) Global Markov transition, rolled:** one map per layer, context enters only via the initial state (the last-prompt-token activation). Tests whether a single universal token-dynamics law + the context-as-initial-condition suffices. Weakness: context information not retained in the rolled state can never re-enter, and rollout errors compound un-anchored.
- **(b) Context-conditioned transition, rolled:** the fixed context vector conditions the map at EVERY step. Tests whether the rolled state is a sufficient statistic for the dynamics; any (b) - (a) gap measures context information the stepwise state loses, and c acts as an anchor against compounding drift. TWO sub-arms with distinct hypotheses, capacity-matched within the linear family:
  - **(b1) ADDITIVE conditioning:** h_{t+1} = A h_t + B c (+ bias) — one global operator A for all contexts; the context contributes a per-context drift. The MLP analogue is concat([h_t, c]) input.
  - **(b2) OPERATOR-VALUED / multiplicative conditioning — the context RESHAPES the map itself:** h_{t+1} = A(c) h_t, the transition operator a function of the context (hypernetwork / bilinear / selective-SSM family). A full d x d A(c) is intractable — use structured forms, e.g. FiLM-diagonal scaling, a low-rank delta A(c) = A + U(c) V(c)^T, or a mixture of K base operators with context-dependent gating ("the context selects the dynamical regime"); the planner picks 1-2 forms and matches parameter counts to (b1). Mechanistic motivation: in the real transformer the token->token transition runs through attention whose weights depend on the prefix, so the true transition operator IS context-dependent; (b2) is the faithful reduced model, (a)/(b1) are its fixed-operator simplifications. NOTE: the concat-MLP implicitly reshapes its Jacobian with c, so it cannot separate b1-vs-b2 — the crisp additive-vs-multiplicative read is the capacity-matched LINEAR pair.
- **(c) Direct context -> trajectory:** per-horizon maps c -> h_{t+k} (k = 1..K), no recursion, no error compounding — the trajectory-resolved generalization of #779, whose mean-answer-profile map is exactly this arm's output averaged over the horizon (so #779 is a nested baseline). Tests whether "dynamics" adds anything over static per-horizon regression: if (c) matches or beats (a)/(b), stepwise dynamics structure buys nothing for forecasting.

#841 is the depth-axis sibling (per-layer maps h_l -> h_{l+1} across layers at a fixed position); this is the token-position axis, with the same "one mapping per layer, compare function classes" structure.

## Key design considerations (planner refines)

- **No token-identity inputs, by construction (all three arms).** Deployed maps see only activations: the previous position's rolled state (a), plus the fixed CONTEXT vector (b, c) — the context vector is available without generation (it is computed from the prompt), so conditioning on it does NOT violate the no-generation constraint. A token-informed arm (map also receives the true t+1 input embedding) is permitted ONLY as a diagnostic ceiling — how much of the next state is even predictable if the token were known — never as the headline arm; at forecast time there is no generated token to condition on.
- **Context-vector definition:** default to the parent line's context representation (the last-prompt-token residual activation at the map's layer, per #779/#841); the planner may add the #779 alternatives (mean-over-prompt) as a secondary sweep only if cheap.
- **Supervision comes from cached/true continuations at FIT time only.** Fit and eval targets are true per-position activations from the model's actual on-policy completions on real chat contexts (LMSYS/WildChat lineage, matching #779/#841); the no-generation constraint binds at INFERENCE/rollout time, not at training time.
- **Multi-step rollout is the load-bearing eval.** Single-step Delta R² alone can look good while autoregressive rollout diverges (error compounding). Report skill as a function of rollout horizon k from the last prompt token, against the true answer-position activations, with the copy-previous (frozen last-prompt-state) roll as the null — note the frozen null at horizon k IS approximately #779's context-side read, which makes the #779 benchmark natural.
- **Behavior read-out benchmark.** Project rolled-forward states (or their mean over the rolled horizon) onto the reused persona directions / read-out protocol from #779/#841 and compare trait-expression forecasting against #779's direct context->trait predictor g and the context->answer-profile map. This is the "predict if the model will exhibit a behavior without generating" use case.
- **Delta framing is mandatory for the single-step read:** residual streams are highly autocorrelated across positions, so raw state R² is trivially high; score against identity/copy-previous and affine baselines (same protocol shape as #841, seed 42).
- **Data/footprint:** per-position activations at all 28 layers over prompt+answer spans is a large capture (the #779/#841 cached tensors are last-prompt-token-only); size the capture and route per the §9 footprint rules; sub-sample contexts/positions as needed.
- **Positioning:** deep literature search required before the plan (new-direction rule) — token-position residual-stream dynamics, state-space/world-model views of transformer forward passes, activation-forecasting / early-exit / future-token-probing work (e.g. future lens-style probes predicting future-token content from current activations).

## Constraints

- No fine-tuning; generation only to produce the fit/eval supervision corpus (on-policy completions + activation capture), never at forecast time.
- Reuse the #841 fit/eval code shape and the #779 read-out rig wherever fit-for-purpose (artifact-reuse checklist applies).
