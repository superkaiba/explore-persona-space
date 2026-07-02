---
title: 'Layer-to-layer activation dynamics on real chat contexts: per-layer predictability
  atlas of the residual-stream update + does a dynamics-pretrained predictor beat
  #779''s direct context->trait predictor'
kind: experiment
tags: []
created_at: '2026-07-02T06:56:23Z'
has_clean_result: false
parent_id: 779
origin_prompt: '# Motivation - We''ve found that there is a map from context vector
  to answer profile - Potentially there could be a map from each activation to the
  next one (not sure if per position or a general map) - I want to see: how well can
  we actually predict this, and how well can we predict this for the purposes of predicting
  if a model will exhibit a specific behavior - Simply: take a bunch of contexts from
  LMSYS/WILDCHAT, train a linear map to predict next activation from previous activation,
  also try a small MLP, potentially position information in input/output, wait actually
  should we train a RNN/LSTM/SSM?, try at each layer and see what''s best - Also need
  a deep literature search to see what has been done before. Assess the novelty and
  likelihood that it will work. [after lit review + discussion:] okay I think it''s
  worth running, can we benchmark against 779?'
---
## Overview / Motivation

We have a map from context vector to answer profile (#779). This task tests the next structural object: a map from each activation to the next one — the residual-stream depth dynamics h_ℓ → h_{ℓ+1} — fit as a learned predictor (linear / MLP / sequence model over depth) on real chat contexts, then evaluated for whether the learned dynamics add trait-expression-forecasting value over #779's static predictors on the identical rig.

A deep literature search (four-slice sweep, 2026-07-01; full reports in `artifacts/lit-review-2026-07-01.md`) found the two halves separately covered but the combination open:

- Adjacent-layer transformations are known ~0.99 Procrustes-linear, but this is residual/identity-dominated and collapses once the residual is removed (arXiv 2405.12250). Trained cross-layer linear maps exist but target the final layer / logit space (2303.09435 Jump-to-Conclusions; 2303.08112 tuned lens; 2409.14091 N-NJTC, which benchmarks the identity-shortcut baseline). ReSAE (2605.27819) fits per-position affine h_ℓ→h_{ℓ+1} maps as an SAE pre-pass. Nobody reports a per-layer predictability atlas of the UPDATE Δ_ℓ with a function-class comparison on chat data.
- The depth-as-dynamical-system framing is active but analytical only (2502.12131, 2605.14258 per-layer Jacobian spectra; 2606.09287 trajectory geometry) — no learned surrogate.
- MLP/RNN/SSM surrogates over the depth axis: no prior instance (MOHAWK 2408.10189 matches hidden states in transformer→SSM distillation, but along the token axis). Looped-transformer scaling (2604.21106, recurrence-equivalence exponent φ≈0.46) predicts a single shared map recovers only partial capacity — expect per-layer maps to win, with a near-stationary middle band (2406.19384).
- Behavior forecasting from activations is a strong STATIC-probe field (refusal direction 2406.11717 near-ceiling; sleeper-agent defection probes AUROC>0.99; 2606.11172 / 2606.11445 / 2511.04527 forecast future behavior from single-snapshot reads). NO prior work shows a dynamics/next-activation model adds behavior-predictive value over a static probe.
- EAGLE (2401.15077) is the canonical next-activation predictor (token axis): one FC + one decoder layer; its ablations vary INPUT conditioning, not function class. EAGLE-3 (2503.01840) abandoned exact feature regression as a constraint — lesson: include an end-to-end behavior-readout arm, use multi-layer fusion, and train any rolled-forward map on its own rollouts.

Key framing constraint (data-processing inequality): a probe on f_ℓ(h_ℓ) cannot carry more information about the current model's behavior than a probe on h_ℓ, which is free from the same forward pass. The dynamics map's pre-registered win conditions are therefore NOT raw accuracy but (a) LABEL EFFICIENCY — the dynamics fit trains unsupervised on unlimited unlabeled chat contexts while trait labels cost rollouts + judge calls, so a dynamics-pretrained predictor may reach a given within-condition r with fewer labeled contexts than #779's direct predictor g trained from scratch; (b) OOD / held-out-behavior transfer, the documented static-probe failure mode (2411.03343); plus (c) the structure question Q1 answers regardless.

## Goal

On Qwen-2.5-7B-Instruct, (Q1) characterize the per-layer predictability of residual-stream depth dynamics on real chat contexts — fit per-layer maps f_ℓ: h_ℓ → h_{ℓ+1} at the last-prompt-token position (function classes: identity baseline, affine/ridge, MLP, and a sequence model over the depth axis) on the #779 LMSYS activation corpus and report held-out variance explained of the update Δ_ℓ = h_{ℓ+1} − h_ℓ per layer, relative to the identity and affine baselines — and (Q2) test whether dynamics-derived features or dynamics-pretraining add trait-expression-forecasting value over #779's matched-capacity direct context→trait predictor g on the identical monitoring rig (same traits, conditions, judge, within-condition Pearson r), with label-efficiency curves and held-out-behavior transfer as the pre-registered win conditions rather than absolute accuracy alone.

## Benchmark spec — versus #779 (parent)

Benchmark on #779's own rig and in-repo measured numbers, NOT Persona Vectors' published table (#779's rig-validation gate missed 4/6 trait-mode cells → LOW confidence on absolute anchoring; internally-consistent comparisons on the shared rig remain valid):

- **Numbers to beat (system mode, within-condition Pearson r, oracle-selected layers):** direct predictor g — sycophancy 0.91, hallucination 0.86 (CI-separated from the oracle in 4/6 cells; beat every method at point estimate in 6/6). Context rows: raw PV projection (evil 0.17 at oracle layer, 0.50 at its own layer 20; sycophancy 0.60; hallucination 0.56 system), learned map h (cleared +0.05 only in 2/6 many-shot cells).
- **Reuse (run the artifact-reuse (a)-(h) fitness check):** HF `superkaiba1/explore-persona-space-data/issue779_monitoring/` — `analysis_tensors/` (c_x and v(x) at all 28 layers, 5000 LMSYS contexts + eval-pass tensors + `pass_a/` judge scores), `r_b/` (3 traits × 28 layers), `raw_completions/` (39 cell files). Scripts: `scripts/issue779_common.py` (judge rubric, system prompts, templates), `scripts/issue779_collect.py`. Protocols inherited verbatim: LOCO direct-predictor fit, within-condition Pearson r with std<1 condition prune, bootstrap CI resampling conditions, Sonnet graded 0-100 N=5 drop-never-coerce judge.
- **Caveat inherited from #779:** oracle layer selection materially disadvantaged the raw projection there. Any per-layer argmax in THIS task's headline must use selection-symmetric nulls / held-out-split layer selection (`.claude/rules/selection-symmetric-nulls.md`).

## Proposed design sketch (planner refines)

- **Q1 (free analysis on cached tensors, no new GPU capture for the core):** the #779 tensors give the last-prompt-token depth trajectory h_1..h_28 for 5000 LMSYS contexts + eval contexts. Fit per transition ℓ=1..27: identity, ridge (closed-form, vectorized), small MLP (vectorized via `analysis/vectorized_mlp_skill.py` — many-cell fits are OVERHEAD-bound, vectorize before GPU), and one sequence model over depth (GRU or SSM consuming the trajectory + layer index). Held-out split over contexts. Metric: R²/cos of Δ_ℓ (the update), per layer; report the per-layer atlas + how much the sequence model closes the gap over per-layer maps. The pointwise-map residual at each layer IS the measurement of cross-position (attention) lossiness at this position — report it; a per-position extension over full prompts needs new capture and is a stretch arm only.
- **Q2 (judge-free reuse of #779 eval frame):** candidate dynamics-derived features: per-layer prediction residuals ‖Δ̂_ℓ − Δ_ℓ‖ (surprise), dynamics-map hidden state, local Jacobian projections onto r_B. Arms: (i) g baseline reproduced (ridge/MLP LOCO on c_x, #779 protocol verbatim); (ii) g + dynamics features (same capacity budget); (iii) dynamics-pretrained encoder → small trait head (EAGLE-3 lesson: also an end-to-end variant trained against the trait readout, not raw-state L2); (iv) multi-layer-fusion g (concat 3 layers → FC), the cheap EAGLE-3 upgrade to #779's single-layer g. Primary comparisons: label-efficiency curve (within-condition r vs N labeled contexts, N ∈ {50, 200, 1000, all}) and leave-one-behavior-out transfer; absolute within-condition r reported as the headline companion.
- **Explicit non-goals this task:** token-axis forecasting (EAGLE territory, sampling-uncertainty-dominated) and fine-tuning-outcome prediction from dynamics (natural follow-up toward the leakage-theory line, filed separately if Q1/Q2 land).

## Constraints / invariants

- Identity baseline h_{ℓ+1} = h_ℓ and per-layer affine are mandatory comparators; the target is Δ_ℓ, never raw-state R² (2405.12250 trap).
- DPI discipline: no claim that dynamics add information over static probes; claims are label efficiency, transfer, and structure.
- Train/eval context disjointness inherited from #779 (eval contexts never seen by any fit).
- Judge: `claude-sonnet-4-5-20250929` graded 0-100, N=5, Batch API, drop-never-coerce (only needed if new rollouts are generated; the core Q2 reuses #779's `pass_a/` scores).
- Compute: core Q1+Q2 is 0-GPU-h-class analysis over cached tensors (vectorized fits; route per compute-character carve-out — a many-epoch GD fit phase >15-30 min goes to a small GPU lane, vectorized first). New capture only for the optional per-position arm.
- No model fine-tuning anywhere in this task.

## Provenance

- Parent: #779 (benchmark + artifact source). Informed by #778 (PV replication), #658/#742/#761/#763 (base-side predictor line), #493 (predictor bake-off), #502 (mid-late-layer context→behavior signal).
- Verbatim originating prompt (2026-07-01 chat):

> # Motivation
> - We've found that there is a map from context vector to answer profile
> - Potentially there could be a map from each activation to the next one
>     - not sure if this should be per position or just a general map
> - I want to see:
>     - how well can we actually predict this
>     - how well can we predict this **for the purposes of predicting if a model will exhibit a specific behavior**
> - Simply:
>     - Take a bunch of contexts from LMSYS/WILDCHAT
>     - train a linear map to predict next activation from previous activation
>     - also try to train a small MLP
>     - potentially there should be some position information included in the input (and output)
>     - wait actually should we train a RNN/LSTM/SSM?
>     - Try at each layer and see what's best
> - Also need to do a deep literature search to see what has been done before
>
> Assess the novelty and likelihood that it will work

> [after lit review + discussion of the DPI framing, EAGLE/EAGLE-3 lessons, and #779's direct-predictor result:] okay I think it's worth running, can we benchmark against 779?
