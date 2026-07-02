---
title: 'Layer-to-layer activation dynamics on real chat contexts: per-layer predictability
  atlas of the residual-stream update + does a dynamics-pretrained predictor beat
  #779''s direct context->trait predictor'
kind: experiment
tags:
- activation-dynamics
- context-geometry
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
goal: 'On Qwen-2.5-7B-Instruct, (Q1) characterize the per-layer predictability of
  residual-stream depth dynamics on real chat contexts — fit per-layer maps f_l: h_l
  -> h_{l+1} at the last-prompt-token position (identity baseline, affine/ridge, MLP,
  and a sequence model over the depth axis) on the #779 LMSYS activation corpus and
  report held-out variance explained of the update Delta_l = h_{l+1} - h_l per layer
  relative to the identity and affine baselines — and (Q2) test whether dynamics-derived
  features or dynamics-pretraining add trait-expression-forecasting value over #779''s
  matched-capacity direct context->trait predictor g on the identical monitoring rig
  (same traits, conditions, judge, within-condition Pearson r), with label-efficiency
  curves and held-out-behavior transfer as the pre-registered win conditions rather
  than absolute accuracy alone.'
---
## Overview / Motivation

We have a map from context vector to answer profile (#779). This task tests the next structural object: a map from each activation to the next one — the residual-stream depth dynamics h_ℓ → h_{ℓ+1} — fit as a learned predictor (linear / MLP / sequence model over depth) on real chat contexts, then evaluated for whether the learned dynamics add trait-expression-forecasting value over #779's static predictors on the identical rig.

Two literature passes ground this task (full reports in `artifacts/lit-review-2026-07-01.md` and `artifacts/lit-review-2026-07-02-deep-dive.md`):

**Broad sweep (2026-07-01):** the two halves are separately covered but the combination is open. Adjacent-layer transformations are ~0.99 Procrustes-linear, but this is residual/identity-dominated and collapses once the residual is removed (2405.12250). Trained cross-layer maps target the final layer / logit space (2303.09435; 2303.08112 tuned lens; 2409.14091 N-NJTC with the explicit identity-shortcut baseline); ReSAE (2605.27819) fits per-position affine h_ℓ→h_{ℓ+1} maps as an SAE pre-pass. The depth-as-dynamical-system framing is analytical only (2502.12131; 2605.14258 Jacobian spectra; 2606.09287). MLP/RNN/SSM surrogates over depth have no prior instance; looped-transformer scaling (2604.21106, φ≈0.46) predicts per-layer maps beat one shared map, with a near-stationary middle band (2406.19384). Behavior forecasting from activations is a strong STATIC-probe field (2406.11717 refusal direction near-ceiling; sleeper-agent probes AUROC>0.99; 2606.11172 / 2606.11445 / 2511.04527 single-snapshot future-behavior probes); no prior work shows a dynamics model adds behavior-predictive value over a static probe.

**Deep dive (2026-07-02, full-paper reads):** sharpens the whitespace and the design.
- Closest residual-as-feature precedents: 2606.05346 uses the **residual of a rolling linear trajectory extrapolation over hidden states as a behavioral feature** (near-orthogonal to surprisal; displacement ‖Δ‖ and trajectory-deviation DISSOCIATE, opposite signs); 2605.05134 fits **Koopman operators per regime (factual vs hallucinated) and classifies by the one-step prediction-error GAP**. Neither learns the map nor runs on the internal per-layer residual stream.
- Closest system: **NextLat (2511.05963)** — a plain-MLP latent dynamics model predicting the next FINAL-layer state given the next token, trained as an auxiliary loss (SmoothL1 rollout + KL-in-decode-space) that reshapes the base model; token-axis, not depth-axis, and not post-hoc on a frozen model. #841 must position against it explicitly.
- The sharpest design tension: **EAGLE-3 (2503.01840) abandoned raw feature regression deliberately** — the feature loss made EAGLE's data-scaling curve FLAT, and dropping it alone lifted acceptance length τ 4.05→5.37 (+33%); **ReSAE independently shows raw variance-explained and functional (CE) recovery move in opposite directions.** Raw-Δ fit quality must never be the sole headline; every strong activation-regression method pairs SmoothL1 with a decode/behavior-space term (EAGLE +0.1·CE, HASS +Top-K, NextLat +KL, MOHAWK +logit-KD).
- Structure priors: the Δ-Jacobian is ~98% rotational (complex-eigenvalue) at every depth with three depth regimes (2605.14258) → ridge misses it, MLP/sequence should win early/mid, and the linear→MLP gap per layer is a nonlinearity readout. ‖Δ_ℓ‖/‖h_ℓ‖ declines 0.963→0.518 with depth and residual norms grow ~exponentially (≈1.045×/layer GPT2-XL; LN-placement-dependent per 2502.02732) → per-layer normalization mandatory, measured on Qwen directly.
- Compounding error in rolled-out activation predictors is FRONT-LOADED (EAGLE n-α: the first off-manifold step dominates); the validated fix is detached ~3-step self-rollout training (HASS: fixed-data self-rollout τ 5.15 > on-policy teacher-forced τ 4.94).
- Why predictable at all: future-relevant content is mostly a byproduct of local next-token computation at small scale ("breadcrumbs", 2404.00859) and belief states are LINEARLY represented in the residual stream (2405.15943) — but pre-caching grows with scale (a 7B caveat), and future-predictive content peaks at MID depth (2311.04897) and is often distributed across layers → multi-layer input arm.
- Positioning gap claimable: no paper frames the transformer layer update as an explicit learned ℓ→ℓ+1 prediction + residual (no predictive-coding-of-transformers instance; no Dreamer-style latent dynamics on the residual stream).

Key framing constraint (data-processing inequality): a probe on f_ℓ(h_ℓ) cannot carry more information about the current model's behavior than a probe on h_ℓ, which is free from the same forward pass. The dynamics map's pre-registered win conditions are therefore NOT raw accuracy but (a) LABEL EFFICIENCY — the dynamics fit trains unsupervised on unlimited unlabeled chat contexts while trait labels cost rollouts + judge calls; (b) OOD / held-out-behavior transfer, the documented static-probe failure mode (2411.03343); plus (c) the structure question Q1 answers regardless.

## Goal

On Qwen-2.5-7B-Instruct, (Q1) characterize the per-layer predictability of residual-stream depth dynamics on real chat contexts — fit per-layer maps f_ℓ: h_ℓ → h_{ℓ+1} at the last-prompt-token position (identity baseline, affine/ridge, MLP, and a sequence model over the depth axis) on the #779 LMSYS activation corpus and report held-out variance explained of the update Δ_ℓ = h_{ℓ+1} − h_ℓ per layer, relative to the identity and affine baselines — and (Q2) test whether dynamics-derived features or dynamics-pretraining add trait-expression-forecasting value over #779's matched-capacity direct context→trait predictor g on the identical monitoring rig (same traits, conditions, judge, within-condition Pearson r), with label-efficiency curves and held-out-behavior transfer as the pre-registered win conditions rather than absolute accuracy alone.

## Benchmark spec — versus #779 (parent)

Benchmark on #779's own rig and in-repo measured numbers, NOT Persona Vectors' published table (#779's rig-validation gate missed 4/6 trait-mode cells → LOW confidence on absolute anchoring; internally-consistent comparisons on the shared rig remain valid):

- **Numbers to beat (system mode, within-condition Pearson r, oracle-selected layers):** direct predictor g — sycophancy 0.91, hallucination 0.86 (CI-separated from the oracle in 4/6 cells; beat every method at point estimate in 6/6). Context rows: raw PV projection (evil 0.17 at oracle layer, 0.50 at its own layer 20; sycophancy 0.60; hallucination 0.56 system), learned map h (cleared +0.05 only in 2/6 many-shot cells).
- **Reuse (run the artifact-reuse (a)-(h) fitness check):** HF `superkaiba1/explore-persona-space-data/issue779_monitoring/` — `analysis_tensors/` (c_x and v(x) at all 28 layers, 5000 LMSYS contexts + eval-pass tensors + `pass_a/` judge scores), `r_b/` (3 traits × 28 layers), `raw_completions/` (39 cell files). Scripts: `scripts/issue779_common.py` (judge rubric, system prompts, templates), `scripts/issue779_collect.py`. Protocols inherited verbatim: LOCO direct-predictor fit, within-condition Pearson r with std<1 condition prune, bootstrap CI resampling conditions, Sonnet graded 0-100 N=5 drop-never-coerce judge.
- **Caveat inherited from #779:** oracle layer selection materially disadvantaged the raw projection there. Any per-layer argmax in THIS task's headline must use selection-symmetric nulls / held-out-split layer selection (`.claude/rules/selection-symmetric-nulls.md`).

## Proposed design sketch (planner refines; deep-dive imports marked ▸)

- **Q1 (free analysis on cached tensors, no new GPU capture for the core):** the #779 tensors give the last-prompt-token depth trajectory h_1..h_28 for 5000 LMSYS contexts + eval contexts. Fit per transition ℓ=1..27: identity, ridge (closed-form, vectorized), small MLP (vectorized via `analysis/vectorized_mlp_skill.py` — many-cell fits are OVERHEAD-bound, vectorize before GPU), and one sequence model over depth (GRU or SSM consuming the trajectory + layer index). Held-out split over contexts.
  - ▸ Targets: Δ_ℓ raw AND per-block RMS-normalized (ReSAE S_m; ‖Δ‖ shrinks relative to ‖h‖ with depth, so unnormalized Δ-R² is early-layer-dominated); report both. Identity null in Δ-space = predict-zero, stated explicitly.
  - ▸ Metrics: coordinate-averaged R² on Δ (the JTC/N-NJTC/ReSAE convention) + generalized-Procrustes linearity + adjacent-layer cosine; median AND tail percentiles of per-context Δ-error (heavy tails, 2405.12250's feature-triggering regime).
  - ▸ Loss: SmoothL1 (the EAGLE/HASS/NextLat standard) with ridge closed-form as the affine baseline (bias term + held-out calibration split per ReSAE).
  - ▸ Controls: base-vs-Instruct Qwen comparison (is Δ-predictability learned? — the OLMo-step-0 idea from 2605.14258); last-token-vs-pooled-positions validation before comparing numbers to the pooled-position literature; measure Qwen's own ‖Δ_ℓ‖/‖h_ℓ‖ and norm-growth curve first (LN-placement-dependent, 2502.02732).
  - ▸ Expected shape (priors to confirm/refute): learned-map margin over identity concentrates early/mid, identity-dominated late (N-NJTC, Jacobian regimes); MLP > ridge where the rotational Δ-structure lives; no symmetric/PSD-only maps or SVD-only Δ summaries.
  - The pointwise-map residual at each layer IS the measurement of cross-position (attention) lossiness at this position — report it; a per-position extension over full prompts needs new capture and is a stretch arm only.
- **Q2 (judge-free reuse of #779 eval frame):**
  - ▸ Features: BOTH displacement ‖Δ_ℓ‖ and trajectory-deviation/prediction-residual ‖Δ̂_ℓ − Δ_ℓ‖ per layer (they dissociate with opposite signs, 2606.05346), dynamics-map hidden state, local Jacobian projections onto r_B.
  - ▸ Regime-conditional arm (Koopman template, 2605.05134): fit dynamics maps separately on trait-eliciting vs neutral contexts; the per-context residual GAP between the two maps is the forecasting feature.
  - Arms: (i) g baseline reproduced (ridge/MLP LOCO on c_x, #779 protocol verbatim); (ii) g + dynamics features (same capacity budget); (iii) dynamics-pretrained encoder → small trait head, plus ▸ an end-to-end variant trained against the trait readout with a behavior-space auxiliary term (SmoothL1 + readout loss — the EAGLE-3/ReSAE lesson: raw-vector regression alone under-weights behaviorally-relevant directions and blocks data scaling); (iv) multi-layer-fusion g (concat low/mid/high layers → FC — EAGLE-3 fusion; future-predictive content peaks mid-depth per Future Lens and is distributed per belief-geometry).
  - ▸ If any arm rolls the map forward multi-step: detached ~3-step self-rollout training (HASS) + a divergence-horizon metric; input-noise U(−0.1,0.1) as the cheap alternative.
  - Primary comparisons: label-efficiency curve (within-condition r vs N labeled contexts, N ∈ {50, 200, 1000, all}) and leave-one-behavior-out transfer; absolute within-condition r reported as the headline companion.
- **Explicit non-goals this task:** token-axis forecasting (EAGLE territory, sampling-uncertainty-dominated) and fine-tuning-outcome prediction from dynamics (natural follow-up toward the leakage-theory line, filed separately if Q1/Q2 land).

## Constraints / invariants

- Identity baseline h_{ℓ+1} = h_ℓ (= predict-zero in Δ-space) and per-layer affine are mandatory comparators; the target is Δ_ℓ, never raw-state R² (2405.12250: raw is 0.99-linear tautologically; Instruct models even more so).
- ▸ Δ-R² alone is never the headline: validate any fit-quality claim against a functional/behavior readout (ReSAE's EV↔CE divergence; EAGLE-3's flat-scaling case against raw-state regression).
- DPI discipline: no claim that dynamics add information over static probes; claims are label efficiency, transfer, and structure.
- Train/eval context disjointness inherited from #779 (eval contexts never seen by any fit).
- Judge: `claude-sonnet-4-5-20250929` graded 0-100, N=5, Batch API, drop-never-coerce (only needed if new rollouts are generated; the core Q2 reuses #779's `pass_a/` scores).
- Compute: core Q1+Q2 is 0-GPU-h-class analysis over cached tensors (vectorized fits; route per compute-character carve-out — a many-epoch GD fit phase >15-30 min goes to a small GPU lane, vectorized first). New capture only for the optional per-position arm and the base-model control (one activation-caching pass on base Qwen-2.5-7B over the same contexts).
- No model fine-tuning anywhere in this task.

## Provenance

- Parent: #779 (benchmark + artifact source). Informed by #778 (PV replication), #658/#742/#761/#763 (base-side predictor line), #493 (predictor bake-off), #502 (mid-late-layer context→behavior signal).
- Literature artifacts: `artifacts/lit-review-2026-07-01.md` (four-slice broad sweep), `artifacts/lit-review-2026-07-02-deep-dive.md` (full-paper deep dive: depth-axis maps, token-axis feature forecasting, prediction-error-as-feature + norm facts).
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

> [2026-07-02:] Do a deep literature dive on predicting next activation from current activation and fold in your findings
