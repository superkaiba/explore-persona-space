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
goal: 'On Qwen-2.5-7B-Instruct, characterize how well the residual-stream next-activation
  map can be learned and how much trait-relevant signal predicted activations retain:
  fit per-layer maps f_l: h_l -> h_{l+1} (identity / ridge / MLP / depth-sequence
  classes, target Delta_l, SmoothL1) at the last-prompt-token position on the #779
  LMSYS corpus, then on #779''s exact monitoring rig project rolled-forward predicted
  activations onto the persona directions at the trait read-out layers and benchmark
  the resulting trait monitor (within-condition Pearson r, #779 protocol verbatim)
  against #779''s measured rows — raw projection at the source layer (the matched-information
  comparison), raw projection at the target layer (the transport ceiling), a direct-hop
  ridge map, and the learned map h / direct predictor g reference rows — reporting
  the per-layer Delta-predictability atlas, the trait-signal retention curve vs prediction
  horizon, and the Delta-R2-vs-retention divergence per (layer, function class).'
relates_to:
- spec-context-as-vector
---
## Overview / Motivation

We have a map from context vector to answer profile (#779). This task tests the next structural object: the residual-stream next-activation map h_ℓ → h_{ℓ+1}, fit as a learned predictor on real chat contexts, and evaluated by how much trait-relevant signal the predicted activations retain on #779's monitoring rig. Scope narrowed 2026-07-02 (user): next-activation prediction + the #779 benchmark ONLY — the label-efficiency / dynamics-pretraining / residual-feature apparatus is deferred (see § Deferred extensions).

Two literature passes ground this task (full reports in `artifacts/lit-review-2026-07-01.md` and `artifacts/lit-review-2026-07-02-deep-dive.md`):

**Broad sweep (2026-07-01):** the two halves are separately covered but the combination is open. Adjacent-layer transformations are ~0.99 Procrustes-linear, but this is residual/identity-dominated and collapses once the residual is removed (2405.12250). Trained cross-layer maps target the final layer / logit space (2303.09435; 2303.08112 tuned lens; 2409.14091 N-NJTC with the explicit identity-shortcut baseline); ReSAE (2605.27819) fits per-position affine h_ℓ→h_{ℓ+1} maps as an SAE pre-pass. The depth-as-dynamical-system framing is analytical only (2502.12131; 2605.14258 Jacobian spectra; 2606.09287). MLP/RNN/SSM surrogates over depth have no prior instance; looped-transformer scaling (2604.21106, φ≈0.46) predicts per-layer maps beat one shared map, with a near-stationary middle band (2406.19384). Behavior forecasting from activations is a strong STATIC-probe field; no prior work evaluates predicted activations as a behavior-monitoring surface.

**Deep dive (2026-07-02, full-paper reads):**
- Closest residual-as-feature precedents: 2606.05346 (residual of a rolling linear trajectory extrapolation as a behavioral feature; displacement and trajectory-deviation DISSOCIATE with opposite signs) and 2605.05134 (Koopman operators per regime; prediction-error gap classifies hallucination). Neither learns the map nor runs on the internal per-layer residual stream.
- Closest system: NextLat (2511.05963) — MLP latent-dynamics model predicting the next FINAL-layer state given the next token, as an auxiliary training loss reshaping the base model; token-axis, not post-hoc on a frozen model. Position against it explicitly.
- The sharpest design lesson: EAGLE-3 (2503.01840) abandoned raw feature regression deliberately (feature loss ⇒ FLAT data-scaling; dropping it alone lifted τ 4.05→5.37), and ReSAE shows raw variance-explained and functional (CE) recovery move in OPPOSITE directions ⇒ Δ-fit quality is never the sole headline; the behavior-space evaluation (Stage 1) is the function-class comparison that counts.
- Structure priors: the Δ-Jacobian is ~98% rotational at every depth with three depth regimes (2605.14258) → ridge should miss structure an MLP captures, most early/mid; identity dominates late. ‖Δ_ℓ‖/‖h_ℓ‖ declines 0.963→0.518 with depth; residual norms grow ~exponentially and the curve is LN-placement-dependent (2502.02732) → per-layer normalization, measured on Qwen directly.
- Compounding in rolled-out activation predictors is FRONT-LOADED (EAGLE n-α: the first off-manifold step dominates); validated fix = detached ~3-step self-rollout training (HASS: fixed-data self-rollout τ 5.15 > on-policy teacher-forced τ 4.94); cheap alternative = input-noise U(−0.1,0.1).
- Why predictable at all: future-relevant content is mostly a byproduct of local computation at small scale (breadcrumbs, 2404.00859); belief states are linearly represented in the residual stream (2405.15943); future-predictive content peaks MID-depth (2311.04897).
- Positioning gap claimable: no paper frames the transformer layer update as an explicit learned ℓ→ℓ+1 prediction + residual, and no one has evaluated predicted activations as a trait-monitoring surface.

Framing (data-processing inequality): a projection of the PREDICTED target-layer state carries at most the information in the source-layer state, so the transport read is bounded by the actual-target-layer projection (the ceiling row). The honest winnable comparison is at MATCHED source information: does transporting layer-ℓ information through learned dynamics and reading it at the trait layer beat reading layer ℓ directly? This is the exact analogue of #779's R1 (which asked it for the answer-profile map and got "no"); either answer is informative, and the Δ-R²-vs-retention divergence is a finding in itself.

## Goal

On Qwen-2.5-7B-Instruct, characterize how well the residual-stream next-activation map can be learned and how much trait-relevant signal predicted activations retain: fit per-layer maps f_ℓ: h_ℓ → h_{ℓ+1} (identity / ridge / MLP / depth-sequence classes, target Δ_ℓ, SmoothL1) at the last-prompt-token position on the #779 LMSYS corpus, then on #779's exact monitoring rig project rolled-forward predicted activations onto the persona directions at the trait read-out layers and benchmark the resulting trait monitor (within-condition Pearson r, #779 protocol verbatim) against #779's measured rows — raw projection at the source layer (the matched-information comparison), raw projection at the target layer (the transport ceiling), a direct-hop ridge map, and the learned map h / direct predictor g reference rows — reporting the per-layer Δ-predictability atlas, the trait-signal retention curve vs prediction horizon, and the Δ-R²-vs-retention divergence per (layer, function class).

## Design — two stages, both on #779's cached tensors

**Stage 0 — learn the next-activation maps (unsupervised).** On the 5000 LMSYS contexts' last-prompt-token depth trajectories h_1..h_28 (cached at all 28 layers in #779's `analysis_tensors/`), fit per transition ℓ=1..27:
- Function classes: **identity** (Δ̂=0 null, stated explicitly), **ridge** (closed-form affine, bias term, held-out calibration split per ReSAE), **small MLP** (2-layer, vectorized via `analysis/vectorized_mlp_skill.py` — many-cell fits are OVERHEAD-bound, vectorize before GPU), **depth-sequence model** (GRU or SSM consuming the trajectory + layer index).
- Loss: SmoothL1 on Δ_ℓ (the EAGLE/HASS/NextLat standard). Held-out context split (4500/500), eval contexts NEVER used in any fit.
- Atlas metrics: coordinate-averaged held-out R² on Δ_ℓ, raw AND per-block RMS-normalized (‖Δ‖ shrinks vs ‖h‖ with depth; unnormalized Δ-R² is early-layer-dominated); median AND tail percentiles of per-context Δ-error (heavy tails); adjacent-layer cosine + Procrustes linearity as complementary reads. Measure Qwen's own ‖Δ_ℓ‖/‖h_ℓ‖ / norm-growth curve first.
- Multi-hop variants for Stage 1: iterated one-step composition f_{ℓ*−1}∘…∘f_ℓ AND direct-hop ridge ℓ→ℓ* (JTC-style) per (source ℓ, target ℓ*). If a rolled-forward class is trained multi-step, use detached ~3-step self-rollout (HASS) and report a divergence-horizon metric.

**Stage 1 — behavior-space benchmark on the #779 rig.** Same single-variable-change structure as #779's R1: the manipulated variable is the PROJECTION INPUT; directions r_B, eval frame, judge scores, metric, and prune/bootstrap protocol are #779-verbatim (reused from `analysis_tensors/pass_a/` + `r_b/` — zero new rollouts, zero judge calls). Per trait, target layer ℓ* fixed A PRIORI from #779 (evil 20 — its own best layer, not the oracle-shared 14; sycophancy 26; hallucination 17); source layers ℓ on a grid below ℓ*. Rows per trait × mode:
1. **Dynamics-transported projection** ⟨ĥ_{ℓ*}(x), r_B(ℓ*)⟩, ĥ from iterated one-step maps, per function class.
2. **Raw projection at the source layer** ⟨h_ℓ, r_B(ℓ)⟩ — the matched-information fair fight.
3. **Raw projection at the target layer** ⟨h_{ℓ*}, r_B(ℓ*)⟩ — the transport ceiling (DPI bound).
4. **Direct-hop ridge** ℓ→ℓ* — is composing one-step maps worse than one long hop?
5. **#779 reference rows** — learned map h and direct predictor g (context: g is the practical upper reference at sycophancy 0.91 / hallucination 0.86 system-mode; the transport read is not expected to beat g).

Headline deliverables: (a) the **trait-signal retention curve** — fraction of the ceiling row's r retained vs prediction horizon k = ℓ* − ℓ, per class (expect front-loaded loss at hop 1 per EAGLE n-α); (b) the **matched-information verdict** — row 1 vs row 2; (c) the **Δ-R²-vs-retention scatter** per (layer, class) — if classes order differently in Δ-space and behavior space, that is the ReSAE/EAGLE-3 divergence realized and the headline figure; (d) function-class ordering in behavior space.

**Kill criterion:** if one-step (k=1) predicted states at the trait layers retain <~50% of the ceiling r for ALL function classes and traits, multi-hop transport is dead — report the Stage-0 atlas + the one-step table and stop.

## Benchmark spec — versus #779 (parent)

Benchmark on #779's own rig and in-repo measured numbers, NOT Persona Vectors' published table (#779's rig-validation gate missed 4/6 trait-mode cells → LOW confidence on absolute anchoring; internally-consistent comparisons on the shared rig remain valid):

- **Reference numbers (system mode, within-condition Pearson r):** direct predictor g — sycophancy 0.91, hallucination 0.86; raw PV projection — evil 0.50 at its own layer 20, sycophancy 0.60, hallucination 0.56; learned map h cleared +0.05 only in 2/6 many-shot cells.
- **Reuse (run the artifact-reuse (a)-(h) fitness check):** HF `superkaiba1/explore-persona-space-data/issue779_monitoring/` — `analysis_tensors/` (c_x at all 28 layers, 5000 LMSYS contexts + eval-pass tensors + `pass_a/` judge scores), `r_b/` (3 traits × 28 layers), `raw_completions/`. Scripts: `scripts/issue779_common.py`, `scripts/issue779_collect.py`. Protocols inherited verbatim: within-condition Pearson r with std<1 condition prune, bootstrap CI resampling conditions, Sonnet graded 0-100 N=5 drop-never-coerce judge scores (reused, not re-run).
- **Layer-selection discipline:** target layers fixed a priori from #779's per-trait curves (`step0_oracle.json`); any NEW per-layer argmax in a headline must use selection-symmetric nulls / held-out-split selection (`.claude/rules/selection-symmetric-nulls.md`) — the oracle-layer artifact is #779's binding lesson.

## Constraints / invariants

- Identity (predict-zero in Δ-space) and per-layer affine are mandatory comparators; the target is Δ_ℓ, never raw-state R² (2405.12250: raw is 0.99-linear tautologically; Instruct models even more so).
- Δ-R² alone is never the headline: the behavior-space evaluation (Stage 1) is the function-class comparison that counts (ReSAE EV↔CE divergence; EAGLE-3 flat-scaling case). No symmetric/PSD-only maps or SVD-only Δ summaries (the Δ-structure is rotational).
- DPI discipline: the transport read is bounded by the target-layer ceiling; no claim of information beyond the source layer. The matched-information comparison (row 1 vs row 2) is the only winnable fight and is framed as such.
- Train/eval context disjointness inherited from #779 (eval contexts never in any fit, including the calibration split).
- Position-distribution caveat: all depth-axis precedents fit on pooled random positions; this task fits last-prompt-token only — run the cheap per-position-vs-last-token validation before comparing R² magnitudes to the literature.
- Compute: 0-GPU-h-class (vectorized regressions over cached tensors; depth-GRU minutes on one small GPU via the compute-character carve-out). No model fine-tuning, no new capture, no judge spend.

## Deferred extensions (explicitly out of scope this task)

- Dynamics-residual features into a g-style predictor; label-efficiency curves; dynamics-pretrained encoders (the former Q2 apparatus).
- Regime-conditional residual-gap arm (trait-on vs trait-off dynamics, Koopman template 2605.05134).
- Base-vs-Instruct learned-dynamics control (needs one base-model activation-caching pass).
- Per-position maps over full prompts; token-axis forecasting (EAGLE territory).
- Fine-tuning-outcome prediction from dynamics (the leakage-theory follow-up; file as a child if this task lands).

## Provenance

- Parent: #779 (benchmark + artifact source). Informed by #778 (PV replication), #658/#742/#761/#763 (base-side predictor line), #493 (predictor bake-off), #502 (mid-late-layer context→behavior signal).
- Literature artifacts: `artifacts/lit-review-2026-07-01.md` (four-slice broad sweep), `artifacts/lit-review-2026-07-02-deep-dive.md` (full-paper deep dive).
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

> [after lit review + discussion:] okay I think it's worth running, can we benchmark against 779?

> [2026-07-02:] Do a deep literature dive on predicting next activation from current activation and fold in your findings

> [2026-07-02, scope narrowing:] we're only interested in the next activation prediction part and benchmarking against 779. What experiment will you run for that
