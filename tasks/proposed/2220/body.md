---
title: 'Read-write duality: does the behavior-prediction map''s read direction steer
  as well as the mean-difference persona vector?'
kind: experiment
tags: []
created_at: '2026-08-10T20:23:13Z'
has_clean_result: false
parent_id: 1739
origin_prompt: We have a mapping from context -> behavior expression, for evil/hallucination/sycophancy.
  If we have this mapping (averaged over queries but at context vector position),
  can we find a context vector which maximally expresses evil and insert it to make
  the model be evil?
workflow: v1
goal: Test whether the fitted context-to-behavior map's read direction (the whitened
  ridge-weight direction) causally induces the behavior when injected at the last
  context token, as strongly as the mean-difference persona vector, for evil, hallucination,
  and sycophancy.
relates_to:
- spec-steering
- identity-cb-duality
---
# Read↔write duality: does the behavior-prediction map's read direction steer as well as the mean-difference persona vector?

## Goal
Test whether the direction that best PREDICTS a behavior from the context vector — the fitted context→behavior map's read direction `d_read = Σ^{-1/2} w` (normalized whitened ridge weight) — CAUSALLY induces that behavior when injected at the last context token, as strongly as the standard mean-difference persona vector, across evil/misalignment, hallucination, and sycophancy.

## Motivation
We already have a mapping from a model's context vector to its behavior expression (task #1739): a linear ridge predictor from `v_C` (the residual-stream activation at the last prompt token, per layer) to a graded 0–100 judged behavior score, fit for evil, hallucination, and sycophancy on on-policy WildChat rollouts. This map is a **read** — it decodes how strongly a context will express a behavior.

The question here is whether that read direction is also a **write**: if we take the context vector that maximally increases the predicted behavior and inject it into the residual stream, does the model actually express the behavior? And does it do so as well as the mean-difference persona vector (`r_B`, arXiv 2507.21509) — the steering direction the field already trusts?

This is the **stronger version of the causal test** already run on the map line:
- #1415 injected an *observed* context-vector difference at the last context token → evil moved +6.2 judged points at layer 14 (2 seeds), ≈21% of the context-swap ceiling.
- #1776 / #2094 found the fitted linear map "reads a correlate, not a cause": the map predicted ≈none of that realized causal shift at the read-out layer (predicted-vs-realized cosine ≈ 0.00), and context-end is the only single position where activation edits move behavior.

So the correlational map failed a causal test built from *observed* differences. This experiment asks the sharper question: does the map's *own argmax direction* — the direction the map itself says maximizes the behavior — steer, and how does it stack up against the trusted mean-difference vector? A clean read↔write result either rehabilitates the map as a causal handle or confirms that prediction and control live on different geometry.

## Formalization
- `v_C(x) ∈ R^d`: residual-stream activation at the last prompt token of context `x`, at layer ℓ (Qwen-2.5-7B, d=3584, 28 layers). `v_P`: the prefix vector (a prefix's `v_C` averaged over queries).
- `s_B(x) ∈ [0,100]`: graded Sonnet-judge score for behavior B on the model's **on-policy** response to `x`.
- **Map** (from #1739, ridge on *whitened* context vectors): `ŝ_B(v) = wᵀ Σ^{-1/2} (v − μ) + b`.
- **Read direction** = the input-space gradient of the predicted score, i.e. the fixed-norm argmax of `ŝ_B`:  `d_read_{B,ℓ} = ∇_v ŝ_B = Σ^{-1/2} w`, normalized. Under a fixed-norm constraint this IS "the context vector that maximally expresses B." (A linear map's argmax is unbounded without the constraint; the fixed-norm ball makes it a direction. The whitening `Σ^{-1/2}` is load-bearing — the raw ridge weight `w` is NOT the score gradient when the map is fit on whitened features.)
- **Write test**: inject `α · d̂` at a position set `P` during generation (`DeltaHook`), layer ℓ, where `P ∈ {context token (last prompt token), answer tokens (all generated positions)}`; measure Δ(on-policy judged behavior rate) and Δ(teacher-forced margin) vs baselines, inside the coherence-gated α region, on held-out queries, ≥2 seeds.
- **Claim**: read↔write duality holds for B iff the map-read direction injected at the context token achieves a judged behavior-rate lift ≥ the mean-difference persona-vector baselines at matched injection norm — INCLUDING the persona vector at its native answer-token steering regime — and decisively beats the shuffled-map and random controls.

## Design

Injection is a **two-factor grid: DIRECTION × POSITION.** This is the core of the design — it lets us attribute any behavior lift to the direction, the injection locus, or their interaction.

**Direction factor** (per behavior, per layer ℓ; all unit-normalized so α = matched injection norm):
1. **Map-read, context arm** — `d_read = Σ_C^{-1/2} w` from #1739's whitened-`v_C` ridge fit.
2. **Map-read, prefix arm** — same construction from the prefix (query-averaged `v_P`) map. *(both-arms rule)*
3. **Standard persona vector `r_B`** — mean-difference of contrastive activations (arXiv 2507.21509 recipe; #623/#779 vectors). The field's trusted steering direction.
4. **Same-data mean-diff** — mean-difference between the highest- and lowest-behavior-score contexts *in the map's own training data*. Isolates whether any advantage comes from the regression accounting for covariance (arm 1) vs a raw high/low contrast.
5. **Shuffled-map control** (#1739 arm20) — identical fit pipeline with behavior labels shuffled; a direction with no genuine behavior signal.
6. **Random direction** and **α=0 no-injection** — matched-norm noise floor and the true floor.

**Position factor** — WHERE the direction is injected during generation:
- **Context token** — the single last-prompt-token position (`DeltaHook` last-context-token prefill). This is the map's native locus (the map reads `v_C` there) and the only single position #2094 found moves behavior.
- **Answer tokens** — all generated positions (`DeltaHook` `all_positions`). This is the persona vector's native steering regime (arXiv 2507.21509 steers every generated token).

**The decisive cells** (the 2×2 over the two focal directions, plus the canonical baseline the field uses):
- **persona vector `r_B` @ answer tokens** — canonical persona-vector steering; the strongest expected baseline (its home turf).
- **map-read direction @ context token** — our proposal (its home turf).
- **persona vector `r_B` @ context token** — matched-position control: does the trusted direction work on the map's locus?
- **map-read direction @ answer tokens** — matched-position control: does our direction work on the persona vector's regime?

Reading the 2×2: a same-direction difference across positions isolates the POSITION effect; a same-position difference across directions isolates the DIRECTION effect. A clean read↔write-duality win is map-read @ context ≥ `r_B` @ answer at matched norm.

### Injection protocol (shared across cells)
- **Layer:** sweep; inject the layer-ℓ direction at layer ℓ. Expect the effect to peak near layer 14 for evil (#1415), which may differ from the map's best read-out layer — that read/write layer mismatch is itself a result.
- **Mode:** `DeltaHook` — last-context-token prefill for the context-token position; `all_positions` for the answer-token position.
- **Dose:** sweep the injected norm α → a dose-response curve per (direction, position) cell. Matched norm across cells is what makes "steers better" a fair claim.
- **Coherence gate:** compare only within the α region where generations stay coherent, so gibberish is never scored as behavior (persona-vectors and #1415 both use one).

### Measurement (dual-DV rule)
- **Primary:** on-policy **judged behavior rate** (Sonnet `claude-sonnet-4-5-20250929`, graded 0–100 → rate) on held-out queries disjoint from the map-fit and persona-extraction sets.
- **Secondary:** the teacher-forced fixed ±completion-pool **margin** (non-saturating continuous companion).
- Coherence score as a gating covariate.

### Success criterion / headline
Read↔write duality holds for behavior B iff the **map-read direction @ context token** drives a judged behavior-rate increase ≥ the persona-vector baselines at matched norm inside the coherence-gated region — critically including **`r_B` @ answer tokens** (the persona vector's canonical steering regime), not only `r_B` @ context token — reported as peak effect AND dose-response area, while decisively beating shuffled/random controls. Report the **full direction×position grid** per behavior × {context, prefix} map arm, ≥2 seeds, with a **selection-symmetric null band** on the best-layer/best-dose pick so the argmax isn't a fishing artifact.

## Reuse (search-before-building)
- **Maps:** reuse #1739's fitted ridge maps + whitening (`src/explore_persona_space/experiments/issue_1739/{arms,fits,capture}.py`; arm4_ridge_ctx, arm20 shuffled control). Recompute `Σ^{-1/2}w` from the stored fit if the direction isn't already materialized.
- **Persona vectors:** `src/explore_persona_space/artifacts/directions.py`; #623/#779 extracted `r_B`.
- **Injection:** `experiments/issue1415/steering.py::DeltaHook`, generalized `experiments/issue2094/hooks.py::PositionEditHook`.
- **Eval:** #1739 on-policy WildChat + Sonnet batch judge; #779 per-trait rubrics/assets; hallucination alias-match + `HALLU_ABSTAIN_RUBRIC`.
- No training. Generation + judge only.

## Both mapping arms (explicit)
Per the project's prefix-mapping-AND-context-mapping rule, this experiment runs BOTH map arms as paired conditions: the map-read direction is derived from the context vector `v_C` (arm 1) AND from the prefix vector `v_P` (arm 2), and both are injected and reported. Running only one arm would be an explicit, stated deviation — it is not the default here.

## Routing / provenance
- **Question relation:** substantially-different / new question (read↔write duality of the map) — filed as a new child of #1739 (the map being inverted), sibling of #1415/#1776/#2094 (the causal-test line).
- Closest standing open question: `docs/open_questions.md` §1.4 `q:spec-steering` ("Does a steering vector reach the same state?") and §4.3 `q:identity-cb-duality`.

## Plan-time open items (for /adversarial-planner)
- **Literature review** (mandatory for a new direction): ground the read-direction-as-steering-vector idea against DiffMean, logistic-regression-probe-as-steering-vector, and unsupervised steering (e.g. MELBO); position vs persona vectors (arXiv 2507.21509). Name the closest prior formalization of "regress a behavior score on activations, then steer with the weight direction."
- **Compute sizing:** the behavior × direction(6) × layer × dose × seed × held-out-query generation grid needs a MEASURED pilot and vectorized vLLM batching before launch; confirm the cheap-band (<20 GPU-h) estimate.
- **Fit well-posedness:** confirm `n_train` vs `d` for any re-fit; reuse #1739's fits rather than re-fitting under-determined.
- **Hyperparameter grounding:** layer range, α grid, coherence-gate threshold, judge draw count — each with a Source (prior issue or paper).

## Provenance (verbatim)
Originating research prompt: "We have a mapping from context -> behavior expression. For evil, hallucination, sycophancy. If we have this mapping (averaged over queries but at context vector position), can we then find a context vector which maximally expresses evil and in some way insert that to make the model be evil?" Dispatch directive: "just run it in background."
