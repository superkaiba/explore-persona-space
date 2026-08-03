---
title: Jacobian context→answer map (J_C→A) vs fitted ridge M + J-space workspace mediation
  on Qwen-2.5-7B
kind: experiment
tags: []
created_at: '2026-07-28T21:44:34Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'don''t treat it as a kill switch. just run it then continue till the
  end no matter what (thread: J-space paper vs our context→answer mapping; ''ok combine
  everything we''ve discussed so far into a plan and estimate the GPU h and wall clock
  time'')'
workflow: v1
goal: 'Compute the averaged causal Jacobian J_{C→A} on the #779 corpus; measure what
  fraction of the fitted ridge map M''s predictive power is causal; test which map
  predicts steered-generation interventions and off-distribution transfer; and test
  whether the context→answer channel reads from / writes into the J-space verbalizable
  workspace (Anthropic 2026), all phases unconditional.'
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Jacobian context→answer map (J_{C→A}) vs fitted ridge map M + J-space workspace mediation (Qwen-2.5-7B)

## Goal

Compute the averaged causal Jacobian J_{C→A} on the #779 corpus; measure what fraction of the fitted ridge map M's predictive power is causal; test which map predicts steered-generation interventions and off-distribution transfer; and test whether the context→answer channel reads from / writes into the J-space verbalizable workspace (Anthropic 2026), all phases unconditional.

## Provenance

- Origin: user chat 2026-07-28 (J-space paper discussion → J_{C→A} design → combination program).
- Verbatim user directive on execution semantics: "don't treat it as a kill switch. just run it then continue till the end no matter what" — Phase 1 is a diagnostic read only; ALL phases run unconditionally regardless of any intermediate outcome. No phase gates on another phase's scientific result (engineering gates like the parity check still block, since downstream numbers are invalid without them).
- Earlier verbatim asks in the same thread: "could we do jacobian from context to answer vectors?", "can we combine our mapping with theirs somehow?", "does this include seeing if our mapping reads from the same space as the J space?", "ok combine everything we've discussed so far into a plan and estimate the GPU h and wall clock time".

## Design (phases; all unconditional)

Both mapping arms run throughout per the standing rule: prefix-based (perturbation broadcast over prefix span only) AND context-based (prefix + user query span). Both arms come from the same backward passes (position-subset sums).

- **Phase 0 — infrastructure.** (0.1) Adapt anthropics/jacobian-lens `fit()` to Qwen-2.5-7B; fit their J_ℓ (full d×d per layer, ~1k web-text prompts × 128 tok, frozen weights, all layers per backward). (0.2) J-space dictionary: lens vectors J_ℓᵀ·W_U rows; projection operator = sparse nonnegative coding (NNLS) reconstruction energy (honest) + top-active-linear-span (cheap upper bound) — J-space is a cone, not a subspace; both operationalizations reported. (0.3) Parity rig: teacher-forced re-run of stored #779 (context, answer) pairs must reproduce the stored answer-summary shards before any J number is trusted (engineering gate).
- **Phase 1 — directional diagnostic (NOT a kill switch).** Top-20–50 right/left singular direction pairs of the fitted M as backward seeds × ~1k pairs → per-direction table: claimed gain σ_i vs measured causal gain vs direction alignment (cos to u_i). Recorded and carried forward as interpretation context only.
- **Phase 2 — J_{C→A} sketch then full.** 256 random seeds × ~1.5k pairs (low-rank sketch), THEN the full-rank 3,584-seed round (unconditional per the user directive). Reads: fraction of M's held-out R² recovered by J on the same held-out 1,000 contexts; identity+bias baseline + kNN retrieval (standing rule); operator battery vs M per scripts/issue1345_operator_comparison.py conventions, restricted to the on-support (top-variance v(C)) subspace.
- **Phase 3 — steered-regeneration ground truth.** Activation steering during generation (broadcast αΔ at layer ℓ, Δ = persona/context difference vectors + top map directions): ~200 contexts × 3 scales × 2–4 directions × K=5 samples. DVs per the dual-DV rule: judge-scored on-policy behavior change PRIMARY (graded 0-100, Sonnet judge, Batch API, drop-never-coerce) + continuous answer-summary shift SECONDARY. Headline read: J·αΔ vs M·αΔ vs measured shift — which map predicts interventions.
- **Phase 4 — J-space mediation (flagship).** Read side: projection energy of M's top right-singular directions into the J-space cone vs matched random-subspace nulls; refit M on P_J·v(C) vs (I−P_J)·v(C) vs full; causal split J_{C→A}·δ_J vs δ_⊥. Write side: do M's column space and the ACTUAL Phase-3 steered answer shifts land in J-space at L′ (a behavior shift landing outside J-space = non-reportable shift, safety-relevant either way). Word-level: NNLS decomposition of persona difference vectors — are active lens words trait-relevant?
- **Phase 5 — transfer + free analyses.** (a) LMSYS-fit J and M evaluated on WildChat / persona batteries (causal components should transfer; correlational decay). (b) Retrospective leakage re-read: J-space-projected context similarity vs raw context-geometry similarity as predictor of EXISTING measured implant leakage (0 GPU-h, existing eval JSONs). (c) J-lens vocab decoding of M's directions + persona vectors (word clouds). (d) Chain-composition judge-free DV: lens(J_{L′}·M·v(C)) predicted answer vocabulary vs actual generated answers.

## Reuse

- Corpus + targets: #779 persisted (context, on-policy answer) pairs — rollout text on HF (issue779_monitoring prefix), per-layer answer summaries (4 pooling variants, sharded .pt), prefix activations; fitted-map recipes in eval_results/issue_779/fitter-fair-comparison*/ (n_train 3.6k/10k/50k/150k–1M ladders, shared held-out 1,000). Ridge M deterministically refittable from stored summaries.
- Estimator: anthropics/jacobian-lens (open reference implementation) — adapt target (pooled answer summary at L′) + source restriction (context/prefix span, pooling-calibrated broadcast sum). https://github.com/anthropics/jacobian-lens
- Instruments: analysis/mapping_baselines (identity_bias_predict, knn_retrieval), scripts/issue1345_operator_comparison.py battery, existing steering + judge rigs.
- Standard reuse fitness checklist (.claude/rules/artifact-reuse.md) applies at plan time.

## Compute (chat-grade arithmetic — UNMEASURED, ±2–3×; plan §9 re-derives from measured 1-cell pilots per plan-compute-sizing)

- P0.1 ~5–8 H100-h; P0.3 <0.5; P1 <1 (~20 min); P2 sketch ~3–4, full-rank +20–30 (~3–4 h wall on 8×H100); P3 ~2–5 GPU-h + Batch-API judge (no GPU held during judging — release/downsize pod per GPU-width rules); P4 ~1–2; P5 ~1–3.
- Total: ~35–50 H100-h committed up front (full-rank round unconditional). Compute wall ~1 day on one 8-GPU node; calendar ~3–5 days (engineering-dominated: jlens adaptation, steering rig, NNLS projection, parity checks).

## Measurement notes

- Pooling/normalization parity between J and M is a hard validity requirement (same summary variant of the 4 stored).
- Operator comparisons restricted to on-support subspace (M undefined off-support; naive full-space comparison manufactures fake disagreement).
- All J-space overlap numbers reported against matched nulls (selection-symmetric-nulls discipline).
- Teacher-forced J captures the fixed-text (representation) component only; the generation component is non-differentiable — Phase 3 regeneration is the total-effect ground truth. Expected a priori: M wins in-distribution held-out R² by construction; J's value is causal decomposition, transfer, and intervention prediction.
