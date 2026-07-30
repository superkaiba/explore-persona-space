---
title: Linear/operator characterization of the four context→answer map arms (consistency,
  channels, null spaces, endomorphism reads)
kind: experiment
tags: []
created_at: '2026-07-28T21:35:32Z'
has_clean_result: false
parent_id: 1092
origin_prompt: 'run both in background with ahppy coder (2026-07-28; two-task split
  per: ''can it not be all one task? Or at least separate the nonlinear out'')'
workflow: v1
goal: Characterize the fitted linear context→answer operators across the four arms
  (prefix-end, context, bare-query, query-averaged) as operators — algebraic consistency,
  per-arm channel structure and trait tables, null-space ceilings with causal-inertness
  tests, and gate-conditional endomorphism reads (decoded SVD, eigen/copying, preservation
  map, trait gain matrix, fixed point) — all noise-ceiling-gated, per Task A of docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Linear/operator characterization of the four context→answer map arms

## Goal

Characterize the fitted linear context→answer operators across the four arms (prefix-end, context, bare-query, query-averaged) as operators — algebraic consistency, per-arm channel structure and trait tables, null-space ceilings with causal-inertness tests, and gate-conditional endomorphism reads (decoded SVD, eigen/copying, preservation map, trait gain matrix, fixed point) — all noise-ceiling-gated, per Task A of docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md

## Plan basis

Execute **Task A** of `docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md` — sections §2 (Q1 algebraic consistency), §4 (Q3 channel structure), §5 (Q4 ceilings and null spaces), §6 (Q5 endomorphism reads), under the §8 validity gates and the §9 reuse policy. The plan doc is the scope contract; the adversarial planner refines it into the executable plan. Companion lit grounding: `docs/ideas/2026-07-06-context-answer-map-analyses.md` (methods survey + Round 3 null-space section); measured anchors: `docs/results_summaries/2026-07-22-prefix-query-context-answer-map-consolidated.md`.

## Work items (from the plan doc)

1. **Q1 consistency:** Jensen/commutativity gap (avg-then-map vs map-then-avg, zero new fits); e(p)→v̄_P chain test (state deficit vs readout deficit for the prefix-end arm); joint-vs-marginal prefix operator (crossed-design identification check).
2. **Q3 channels, all four arms on shared folds:** held-out predictable-variance spectra; ρ₁² linear maximal-correlation ceilings per layer; per-trait-direction held-out R² tables (directions fixed on train folds); cross-arm principal angles vs spectrum-matched nulls.
3. **Q4 null spaces:** co-kernel ceiling per behavior (fraction of each trait read-out outside range(M), per arm); LEACE-erase / injection causal tests on candidate kernel directions vs top singular directions (n ≫ d arms only; prefix arm rank-limited by 1,145 prefixes — no prefix-side kernel claims).
4. **Q5 endomorphism reads (gate-conditional):** decoded top singular pairs (tuned lens + trait cosines); eigen/copying detector + near-eigenvector trait test; (cos, gain) preservation map per direction; trait gain matrix G = U_Bᵀ·W·V_B per arm; almost-invariant subspaces; affine fixed point x* = (I−W)⁻¹b decoded.
5. **Multi-draw decode-noise ceiling:** K≥5 answer draws per context on a ~2K-context subset; per-direction noise floor gating every per-direction R² read. PERSIST the draws — the sibling nonlinearity task reuses them.

## Scope / constraints

- Reuse banked artifacts per plan §9: banked ridge operators (#779 up-to-1M-row, #1092 21,193-row crossed) are the objects of study; held-out reads refit cheaply per fold from banked activation stores (per-row held-out predictions were not persisted). Artifact-reuse fitness checks apply.
- Both mapping arms (prefix-based AND context-based) are in the design by construction; every fitted map reports the standing identity+bias baseline + kNN-retrieval reads.
- Validity gates (§8) bind every read: group-level folds by prefix id, λ discipline (df(λ), ~3-λ reruns, same-λ refits inside permutation draws), permutation + matched-n nulls, non-normality gate before any eigen read, train-fold-only direction selection, n_train vs d stated per fit.
- Estimated 5–12 GPU-h total; the majority of the work is 0-GPU on banked stores. GPU items: causal steering tests, multi-draw generation, batched permutation nulls.

## Provenance

- Verbatim dispatch: "run both in background with ahppy coder" (2026-07-28 chat), following "can it not be all one task? Or at least separate the nonlinear out" — the two-task split is encoded in plan §9 "Execution shape: two tasks."
