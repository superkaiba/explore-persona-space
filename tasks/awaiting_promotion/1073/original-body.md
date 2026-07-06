---
title: 'Decoding-regime robustness of the context→answer map: single greedy vs single
  stochastic vs 10-rollout-averaged activations'
kind: experiment
tags: []
created_at: '2026-07-06T04:17:21Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'file it as its own task and run in the background with happy coder
  (preceding ask: How do mapping and activations averaged over rollouts compare to
  mapping and activations for single rollout with deterministic vs single rollout
  with stochastic)'
workflow: v1
goal: 'Measure how the #779 context→answer mapping and its answer-side activations
  change with decoding regime — single greedy (temperature 0) vs single stochastic
  (temp 1.0, top_p 0.95, seed 42) vs 10-rollout-averaged, everything else held at
  the #779 recipe, in BOTH prefix-based and context-based mapping arms — to test whether
  single-greedy targets are an adequate stand-in for rollout-averaged ones (theory
  plan C11/C12).'
relates_to:
- spec-context-as-vector
---
# Decoding-regime robustness of the context→answer map: single greedy vs single stochastic vs 10-rollout-averaged activations

## Goal

Measure how the #779 context→answer mapping and its answer-side activations change with decoding regime — single greedy (temperature 0) vs single stochastic (temp 1.0, top_p 0.95, seed 42) vs 10-rollout-averaged, everything else held at the #779 recipe, in BOTH prefix-based and context-based mapping arms — to test whether single-greedy targets are an adequate stand-in for rollout-averaged ones (theory plan C11/C12).

## Overview / Motivation

#779's context→answer monitoring map compared 1-rollout stochastic targets against a 10-rollout-averaged target variant (levels shift +0.02 to +0.07, pattern unchanged; `tasks/awaiting_promotion/779/body.md` Results) but never ran a deterministic (greedy, temperature 0) arm. The greedy-decoding activation data that exists elsewhere (#928 OpenThinker2-7B CoT map, #833 on-policy map re-test, #813 substrate-dependence map) shares no model+substrate with the stochastic map line (#779/#823/#922/#952/#958), so no clean greedy-vs-sampled contrast exists anywhere in the project. `docs/theory_assumption_test_plan.md` items C11 (:35) and C12 (:36) name exactly this alignment — "primary theory test aligns the activation and the behavior measurement on the SAME greedy answer; temperature sampling a labeled robustness extension; separate greedy vs stochastic reporting" — as planned-but-unexecuted PHASE-2+ work. `docs/glossary_context_answer_map.md:32` already formalizes the axis (single-draw `v_A` vs rollout-averaged `v̄_A^r`). This task closes the axis with decoding regime as the single manipulated variable on one substrate.

## Design (capture-time notes — planner refines)

- **Three arms, decoding regime of the answer-side rollout as the single manipulated variable** (same model Qwen-2.5-7B-Instruct, same LMSYS contexts, same layer set, same fit recipe as #779):
  - **(a) 10-rollout-averaged targets** — REUSE #779's persisted 10-rollout-target variant (no new compute).
  - **(c) single stochastic rollout** — REUSE #779's headline data (temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, n=1; `scripts/issue779_collect.py:558` convention).
  - **(b) single greedy rollout (temperature 0)** — NEW compute: one greedy decode per context over the SAME contexts via `scripts/issue779_collect.py` (SamplingParams already parametrized; single-draw path at `:279`), then activation capture + map fits under the identical #779 recipe.
- **Mapping arms (project capture default):** compute the mapping BOTH prefix-based (the prefix is everything before the user query) AND context-based (the context is the prefix plus the user query), as paired arms of the same design, and report both.
- **Readouts:** per-arm map quality on the #779 metrics, plus cross-arm agreement — correlation of per-context targets and per-context map predictions across (a)/(b)/(c) — answering whether single-greedy is an adequate stand-in for rollout-averaged targets and how far a single stochastic draw sits from both.
- **Reuse (fitness-check per `.claude/rules/artifact-reuse.md`):** #779 activation stores / eval JSONs / context set, `scripts/issue779_collect.py`, the #779 fit code. New code only for the greedy decoding config and the cross-arm agreement analysis.
- **Cost expectation:** low single-digit GPU-h — one greedy generation + capture pass over existing contexts plus fits; arms (a) and (c) are precomputed.

## Anchors / lineage

- Parent: #779 (context→answer monitoring map; its body already carries the (a)-vs-(c) leg).
- `docs/theory_assumption_test_plan.md` C11/C12 — the planned-not-run greedy-vs-temperature items this executes.
- `docs/glossary_context_answer_map.md:32` — single-draw `v_A` vs rollout-averaged `v̄_A^r`.
- NOT this axis (disambiguation): #813 `per_example_vs_averaged` (query-averaging grain of fits), #931 draw-averaged denominator (training-row subsampling), `eval_results/extraction_method_comparison/` (last-token vs mean-response position).

## Provenance

Originating user prompts (verbatim, same chat, 2026-07-05):
1. "Do we have an experiment/prexompute data for this: How do mapping and activations averaged over rollouts compare to mapping and activations for single rollout with deterministic vs single rollout with stochastic"
2. "file it as its own task and run in the background with happy coder"
