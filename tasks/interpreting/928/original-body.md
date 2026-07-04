---
title: 'Thinking-model CoT decomposition of the context→answer activation map: composed
  vs joint vs CoT-augmented'
kind: experiment
tags:
- from-810
- thinking-model
- cot-decomposition
created_at: '2026-07-03T11:47:21Z'
has_clean_result: false
parent_id: 810
origin_prompt: 'We''ve found a mapping from context to mean answer. This is on a non
  thinking model. For a thinking model we could either have: - Mapping from context
  to CoT, mapping from CoT to answer (which compose into mapping from context to answer)
  - Mapping from context to (CoT + answer) - Mapping from context + CoT to answer.
  This should be pretty easy to test with our existing setup, just using a thinking
  model. Pick one as similar as possible to Qwen2.5-7B (or if Qwen2.5-7B itself is
  a thinking model). Run both the averaged over query context vectors and the individual
  context + query context vectors. For averaged over query you can average over query
  vector or average over CoT vector. Try mean pool, max pool, and boundary tokens
  as summaries for each part (context, CoT, answer).'
workflow: v1
goal: On a thinking model chosen for maximal similarity to Qwen2.5-7B-Instruct, determine
  how the base-model linear context→mean-answer activation map (#810) decomposes across
  the chain-of-thought — comparing (1) context→CoT composed with CoT→answer, (2) context→(CoT+answer)
  jointly, and (3) (context+CoT)→answer, against the direct context→answer baseline
  — under mean-pool / max-pool / boundary-token summaries for each part (context,
  CoT, answer), in both the averaged-over-query regime (averaging over the query vector
  or over the CoT vector) and the individual context+query regime, scored by held-out
  R² per layer on the parent line's folds.
relates_to:
- spec-context-as-vector
---
## Goal

On a thinking model chosen for maximal similarity to Qwen2.5-7B-Instruct, determine how the base-model linear context→mean-answer activation map (#810) decomposes across the chain-of-thought — comparing (1) context→CoT composed with CoT→answer, (2) context→(CoT+answer) jointly, and (3) (context+CoT)→answer, against the direct context→answer baseline — under mean-pool / max-pool / boundary-token summaries for each part (context, CoT, answer), in both the averaged-over-query regime (averaging over the query vector or over the CoT vector) and the individual context+query regime, scored by held-out R² per layer on the parent line's folds.


## Overview / Motivation

#810 (parent; #722/#658 line) established a linear map from base-model context activation summaries to the mean-answer activation summary on Qwen2.5-7B-Instruct — a non-thinking model. A thinking model inserts a CoT between context and answer, so the single map factors in at least three testable ways:

1. **Two-stage composition:** context → CoT-summary, and CoT-summary → answer-summary (these compose into a context → answer map; compare its held-out R² to the direct map).
2. **Joint target:** context → (CoT + answer) summary (CoT and answer summarized jointly / stacked as one target).
3. **CoT-augmented input:** (context + CoT) → answer (does conditioning on the realized CoT summary improve answer prediction over context alone?).

Comparing (1)/(2)/(3) against the direct context→answer map (fit on the same rollouts) measures whether the CoT mediates the context→answer mapping (composition ≈ direct; CoT summary adds predictive information) or is bypassed (direct ≫ composed; CoT adds nothing beyond the context).

## Model choice (planner grounds the pick; user criterion: as similar as possible to Qwen2.5-7B)

Qwen2.5-7B(-Instruct) is NOT a thinking model (no thinking mode; QwQ / Qwen3 are the Qwen thinking lines). Candidates, in order of lineage-closeness to the project's base model — planner verifies each on HF and picks:

- **OpenThinker2-7B / OpenThinker-7B** — Qwen2.5-7B-Instruct fine-tuned on OpenThoughts reasoning traces; same base checkpoint as the project's model (closest lineage).
- **DeepSeek-R1-Distill-Qwen-7B** — R1 distilled onto Qwen2.5-Math-7B (same architecture family, math-specialized base; the most-studied 7B open thinking model).
- **Qwen3-8B with thinking enabled** — different generation, but the hybrid thinking toggle gives a within-model thinking-ON vs thinking-OFF control arm (same weights, thinking off ≈ the non-thinking regime) — attractive as a secondary/control arm, not the primary pick under the similarity criterion.

The chosen model must emit an explicit, parseable CoT segment (e.g. `<think>...</think>`) so the CoT span can be token-indexed for summaries; assert the delimiter tokenization in-process.

## Design sketch (planner refines)

- **Reuse the existing setup:** the #810/#722/#658 pipeline — context grid, question set, activation-capture + summary code, held-out R²-over-mean-per-layer DV. The single new variable is the model (plus the CoT segmentation it necessitates).
- **Three parts per rollout:** context, CoT (thinking span), answer (post-thinking response), segmented by the model's thinking delimiters.
- **Summaries per part (crossed):** mean pool, max pool, boundary tokens (the part-final / boundary token(s), analogous to #810's turn-boundary summary). #810 found mean — not max-pool — carries the map on the answer side; that prior exists for the answer part but NOT for the CoT part, so the full cross is warranted.
- **Two regimes, as in the parent line:**
  - **Averaged-over-query:** context vectors averaged over queries. For the CoT-involved maps the averaging can be taken over the query vector or over the CoT vector — run both (user-specified).
  - **Individual context + query:** per-(context, query) vectors, no averaging.
- **Fits:** linear/ridge maps under GROUP-level held-out folds (held-out contexts, not pointwise LOO — `.claude/rules/ood-generalization-folds.md`); score composed (1), joint (2), CoT-augmented (3), and the direct context→answer baseline (0) on the same folds.
- **Compute character:** the summary × layer × mapping × fold grid is a many-cell linear-fit battery — batch via the Gram/dual-space recipe (`.claude/rules/vectorize-many-cell-fits.md`); never a serial per-cell loop. Generation of thinking rollouts via vLLM batched.

## First step (per the new-direction rule)

Thorough literature review (arXiv MCP + web) on linear readouts / probes over reasoning traces and CoT-mediation analyses of hidden states — name the closest prior formalizations — then finalize the formalization in the Goal before any capture code.

## Provenance

Verbatim originating prompt recorded in frontmatter `origin_prompt:`.
