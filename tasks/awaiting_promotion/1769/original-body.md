---
title: 'Prefill-vs-decode behavioral commitment: does r_B steering only during decode
  change judged behavior expression in a dense full-attention model'
kind: experiment
tags: []
created_at: '2026-07-28T20:23:10Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'add these to the existing experiments: 2. Prefill-vs-decode behavioral
  commitment (~4-8 GPU-h) - the only existing evidence is one hybrid-MoE paper; our
  rig is nearly the instrument.'
workflow: v1
goal: 'Determine when behavioral commitment happens in Qwen-2.5-7B-Instruct: does
  steering a persona-vector direction r_B only during prefill vs only during decode
  vs both change on-policy judged behavior expression - i.e., is the answer''s behavior
  already committed in the pre-answer context state, or decided token-by-token during
  generation?'
relates_to:
- spec-steering
- spec-context-as-vector
---
## Goal

Determine when behavioral commitment happens in Qwen-2.5-7B-Instruct: does steering a persona-vector direction r_B only during prefill vs only during decode vs both change on-policy judged behavior expression - i.e., is the answer's behavior already committed in the pre-answer context state, or decided token-by-token during generation?

## Overview / Motivation

The residual-stream direction-taxonomy review (docs/lit_reviews/residual-stream-direction-taxonomy-integration.md @ a74c9d54; review gap rank 17, OPEN) found exactly ONE piece of evidence on prefill-vs-decode behavioral commitment: steering only during decoding had zero effect (p > 0.35), implying prefill-time commitment — but measured only in a gated-linear-attention hybrid MoE (2603.16335). No dense full-attention transformer measurement exists. Qwen-2.5-7B is dense full-attention, and this project's rig (persona-vector r_B bank per layer from #779, judged on-policy behavior expression per the project judge, behavior-eliciting context batteries, activation-steering harness) is nearly the complete instrument already.

This also directly complements the transport line's central finding (#1092: answer-state transport runs through the query-bearing context state; the prefix-end state carries disposition but little answer content): if behavior is committed at prefill, pre-answer monitoring (the #779/#1092/#1739 line) is monitoring the decision itself; if decode-time steering works, the pre-answer state is only a prior.

## Design sketch (planner refines)

- Conditions per behavior (evil / sycophancy / hallucination, the r_B bank): steer with alpha * r_B at the steering-validated layer under 4 arms — prefill-only, decode-only, both, neither — on a behavior-eliciting context battery (reuse the #779/#1092 eval batteries); N rollouts per context per arm; judged expression rate + graded score per the project judge (claude-sonnet-4-5-20250929).
- Dose ladder over alpha (2-3 values) so a null decode-only arm is interpretable against a working both-arm (manipulation check), not a dead-steering artifact.
- Both mapping arms rule: N/A (no representation map fitted) — this is a causal-timing intervention experiment; state exemption in plan.
- Describe interventions mechanistically per brief-hygiene rules (adds a direction vector to hidden activations at layer l during specified token ranges).
- est 4-8 GPU-h (vLLM batched generation across arms x behaviors x doses; steering hooks; judge via Batch API).

## Provenance

Filed from user chat 2026-07-28 (taxonomy-review integration follow-ups). Origin prompt (verbatim): "add these to the existing experiments: ... 2. Prefill-vs-decode behavioral commitment (~4-8 GPU-h) — the only existing evidence is one hybrid-MoE paper; our rig is nearly the instrument."
