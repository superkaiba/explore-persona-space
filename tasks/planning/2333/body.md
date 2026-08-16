---
title: 'Snowball test: do context-vector patch effects run through the first few answer
  tokens? First-k token patches + prefills vs ce patch, both models'
kind: experiment
tags: []
created_at: '2026-08-16T19:44:45Z'
has_clean_result: false
parent_id: 2162
origin_prompt: 'Can you also run this on both models: patching at context vector at
  all layers vs patching at first 1/2/3 answer tokens at all layers (excluding context
  vector) vs prefilling first 1/2/3 tokens, on the most successful causal patching
  experiments; donor schemes: BOTH mediation form and B''s-answer-start form'
workflow: v1
backend: runpod
goal: 'Test the snowball explanation of context-vector patch causality on Qwen2.5-7B-Instruct
  and Qwen3.5-9B (thinking disabled): compare the all-layer context-end patch against
  patching only the first 1/2/3 generated answer-token positions (ce excluded) and
  against prefilling the first 1/2/3 tokens, under two donor schemes (mediation form:
  content produced under the ce patch with the patch then removed; B''s-answer-start
  form: the target context''s own reference answer opening), measured by F_beh/F_act
  vs scheme-matched shuffled-donor nulls on the pre-registered most-successful cells
  (#2162''s 5 stored-and-used + #2094''s pirate matched-query pairs).'
relates_to:
- spec-context-as-vector
- spec-steering
---
## Goal

Test the snowball explanation of context-vector patch causality on Qwen2.5-7B-Instruct and Qwen3.5-9B (thinking disabled): compare the all-layer context-end patch against patching only the first 1/2/3 generated answer-token positions (ce excluded) and against prefilling the first 1/2/3 tokens, under two donor schemes (mediation form: content produced under the ce patch with the patch then removed; B's-answer-start form: the target context's own reference answer opening), measured by F_beh/F_act vs scheme-matched shuffled-donor nulls on the pre-registered most-successful cells (#2162's 5 stored-and-used + #2094's pirate matched-query pairs).

## Design essentials (user-specified; planner elaborates, deviations named)

- **Pair sets (pre-registered "most successful"):** #2162's 5 stored-and-used cells (the instruction-format family, 180 directed pairs, context-end slot) + #2094's matched-query bare-to-pirate pairs (its strongest clean cell: context-end, all-layer replace, F_beh 0.63).
- **Arms per model:**
  1. ce all-layer full-state replace — positive control (banked for Qwen2.5 in #2162/#2094; reuse, do not regenerate unless a fitness check fails);
  2. first-k generated-token positions, k in {1,2,3}, all layers, ce slot EXCLUDED, donor schemes (a) and (b);
  3. prefill first-k tokens, k in {1,2,3}, schemes (a) (tokens generated under the ce patch, then patch dropped, free continuation) and (b) (B's greedy reference answer tokens), free continuation after the prefill.
- **Donors/references:** greedy reference rollout per context for scheme (b) tokens and states; scheme (a) states/tokens captured from a ce-patched A forward. Activation donors are the realized states at answer positions 1..k, all layers.
- **Metrics:** F_beh primary (judged, #2162 recipe: Sonnet graded 0-100, netted per banked convention) + F_act; scheme-matched shuffled-donor nulls; floors/ceilings from anchors; anchor-separation exclusion and pair-clustered bootstrap as banked. Judge whole response; continuation-only companion read for prefill arms (the prefilled tokens are the donor's, not the model's).
- **Decoding:** evaluated arms K=5 temp 1.0, max_new_tokens 2048 + cap-hit regen trigger; greedy for reference rollouts.
- **Qwen3.5-9B leg:** thinking disabled (enable_thinking=False), position pinning per #2329's template re-pin; REUSE #2329's frozen bank/anchors/ce-control where landed at dispatch time, else self-generate the needed slice (5 cells + a pirate-pair analog + anchors) — small; do not block on #2329.
- **Compute:** ~12 first-k arms x ~195 pairs x K5 x {steered, null} ~= 23k rollouts per model + references/anchors; ~2-4 GPU-h per model (1x H100, batched hooked generation); judging via Batch API at 2k shards. Well under 20 GPU-h total.

## Provenance

Origin prompt (verbatim, user 2026-08-16): "Can you also run this on both models: Motivation: We've found that patching the context vector has a causal effect on the entire answer. One potential explanation is that context vector affects first few tokens -> next few tokens maintain consistency -> it snowballs. We want to test this directly by patching/prefilling the first few tokens of the answer vs patching the context vector. Methodology: Take most successful causal patching experiments. Compare: patching at context vector at all layers / patching at first token at all layers (doesn't include context vector) / first 2 tokens / first 3 tokens / Prefilling first 1/2/3 tokens." Donor-scheme decision (user, same date): run BOTH the mediation form and the B's-answer-start form. Parents/precedent: #2162 (survivor cells, F machinery, judge recipe), #2094 (pirate pairs, hooks, all-layer replace), #2329 (Qwen3.5 integration), #1415 (steering rig lineage).
