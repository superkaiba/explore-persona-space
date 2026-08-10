---
title: 'Representation-level effects of the 21 minimal-pair information types: context-vector
  shift, answer-vector shift, and map discrimination'
kind: experiment
tags: []
created_at: '2026-08-10T05:01:23Z'
has_clean_result: false
parent_id: 2162
workflow: v1
goal: 'On Qwen-2.5-7B-Instruct, using the frozen 21-type minimal-pair context bank
  and banked state bank from the parent patch sweep (#2162), measure with NO patching
  how changing exactly one information type in the context shifts (a) the context
  representation (context-end vector AND prefix-end state, both arms: per-type shift
  magnitude, within-type direction consistency, cross-type geometry) and (b) the realized
  on-policy answer representation v_A (teacher-forced answer activations over the
  banked floor/ceiling rollout text under each unpatched context), and (c) whether
  the banked context-to-answer ridge maps (#779 context-end, #1738 prefix-end; no
  new map training unless fitness fails) DISCRIMINATE the paired contexts — per pair,
  does f(v_C(A)) land closer to v_A(A) than to v_A(B) (paired two-alternative retrieval
  accuracy + cosine/euclidean similarity margin), reported per information type with
  a ranking of which types the map discriminates worst, alongside the identity+learned-bias
  baseline and pool-level kNN retrieval, against a shuffled-pair null.'
relates_to:
- spec-context-as-vector
---
# Representation-level effects of the 21 minimal-pair information types: context-vector shift, answer-vector shift, and map discrimination

## Goal

On Qwen-2.5-7B-Instruct, using the frozen 21-type minimal-pair context bank and banked state bank from the parent patch sweep (#2162), measure with NO patching how changing exactly one information type in the context shifts (a) the context representation (context-end vector AND prefix-end state, both arms: per-type shift magnitude, within-type direction consistency, cross-type geometry) and (b) the realized on-policy answer representation v_A (teacher-forced answer activations over the banked floor/ceiling rollout text under each unpatched context), and (c) whether the banked context-to-answer ridge maps (#779 context-end, #1738 prefix-end; no new map training unless fitness fails) DISCRIMINATE the paired contexts — per pair, does f(v_C(A)) land closer to v_A(A) than to v_A(B) (paired two-alternative retrieval accuracy + cosine/euclidean similarity margin), reported per information type with a ranking of which types the map discriminates worst, alongside the identity+learned-bias baseline and pool-level kNN retrieval, against a shuffled-pair null.

## Design sketch (capture-time; planner refines)

Sibling of the #2162 patch-only causal sweep, on the SAME frozen context bank — but with NO patching. Three DV families, per information type:

1. **Context-vector shift.** For each minimal pair (A, B) differing in exactly one information type: Δv_C = v_C(B) − v_C(A). Report per-type shift magnitude (norm, relative to within-type baseline variability), within-type direction consistency (pairwise cosine of Δv_C across pairs of the same type), and cross-type geometry (are different information types' shift directions distinct?). **Both mapping arms per the standing rule: context-end vector v_C AND prefix-end state, as paired arms** (the #2162 state bank carries both slots at every layer).
2. **Answer-vector shift.** Δv_A = v_A(B) − v_A(A), where v_A(c) is the mean answer activation over on-policy generations under unpatched context c. #2162's floor rollouts (unpatched under A) and generate-under-donor ceiling rollouts (unpatched under B) are the on-policy text under each context — answer activations come from teacher-forced passes over that banked rollout text (cheap GPU; no new generation unless coverage gaps surface).
3. **Map discrimination.** Using the BANKED context→answer ridge maps (#779 context-end 963k L14/L19; #1738 prefix-end L14/L19/L26 — no new map training unless the artifact-reuse fitness check fails): per pair, does f(v_C(A)) land closer to v_A(A) than to v_A(B)? Report paired two-alternative retrieval accuracy + the similarity margin (cosine AND euclidean), per information type, with a ranking of which types the map discriminates WORST. Alongside, per the standing mapping rules: the identity+learned-bias baseline (v̂ = x + b) and the pool-level kNN-retrieval read (`analysis/mapping_baselines.knn_retrieval`, chance stated). Null: shuffled-pair assignment.

Key reuse (fitness-check per `.claude/rules/artifact-reuse.md` before any regeneration):
- Frozen bank + state bank: `issue2162_ctxinfo/analysis_tensors/vc_bank/` (incl. `bank.json`) on `superkaiba1/explore-persona-space-data` — 1,404 contexts × 2 slots (v_ce, v_pe) × 28 layers.
- Rollout text: `issue2162_ctxinfo/raw_completions/{anchors,grid,stage2}` (floor + ceiling arms are the unpatched on-policy text under A and B respectively).
- Banked maps: #779 context-end ridge (L14/L19), #1738 prefix-end ridge (L14/L19/L26).

Relation to parent: #2162 answered which types are causally USABLE when injected at one position (and, via the read probe, whether they are linearly DECODABLE there). This task answers a different question — how much each type MOVES the context and answer representations, and whether the fitted context→answer map PRESERVES the distinction (map fidelity per information type). A read-probe AUC near 1 with a map that cannot discriminate the paired answers would dissociate encoding from map fidelity.

## Provenance

Verbatim originating prompt (user chat, 2026-08-09):

> can you run almost that same experiment but instead looking at how changing that information in the context:
> - changes the context vector
> - changes the answer vector
> - whether our mapping can distinguish well between both contexts (i.e. can the mapping correctly identify the answer vector corresponding to each context + which it's worse at (some kind of similarity metric))

("that same experiment" = #2162, the 21-type minimal-pair patch-only sweep.)
