---
title: 'Causal test of the context/answer vectors: does steering with V_c'' - V_c
  shift on-policy answers from V_a toward V_a''?'
kind: experiment
tags: []
created_at: '2026-07-16T08:08:10Z'
has_clean_result: false
origin_prompt: Do these vectors V_c and V_a have meaning? Like are they causal in
  a way you'd expect or seeing what logit lens produces on them? For example if you
  take a context, c and steer it with V_c' - V_c, does that cause a shift in the average
  answer tokens from V_a to V_a'?
workflow: v1
goal: 'On Qwen-2.5-7B-Instruct, test whether the context vector V_c (last-context-token
  residual activation) is causally sufficient for the answer state and behavior it
  predicts: patch context c with Delta = V_c(c'') - V_c(c) (matched-query persona/instruction
  swaps and cross-query swaps; prefix-based AND context-based arms; layer + position
  sweeps), generate on-policy, and measure (a) the realized mean answer activation''s
  shift from V_a(c) toward V_a(c'') and its agreement with the fitted map''s counterfactual
  prediction f(V_c + Delta), and (b) the judge-scored behavior shift (graded 0-100
  primary + rate companion), against norm-matched random-Delta and shuffled-pair nulls,
  a full-prefix KV-swap ceiling, and the persona-vector r_B steering baseline, with
  a logit-lens top-token readout of V_c, V_a, Delta, and the #922 slow modes as descriptive
  companion.'
relates_to:
- spec-context-as-vector
- spec-steering
---
# Causal test of the context/answer vectors: does steering with V_c′ − V_c shift on-policy answers from V_a toward V_a′?

## Goal

On Qwen-2.5-7B-Instruct, test whether the context vector V_c (last-context-token residual activation) is causally sufficient for the answer state and behavior it predicts: patch context c with Delta = V_c(c') - V_c(c) (matched-query persona/instruction swaps and cross-query swaps; prefix-based AND context-based arms; layer + position sweeps), generate on-policy, and measure (a) the realized mean answer activation's shift from V_a(c) toward V_a(c') and its agreement with the fitted map's counterfactual prediction f(V_c + Delta), and (b) the judge-scored behavior shift (graded 0-100 primary + rate companion), against norm-matched random-Delta and shuffled-pair nulls, a full-prefix KV-swap ceiling, and the persona-vector r_B steering baseline, with a logit-lens top-token readout of V_c, V_a, Delta, and the #922 slow modes as descriptive companion.

## Overview / Motivation

The context→answer mapping line (#779, #823, #841, #922, #1092) is entirely correlational: every read is a probe / regression / teacher-forced predictability claim (#823 flags this explicitly — "representation-level claim, not an on-policy behavioral one"). The open question is whether the two vectors the line is built on have causal standing:

- **V_c** = last-context-token residual-stream activation (per layer) — the context summary the maps take as input.
- **V_a** = mean residual-stream activation over the answer span (the "answer profile") — the map target.

The causal test: take a context c, intervene on its state with Δ = V_c(c′) − V_c(c), generate on-policy, and ask whether the realized answer state shifts from V_a(c) toward V_a(c′) — and whether behavior shifts with it. A descriptive companion asks what the vectors decode to under the logit lens.

## Evidence context (what is already known, all read-out-level)

- V_c shifts are interpretable: appending a behavioral instruction moves V_c along one dominant direction (cosine 0.75–0.90 to a single direction at layers 7–21) aligned with the response-space behavior direction (#685) — but the behavioral change there is caused by the instruction *text*, not by any vector intervention.
- V_a carries trait content: the persona direction sits at the 99.7–99.9th variance percentile of the answer profile, held-out per-direction R² 0.79–0.87 (#779); the rolled map's slow modes concentrate trait directions and context identity 30–125× above a random-subspace null (#922).
- Sharp caveat: the fitted V_c→V_a map reads *answer-content match*, not self-generation dynamics — a content-identity baseline beats the context→answer map, and each fitted map is style-specific (#823). A large part of the map's R² may therefore not transport under intervention.
- The query-bearing state dominates: the query factor carries 78–94% of trait-direction variance; the prefix factor ≤5% (#1092).
- Causality untested in-house: open-questions 1.4 (steering) is LOW/untested; #816 (Persona Vectors steering / preventative-steering reproduction, on_hold) holds the coherence-gated steering recipe that serves as the positive baseline here.

## Design sketch (planner refines)

1. **Vectors.** V_c(x) per layer at the last context token; V_a(x) = mean answer-span activation (the #779/#823 definitions). BOTH mapping arms per the standing rule: prefix-based (prefix-only last token) AND context-based (prefix+query last token).
2. **Interventions.** Per context pair c → c′: add Δ = V_c(c′) − V_c(c) at the last context token (position sweep: last token only vs all answer positions during generation); matched-query pairs (same query, different persona/instruction — the #685 construction) vs cross-query pairs; layer sweep centered on the 7–21 band (#685).
3. **Ceiling and floors.** Full-prefix KV-cache swap as the ceiling (actually changing the context); norm-matched random-Δ and shuffled-pair-Δ nulls; persona-vector r_B steering as the literature-anchored positive control (#816 recipe, coherence-gated strength per arXiv 2507.21509).
4. **DVs (dual-DV rule).** (a) Geometric: projection of the realized steered answer profile onto (V_a(c′) − V_a(c)), plus agreement with the fitted map's counterfactual prediction f(V_c + Δ) vs f(V_c) — the map-transportability test. (b) Behavioral: graded 0–100 judge score primary + binary rate companion (standard Sonnet judge), with a coherence gate on generations.
5. **Descriptive companion (near-zero GPU).** Logit-lens / top-token unembedding of V_c, V_a, Δ, the map fixed point, and the #922 slow modes (top-k promoted tokens per vector).

## Hypotheses

- **H1 (sufficiency):** last-token steering moves answer state and behavior in the predicted direction but under-delivers vs the KV-swap ceiling — quantifying how much of the context's causal effect routes through the last-token state vs the earlier-position KV cache. This is the causal version of open-q 1.1 ("can a context be treated as a vector?").
- **H2 (transportability):** steered answers land closer to f(V_c + Δ) than to f(V_c); per #823, expect partial transport (the content-indexed component of the map's R² may not move under intervention).
- **H3 (pair construction):** matched-query Δ transports better than cross-query Δ (per #1092's query-factor dominance).

## Closest prior work to verify at planning (lit-review-first rule)

Persona Vectors steering (arXiv 2507.21509); function vectors (arXiv 2310.15213); task vectors (arXiv 2310.15916); In-Context Vectors (arXiv 2311.06668); activation-patching methodology. The function-vector / ICV literature is exactly "context as a vector, causally transplanted" — the planner names where this design differs (persona/behavior contexts, answer-profile DV, map-transportability test).

## Relations

- Anchors: q:spec-context-as-vector (1.1), q:spec-steering (1.4).
- Siblings: #685, #779, #823, #841, #922, #1092.
- #816 (on_hold) is the steering-recipe baseline; consider reviving alongside or folding its steering leg in here.
