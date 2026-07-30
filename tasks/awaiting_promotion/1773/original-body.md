---
title: 'SAE feature description + categorization pipeline: evidence builder, five
  judged axes, and validation harness'
kind: experiment
tags: []
created_at: '2026-07-28T21:22:22Z'
has_clean_result: false
parent_id: 1482
origin_prompt: 'deep literature review on autointerpretability methods to describe
  SAE features -> design a pipeline for describing + categorizing along axes; Batch
  API + Claude Sonnet 4.5; then: is this running already and will the running SAE
  feature experiments use it already?'
workflow: v1
goal: Build and validate a production pipeline that describes and categorizes SAE
  features (mechanical axes + five judged axes + a validation harness), producing
  a per-feature table every map round joins against, and determine whether the judged
  axes are trustworthy enough to carry a headline or remain a search index only.
relates_to:
- spec-context-as-vector
---
## Goal

Build and validate a production pipeline that describes and categorizes SAE features (mechanical axes + five judged axes + a validation harness), producing a per-feature table every map round joins against, and determine whether the judged axes are trustworthy enough to carry a headline or remain a search index only.

## Overview / Motivation

Full design: `docs/lit_reviews/autointerp-pipeline-plan.md` @ 92c73f9b98 (five-agent literature sweep, notes in `docs/lit_reviews/notes/autointerp-rq{1..5}-*.md`, every arXiv id resolved in-session).

The three findings that force the design:
1. Explainer capability is SATURATED — Claude 3.5 Sonnet (0.75 fuzz/0.75 detect), Llama-3.1-70B (0.76/0.74) and a human annotator (0.75/0.74) are indistinguishable against a 0.51 random-explanation floor (2410.13928). Evidence design is the only remaining lever.
2. Input-plus-output evidence is the largest measured gain: 56.6 input / 49.2 output (max-activating alone) and 50.1 / 56.5 (logit-lens alone) vs 66.6 / 64.9 combined (2501.08319); the output side is one matrix product per feature.
3. Labels are a SEARCH INDEX, not evidence — top-scoring explanations re-evaluate at F1 ~0.6 with little causal efficacy (2309.10312); explanation-driven simulation of a layer is statistically indistinguishable from zero-ablating it (2501.18838). Every headline stays on the mechanical axes.

## Scope (phases, per the plan doc)

- Phase 0 — mechanical axes over all features (logit footprint, density, within-answer + cross-query persistence, nuisance load incl. the scaffold-projected r_B read, decoder neighbours, Matryoshka tier, per-arm variance shares). Free, no LLM.
- Phase 1 — THE BUILD ITEM: an all-features evidence builder. 40 quantile-stratified activating windows (32 tok, delimiter-marked peak, sink/BOS excluded) + 20 non-activating + 5 near-miss neighbour windows + the output-token block + a statistics block. The existing `phase_scan` builds evidence for 300 sampled features and cannot be scaled by parameter change (it keys its candidate buffer on a -1-filled samp_pos array).
- Phase 2 — description via Batch API + claude-sonnet-4-5-20250929, 1 draw, NOT length-capped.
- Phase 3 — five axes (abstraction / speaker_property / content_type / functional_role / interpretable), ONE AXIS PER CALL, forced single choice, label order permuted, N=5 draws majority vote. Per-axis input matrix + the two blinding rules (blind to the DV; blind each axis to its own mechanical validator) are in plan doc section 3b.
- Phase 4 — validation: detection + fuzzing on a ~1k stratified sample, a neighbour-distractor DISCRIMINATION score (detection is provably invariant to label collision), shuffled-label AND random-init controls, the alt-test on a human-annotated subset, per-axis non-judge validators, and PRECISION AT LOW BASE RATE as the headline metric for identity-disposition (comparable target classes run 1-8%).

## Cost

Batch API, Sonnet 4.5: 16,384 restricted features ~= $1.2-1.9k all-in (describe + 5 axes x 5 draws + validation sample). 131,072 full dictionary ~= $9-15k, which is the cell that would force a local open-weight explainer instead. Mechanical axes ~0. Evidence building is one streamed pass per dictionary (no API).

## Build vs adopt

Build only the evidence builder. KEEP the in-house judge dispatch unchanged (Batch routing, rubric-keyed caching, transport-vs-content drop split, the 2,000-item shard ceiling learned from a ~9h starvation incident). ADOPT Delphi's detection/fuzzing scorers IF a custom BatchTopK encoder registers without forking — resolve by reading `delphi/sparse_coders/` + `delphi/__main__.py` BEFORE committing to that leg. Adopt the Neuronpedia export as a free auxiliary baseline only (median one word, token-level, different corpus).

## Acceptance criteria

1. Phase 0 table emitted for the layer-19 restricted dictionary and joinable by feature id.
2. Evidence builder runs over all 16,384 restricted features within one streamed pass, with a recorded per-feature evidence manifest.
3. Phase 2 + 3 complete for 16,384 features with per-axis test-retest kappa reported, drops split content-vs-transport, 0 coerced returns.
4. Phase 4 reports, per axis: detection + fuzzing + discrimination scores on the sample, both controls, and precision at low base rate for identity-disposition.
5. A written verdict on whether the judged axes are trustworthy enough to carry any headline, or remain a search index only.

## Provenance

Filed from user chat 2026-07-28 after the assistant delivered the plan and the user asked whether it was running. Origin prompt (verbatim): "do a deep literature review on autointerpretability methods to describe SAE features. Then use those to design a pipeline for both describing the SAE features and separating them into categories (along different axes as we described) -- Claude API credits are not a problem -- we want to use the batch API for this though and probably Claude Sonnet 4.5. Come back with a clear and concise plan" -> "okay is this running already and will the running SAE feature experiments use it already?"
