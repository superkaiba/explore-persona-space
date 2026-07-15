---
title: Does RLVR post-training change the base→post-trained context→answer map more
  than SFT/DPO? (separated-stage ladder)
kind: experiment
tags: []
created_at: '2026-07-15T08:33:12Z'
has_clean_result: false
parent_id: 825
origin_prompt: 'Help me to run an issue to test the RLVR part of next steps here:
  [#825 report — Next Steps: could check if doing RLVR changes the base -> instruct
  mapping more (the model I''m using is too old to have done RLVR)]'
workflow: v1
---
# Does RLVR post-training change the context→answer map more than SFT/DPO?

## Goal

Determine whether RLVR-style RL post-training changes the linear context→answer-profile map more than SFT/DPO post-training, using a released post-training stage ladder with separated checkpoints (primary candidate: Llama-3.1-8B base → Tulu-3-8B-SFT → Tulu-3-8B-DPO → Tulu-3-8B final, where the final stage is RLVR; arXiv 2411.15124 coined RLVR and released all stages). Two dependent reads per stage, both computed with the #779/#825 recipe: (a) within-stage held-out R² of the per-example ridge map c_x → v(x); (b) the reparameterization test (#825 Result 2): fit a general linear change of coordinates between the base model and stage-k on identical text and test whether the base map, reparameterized, matches the stage-k map's predictive power. The question "does RLVR teach or elicit?" is operationalized as: does the reparameterization gap (within-stage R² minus reparameterized-base R²) grow specifically at the RLVR stage, relative to the SFT and DPO stages?

## Background (parent line)

- #779: linear map from a single context vector to the mean answer vector in Qwen2.5-7B-Instruct, held-out R² ≈ 0.7.
- #825: the map already exists in pretrained Qwen2.5-7B at ~87% of instruct strength (0.588 vs 0.673 at the shared best layer); post-training reparameterizes the existing map by a general linear map (reparameterized base map matches the instruct map's predictive power on instruct text); the map survives chat-template removal; it does not hold for generic stories, the user turn, or generic next-span prediction.
- Open question this task answers (#825 Next Steps): Qwen2.5 predates RLVR, so the elicitation-not-teaching read is confined to SFT/DPO-era post-training. If RLVR genuinely teaches new capabilities, the stage-k map after RLVR should NOT be reconstructible from the base map by a linear change of coordinates.

## Hypotheses

- H-elicit: reparameterized-base R² ≈ within-stage R² at every stage including post-RLVR — RLVR adds no linearly-new map structure (extends #825's elicitation read to RL post-training).
- H-teach: the reparameterization gap grows specifically at the RLVR stage (final vs DPO delta exceeds the SFT and DPO deltas).
- H-generic: the gap grows uniformly with post-training depth regardless of stage type (post-training dose, not RLVR specifically).

## Proposed design (planner refines; single-variable ladder)

- **Models (4):** the base + three stage checkpoints of one released ladder. Primary candidate: `meta-llama/Llama-3.1-8B`, `allenai/Llama-3.1-Tulu-3-8B-SFT`, `allenai/Llama-3.1-Tulu-3-8B-DPO`, `allenai/Llama-3.1-Tulu-3-8B` (post-RLVR). Planner verifies checkpoint availability + recipe provenance on HF before committing; if a better separated-RLVR ladder exists (e.g. a newer release with an isolated RLVR stage on a stronger base), ground the swap in the plan.
- **Data:** the same 5,000 LMSYS user turns as #825 for comparability; answers generated on-policy per model (vLLM). Because Tülu-3's RLVR stage trains on verifiable domains (math/precise-instruction), ADD a verifiable-domain prompt subset (e.g. GSM8K questions) as a second eval distribution — the RLVR-specific map change may be domain-local.
- **Mapping arms (BOTH, per standing rule):** prefix-based (activation at the end of everything before the user query) AND context-based (activation at the end of prefix + user query). Report both; neither is dropped silently.
- **Recipe (reuse #825/#779 pipeline verbatim):** v_C = activation at end of context (per arm above); v_A = mean activation over the answer span; layer sweep in each model; 5-fold held-out ridge; variance-weighted R² over hidden dims; shuffled context/answer-pairing baseline.
- **Reparameterization test:** fit context-side and answer-side linear change-of-coordinates between base and stage-k on identical text (same construction as #825 Result 2), then evaluate the reparameterized base operator on stage-k text vs the stage-k operator.
- **Template arms:** chat-templated + plain "User:/Assistant:" transcripts (as in #825 Result 3), at least for the context arm.

## Measurement / DVs

- Primary: held-out variance-weighted R² per (stage, arm, layer, eval-distribution), with the shuffled-pairing baseline band.
- Headline contrast: reparameterization gap per stage — Δ_k = R²_within(stage k) − R²_reparam-base(stage k) — and whether Δ_RLVR > Δ_DPO, Δ_SFT beyond fold variability.

## Scope caveats to carry forward

- Base-family change (Llama vs Qwen) means absolute R² is not comparable to #825; the DV is the within-ladder stage deltas.
- Tülu-3's RLVR is a narrow-domain final stage; a null on LMSYS-general prompts with an effect on the verifiable-domain subset is an expected, reportable outcome, not a failure.

## Compute estimate

~15–25 GPU-h: 4 models × (5k on-policy generations via vLLM + teacher-forced activation extraction, 1 GPU each) + CPU-side vectorized ridge/reparameterization fits (reuse the batched fit helpers; no serial per-cell loops).

## Provenance

- origin: user chat request on #825's Next Steps (RLVR item), 2026-07-15.
