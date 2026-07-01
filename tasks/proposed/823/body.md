---
title: Does the base context→answer-profile map predict the model's own output or
  context-side processing? (external-answer refit)
kind: experiment
tags: []
created_at: '2026-07-01T23:41:29Z'
has_clean_result: false
parent_id: 722
origin_prompt: "## Motivation\n- We showed that there is a mapping from context vector\
  \ to answer profile -> when the answer was generated from that context\n- It is\
  \ also interesting to see if that mapping holds for answers not generated from the\
  \ model\n    - i.e. is the model just predicting what its assistant will say\n \
  \       - or somehow predicting in general the activations\n    - what I'm trying\
  \ to get at I think is is the model really just predicting its internals or also\
  \ the external output\n        - if it can predict the answer profile for output\
  \ generated from another model then this indicates it's not really predicting the\
  \ external\n    - This is also interesting to characterize the finetuned mapping\
  \ M+ -> should we be measuring it on the generated responses? Currently we are measuring\
  \ it on the pre-finetuned responses\n## Methodology\n- Take pre-finetuned model\n\
  - Compute:\n    - context vector -> answer profile mapping for answers generated\
  \ from the pre-finetuned model\n    - context vector -> answer profile mapping for\
  \ answers generated from Claude 4.5 Sonnet (with weird behavior prompt so that answers\
  \ aren't too similar)\n- Use held-out evaluation context setup from 722\n- Measure\
  \ (at all layers):\n    - which mapping is a better predictor on the held-out contexts\n\
  \        - if own answer mapping is better -> model is actually predicting its output\n\
  \        - if they are both similar -> model is just predicting its internals\n\
  \    - can one mapping predict the other? --> almost definitely not but why not\
  \ check\n\n[Design decisions confirmed in chat 2026-07-01: three external arms —\
  \ Sonnet with context C + weird-style instruction, Sonnet with context C plain,\
  \ and same-answer-text-across-contexts (bare-probe Sonnet); new child task of #722;\
  \ M+ re-measurement on finetuned-model generations out of scope (named follow-up);\
  \ no MLP training anywhere (ridge only); keep the Betley-pinned 48-probe pool (not\
  \ WildChat/UltraChat).]"
goal: 'Determine whether the strong base context→answer-profile map (c_C → v0, held-out
  skill-over-mean R² 0.74–0.80 in #722) predicts the model''s own output content or
  merely context-side processing of arbitrary answer text, by refitting the identical
  LOCO-ridge harness on answer profiles computed from externally-generated (Claude
  Sonnet) and context-blind fixed answers teacher-forced through the frozen base model.'
---
# Does the base context→answer-profile map predict the model's own output or context-side processing? (external-answer refit)

## Goal

Determine whether the strong base context→answer-profile map (c_C → v0, held-out skill-over-mean R² 0.74–0.80 in #722) predicts the model's own output content or merely context-side processing of arbitrary answer text, by refitting the identical LOCO-ridge harness on answer profiles computed from externally-generated (Claude Sonnet) and context-blind fixed answers teacher-forced through the frozen base model.

**Formalized question.** For answer source s ∈ {own, sonnet-weird, sonnet-plain, fixed}, define v_s(C) = mean over the 48 pinned probes of (mean answer-token residual-stream activation of answer a_s(C, q) teacher-forced through frozen base Qwen-2.5-7B-Instruct under context C), per layer 0–27, over the 50-context #594 battery. Fit M_s: c_C → v_s(C) with the identical corrected LOCO-ridge harness (#722 DV(0)). Compare held-out skill-over-mean R² across s, plus the cross-prediction matrix (M_s scored on v_{s′} targets).

**Competing hypotheses (pre-registered):**

- H1 (own-output prediction): R²_own ≫ R²_sonnet ≈ R²_fixed — the predictable variance is own-policy content.
- H2 (pure processing): R²_own ≈ R²_sonnet ≈ R²_fixed — answer content contributes ~nothing; the map captures context-side processing. The fixed-text arm has zero content signal by construction, so R²_fixed is the processing floor.
- H3 (general content prediction): R²_own ≈ R²_sonnet-plain > R²_fixed — the map predicts context-appropriate content from any policy, not just its own.
- Orderings such as R²_fixed > R²_own are admissible (own content may inject unpredictable variance) and interpreted accordingly.

## Overview / Motivation

#722 (DV 0, inherited from #658) established that a linear map M: c_C → v0 — from the last-input-token base context vector to the answer profile (mean answer-span residual activation, averaged over 48 probes) — is strong: held-out skill-over-mean R² 0.74–0.80 at mid-to-late layers over the 50-context battery. But v0 was always computed on answers the base model itself generated (greedy, temperature 0), so the result is ambiguous between (a) the context vector encoding what the model's own policy will say, and (b) the context vector predicting how the model processes whatever answer text appears in that context. This experiment separates the two readings with answers not generated by the model. The answer also bears on #722's M⁺ arm: whether the finetuned map should be measured on the finetuned model's own generations rather than frozen base responses (out of scope here — see Named follow-up).

## Design (4 arms; single manipulated variable = answer source)

| Arm | Answer text | Generator sees | N completions |
|---|---|---|---|
| A own (baseline) | existing frozen greedy R | Qwen under C | 0 (reuse store) |
| B1 sonnet-weird | Sonnet 4.5, weird-style instruction | C's system prompt + prefix messages + probe + style instruction (instruction NOT in the teacher-forced text) | 2400 |
| B2 sonnet-plain | Sonnet 4.5, plain | C's system prompt + prefix messages + probe | 2400 |
| C fixed | Sonnet 4.5, bare probe only; same text under every context | probe only | 48 |

All arms teacher-force through the SAME frozen base model under the SAME 50 contexts × 48 probes, with the same span/mean/probe-averaging recipe and the same c_C (input-side, hence teacher-independent — reused unchanged). B2 vs B1 isolates the style-divergence knob; C is the decisive processing-only control.

**Probe pool (pinned, all arms):** the Betley-pinned 48 preregistered neutral probes (`preregistered_evals.yaml` paraphrases excluding the Betley main-8; `data/issue594/battery.json meta.probe_pool_hash = ad687bec…`). NOT WildChat (WildChat is a context family only, `f2_wc_`) and NOT the UltraChat pool — the stored c_C is probe-dependent and Betley-pinned (`issue658_extract_base_store.py` hard-asserts `--cc-recompute-last` for any non-Betley pool), so keeping this pool licenses reusing `context_vectors_mean.pt` unchanged.

**DVs (per layer 0–27), ridge ONLY — no MLP training anywhere (explicit user directive):**

1. Held-out skill-over-mean R² per arm (LOCO ridge, full-dimension target) + label-shuffle null per arm.
2. Paired ΔR² (own − each external arm) with family-clustered bootstrap CI (resample the 7 context families, mirroring #722).
3. Cross-prediction matrix: M_s trained on arm s, scored on arm s′ targets (including the identity baseline v_s′(C) := v_own(C)).

**Validity diagnostics (report, cheap):** (i) text decorrelation — lexical/embedding similarity of B1/B2 answers vs own R (verifies the style instruction actually decorrelated the text); (ii) OOD covariate — mean answer-span log P under base Qwen per arm (rules out "external arm fails only because the text is wildly off-distribution"); (iii) cos(v_s(C), v_own(C)) distributions.

## Reuse map (verified to exist)

- Contexts + probes: `data/issue594/battery.json` via `scripts/issue594_common.py::load_battery` / `messages_for_instance` (50 instances, 7 families, probe_pool_n=48, hash-pinned).
- c_C store: HF `issue594_context_geometry/analysis_tensors/context_vectors_mean.pt` (50, 28, 3584) — loader `scripts/issue658_common.py::load_cc_last_store`.
- Own-answer baseline: `data/issue_658/store/v0_summaries.pt` (+ HF `issue658_theory_assumptions/store/`); published numbers `eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json`. Arm A is REFIT through the new driver from the stored summaries (harness identity), cross-checked against the published JSON.
- Teacher-forced extraction hook: `scripts/issue658_extract_base_store.py::capture_v0_for_context(model, tokenizer, instance, probes, completions, ...)` — already accepts external `completions`; span capture `AnswerSpanCapture`, recipe `issue658_common.py::summarize_answer_span` (`mean`), `V0_MAX_NEW_TOKENS=512`; memory-safe layer hooks `src/explore_persona_space/analysis/extraction.py::extract_layer_activations`.
- Fit harness: `src/explore_persona_space/analysis/vectorized_mlp_skill.py` — `ridge_predict_loco_centered`, `skill_over_mean_r2`, `SkillVariantSpec` (shuffle) ONLY; the module's MLP entry points are NOT used. Driver template: `scripts/issue658_proper_rb_chain.py::main()`.
- Sonnet generation: `src/explore_persona_space/llm/api_dispatch.py::dispatch_calls` (model `claude-sonnet-4-5-20250929`, multi-org, cached; ~4.9k calls total — sync fan-out territory); generation precedent `scripts/i528_phase1_generate_RPos.py`.
- Launch: `scripts/dispatch_issue.py launch --issue <N> --intent eval`, cloning `scripts/issue667_dispatch.py`'s phase/sentinel/per-cell-upload pattern.
- Storage: per-context summaries + per-(context, probe) means only (≈10 MB + ≈460 MB fp16 per arm) → HF `issue<N>_external_answer_profiles/`; raw Sonnet answers → `raw_completions/`. NEVER materialize full answer spans (#658's spans are ~142 GB; stream if ever needed).

**Cost estimate:** API ~4,850 Sonnet completions (0 GPU). GPU: 3 external arms × 2400 teacher-forced forwards with 28-layer capture ≈ 4–8 GPU-h on 1× A100 (`--intent eval`). Analysis: vectorized, minutes.

## Constraints / standing-rule notes

- Teacher-forced activation reads are the legitimate teacher-forced use (#432→#456 bars behavioral leaderboards, not span reads) — carry the flag as #722 did.
- No training rows are constructed ⇒ contrastive-negatives / on-policy-completions rules N/A.
- Sonnet-generated answer text is tier-3 by construction, but the external-LLM provenance IS the manipulated variable — justify in plan §-assumptions.
- Lit-review + Goal formalization first per the new-direction rule. Candidate anchors: LLM self-prediction / introspection (Binder et al.), the Persona Vectors A3.3 E ≈ r_Bᵀv predictor line; the planner runs the full arXiv-MCP sweep.
- Verification gate: the arm-A refit must reproduce `skill_over_mean.json` per-layer R² to ≤1e-6 (`assert_matches_reference` contract) BEFORE any cross-arm read.
- The text-decorrelation diagnostic must show B1 ≉ A; a failed style knob is reported, never hidden.

## Named follow-up (out of scope here)

- Re-fit M⁺ on answers the finetuned model itself generated (per-adapter on-policy generation over the #537 adapter fleet), replacing the frozen-base-response measurement of #722's M⁺ arm — file as a separate follow-up once this task lands.
