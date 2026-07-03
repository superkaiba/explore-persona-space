---
title: 'Context/query decomposability of the on-policy answer mapping: M_A (context->answer)
  + M_B (query->answer) vs M_C (context+query->answer)'
kind: experiment
tags: []
created_at: '2026-07-03T10:36:02Z'
has_clean_result: false
parent_id: 658
origin_prompt: 'Help me to plan this:


  We want to see if there is some decomposition of single query mapping into the "context"
  portion and the "query" portion


  For this we can:

  - directly predict the answer just from the context portion

  - directly predict the answer just from the query portion (with blank context -->
  make sure to not insert system prompt -- try empty system prompt but also removing
  the system prompt part of chat template completely......, also potentially just
  masking out the tokens of the context with the query in there)

  - context and query must be disjoint token sets

  See if by some combination of these 2 mappings we can get better performance

  Also train the context + query -> answer mapping and analyze the relationship between
  this mapping and the 2 other mappings


  probably good to use our diverse contexts + ultrachat queries setup so you get:

  M_A: context -> answer

  M_B: query -> answer

  M_C: context + query --> answer

  with matched contexts and queries

  All generations should be on policy

  Evaluate mappings on LOFO context + OOD queries'
workflow: v1
goal: 'Determine whether the (context, query) -> answer conditional of Qwen-2.5-7B-Instruct
  decomposes additively into context-only and query-only components: train three matched
  LoRA mappings on on-policy base-model answers over a diverse-contexts x UltraChat
  grid (M_A: context->answer; M_B: query->answer with blank context; M_C: context+query->answer)
  and measure (1) how much of the [best-single-mapping -> base-full-conditional] cross-entropy
  gap a combination of M_A and M_B closes on held-out cells (LOFO context families
  x held-out/OOD queries; primary DV: length-normalized teacher-forced cross-entropy
  of fresh on-policy reference answers; secondary DV: judge-scored behavioral-profile
  agreement), and (2) the relationship of M_C to {M_A, M_B} in function space and
  weight space (per-layer regression of dW_C onto span{dW_A, dW_B}).'
---
## Overview / Motivation

Is the base model's context-conditioned answer mapping (context, query) → answer decomposable into a context-only portion and a query-only portion? If a combination of two partial-input trained mappings recovers most of the full mapping's predictive performance on held-out cells, the context contribution is a separable factor of the conditional answer distribution — direct mechanistic support for treating base-side context structure as a standalone driver in the leakage-prediction line (#658/#742/#761/#763/#810) and the pre-fine-tuning context-geometry theory (Overleaf leakage-theory paper). If it does not decompose, the interaction residual is itself the finding: the mapping's context×query coupling is where the full model earns its performance.

## Goal

Determine whether the (context, query) -> answer conditional of Qwen-2.5-7B-Instruct decomposes additively into context-only and query-only components: train three matched LoRA mappings on on-policy base-model answers over a diverse-contexts x UltraChat grid (M_A: context->answer; M_B: query->answer with blank context; M_C: context+query->answer) and measure (1) how much of the [best-single-mapping -> base-full-conditional] cross-entropy gap a combination of M_A and M_B closes on held-out cells (LOFO context families x held-out/OOD queries; primary DV: length-normalized teacher-forced cross-entropy of fresh on-policy reference answers; secondary DV: judge-scored behavioral-profile agreement), and (2) the relationship of M_C to {M_A, M_B} in function space and weight space (per-layer regression of dW_C onto span{dW_A, dW_B}).

## Design sketch (pre-plan; /adversarial-planner refines every value)

**Data grid.** Reuse the diverse-contexts × UltraChat setup from the predictor line: contexts c_i grouped into families (the LOFO fold unit, per `.claude/rules/ood-generalization-folds.md` — group-level folds, never pointwise LOO), UltraChat queries q_j with train / held-out splits, plus one genuinely-OOD query corpus as the harder transfer read. Context and query occupy disjoint token spans (context = system turn, query = user turn, never interleaved), which makes the masking presentation below well-defined.

**Reference answers (on-policy).** a_ij ~ π₀(·| c_i, q_j) sampled on-policy from the base model over the training grid; FRESH samples drawn on the eval grid (never reuse training-grid answers for eval). Sampling temperature and samples-per-cell grounded at plan time (§11). All rollout text persists to the HF data repo per upload policy.

**Trained mappings — matched LoRA recipe, same cells, same answers, single-variable difference = input presentation:**
- M_A: input c_i only (no query) → a_ij. Learns the query-marginal p̂(a|c).
- M_B: input q_j only (no context) → a_ij. Learns the context-marginal p̂(a|q). The null-context presentation is itself ablated 3 ways: (i) chat template with explicitly empty system prompt (note: the Qwen template silently inserts a default system prompt when none is given — must be suppressed, not defaulted); (ii) system-prompt block removed from the chat template entirely; (iii) full (c,q) token sequence with the context-span tokens attention-masked in place (query positions preserved). Train+eval presentation always matched; run the 3-way ablation on one fold first, carry the representative presentation into the full sweep.
- M_C: input (c_i, q_j) → a_ij. The full mapping (self-distillation of the base conditional); anchor for the weight-space analysis.

**Combination test (function space primary, weight space secondary).**
- Function space: per-token log-linear / product-of-experts combination, e.g. ℓ_comb = ℓ_B(q) + λ·(ℓ_A(c) − ℓ_uncond), with λ swept on a validation fold (never fit on eval cells); additive-mixture control α·p_A + (1−α)·p_B.
- Weight space: apply ΔW_A + ΔW_B merged; per-layer least-squares regression of ΔW_C onto span{ΔW_A, ΔW_B} (R², cosines); note the merged model's input-presentation ambiguity (trained on different input formats) is an acknowledged interpretive caveat — the regression/geometry read is the primary weight-space result.

**Evaluation.** Held-out grid = LOFO context family × {held-out UltraChat, OOD corpus} queries. PRIMARY DV: length-normalized teacher-forced cross-entropy of fresh on-policy reference answers under each mapping/combination — a proper scoring rule, so mapping comparisons estimate KL-to-π₀(·|c,q) differences; this is a prediction/regression construct, so the continuous log-prob DV is the natural primary and the judge-scored read is the companion (mirrors the graded-primary-for-ranking-targets extension). SECONDARY DV: judge-scored behavioral-profile agreement (project judge claude-sonnet-4-5-20250929) between each mapping's free generations and π₀(·|c,q), per context family. Baselines: untrained partial-input reads π₀(·|c), π₀(·|q) (what training buys), unconditional answer model (floor / correction term), base full conditional π₀(·|c,q) (ceiling by construction).

**Headline quantity.** Decomposability index D = [CE(best single) − CE(best combined)] / [CE(best single) − CE(π₀ full)] on LOFO × OOD cells — the fraction of the single→full gap closed by combining — plus the interaction residual CE(best combined) − CE(π₀ full) and the M_C relationship reads.

## Hypotheses

- H1 (additive): the best M_A⊕M_B combination closes most of the gap (D high) — the context portion is a separable factor of the conditional.
- H2 (interaction-dominant): combination ≈ best single (D low) — the mapping does not factorize; the residual is context×query interaction.
- H3 (consistency): weight-space containment of ΔW_C in span{ΔW_A, ΔW_B} tracks the function-space D (the two decomposability reads agree).

## Assumptions from the planning chat (user can override any of these)

- **Disjointness** read as disjoint token SPANS (system turn vs user turn, no interleaving), not literal vocabulary disjointness and not topic filtering. (Asked 2026-07-03, no response within window — recommended default taken.)
- **Combination** lives in function space (primary) + weight space (secondary analysis), per the design sketch.
- **OOD queries** = held-out UltraChat AND one genuinely-OOD query corpus, both crossed with LOFO families.

## Open design points (clarifier/planner)

- Sampling temperature + samples-per-cell for a_ij; grid sizes (n contexts, n families, n queries); LoRA recipe grounding with `Source:` per hyperparameter (§11).
- Whether M_A's "no query" presentation needs the same 3-way null-presentation ablation as M_B (recommend: yes, but on one fold only).
- Exact λ/α-fitting protocol (validation fold definition inside the LOFO scheme; selection-symmetric handling if any argmax-over-λ appears in a headline vs null band, per `.claude/rules/selection-symmetric-nulls.md`).
- OOD query corpus choice (Dolly/WildChat-style; ground at plan time).
- Query-bank size: the existing UltraChat pool is 48 probes (built to 1:1-match the Betley probes); TRAINING M_B (query→answer) and reading OOD-query generalization likely needs a substantially larger query set — `issue594_build_probes_ultrachat.py` already streams 20k UltraChat rows, so extending the filter to a few hundred–few thousand queries is cheap; planner sizes the grid (50 ctx × n_q) jointly with compute.
- Contrastive-negatives rule: this is a mapping-decomposition/distillation experiment, not a persona-behavior implant — planner states the exemption argument explicitly (M_B trained on context-marginal answers does shift default-context behavior; that shift is measured, not a confound, but the critic should stress-test this).
- Compute sizing: rough order 25–40 GPU-h (grid generation via vLLM + ~15–20 LoRA trainings across folds/ablations + batched teacher-forced scoring); planner sizes §9 properly, vectorized scoring mandatory.

## Candidate prior work (abstract-verified 2026-07-03 via arXiv MCP; planner re-verifies and extends)

- 2209.15189 — Learning by Distilling Context. The M_A/M_B construction IS context distillation (internalize a context into weights via self-distillation); never trains matched partial-input models to test additive combination.
- 2602.12275 — On-Policy Context Distillation. Closest to the on-policy answer-sampling choice; single-model internalization, no decomposition test.
- 2304.08467 — Gist tokens. Soft-prompt analogue of context internalization; compression for efficiency, no separability claim.
- 2211.12485 — HyperTuning; 2602.06358 — SHINE. Context→adapter hypernetworks (M_A as a learned context→weights map); no query-only twin, no composition test.
- 2212.04089 — Task Arithmetic; 2305.12827 — Task Arithmetic in the Tangent Space. The weight-space additive-composition machinery + the only existing theory of when two fine-tune deltas compose (weight disentanglement); cross-task, not a context/query factorization of one conditional.
- 2105.03023 — DExperts; 2307.03214 — PREADD. Decode-time logit combination templates; PREADD literally combines a raw-prompt and prefix-prompt distribution, but for attribute control from the same model, never testing recovery of the full joint conditional.
- 2310.15916 — In-Context Learning Creates Task Vectors; 2212.07677 — Transformers learn in-context by gradient descent. Representational/mechanistic context-vs-query factorization claims for ICL; not a trained-model decomposability test of the answer distribution.
- Sibling persona papers 2507.21509 / 2506.19823: decompose behavior along persona/trait axes in activation space — neither addresses context-vs-query decomposition of the answer distribution.
- Bottom line: the ingredients exist separately; the matched-partial-LoRA combination test with held-out-family × held-out-query generalization and a measured interaction residual appears unanswered — genuinely open.

## Reusable artifacts (repo scout 2026-07-03; planner runs the artifact-reuse (a)-(i) fitness check on each)

- Context bank (the LOFO unit): `scripts/issue594_common.py` — `load_battery()` on `data/issue594/battery.json` (HF mirror `issue594_context_geometry/inputs/battery.json`); 50 contexts, seed-42 deterministic, 7 families = persona 14 / WildChat 10 / ICL 8 / rephrase 6 / format 5 / behavior 5 / default 2 — ready-made LOFO groups; `messages_for_instance(instance, probe)` already composes the context+query chat input; `DEFAULT_MODEL = Qwen/Qwen2.5-7B-Instruct`.
- UltraChat queries: `data/issue594/probes_ultrachat.json` (48 probes; built by `scripts/issue594_build_probes_ultrachat.py` from HuggingFaceH4/ultrachat_200k with clean-single-turn-English filtering, length-matching to the 48 Betley probes, and disjointness asserts). 48 is likely too thin to TRAIN M_B — see the query-bank-size open point.
- Cached on-policy base generations: HF `superkaiba1/explore-persona-space-data:issue658_theory_assumptions/{store,raw_completions}` — 50 ctx × 48 probes, Betley + UltraChat arms, judged completions. Candidate training-grid answer source if the fitness check passes; fresh eval-grid samples are still required either way.
- On-policy generation pattern: `scripts/issue658_extract_base_store.py` (vLLM batched generate over the context×probe grid, then teacher-force back through HF).
- Teacher-forced LN log-prob scoring: `src/explore_persona_space/eval/margin.py` — `score_answer_logprobs_batched()` (L124), `compute_tf_margin()` (L233); OOM-safe fp32 log-softmax, dynamic batching.
- Vectorized fold fits: `src/explore_persona_space/analysis/vectorized_mlp_skill.py`; canonical LOFO fold implementation: `scripts/issue778_null_battery.py:149` `_leave_one_family_out(...)`.
- Setup documentation of record: #658 body `## Methodology` (the 50-ctx/7-family × 48-probe grid), #810 body (the LOFO rationale — pointwise-LOO headlines were within-family interpolation artifacts).

## Provenance

- origin: user chat 2026-07-03 (verbatim originating prompt recorded via --origin-prompt)
- planning assumptions taken after AskUserQuestion timed out; all three marked overridable above
