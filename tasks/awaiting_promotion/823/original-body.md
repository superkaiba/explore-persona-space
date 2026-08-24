---
title: 'Does the per-context map h predict the model''s own output or context-side
  processing? (external-answer refit, #779 substrate)'
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
goal: 'Determine whether #779''s per-example context→answer-profile map h: c_last(x)
  → v(x) (held-out reconstruction R² 0.60–0.63, per-context cosine 0.93–0.96 over
  ~5000 LMSYS contexts) predicts the model''s own output content or merely context-side
  processing of arbitrary answer text, by refitting the identical 5-fold ridge harness
  on answer profiles computed from externally-generated (Claude Sonnet) and content-decoupled
  answers teacher-forced through the frozen base model.'
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
# Does the per-context map h predict the model's own output or context-side processing? (external-answer refit, #779 substrate)

## Goal

Determine whether #779's per-example context→answer-profile map h: c_last(x) → v(x) (held-out reconstruction R² 0.60–0.63, per-context cosine 0.93–0.96 over ~5000 LMSYS contexts) predicts the model's own output content or merely context-side processing of arbitrary answer text, by refitting the identical 5-fold ridge harness on answer profiles computed from externally-generated (Claude Sonnet) and content-decoupled answers teacher-forced through the frozen base model.

**Formalized question.** For answer source s ∈ {own, sonnet-weird, sonnet-plain, decoupled}, define v_s(x) = mean answer-token residual-stream profile of answer a_s(x) teacher-forced through frozen base Qwen-2.5-7B-Instruct after prompt x, per layer 0–27, over the N≈5000 pass_b LMSYS contexts. Fit h_s: cx_last(x) → v_s(x) with #779's ridge (GCV λ selection), 5-fold CV over contexts. Compare held-out pooled reconstruction R² (+ per-context cosine) across s, plus the cross-arm transfer matrix (h_s scored on v_{s′} targets).

**Competing hypotheses (pre-registered):**

- H1 (own-output prediction): R²_own ≫ R²_sonnet ≈ R²_decoupled — the predictable variance is own-policy content.
- H2 (pure processing): R²_own ≈ R²_sonnet ≈ R²_decoupled — answer content contributes ~nothing; h captures context-side processing. The decoupled arm has zero context↔content coupling by construction, so R²_decoupled is the processing floor.
- H3 (general content prediction): R²_own ≈ R²_sonnet-plain > R²_decoupled — h predicts context-appropriate content from any policy, not just its own.
- R²_decoupled > R²_own is admissible (own content injects unpredictable variance) and interpreted accordingly.

## Overview / Motivation

#779 fit a behavior-agnostic map h: c_last(x) → v(x) at per-single-context granularity — one un-averaged (last-prompt-token activation → mean-response profile) row per LMSYS context, ~5000 rows — and found held-out reconstruction R² 0.60–0.63 (`eval_results/issue_779/percontext_recon.json`, 5-fold CV; in-sample 0.83–0.86). Every target v(x) in that result is the profile of an answer Qwen itself generated, so the result is ambiguous between (a) the pre-generation context state encoding what the model's own policy will say, and (b) the context state predicting how the model processes whatever answer text follows. This experiment separates the two readings by swapping the TARGET's answer source while holding the input cx_last, the contexts, and the fit harness fixed. The answer also bears on whether #722's finetuned map M⁺ should be measured on the finetuned model's own generations (out of scope — see Named follow-ups). This task was originally designed on #722's DV(0) substrate (global 50-context probe-averaged map, n=50 LOCO) and was REBASED onto #779's per-example construct per user direction ("we are interested more in the per-context map"); the #722-based design is preserved in `original-body.md`.

## Design (4 arms; single manipulated variable = answer source)

| Arm | Answer text a_s(x) | Generation | Extraction |
|---|---|---|---|
| A own (baseline) | Qwen's own answer (pass_b `v_x` exists) | 0 | 0 (reuse) |
| B1 sonnet-weird | Sonnet 4.5 answers prompt x + weird-style instruction (instruction NOT in the teacher-forced text) | ~5000 calls | ~5000 TF forwards |
| B2 sonnet-plain | Sonnet 4.5 answers prompt x, plain | ~5000 calls | ~5000 TF forwards |
| C decoupled | Derangement of Qwen's OWN answers: x gets x′'s answer (x′ ≠ x, no fixed points), re-teacher-forced under x | 0 | ~5000 TF forwards |

All arms teacher-force through the SAME frozen base model after the SAME prompts, with the same mean-answer-span recipe and the same cx_last (input-side, hence teacher-independent — reused unchanged from the pass_b bundle). B2 vs B1 isolates the style-divergence knob; C is the decisive processing-only control.

**Arm C vs the shuffle null — a critical distinction the analysis must keep:** #779's existing shuffled-pairing null permutes the (c, v) pairing inside the FIT with no re-extraction — it destroys both content AND processing signal (R² → ~0.12). Arm C is a different object: the deranged TEXT is RE-TEACHER-FORCED under each context, so v_decoupled(x) genuinely carries context x's processing of decoupled content. C is the processing arm, not the null. The fit-level shuffle null is still run per arm as the floor.

**DVs (per layer 0–27, N≈5000 contexts, 5-fold CV over contexts), ridge ONLY — no MLP training anywhere (explicit user directive):**

1. Held-out pooled reconstruction R² per arm (SS_tot on test-fold mean, the #779 Read-1 convention) + per-context cosine, at every layer; fit-level shuffle null per arm.
2. Paired ΔR² (own − each other arm) with a context-level bootstrap CI (LMSYS rows carry no family structure).
3. Cross-arm transfer matrix: h fit on arm s (train folds), scored on arm s′ targets (held-out folds), including the identity baseline v_s′(x) := v_own(x).

**Validity diagnostics (report, cheap):** (i) text decorrelation — lexical/embedding similarity of B1/B2 answers vs own answers (verifies the weird-style knob decorrelated the text); (ii) OOD covariate — mean answer-span log P under base Qwen per arm; (iii) per-context cos(v_s(x), v_own(x)) distributions; (iv) answer-length distributions per arm (length is a nuisance for mean-pooled profiles — match Sonnet max_tokens to pass_b's generation cap).

## Reuse map (verified to exist)

- **Substrate store:** HF `issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt` (6.02 GB bundle: `cx_last`, `cx_mean`, `v_x`, each ~5000×28×3584, + the LMSYS prompt list; local mirrors `data/issue_779/`, `data/issue779_hfstage/`). `cx_last` reused unchanged; `v_x` = arm A. Sha-pin the HF revision at plan time; artifact-reuse fitness check (a)–(h) applies.
- **Fit harness:** `src/explore_persona_space/experiments/issue_779/fit_h.py::ridge_fit_predict` (GCV ridge, multi-output) + the verified fast twin `scripts/issue779_percontext_recon.py::_ridge_fit_predict_fast`; driver template `read1_heldout_recon` (5-fold CV over contexts, all 28 layers, held-out pooled R² + per-context cosine); `fit_h.reconstruction_metrics`. NO MLP entry points.
- **Teacher-forced extraction:** the per-example TF + mean-answer-span capture pattern from `scripts/issue779_collect.py` (pass_b side); arbitrary-completion primitive `scripts/issue667_extract.py::_mean_resp_acts_single`; memory-safe 28-layer hooks `src/explore_persona_space/analysis/extraction.py::extract_layer_activations`.
- **Sonnet generation:** `src/explore_persona_space/llm/api_dispatch.py::dispatch_calls` (model `claude-sonnet-4-5-20250929`, multi-org, cached; ~10k calls → sync fan-out per `docs/api_throughput_guidelines.md`); generation precedent `scripts/i528_phase1_generate_RPos.py`.
- **Launch:** `scripts/dispatch_issue.py launch --issue 823 --intent eval` (1× GPU), cloning the `issue779_collect.py` / `issue667_dispatch.py` phase/sentinel/upload contract.
- **Storage:** per-arm target bundles (`v_sonnetweird`, `v_sonnetplain`, `v_decoupled`, each ~5000×28×3584 fp16 ≈ 1.9 GB) → HF `issue823_external_answer_profiles/analysis_tensors/`; raw Sonnet answers → `raw_completions/`; derangement permutation seed + index map persisted. No full answer-span stacks.

**Cost estimate:** API ~10k Sonnet completions (0 GPU). GPU: 3 arms × ~5000 TF forwards with 28-layer capture ≈ 6–10 GPU-h on 1× A100 (`--intent eval`). Analysis: closed-form ridge, minutes.

**Coordination caveat:** #779 is at `followups_running` (training-source-ablation round live). This task only READS #779's committed pass_b store + library code — no #779 task-state mutation.

## Constraints / standing-rule notes

- Teacher-forced activation reads are the legitimate teacher-forced use (#432→#456 bars behavioral leaderboards, not span reads) — carry the flag as #722/#779 did.
- No training rows are constructed ⇒ contrastive-negatives / on-policy-completions rules N/A.
- Sonnet-generated answer text is tier-3 by construction, but the external-LLM provenance IS the manipulated variable — justify in plan §-assumptions.
- Lit-review + Goal formalization first per the new-direction rule. Candidate anchors: LLM self-prediction / introspection (Binder et al.), predictive-representation work, the Persona Vectors A3.3 predictor line; the planner runs the full arXiv-MCP sweep.
- Verification gates: (1) the arm-A refit must reproduce `percontext_recon.json` Read-1 held-out R² (0.60/0.60/0.63 at the read-out layers; full 28-layer curve within tolerance) BEFORE any cross-arm read; (2) the `_ridge_fit_predict_fast` equivalence gate re-runs in the new driver; (3) the derangement is verified fixed-point-free with the permutation persisted; (4) the text-decorrelation diagnostic must show B1 ≉ A — a failed style knob is reported, never hidden.

## Named follow-ups (out of scope here)

- Re-fit M⁺ on answers the finetuned model itself generated (per-adapter on-policy generation over the #537 adapter fleet), replacing the frozen-base-response measurement of #722's M⁺ arm.
- Read-2-style trait-projection comparison on #779's pass_a eval rig: predicted ⟨h_s(c), r_B⟩ vs true ⟨v_s, r_B⟩ per arm — asks whether the r_B-relevant component specifically is own-output or processing.
