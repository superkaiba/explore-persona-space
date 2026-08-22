---
title: 'Context-position preventative steering: context-extracted directions steered
  at context tokens during finetuning vs the Persona Vectors method'
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-10T21:34:40Z'
has_clean_result: false
parent_id: 2221
workflow: v1
goal: Test whether persona directions extracted at the context position and applied
  ONLY to context tokens during finetuning prevent trait acquisition as well as or
  better than Persona Vectors' response-avg all-token preventative steering (arXiv
  2507.21509 §5.2), across single-layer / middle-band / all-layer conditions, at matched
  coherence and capability, with attribution cells separating extraction position
  from steering position and a probe-based off-direction acquisition check.
relates_to:
- spec-steering
---
# Context-position preventative steering: context-extracted directions steered at context tokens during finetuning vs the Persona Vectors method

## Goal

Test whether persona directions extracted at the context position and applied ONLY to context tokens during finetuning prevent trait acquisition as well as or better than Persona Vectors' response-avg all-token preventative steering (arXiv 2507.21509 §5.2), across single-layer / middle-band / all-layer conditions, at matched coherence and capability, with attribution cells separating extraction position from steering position and a probe-based off-direction acquisition check.

## Design

**Program context.** Experiment 5 of the Persona Vectors (arXiv 2507.21509) reproduce-and-beat program. The paper's preventative steering extracts directions from RESPONSE-averaged activations and adds them at ALL token positions during training (their `steering_intervention` hook is unmasked). This experiment tests the context-side alternative: extract the persona direction at the CONTEXT position and steer ONLY the context tokens during finetuning, across three layer regimes, against the paper's method at matched coherence and capability.

**Mechanistic hypothesis.** The paper's own monitoring result shows trait acquisition manifests as a shift of the context-end resting state along the persona direction, and its preventative-PROMPTING result (trait-eliciting system prompt prepended during training ≈ coef-0.5 steering) shows context-side intervention relieves the training pressure. Context-only activation steering is the continuous generalization: supply the trait signal at the source, let answer tokens compute naturally given the shifted context, so the weights never need to encode the shift. If trait pressure is context-mediated (the context→answer-map view), context-only steering should match or beat all-token steering with fewer side effects on loss-bearing completion tokens.

**Extraction variants (per persona-vectors-recipe.md, with the stated position deviation):**
- E1 (replication): response-avg extraction — the paper's recipe verbatim (7 steps, Sonnet judge carve-out).
- E2: context-end extraction — mean-diff of last-prompt-token activations under positive vs negative system prompts across the extraction questions. **Judge-filter adaptation (structural):** the paper's rollout-level filter reads responses, but all rollouts under one context share one context-end activation; adapt to a CONTEXT-LEVEL filter — keep a positive context's activation only if its rollouts' mean judged trait score is >50 (negatives symmetric, <50); per-arm dropped-context counts reported. Deviation from the paper's rollout-level filter stated in plan §-assumptions.
- E3 (both-arms rule cell): prefix-end extraction (state at end of the system prompt, before the question) — prefix-based AND context-based arms both run per the standing mapping-arms rule (prefix = everything before the user query; context = prefix + query).
- Baseline expectation to beat: the paper's Appendix A.3 position ablation found response-avg extraction beats prompt-position extraction WHEN paired with their usual steering; the novel claim here is that POSITION-MATCHING (context direction applied at context positions) changes that ordering.

**Steering-position arms (training-time hook mask):** S1 all tokens (replication of their unmasked hook); S2 context tokens only; S3 response tokens only (completes the position factorial — if S3 ≈ S1, context adds nothing; if S2 ≈ S1, the pressure is context-mediated).

**Layer arms:** L1 single layer (selected per config family by steering-effectiveness on the held-out eval set — steering regime per recipe rule step 7); L2 middle band (middle third of the stack; exact band planner-finalized); L3 all layers with layer-incremental vectors (v_l − v_{l−1}, their Appendix J.3 correction against cumulative effects — applied to context-position steering identically).

**Config roster (scoped):**
- A (repro): E1 × S1 × L1 — the paper's published single-layer method.
- B (repro): E1 × S1 × L3 — their multi-layer variant.
- C: E2 × S2 × L1 (the user's core ask, single layer).
- D: E2 × S2 × L2 (middle band).
- E: E2 × S2 × L3 (all layers, incremental).
- F (attribution, 1 dataset): E1 × S2 × L1 — response direction at context positions (isolates steering position from extraction position).
- G (attribution, 1 dataset): E2 × S1 × L1 — context direction at all tokens.
- H (anchor, 1 dataset): preventative prompting (their Appendix J.7.2 method) — the token-space sibling of S2.
- I (attribution, 1 dataset): E1 × S3 × L1 — response-only position cell.
- P (rule cell, 1 dataset): E3 × prefix-tokens-only × L1.

**Datasets:** a hard subset of the paper's suite where single-layer prevention was weakest — the three trait-eliciting II sets + Mistake Opinions II (induces all three traits) as the EM-like cell. Core configs (A–E) run on all four; attribution/anchor cells (F–I, P) on one. Real-twin extension (on #2221's suite) is a named follow-up, not in scope here.

**Protocol pins (unchanged from the paper):** Qwen-2.5-7B-Instruct; rs-LoRA r=32, α=64, lr 1e-5, 1 epoch; coefficient sweep per config (3–4 values, pilot-sized); trait expression judge-scored on the held-out 20-question sets (graded 0–100 primary + rate companion); MMLU + coherence (comparisons at matched coherence ≥80, their protocol); narrow-domain task retention for the EM-like cell.

**Off-direction acquisition read (new — the "was the trait learned anyway?" check):** trait score at baseline is not proof the trait wasn't acquired off-direction. On each steered model: (a) a linear probe sweep for trait information in activations (LINEAR only, project rule; trained on the program's judged pool, group folds); (b) the #2221 finetuning-shift monitor reads (context-end shift + mapped read). A model whose trait score is clean but whose probe/monitor reads are elevated is reported as partial prevention, never as success.

**Compute shape (plan-time to refine):** core = 5 configs × 4 datasets × ~3–4 coefficients ≈ 60–80 LoRA finetunes; attribution/anchor cells add ~15–20; each ~1–2 GPU-h ⇒ ~100–200 GPU-h total. Planner scopes the coefficient grid from a pilot; extraction forward passes and probe fits are marginal. Artifact reuse: persona vectors + extraction rollouts from the program's earlier experiments; the judged pool for probes.

## Scope caveats (carried to the clean-result)

- Synthetic-suite primary (comparability with the paper's published curves); realistic preventative steering on the real-twin suite is the named follow-up.
- Positive-only finetunes: faithful replication of the paper's positive-only design (contrastive-negatives replication exemption).
- E2's context-level judge filter is a stated adaptation of the paper's rollout-level filter (structurally forced by shared context activations), not a silent deviation.
- Single trait steered per run, matching the paper; simultaneous multi-trait prevention remains untested here (known gap, candidate follow-up).

## Provenance

Verbatim originating prompts (user, 2026-08-10):
- "i want to reproduce all the persona vectors experiments but show that we can do better with our mapping (especially on REALISTIC and not SYNTHETIC data) or by using just the context vector instead of the answer vectors. what would this look like? let's go one experiment at a time"
- "can we try extracting directions at context vector and only steering there during finetuning, both single layer, middle layers, and all layers, and compare to their method?"
