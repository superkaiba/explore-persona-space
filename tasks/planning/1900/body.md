---
title: 'Leakage-predictor horse race at per-prompt grain: pre-FT geometry vs base
  propensity, with answer-side mediation test'
kind: experiment
tags: []
created_at: '2026-07-30T23:28:44Z'
has_clean_result: false
parent_id: 1768
origin_prompt: 'Help me to plan this experiment with these model organisms: What is
  the best leakage predictor (8 candidate similarity reads pre/post finetuning, map-mediated
  variants); is leakage due to similarity of mean answer vectors with context similarity
  a byproduct; any other suggestions'
workflow: v1
goal: 'On the model-organism fleet at per-prompt grain (the #1768 16,400-real-user-prompt
  corpus), determine which per-context predictor computable BEFORE fine-tuning best
  predicts per-context leakage of the trained behavior (graded judge DV for content
  arms; three-space log P(marker) for marker arms), whether base behavioral propensity
  survives as the champion, and whether context-vector similarity''s predictive power
  is mediated by answer-side similarity; post-FT and delta candidates form a separate
  mechanistic panel, and every similarity candidate is computed against both the training-context-centroid
  and panel-source-context anchors.'
relates_to:
- leak-predictor
- spec-context-as-vector
---
# Leakage-predictor horse race at per-prompt grain: pre-fine-tuning geometry candidates vs base propensity, with an answer-side mediation test

## Provenance

- origin_prompt (verbatim): "Help me to plan this experiment with these model organisms: What is the best leakage predictor: context vector cosine similarity pre-finetuning / context vector cosine similarity post-finetuning / answer vector cosine similarity pre-finetuning / answer vector cosine similarity post-finetuning / similarity between change in context vector pre and post finetuning / similarity between change in answer vector pre and post finetuning / context vector -> apply mapping -> predicted answer vector similarity pre-finetuning / context vector -> apply mapping -> predicted answer vector similarity post-finetuning / Is leakage due to similarity of mean answer vectors and so context vector similarity is just byproduct / Any other suggestions?"
- User clarifications (2026-07-30 chat): grain = per-prompt over the real-user corpus (reuse #1768 stores); behaviors = content + marker (marker via the three-space log-P DV, small TF GPU pass); headline = deployable pre-FT race, post-FT/delta candidates as a separate mechanistic panel; anchors = BOTH training-context centroid AND panel source context, side by side.

## Goal

On the model-organism fleet at per-prompt grain (the #1768 16,400-real-user-prompt corpus), determine which per-context predictor computable BEFORE fine-tuning best predicts per-context leakage of the trained behavior (graded judge DV for content arms; three-space log P(marker) for marker arms), whether base behavioral propensity survives as the champion, and whether context-vector similarity's predictive power is mediated by answer-side similarity; post-FT and delta candidates form a separate mechanistic panel, and every similarity candidate is computed against both the training-context-centroid and panel-source-context anchors.

## Candidate roster

**Deployable panel (pre-FT inputs only; headline race):**

| # | Predictor | Definition (per probe context x, anchor A) | Note |
|---|---|---|---|
| P1 | ctx-sim pre | cos(c0(x), A_ctx) | the classic geometry read |
| P2 | ans-sim pre | cos(v0(x), A_ans) | the mediation counterpart |
| P3 | map-mediated sim pre | cos(M0 c0(x), M0 A_ctx) and cos(M0 c0(x), A_ans) | "context sim as the map sees it" — the structural mediation instrument |
| P4 | whitened gate sim | whitened dot(c0(x), A_ctx), corpus second moment | the #1768 gate read (known weak, median 0.14 — in-family control) |
| P5 | r_B projection direct | (v0(x) − v̄0) · r_B | activation-side propensity |
| P6 | r_B projection through map | (M0 c0(x) − M0 c̄0) · r_B | #1739's predictor transplanted to the leakage target |
| P7 | base behavioral propensity | graded judge score of the BASE model's completion on x | the incumbent champion (#500/#532/#541 line) — must be in the race AND partialled as covariate |
| P8 | write-map cross-arm prediction | ŵ(x) from the #1768 write-predictability map fit on OTHER arms, its magnitude / r_B-alignment | deployable ONLY in the cross-arm-transfer form; within-arm form goes to the mechanistic panel |

**Mechanistic panel (requires the trained model; explains, does not predict):**

| # | Predictor | Definition | Note |
|---|---|---|---|
| M1 | ctx-sim post | cos(c+(x), A+_ctx) | |
| M2 | ans-sim post | cos(v+(x), A+_ans) | |
| M3 | delta-ctx sim | cos(Δc(x), Δc(A)) | expect noise-dominated: #1768 measured median relative context movement 0.025 — keep as a registered near-null |
| M4 | delta-ans sim (matched-text) | cos(Δv_tf(x), Δv_tf(A)) | USE THE MATCHED-TEXT tree — the on-policy version is nearly the outcome itself (circularity); state this in the plan |
| M5 | map-mediated sim post | cos(M+ c+(x), ·) | |
| M6 | weights-carried write magnitude | ‖Δv_tf(x)‖ | from #1768 stores |

Every similarity candidate is computed against BOTH anchors: (a) the arm's training-context centroid (context-side and answer-side means over its actual training rows — δ-tensor provenance), and (b) the panel source context (undefined for bare-context arms; reported as N/A there). Cosine and whitened-dot variants where meaningful; the plan bounds the variant grid.

## Mediation design (the checkbox question)

Is leakage driven by answer-side similarity, with context similarity a byproduct (v ≈ Mc makes them correlated through the map)? Reads: (1) partial Spearman — leakage vs P1 controlling P2, and leakage vs P2 controlling P1, per arm; (2) the structural read — if P3 (context sim pushed through M0) recovers P2's ranking and absorbs P1's partial signal, context similarity matters only through where the map sends it; (3) BOTH partialled against P7 (base propensity), the known dominant covariate. Commonality/variance-partitioning summary per arm.

## Design / reuse / cost

- Arms: subset of the #1768 72-arm fleet (content LoRA + full-FT + marker), sized by the judge-spend lever below; arm picks recorded with criteria (span behaviors × training contexts × regimes × methods).
- Predictor inputs: ALL from existing #1768 stores (issue1768_mapshift corpus_capture{,_tf} pooled.pt; maps = the 216 fit JSONs; δ from delta_tf; r_B from #1112/#1315/#1434; gate reads from p9_units; write-map from write_predictability round). Predictors are 0 GPU-h.
- DV, content arms: graded 0–100 judge, N≥3 draws, temp>0, drop-never-coerce, max_tokens ≥300, Batch API, on the EXISTING on-policy raw completions (no new generation). Spend arithmetic is the sizing lever: full grid 48 content arms × 16,400 × 3 draws ≈ 2.4M calls (oversized); plan picks (arms × contexts × draws), e.g. 12–16 arms × ~4,000 stratified contexts × 3 draws ≈ 150–190k calls. Binary rate companion + the standing graded-vs-reference validation.
- DV, marker arms: three-space log P(marker id 83399) at the post-response slot, teacher-forced on the existing on-policy rows — needs a small GPU pass (~1–2 GPU-h) since round 1 stored activations, not token logprobs.
- Statistics: per-arm Spearman per candidate (graded target), paired bootstrap CIs over contexts; per-arm ρ distributions aggregated across arms — never raw pooling across arms (dose confound; within-arm ranking is dose-clean). Map-mediated candidates evaluated only on contexts held out from the map's fit rows (leak-through-M guard) — refit M on a split if needed. "Best predictor" is a max over ~8 candidates ⇒ selection-symmetric null / selection-rides-the-bootstrap per `.claude/rules/selection-symmetric-nulls.md`. Group-level folds (LOFO across arms) for any FITTED combination predictor per `.claude/rules/ood-generalization-folds.md`.
- Both-arms mapping rule: context-based arm only — the prefix arm is degenerate on this corpus (2 distinct prefix strings; the #1768 round-1 stated deviation carries over). If the round-3 on-target prefix stores land first, a per-prefix bridge read becomes a natural cheap extension (NOT in scope v1).
- Positioning: #1739 predicts BASE-model behavior expression through the map (evil/sycophancy/hallucination); this task predicts FINE-TUNING LEAKAGE of trained arms — same toolbox (reuse #1739's projection/regression code where merged), different target. Extends the #658/#742/#761/#763 leak-predictor line by racing geometry vs the incumbent propensity per-context at corpus scale.

## Constraints

- kind: experiment; requires /adversarial-planner before any execution.
- Judge = claude-sonnet-4-5-20250929, Batch API; judge-cache keys carry rubric fingerprint.
- No new training; no new generation for content arms; marker TF pass is the only GPU phase.
