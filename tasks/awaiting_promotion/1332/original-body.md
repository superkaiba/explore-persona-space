---
title: Function-space similarity between per-context fitted context→answer maps as
  a leakage predictor
kind: experiment
tags: []
created_at: '2026-07-15T07:06:20Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'can you add an issue based on the next step? let''s plan it here [next
  step, verbatim from the finalized off-policy mapping result''s Conclusion: ''I think
  potentially leakage could be predicted by a similarity metric between the mappings
  for these different contexts — although this seems similar to KL divergence and
  that didn''t work too well — testing this now'']'
workflow: v1
goal: Test whether function-space similarity between per-context-family fitted linear
  context→answer maps (cross-family transfer R² / prediction agreement on the frozen
  base model, both prefix-based and context-based arms) predicts fine-tuning leakage
  from source persona to target contexts on an existing measured leakage matrix, with
  incremental validity over the activation-cosine, JS-divergence, base-rate-prior,
  and whitened-gate baselines under group-level held-out (LOFO) evaluation.
relates_to:
- leak-predictor
---
## Goal

Test whether function-space similarity between per-context-family fitted linear context→answer maps (cross-family transfer R² / prediction agreement on the frozen base model, both prefix-based and context-based arms) predicts fine-tuning leakage from source persona to target contexts on an existing measured leakage matrix, with incremental validity over the activation-cosine, JS-divergence, base-rate-prior, and whitened-gate baselines under group-level held-out (LOFO) evaluation.

**Formalization (to be sharpened by the planner):** for each context family c (persona/prefix over a shared query bank), fit the per-family linear map h_c: state(c, q) → answer-state(c, q) on the frozen base model. Define the mapping-similarity S(c_i, c_j) in FUNCTION space — cross-family transfer R² (h_{c_i} scored on c_j's rows, symmetrized) and/or held-out prediction agreement R²(ŷ_i, ŷ_j) — never weight-space cosine (#823 matched-half calibration: weight cosine is estimation noise at feasible n). The hypothesis: S(source, target) predicts the measured leakage matrix L(source → target) across target contexts, held-out at the group level (LOFO), with incremental validity over the representation-similarity baselines computed on the identical rows.

## Overview / Motivation

The off-policy mapping result (docs/writeups/offpolicy_mapping_report.md @ 370510c311; #823/#952/#779 line) concluded the context→answer map reads a consistent persona/character rather than the model's own policy. Conclusion's next step (verbatim): "I think potentially leakage could be predicted by a similarity metric between the mappings for these different contexts — although this seems similar to KL divergence and that didn't work too well — testing this now."

What is genuinely new vs the retired predictors: cosine/JS/base-prior compare context REPRESENTATIONS (points); this compares the fitted context→answer TRANSFORMATIONS (maps) — "do these two contexts imply the same context→content computation." The #823/#952 transfer results show map-similarity is measurable, discriminative (style breaks it while content match preserves it), and cheap in function space.

## Design decisions (SETTLED in chat 2026-07-15 — user: 'defaults fine'; planner refines mechanics, not these choices)

1. **Context families:** the marker-leakage personas (e.g. the #474/#532 16-source loc-arm panel + their bystander/target contexts) over a shared query bank — chosen so an EXISTING measured leakage matrix is reusable as ground truth with zero retraining.
2. **Leakage ground truth (reuse, no new training):** the #474/#532 marker leakage matrices (source→target trained−base log P(marker), three-space recipe) as the primary DV; the #545 behavior→behavior testbed matrix as the transfer/OOD test of the predictor.
3. **Maps:** base-model capture per (family, query) → per-family ridge maps, both arms per the standing dual-arm rule — prefix-based (persona text only) AND context-based (persona + query) — reusing the #823/#952 ridge harness (GCV, standardize-X/center-Y).
4. **Similarity metrics (function space only):** symmetrized cross-family transfer R²; held-out prediction-agreement; plus the map-mediated displacement variant. Report against a shuffled-pairing null and the same-family split-half ceiling.
5. **Evaluation:** Spearman + regression of S(source, target) vs L(source, target), LOFO group folds (`.claude/rules/ood-generalization-folds.md`); baselines computed on identical rows: activation cosine (#404 recipe), JS (#458 recipe), base-rate prior, whitened gate (#667), predict-the-mean. Selection-symmetric nulls for any argmax over layers.
6. **Compute:** base-model capture for ~16-30 families × query bank on Qwen-2.5-7B + CPU/GPU ridge fits — rough est 3-8 GPU-h; no training.

**Settled choices (from the filing chat):**
- Families = the marker-line personas (#474/#532 16-source loc-arm panel + bystander/target contexts) as PRIMARY, reusing their measured leakage matrix as ground truth (no retraining); the #545 behavior→behavior matrix as the predictor's transfer/OOD test.
- Similarity = symmetrized cross-family transfer R² (primary) + held-out prediction-agreement (companion), each against a shuffled-pairing null and a same-family split-half ceiling. Function-space only; weight-space descriptive-only with the matched-half noise reference (#823 calibration).
- Pre-registered kill criterion: partial correlation of map-similarity with leakage CONTROLLING for activation cosine (#404 recipe) and JS divergence (#458 recipe) on the identical rows — no incremental validity over the point metrics ⇒ the predictor is a re-dressed representation metric and the headline says so.
- Query bank: one shared neutral bank of ~300–500 queries captured once across all families (map stability floor per #779's scaling curve: usable by ~250 rows, still rising at 3,600); BOTH mapping arms per the standing rule — prefix-based (persona text only; the leakage-relevant arm, since leakage targets are personas) AND context-based (persona + query).

## Constraints / notes

- Predictor-target measurement rules apply (graded/judged DV conventions inherited from the reused leakage artifacts; artifact-reuse fitness checklist binds on the reused matrices + adapters' provenance).
- Function-space similarity ONLY for the headline; weight-space reads may ride along as descriptive with the matched-half noise reference.
- Related prior negatives to position against: #406 (output divergence negatively predicts transfer), #545 (context-geometry predictors don't transfer to behavior→behavior), #500/#532/#541 (base prior beats geometry).

## Provenance

Filed from user chat 2026-07-15 while finalizing docs/writeups/offpolicy_mapping_report.md; planned interactively in-chat before any /issue run.
