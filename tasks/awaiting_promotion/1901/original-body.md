---
title: 'Characterize mapping-quality metrics: 1M-context linear + nonlinear maps vs
  baselines on every metric'
kind: experiment
tags: []
created_at: '2026-07-30T23:49:53Z'
has_clean_result: false
origin_prompt: 'Run this in background with happy coder:

  ## Motivation

  We have come up with a bunch of different metrics for measuring the quality of the
  mapping. I want to characterize each and see how the 1m context linear and nonlinear
  mapping on them vs baselines, as well as what each metric is measuring exactly


  - [ ] [[I think a good baseline would be learned bias]], W = identity - another
  metric besides R^2 would be e.g. P(answer summary is in k nearest neighbors of prediction)'
workflow: v1
goal: Characterize every mapping-quality metric in use (held-out R², kNN retrieval
  acc@k euclidean+cosine, and companions) — what construct each measures, what it
  rules out, where they dissociate — and report the ~1M-context (n≈963k LMSYS) linear
  and nonlinear context→answer mappings against the full baseline ladder (constant
  train-mean; W = identity; identity + learned bias) on ALL metrics, in both the context-based
  and prefix-based arms.
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Characterize mapping-quality metrics: 1M-context linear + nonlinear maps vs baselines on every metric

## Motivation

We have accumulated several metrics for measuring the quality of a fitted representation mapping (held-out R², identity-family baselines, kNN retrieval), and they demonstrably dissociate (first measurements 2026-07-22: on the prefix-level 50-context battery map, identity+bias scored pooled-OOF R² −6.5 yet retrieval acc@1 0.84 vs the LOFO ridge map's 0.04 — `eval_results/issue_722/identity_bias_knn/`; on the #779 LMSYS single-context map the fitted ridge dominated retrieval, acc@1 0.72 vs 0.50 identity+bias, chance 0.001 — `eval_results/issue_779/identity_bias_knn/`). We want to characterize each metric — what it is measuring exactly — and see how the large-n (~1M-context) linear and nonlinear context→answer mappings score on each metric vs baselines.

## Goal

Characterize every mapping-quality metric in use (held-out R², kNN retrieval acc@k euclidean+cosine, and companions) — what construct each measures, what it rules out, where they dissociate — and report the ~1M-context (n≈963k LMSYS) linear and nonlinear context→answer mappings against the full baseline ladder (constant train-mean; W = identity; identity + learned bias) on ALL metrics, in both the context-based and prefix-based arms.

## Design sketch (planner refines)

- **Baseline ladder:** constant train-mean predictor → W = identity → identity + learned bias (`analysis/mapping_baselines.identity_bias_predict`, b = train-fold mean of y − x) → fitted linear (ridge) → fitted nonlinear (MLP / kernel).
- **Metric battery:** held-out R² (pooled; per-dimension summary), kNN retrieval P(true target within k nearest neighbors of prediction) (`analysis/mapping_baselines.knn_retrieval`; euclidean + cosine, k scaled to the pool, chance = k/n_pool stated — a constant predictor reads exactly chance), plus any other metrics already in use in the #722/#779 line (e.g. cosine(prediction, target)). For EACH metric: a written characterization of what it measures, its invariances, and its failure modes (scale sensitivity, what a constant predictor scores, what a high/low value can and cannot rule out).
- **Both mapping arms (project default):** context-based AND prefix-based maps evaluated on the same battery — neither arm silently dropped.
- **Reuse:** #779 has same-protocol fits banked at n=50k AND n≈963,444; identity+bias/kNN artifacts exist under `eval_results/issue_722/identity_bias_knn/` and `eval_results/issue_779/identity_bias_knn/`. Reuse per `.claude/rules/artifact-reuse.md` rather than refitting where fit-for-purpose; new compute is presumed limited to metric evaluation over banked predictions/targets.
- Group-level held-out folds per `.claude/rules/ood-generalization-folds.md` where a fold structure applies.
