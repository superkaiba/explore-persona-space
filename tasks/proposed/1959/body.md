---
title: 'Per-feature decomposition of the prefix−bare mirror: concentrated or diffuse
  across the SAE dictionary'
kind: experiment
tags: []
created_at: '2026-08-01T05:57:43Z'
has_clean_result: false
parent_id: 1946
origin_prompt: 'Auto-filed from #1946 epm:follow-ups v1 proposal 2 (question_relation:
  substantially-different, auto_run: yes) by the Step 9b autonomous follow-up routing;
  filed-only for manual triage.'
workflow: v1
goal: Determine whether the prefix/bare mirror in per-context map error is carried
  by a small identifiable subset of the 16,384 restricted answer-side SAE features
  or is diffuse across the dictionary, by decomposing the banked per-category prefix−bare
  squared-error differences feature-by-feature.
---
## Goal

Determine whether the prefix/bare mirror in per-context map error is carried by a small identifiable subset of the 16,384 restricted answer-side SAE features or is diffuse across the dictionary, by decomposing the banked per-category prefix−bare squared-error differences feature-by-feature.

## Hypothesis

The mirror's category structure is concentrated: a small feature subset (plausibly language/format-aligned, given the language-stratified follow-up on #1946 localized the M_en flip to bare-arm English magnitude, and this SAE family is known to carry strong language-identity features, #1482) accounts for a disproportionate share of the per-category difference mass.

**Falsification:** contribution spread indistinguishable from diffuse — e.g. the top 1% of features carry ≈1% of the |prefix−bare| per-category difference mass, and a concentration statistic (participation ratio / Gini per contrast) sits inside a permuted-feature null band → the mirror is a bulk property of the maps, and feature-level interpretation of it is a dead end.

## Setup (pre-filled from parent #1946 — exactly ONE change: the analysis grain)

The parent pooled squared error over the 16,384 features per conversation; this task decomposes the SAME banked prediction/target matrices per feature before category aggregation. Same conversations, same arms, same categories, same SAE.

- Model: N/A — no model calls (same as parent).
- Data: banked artifacts — HF `issue1946_sae_percontext/analysis_tensors/sae_space/pred16/{prefix,bare,context}_L19_ridge.npz` + `sae_space/y_holdout/L19.npz` @ data revision `12ab41dc1c4a7163d183697e9c4fa53528904c9b`; judged labels `eval_results/issue_1738/judge_labels/labels.json` (9,925/9,941 labeled; 16 unlabeled tolerated identically to the parent).
- Seeds: same battery conventions (seed 1738 lineage; permuted-feature null seeded fresh, pre-registered).
- Eval: per-feature contribution to each of the 9 named mirror-category deltas + concentration statistics (participation ratio / Gini per contrast) vs a permuted-feature null.
- Config: same as parent EXCEPT the per-feature (not pooled) error decomposition.

## Success / kill criteria

- Success: a pre-registered concentration statistic per contrast lands outside the permuted-feature null band, with the top-contributing feature set enumerated (indices + contribution shares) for a labeled read once the #1773 feature-labeling instrument lands.
- Kill: concentration statistics inside the null band across the 9 mirror categories (the diffuse outcome — itself informative; argues against feature-level interventions on the mirror).

## Compute

0 GPU-h (fp16 pred matrices ≈ 9,941 × 16,384 per cell; vectorized numpy on `cpu-bigmem`/`cpu-mid`, minutes). Batch all per-feature reductions — never a per-feature Python loop.

## Notes

- Artifact premises verified at filing (scoped `HfApi.list_repo_tree` @ `12ab41dc1c4a`: 71 files under `issue1946_sae_percontext/`, all three `pred16/*_L19_ridge.npz` cells + `y_holdout/L19.npz` present).
- Instrument-supersession note: the judged SAE FEATURE labels are frozen (#1773 supersession in flight) — this analysis consumes none and spends nothing on labeling top features; naming what the top features mean waits for #1773's instrument.
- Redundancy screen (parent #1946 VC, 2026-08-01): not-redundant — #1738's SAE-arm per-feature scatter is context-vs-prefix (no bare arm, no category aggregation, no concentration statistic); #1895 is the variance-grain subspace-overlap construct.

## Provenance

Auto-filed by the parent #1946 Step 9b autonomous follow-up routing (`question_relation: substantially-different`, `auto_run: yes`, filed-only — never auto-spawned). Origin: `epm:follow-ups v1` proposal 2 on #1946 (2026-08-01), re-proposing the cap-parked free-analysis candidate that never ran.
