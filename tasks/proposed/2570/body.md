---
title: Re-point paper/poster scaling-figure generators off the superseded cross-store
  join (#1901 finding)
kind: infra
tags: []
created_at: '2026-08-25T09:23:43Z'
has_clean_result: false
parent_id: 1901
origin_prompt: 'Plan v15 analyzer-concerns item 8(ii), reconciler-mandated: route
  the paper/poster generator update (scripts/issue1901_body_figures.py fig_paper_c1_scaling
  + docs/posters/mats_2026/make_plot1_scaling.py: consume mlp_scaling_dense_L19.json,
  or render the old join only with an explicit superseded label) — direct edit if
  trivial, else kind: infra task via file_infra_task.py naming the refuting evidence.'
workflow: v1
---
Re-point the two scaling-figure generators that still render the superseded cross-store join, per #1901's supersession record `eval_results/issue_1901/paper_densify/superseded_cross_store_join.json` (committed on `origin/issue-1901-mlpdense` at `5a3df5a578`, mirrored in the round's fold analysis).

## Goal

`scripts/issue1901_body_figures.py` (`fig_paper_c1_scaling`, feeding `figures/paper/c1_scaling_train_pool.png`) and `docs/posters/mats_2026/make_plot1_scaling.py` still draw the scale7-store ≤25k scaling points (`scaling_ladder_L19.json`, `mlp_scaling_L19.json`) joined with the n1m ≥50k points (`scaling_bigN_acc1_L19.json`) as ONE ladder. #1901's `mlp-scaling-densify` round measured (2026-08-25, on-pod, `epm:progress` 2026-08-25T03:18:29Z + `artifacts/mlpdense-smoke-r3-g1fail.log`) that the two stores' pinned eval pools are DIFFERENT rows — 0/400 scale7 val rows content-match any of the 5,000 pass_b rows (median NN distance 6.63) — so any figure joining scale7 cells with n1m cells on one axis compares curves scored on different held-out rows.

Refuting evidence + replacement: the within-store dense ladder `eval_results/issue_1901/paper_densify/mlp_scaling_dense_L19.json` (all 8 fresh rungs 5k-500k drawn from the n1m store, one pinned val_400/test_1000 pool, whitened-cosine+CSLS primary retrieval; verdict Confirmed, D_gap = +0.0203). Reference rendering: `scripts/issue1901_mlpdense_fold_figures.py` (same branch) → `figures/issue_1901/mlp_scaling_dense_L19.png`.

Required changes:
1. `fig_paper_c1_scaling`: consume `mlp_scaling_dense_L19.json` for the linear + nonlinear curves (or render the old join ONLY with an explicit superseded/different-eval-pool label per the supersession record's disposition); keep the boundary-token control curves (the #1901 boundary round's contribution to the same figure) intact; re-render + commit `figures/paper/c1_scaling_train_pool.{png,pdf,meta.json}`.
2. `docs/posters/mats_2026/make_plot1_scaling.py`: same re-point for the poster plot; NOTE the poster is user-co-edited — regenerate the plot asset only, do not restructure the poster.
3. Preserve the boundary round's figure regression-gate convention where applicable (the prior gate pinned 304 committed points — the join replacement legitimately changes those points; supersede the gate's reference set deliberately, never silently).

Consumers list + disposition: see `superseded_cross_store_join.json` (`consumers`, `disposition`). Also check `docs/posters/mats_2026/csls_rescore.py` (validates vs `scaling_ladder_L19.json`) and `docs/methodology/issue_1901.md` per that record.
