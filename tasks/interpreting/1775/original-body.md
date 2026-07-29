---
title: 'Nonlinearity of the four-arm context→answer maps: rank-r bilinear interaction
  vs kernel/MLP gain'
kind: experiment
tags: []
created_at: '2026-07-28T21:36:04Z'
has_clean_result: false
parent_id: 1092
origin_prompt: 'run both in background with ahppy coder (2026-07-28; two-task split
  per: ''can it not be all one task? Or at least separate the nonlinear out'')'
workflow: v1
goal: 'Determine where the linear story of the context→answer map ends: per-arm nonlinearity
  gain under matched folds (ridge → RFF/Nyström → MLP), residual HSIC/dCor detection,
  fold-structure verification of the banked n50k/n1m fitter comparisons, and the headline
  test of whether a rank-r bilinear prefix×query interaction closes the same ≈0.06
  R² gap the banked 1M nonlinear fits find — per Task B of docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md'
relates_to:
- spec-context-as-vector
- leak-predictor
backend: gcp
---
# Nonlinearity of the four-arm context→answer maps: bilinear interaction vs kernel/MLP gain

## Goal

Determine where the linear story of the context→answer map ends: per-arm nonlinearity gain under matched folds (ridge → RFF/Nyström → MLP), residual HSIC/dCor detection, fold-structure verification of the banked n50k/n1m fitter comparisons, and the headline test of whether a rank-r bilinear prefix×query interaction closes the same ≈0.06 R² gap the banked 1M nonlinear fits find — per Task B of docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md

## Plan basis

Execute **Task B** of `docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md` — sections §3 (Q2 the DMDc layer) and §7 (Q6 where linearity ends), under the §8 validity gates and the §9 reuse policy. The plan doc is the scope contract; the adversarial planner refines it into the executable plan. Sibling task: #1774 (Task A, linear/operator characterization) — reuse its multi-draw answer set and fold definitions.

## Work items (from the plan doc)

1. **Fold-structure check on the banked datapoints (first, cheap, gating):** verify the #779 fitter-fair-comparison splits (n50k, n1m — `eval_results/issue_779/fitter-fair-comparison-n50k/`, `fitter-fair-comparison-n1m/n1m_fits.json`) respect novel-prefix grouping before quoting ridge 0.754–0.760 vs KRR/MLP ≈ 0.81 as the context-arm nonlinearity gain.
2. **Q2 rank-r bilinear interaction:** extend the additive stitch with a = A·p + B·q + Σᵢ (uᵢᵀp)(vᵢᵀq)·wᵢ, sweeping r; project the interaction component onto the persona-vector dictionary + answer PCs. Headline test: does the bilinear model close the same ≈0.06 R² gap the banked nonlinear fits find — is the "nonlinearity" the named prefix×query interaction?
3. **Q6 detection:** residual HSIC / distance correlation between each arm's input and its linear-map residuals, group-respecting permutation p-values — run before any new estimator spend.
4. **Q6 estimation ladder, three remaining arms** (prefix-end, bare-query, query-averaged, at the 21K crossed grain where all four states are banked): ridge → RFF/Nyström kernel ridge → MLP, identical folds + nested-CV tuning on every rung, seed spread reported for MLPs. Nonlinearity gain per arm = held-out R²(nonlinear) − R²(linear), with identity+bias baseline + kNN-retrieval reads per the standing mapping rules.
5. **Noise-ceiling-relative reporting:** score all gains against the per-direction decode-noise floor from #1774's multi-draw set (do NOT regenerate draws).

## Scope / constraints

- Context arm rungs are BANKED — reuse, do not refit: exact-RBF KRR at n=50k (R² 0.807), Nyström KRR at n=963,444 (0.807), MLP w8192/w32768 (0.810/0.813), residual-skip (0.808), plus multilayer n=963k fits with persisted weights (`n1m-nonlinear-map-behavior-readout/`, weights_dir). Artifact-reuse fitness checks apply.
- New nonlinear fits ONLY for the three non-context arms; exact kernels do not scale past n ~ 10⁴ — use RFF/Nyström there.
- The nonlinearity gain is defined only under identical folds and matched nested-CV tuning on both rungs; a gain quoted across mismatched protocols is invalid.
- Both mapping arms (prefix-based AND context-based) are represented by construction across the four-arm design.
- Estimated 7–15 GPU-h (cheap band per round).

## Provenance

- Verbatim dispatch: "run both in background with ahppy coder" (2026-07-28 chat), following "can it not be all one task? Or at least separate the nonlinear out" — the two-task split is encoded in plan §9 "Execution shape: two tasks."
