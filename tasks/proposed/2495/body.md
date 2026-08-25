---
title: 'Correct #825 promoted body: 0.1087 cited as control R² is fraction_of_fulln_ceiling
  (true R²=0.0732)'
kind: infra
tags: []
created_at: '2026-08-23T08:33:39Z'
has_clean_result: false
workflow: v1
---
## Goal
Correct a wrong-metric figure in task #825's promoted clean-result body: the separator/punctuation-control read cited as "R² ≈ 0.1087" is NOT a held-out R² — it is `fraction_of_fulln_ceiling` (0.10868122095179335) from `eval_results/issue_825/base-separator-control/base_sep_to_chat.json` → `decision_support.instruct_reference`; the actual transfer R² in the same artifact is `sep_to_chat_fulln_r2` = 0.0731533298226521, and the committed within-WikiText L19 ridge cell (`eval_results/issue_931/cells_armC_sep.json` → `r2_per_layer_obs[19]`) is −3.17. Apply a NON-Takeaway-flip prose correction via `task.py set-body` where possible; anything touching a bolded Takeaway stays in this task's scope as a body-prose correction with the classification untouched (user-only contract unchanged).

## Evidence
- Refuting artifacts: `eval_results/issue_825/base-separator-control/base_sep_to_chat.json` (both fields present, verified 2026-08-22, task #1901 round generic-boundary-token-control plan v10 §2/§4/M2); `eval_results/issue_931/cells_armC_sep.json`.
- The #1901 round has already retired the downstream consumer: `docs/posters/mats_2026/make_plot1_scaling.py` no longer draws the 0.1087 hline (commits 291da66d/7475bd58/789e64d2), and a MEASURED single-type generic-boundary-token control curve now exists (`eval_results/issue_1901/paper_densify/boundary_token_scaling_L19.json`, branch issue-1901-btokctl) — cite it as the replacement reference where #825's body needs one.
- Residual also worth fixing in the same pass: `scripts/issue1901_body_figures.py` renderer `boundary_hline` kwarg docstring still carries "(instruct R^2 0.1087...)" (pre-existing at 291da66d; flagged r1 g4 Minor).

## Provenance
Filed by the #1901 same-issue follow-up round (generic-boundary-token-control) per plan v10 M2 result-time duty: a round that refutes a claim in a promoted body files the correction task in the same turn as the result summary (CLAUDE.md § inline estimator-validity + record-integrity duties, item 3).
