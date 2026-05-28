---
title: 'Symmetric-baseline rerun corroborates #396''s null: all four predictors fail
  against both assistant and neutral references (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-05-28T06:51:13Z'
has_clean_result: true
parent_id: 396
goal: 'Test whether the 4-predictor null in #396 survives baseline-symmetric construction
  by adding cos-to-baseline + JS-to-assistant predictors on the same 24 personas +
  same recipe; either corroborates the null as predictor-agnostic OR reveals an asymmetric-baseline
  artifact.'
---
# Symmetric-baseline rerun corroborates #396's null: all four predictors fail against both assistant and neutral references (MODERATE confidence)

## Human TL;DR

*Thomas fills this in.*

## TL;DR

- **Motivation:** Parent [#396](https://eps.superkaiba.com/tasks/396) returned a 4-predictor null for marker implantation, all against the same bare-assistant reference state ("You are a helpful assistant."). I worried that anchoring against another persona statement might be hiding a signal that a truly neutral reference (no system prompt) would reveal — so I re-extracted cos and JS predictors for both reference states on the same 24 personas + same DV, and re-correlated.
- **What I ran:** On the same 24 INHERITED_SOURCES personas as #396, I re-extracted base-model predictors against TWO reference states: the bare-assistant prompt (replicates #396) and a truly neutral baseline (no system message — model in pure instruct-mode default). Four predictors total — cos-to-assistant L15, JS-to-assistant, cos-to-neutral L15, JS-to-neutral. Each correlated against #396's headline DV (`logp_end_of_response_diagonal_mean`) via length-partial Spearman + BH-FDR(q=0.05).
- **Results:** Null corroborated across all 4 predictors. Cosine-to-assistant ρ=+0.018, p=0.49; JS-to-assistant ρ=-0.160, p=0.25; cos-to-neutral ρ=+0.184, p=0.24; JS-to-neutral ρ=-0.224, p=0.17. All N=24. None clears the planned |ρ| \>= 0.35 detection threshold. BH-FDR(q=0.05) reject = False for all four. Sign-direction is consistent across baselines (cos slightly positive, JS clearly negative). The neutral baseline gives slightly STRONGER (more negative for JS, more positive for cos) but still subthreshold correlations than the assistant baseline.

    ![Symmetric-baseline rerun: all 4 predictors against headline DV](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2965c59c82f1ccb0f802cac678b234ee4d93cecb/figures/issue_415/hero_4predictor_symmetric_panel.png)

    > **Figure.** *All 4 predictors against trained-LoRA log p(marker) at end-of-response — top row uses the bare-assistant reference, bottom row uses the truly neutral reference; all 4 are subthreshold and consistent in sign.* Each scatter shows 24 trained-source personas (one dot per persona). Top-left: cos-to-assistant ρ=+0.018; top-right: JS-to-assistant ρ=-0.160 (replicates #396 within bfloat16 noise). Bottom-left: cos-to-neutral ρ=+0.184; bottom-right: JS-to-neutral ρ=-0.224 (slightly stronger neutral-baseline cells, still subthreshold).

- **Next steps:** Door is now closed on "reference-state choice was hiding the signal". The lineage's null is robust to whether we anchor against the bare-assistant prompt or against the model's pure-default behavior. Two structural takeaways worth noting. First, per-persona cos-to-assistant and cos-to-neutral are nearly identical (means 0.941 vs 0.934 across personas), suggesting Qwen-Instruct's "no system prompt" hidden state is very close to the "you are a helpful assistant" hidden state — so the two references probe nearly the same baseline. Second, the JS-to-neutral cell is the strongest in the grid at ρ=-0.224, p=0.17 — sign direction (more-distinctive personas implant WORSE) is at least suggestive, but still well below the detection threshold and well above p<0.05. If you want to chase this further, the right move is a larger N (the #296 24 NEW personas, which got dropped from #396 for an unrelated data-availability reason) — not another reference state.

## Details

I wondered whether #396's null might be a reference-state artifact. The setup matters: in #396, both predictor #1 (cos-to-assistant) and predictor #2 (named `js_to_baseline` in the code) used the SAME reference state — the bare-assistant prompt. The variable name was misleading ("baseline" sounded like a different reference); the actual computation anchored both predictors against the same string. The bare-assistant prompt is itself a persona statement, which raised the question: are these predictors measuring distance-from-the-model's-default-behavior, or distance-from-one-specific-helpful-pose? A truly neutral baseline (no system message at all, model in pure instruct-mode default) would tell a different story if the bare-assistant reference was anchoring to the wrong thing.

I expected one of two outcomes. Either the neutral-baseline cells would clear the |ρ| \>= 0.35 threshold while the assistant-baseline cells stayed null — meaning the reference state was a confound and a signal was hidden by anchoring against another persona — or both baselines would return null with broadly similar magnitudes, meaning the predictor framework itself doesn't predict implantation, independent of which neutral pole we measure distance from. The result was the second one.

### Predictor extraction and per-persona replication

I extended #396's `recompute_predictors_i396.py` to extract layer-15 residual stream + next-token output distributions under TWO baseline prompts (bare-assistant + no-system-message), keeping everything else fixed (same 24 trained-source personas, same 20 probe questions, same shared greedy baseline responses, same teacher-force shape). To support the no-system baseline I patched `build_teacher_force_inputs` to accept `sys_prompt=None` → omit the system block entirely (the chat template just emits the user turn + assistant response, with no system block at all). Then I computed all four predictors per persona — cos-to-assistant L15, JS-to-assistant, cos-to-neutral L15, JS-to-neutral.

Per-persona values for the two predictors that ALSO exist in #396 (cos-to-assistant + JS-to-assistant) match #396's stored values within bfloat16 numerical noise (mean absolute difference \~1e-4 across the 5 spot-checked personas — `software_engineer 0.969415 vs 0.969352, villain 0.901779 vs 0.901705, lawyer 0.943903 vs 0.943828, qwen_default 0.941929 vs 0.942016, medical_doctor 0.959322 vs 0.959257`). Replication of #396's predictor extraction is solid.

### What the two baselines actually look like geometrically

A structural observation worth surfacing before the correlations: the two baseline reference states produce nearly identical layer-15 hidden states. Across the 24 personas, mean cos-to-assistant = 0.941 (range 0.833-1.000) and mean cos-to-neutral = 0.934 (range 0.825-1.000); per-persona cosines against the two baselines are extremely similar. Same for JS — mean JS-to-assistant = 0.116, mean JS-to-neutral = 0.093. Qwen-Instruct's "no system prompt" hidden state and its "You are a helpful assistant." hidden state are not that different from each other in either geometric or distributional terms. This itself caps how informative the symmetric-baseline rerun could be: if the two baselines anchor to the same point, the cells inherit the same signal-or-null structure.

### Correlations: 4 predictors x headline DV

![Symmetric-baseline rerun: all 4 predictors against headline DV](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2965c59c82f1ccb0f802cac678b234ee4d93cecb/figures/issue_415/hero_4predictor_symmetric_panel.png)

> **Figure.** *All 4 predictors against trained-LoRA log p(marker) at end-of-response.* Each panel shows N=24 trained-source personas (one dot per persona). Top row replicates #396 (predictors anchored against the bare-assistant baseline); bottom row uses the truly neutral baseline (no system message). None of the 4 cells clears the planned |ρ| \>= 0.35 detection threshold; all have BH-FDR(q=0.05) reject = False; sign direction is consistent across baselines (cos slightly positive, JS clearly negative).

The headline cells (length-partial Spearman):
- Cosine-to-assistant L15: ρ=+0.018, p=0.485
- JS-to-assistant: ρ=-0.160, p=0.252
- Cosine-to-neutral L15: ρ=+0.184, p=0.243
- JS-to-neutral: ρ=-0.224, p=0.165

None clears |ρ| \>= 0.35. None has raw p \< 0.05 even before correction. None survives BH-FDR(q=0.05) across the 4-cell family. The neutral-baseline cells are slightly stronger than their assistant-baseline counterparts (cos: +0.184 vs +0.018; JS: -0.224 vs -0.160), but both stay subthreshold and the direction of the shift is consistent with the structural similarity of the two baselines noted above.

### Full 4x6 forest (other DV surfaces)

![Forest of length-partial rho for all 4x6 cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2965c59c82f1ccb0f802cac678b234ee4d93cecb/figures/issue_415/predictor_x_surface_forest.png)

> **Figure.** *4 predictors x 6 dependent-variable surfaces, length-partial Spearman with 95% bootstrap CIs.* All cells live inside the shaded sub-threshold band |ρ| \< 0.35. Surfaces are the headline `logp_end_of_response_diagonal_mean` plus 5 other trajectory features (`logp_at_k0`, `logp_auc`, `logp_max`, `logp_mean`, and the legacy `substring_match_rate`) inherited from #396's analysis.

The largest absolute cell in the 4x6 grid sits at JS-to-neutral × `logp_max_diagonal_mean` at ρ_partial = -0.339, but its 95% bootstrap CI comfortably crosses zero and the raw p is still >0.05. No cell clears the detection threshold.

### Why this test

Length-partial Spearman is the same test #396 ran: Spearman ρ between each predictor and each DV surface, residualizing both against persona-prompt character length to control for any length confound. BH-FDR(q=0.05) is applied across the 4-cell headline family (one cell per predictor × the single headline DV surface), not the full 4x6 grid — the 6-surface grid is exploratory, the 4-cell headline is the planned-up-front comparison the threshold was set against. 95% bootstrap CIs use 5000 percentile resamples (no BCa accel — for N=24 with no strong skew the percentile interval is close enough to BCa).

### Parameters

| Field | Value |
|---|---|
| Personas | 24 INHERITED_SOURCES_24 (same as #396; subset of the planned 48) |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Hidden layer | 15 (same as #396) |
| Probe questions | 20 (same `EVAL_QUESTIONS_20` as #396) |
| Reference states | bare-assistant: `"You are a helpful assistant."`; neutral: no system message (chat template emits user+assistant turns only) |
| Predictors | cos-to-assistant L15, JS-to-assistant, cos-to-neutral L15, JS-to-neutral |
| Dependent variable surfaces | logp_end_of_response (HEADLINE), logp_at_k0, logp_auc, logp_max, logp_mean, substring_match_rate |
| Statistical test | length-partial Spearman, 95% percentile bootstrap CI (n=5000), BH-FDR(q=0.05) across 4-cell headline family |
| Detection threshold | \|ρ\| \>= 0.35 (matches #396) |
| Compute | 1× H100 for ~7 min wall (model load 80s + baseline generation 65s + question loop 76s + JS derivation ~5 min) |

Confidence: MODERATE — N=24 is the same as #396, the predictor extraction replicates #396 within bfloat16 noise, the verdict is consistent with the parent finding, and the no-system-prompt operationalization of "neutral baseline" is one defensible choice (an empty-string system prompt would be a different choice). The cell-by-cell magnitudes are all subthreshold but two of the four neutral-baseline cells (cos-to-neutral × headline at ρ=+0.184, JS-to-neutral × headline at ρ=-0.224) sit above 0.15 in absolute magnitude, which at N=24 is well within the noise band, so calling them "null" rather than "very small effect under-powered to detect" is a judgment call.

## Reproducibility

**Artifacts:**
- Base-model predictors: [`eval_results/issue_415/base_model_predictors_v2.json`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/eval_results/issue_415/base_model_predictors_v2.json) (4 predictors × 24 personas)
- Analysis summary: [`eval_results/issue_415/analysis_summary.json`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/eval_results/issue_415/analysis_summary.json) (4×6 ρ + p + CI grid)
- Hero figure source: [`figures/issue_415/hero_4predictor_symmetric_panel.png`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/figures/issue_415/hero_4predictor_symmetric_panel.png)
- Forest figure source: [`figures/issue_415/predictor_x_surface_forest.png`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/figures/issue_415/predictor_x_surface_forest.png)
- Per-source DV reused from #396: [`eval_results/issue_396/analysis_summary.json::per_source_aggregation`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/eval_results/issue_396/analysis_summary.json)
- WandB: n/a — pure base-model forward passes, no training, not logged

**Compute:**
- 7 minutes on 1× H100 (RunPod pod-415, terminated after extraction)
- ~$0.35 — single H100, ~7 min wall

**Code:**
- Predictor extraction: [`scripts/recompute_predictors_i415.py`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/scripts/recompute_predictors_i415.py)
- Analysis: [`scripts/analyze_issue415.py`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/scripts/analyze_issue415.py)
- Divergence library patch (sys_prompt=None support): [`src/explore_persona_space/analysis/divergence.py`](https://github.com/superkaiba/explore-persona-space/blob/2965c59c82f1ccb0f802cac678b234ee4d93cecb/src/explore_persona_space/analysis/divergence.py)
- Git commit: `2965c59c82f1ccb0f802cac678b234ee4d93cecb`
- Reproduce:
  ```
  git clone https://github.com/superkaiba/explore-persona-space.git
  cd explore-persona-space && git checkout 2965c59c82f1ccb0f802cac678b234ee4d93cecb
  uv sync
  # On a pod with 1× H100:
  uv run python scripts/recompute_predictors_i415.py --device cuda:0 --batch-size 16
  # Then locally:
  uv run python scripts/analyze_issue415.py
  ```
