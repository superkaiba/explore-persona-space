---
title: 'Symmetric-baseline rerun: cos-to-baseline + JS-to-assistant on #396''s 24
  personas'
kind: experiment
tags: []
created_at: '2026-05-28T06:51:13Z'
has_clean_result: false
parent_id: 396
goal: 'Test whether the 4-predictor null in #396 survives baseline-symmetric construction
  by adding cos-to-baseline + JS-to-assistant predictors on the same 24 personas +
  same recipe; either corroborates the null as predictor-agnostic OR reveals an asymmetric-baseline
  artifact.'
---
## Goal

Test whether the 4-predictor null in #396 survives baseline-symmetric construction by adding cos-to-baseline + JS-to-assistant predictors on the same 24 personas + same recipe; either corroborates the null as predictor-agnostic OR reveals an asymmetric-baseline artifact.

Parent #396 used:
- `cosine_to_assistant_L15` — cosine sim of layer-15 residual stream against the **assistant** baseline prompt (`"You are a helpful assistant."`).
- `js_to_baseline` — Jensen-Shannon divergence of next-token output distribution against the **neutral / generic-system** baseline (no persona).

The two predictors live on **different reference states** (assistant vs neutral). The lineage's hypothesis space is therefore ad hoc — a null could be either a real "geometric distinctiveness doesn't predict implantation" finding OR an artifact of asking the question against the wrong baseline.

## Hypothesis

The 4-predictor null in #396 survives baseline-symmetric construction. I.e., when I compute the full 2×2 grid of {cosine, JS} × {assistant, neutral}, all 4 cells return null at the same threshold. Falsification: if cos-to-baseline OR JS-to-assistant clears the planned |ρ| ≥ 0.35 threshold against the same 24 personas + same headline DV, the lineage's original predictor choice was the wrong baseline pick and the null wasn't general.

## Design (single-variable change from #396)

- **What changes vs #396:** Add 2 new predictors — `cosine_to_baseline_L15` (cosine against neutral / generic-system prompt) + `js_to_assistant` (JS against the same assistant baseline prompt the cosine predictor uses).
- **What stays unchanged:** Same 24 trained personas (`INHERITED_SOURCES_24`), same LoRA recipe (same hyperparameters as #396 — lr=1e-4, r=32, α=64, marker ` ※`, 1050-token training, 2048-token eval cap), same 48-eval-persona panel, same headline DV (`logp_end_of_response_diagonal_mean`), same 6-surface DV grid, same length-partial Spearman + BH-FDR(q=0.05) statistical scaffolding.
- **Reuse from #396:** No retraining. The 24 LoRA adapters are already on HF model repo (`superkaiba1/explore-persona-space` at commit `767a6824`). Only the base-model predictor extraction (Phase A) needs to re-run, on the base Qwen-2.5-7B-Instruct (no adapter loading), to extract layer-15 residuals + next-token output distributions under both baseline prompts (assistant + neutral). Eval phase (Phase B) uses the same `logprob_*.json` files already on HF data repo (`superkaiba1/explore-persona-space-data/issue396/logprob/`).

## Expected resource footprint

- **Compute:** 1× H100 for ~30 min (base-model only, no LoRA loading) for cos-to-baseline + JS-to-assistant extraction across 24 personas + 2 baseline prompts × 20 probe questions. ~$1.
- **Wall time:** ~45 min total including pod provision + bootstrap.
- **No new training.** No new evals beyond the 2 new predictor extractions.

## Acceptance criteria

1. `eval_results/issue_<N>/base_model_predictors_v2.json` populated with all 4 predictors (cos-to-assistant, cos-to-baseline, JS-to-baseline, JS-to-assistant) per persona at layer 15.
2. Length-partial Spearman ρ + BCa 95% bootstrap CI + p-value + BH-FDR(q=0.05) reject computed for all 4 predictors × 6 DV surfaces (same scaffolding as #396).
3. Hero figure: 4-panel scatter (one per predictor) against headline DV, raw + length-partial overlay. Caption names which predictor uses which baseline.
4. Headline finding: either (a) all 4 predictors still null at the same threshold (corroborates #396's claim that geometric / distributional distinctiveness — regardless of reference state — doesn't predict implantation success) or (b) the asymmetric-baseline cells in #396 hid a real signal that the symmetric construction reveals.

## Parent context

- Parent task: #396 (`tasks/awaiting_promotion/396/`).
- Construction-inconsistency call-out: Thomas surfaced this 2026-05-28 after seeing the #396 panel and asking why cos uses assistant baseline while JS uses neutral baseline. The parent body now flags this as a known limitation in `### Methodology corrections` and points here.

## Out of scope

- Adding more predictors beyond cos × baseline + JS × assistant (e.g., L0 / L31 layer sweep, gradient-based predictors past first-step). Those are separate follow-ups.
- Re-running the LoRA training. The adapters are fixed; only the base-model predictor extraction changes.
- Re-running the trajectory log-prob eval. The per-source `logprob_*.json` files are already on HF data repo and can be reused.
