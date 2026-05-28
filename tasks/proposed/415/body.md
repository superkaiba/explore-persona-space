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

Test whether the 4-predictor null in #396 is a **reference-state artifact** by recomputing the two distributional/geometric predictors against a truly neutral baseline (no system prompt) and rerunning the same correlation analysis on the same 24 personas + same shipped recipe.

**Corrected premise (2026-05-28).** I originally framed this as fixing a "construction asymmetry" between cos-to-assistant and JS-to-baseline. After re-reading #396's `recompute_predictors_i396.py`, both predictors actually reference the **same** bare-assistant prompt (`"You are a helpful assistant."`). The variable name `js_to_baseline` is misleading — "baseline" there means "the bare-assistant baseline I picked", not "a generic / neutral reference". So there is **no asymmetry between #1 and #2 in #396**.

The reshaped question this follow-up tests is more honest and more interesting: **the bare-assistant prompt is itself a persona statement**. Geometric/distributional distance from it is measuring distance-from-one-helpful-pose, not distance-from-the-model's-default-behavior. If a truly neutral baseline (no system prompt at all, model in pure instruct-mode default) reveals signal where the bare-assistant baseline showed null, the lineage's reference-state pick was wrong; if both baselines return null, the predictor lineage is robustly uninformative regardless of which reference we anchor against.

## Hypothesis

The 4-predictor null in #396 (against bare-assistant baseline) survives replacement of the reference with a truly neutral baseline. I.e., the new predictors `cosine_to_neutral_L15` and `js_to_neutral` will also return |ρ| \< 0.35 against the same headline DV across all 6 surfaces, BH-FDR(q=0.05) reject = False. **Falsification:** if either new predictor clears |ρ| ≥ 0.35 against the headline DV `logp_end_of_response_diagonal_mean`, the lineage's bare-assistant reference was hiding a real geometric/distributional signal that a neutral reference exposes.

## Design (single-variable change from #396)

- **What changes vs #396:** Add 2 new predictors against a neutral baseline (no system prompt, model in pure instruct-mode default):
  - `cosine_to_neutral_L15` — cosine sim of layer-15 residual stream against the neutral-baseline hidden state.
  - `js_to_neutral` — JS divergence of next-token output distribution against the neutral-baseline output distribution.
- **What stays unchanged:** Same 24 trained personas (`INHERITED_SOURCES_24`), same LoRA recipe (same hyperparameters as #396 — lr=1e-4, r=32, α=64, marker ` ※`, 1050-token training, 2048-token eval cap), same 48-eval-persona panel, same headline DV (`logp_end_of_response_diagonal_mean`), same 6-surface DV grid, same length-partial Spearman + BH-FDR(q=0.05) statistical scaffolding.
- **Reuse from #396:** No retraining. The 24 LoRA adapters are already on HF model repo (`superkaiba1/explore-persona-space` at commit `767a6824`). Only the base-model predictor extraction (Phase A) needs to re-run, on the base Qwen-2.5-7B-Instruct (no adapter loading), to extract layer-15 residuals + next-token output distributions under the neutral-baseline prompt (in addition to the bare-assistant baseline already cached). Eval phase uses the same `logprob_*.json` files already on HF data repo (`superkaiba1/explore-persona-space-data/issue396/logprob/`).
- **Implementation:** Modify `recompute_predictors_i396.py` to ALSO extract hidden states + next-token distributions under the no-system-prompt condition, then add two new predictors against this second reference centroid. The persona axis grows from 49 (48 panel + assistant baseline) to 50 (48 panel + assistant + neutral baseline). One additional sweep adds ~20 forward passes (one per probe question, no persona axis multiplication needed for the new baseline).

## Expected resource footprint

- **Compute:** 1× H100 for ~30 min (base-model only, no LoRA loading) for the extended extraction. ~$1.
- **Wall time:** ~45 min total including pod provision + bootstrap.
- **No new training.** No new evals beyond the 2 new predictor extractions.

## Acceptance criteria

1. `eval_results/issue_415/base_model_predictors_v2.json` populated with all 4 predictors (cos-to-assistant, cos-to-neutral, JS-to-assistant, JS-to-neutral) per persona at layer 15.
2. Length-partial Spearman ρ + BCa 95% bootstrap CI + p-value + BH-FDR(q=0.05) reject computed for all 4 predictors × 6 DV surfaces (same scaffolding as #396).
3. Hero figure: 4-panel scatter (one per predictor) against headline DV. Caption names which predictor uses which baseline.
4. Headline finding: either (a) all 4 predictors still null at the same threshold (corroborates #396's claim that geometric/distributional distinctiveness — regardless of reference state — doesn't predict implantation success) or (b) the neutral-baseline cells reveal a real signal that the bare-assistant baseline hid.

## Parent context

- Parent task: #396 (`tasks/awaiting_promotion/396/`).
- Reference-state question call-out: Thomas surfaced this 2026-05-28 after seeing the #396 panel and asking why cos uses "assistant" while JS uses "baseline" — turned out (per code inspection) both predictors use the same bare-assistant prompt, so the real question is whether the bare-assistant choice itself is biased and a neutral reference would tell a different story.

## Out of scope

- Adding more predictors beyond cos × neutral + JS × neutral (e.g., L0 / L31 layer sweep, gradient-based predictors past first-step, marker-prior). Those are separate follow-ups.
- Re-running the LoRA training. The adapters are fixed; only the base-model predictor extraction changes.
- Re-running the trajectory log-prob eval. The per-source `logprob_*.json` files are already on HF data repo and can be reused.
