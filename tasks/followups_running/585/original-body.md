---
title: 'Re-run the #504 calibration table at true fractions on the verified per-fraction
  smoke snapshots'
kind: experiment
tags: []
created_at: '2026-06-11T03:15:47Z'
has_clean_result: false
parent_id: 549
goal: 'Measure the true training-maturity curve of the #504 saturated-anchor smoke
  cell by re-evaluating its six verified per-fraction adapter snapshots through the
  fixed distinct-id eval path, replacing the stale six-reads-of-one-adapter calibration
  table with real per-fraction numbers.'
relates_to:
- implant-learning-speed
- leak-contrastive-negatives
---
## Goal

Measure the true training-maturity curve of the #504 saturated-anchor smoke cell by re-evaluating its six verified per-fraction adapter snapshots through the fixed distinct-id eval path, replacing the stale six-reads-of-one-adapter calibration table with real per-fraction numbers.

## Hypothesis

The published flat calibration table (5.47/5.89/5.01/6.08/5.63/7.17 nats across nominal fractions 0.08-1.00) is six reads of the fraction-0.08 adapter (per #549's AFFECTED finding); with distinct ids the true curve shows a real maturity gradient, with late-fraction source implant strength likely well above the step-6 value of ~5.5 nats.

## Falsification

If the distinct-id re-eval reproduces the same flat ~5-7 nat band across fractions (within run-to-run noise), the stale-serve explanation for the flatness is wrong and #549's AFFECTED mechanism needs re-examination — the code-read + replay legs would conflict with fresh ground truth.

## Setup (one variable changed vs the run that produced phase0_calibration_v4.json)

- Model: Qwen-2.5-7B + LoRA snapshots `adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{0.08,0.16,0.33,0.50,0.75,1.00}` (HF model repo superkaiba1/explore-persona-space — all 6 verified present with adapter_model.safetensors + adapter_config.json).
- Eval: `issue-534:src/.../eval_trajectory.py` via `--eval-only` + `--fraction-manifest`, with `assert_source_delta_g_matches_manifest` armed. Same probe set, same DV (per-fraction log P(marker) trained − base, alongside the EOS-margin logit per the marker-leakage rule), same rig.
- The ONLY change: distinct `lora_int_id` per checkpoint (the 298877f9c fix path) instead of constant id 1.
- NOTE: blocked on (or bundled with) the fix-port child if executed from main; alternatively run from the issue-534 branch.

## Scope exclusion

The 10-cell production-grid correction is NOT in scope: its per-fraction snapshots do not exist on HF (verified via list_repo_files — final adapters only), so that correction costs ~10 GPU-h retrain+eval and is a separate decision, best taken after this result and after the user-owned #504 body-correction call.

## Cost

~0.5-1 GPU-hour on 1× H100 (`eval` intent), eval-only.

## Provenance

Filed from #549's epm:follow-ups v1 proposal 2 (auto_run: yes, question_relation: substantially-different, execution: filed-only — runs only on manual triage via /issue). The PM may instead route this as a followup-scope ON #504 (the corrected numbers answer #504's question).
