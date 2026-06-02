---
title: Epoch-resolved on-policy marker transfer — overtraining vs construct (#460
  follow-up)
kind: experiment
tags: []
created_at: '2026-06-01T23:14:29Z'
has_clean_result: false
parent_id: 460
goal: 'Determine whether #460''s universal-ceiling on-policy marker transfer (log
  P(marker) approx 0 everywhere, divergence rho=-0.11 null) is an overtraining artifact
  or intrinsic to the marker-at-end construct, by resolving marker log P and divergence->transfer
  rho across training amount (adapters saved at epochs 1/2/3/5, 16x16 crosseval at
  each level).'
---
## Goal

Determine whether #460's universal-ceiling on-policy marker transfer (log P(marker) approx 0 everywhere, divergence rho=-0.11 null) is an overtraining artifact or intrinsic to the marker-at-end construct, by resolving marker log P and divergence->transfer rho across training amount (adapters saved at epochs 1/2/3/5, 16x16 crosseval at each level).

## Background

#460 (parent) measured on-policy marker cross-transfer the corrected way (marker appended after the model's own base-generated response R, continuous log P(※) at the slot after R, loss masked to only the marker token). Result: transfer saturated at ceiling everywhere, so base-model divergence had nothing to predict (ρ=−0.11). But the implant recipe had to be escalated (the minimal 1-epoch/30-row recipe implanted 0%; we used 5 epochs × 300 rows), and #460 only measured the endpoint. Two competing explanations:
- **Overtraining:** 5 epochs × 300 rows with loss on a single token drove log P(※)→0 and washed out a real graded divergence effect.
- **Construct:** marker-at-end conditions the marker on R (a transformation-invariant natural response), so "append ※ after any response" transfers universally by design, regardless of training amount.

#460 cannot separate these (endpoint only).

## Design (reuse #460's rig; the ONLY change is training-amount resolution)

- Same 16 conditions (A1-A5, B1-B5, C1, D1-D5), same on-policy R (reuse `superkaiba1/explore-persona-space-data` @ `issue460_marker_at_end/`), same marker (` ※`, token 83399), same marker-at-end / loss-on-marker-only recipe, same 300 rows/condition.
- Train each condition to 5 epochs, **saving the adapter at epochs {1, 2, 3, 5}** (4 checkpoints/condition).
- **Crosseval the 16×16 ΔG log-prob matrix at each of the 4 checkpoint levels** (4 matrices).
- If cheap to wire into the existing trainer: log per-step marker log-prob on a fixed probe batch to WandB (the within-training dynamics trajectory). Secondary; the 4 checkpoint levels are the primary trajectory.
- Predictor: reuse #406's `D_matrix.json` (base-model forward-KL divergence between the 16 transformations).

## DVs / analysis (per training level)

1. **Saturation:** off-diagonal `g_logprob` mean, sd, fraction within 0.1 nat of 0; diagonal (self-implant) strength.
2. **Divergence→transfer:** length-partial Spearman ρ(D, g_logprob) and ρ(D, delta_g), with bootstrap CI.
3. **Curves:** ρ vs training amount, and saturation-fraction vs training amount.

## Hypothesis

If at low training (epoch 1-2) `g_logprob` sits in a mid-band (unsaturated) AND ρ(D, g_logprob) is meaningfully negative (divergence predicts transfer), then #460's null is an overtraining artifact and #406's effect is real but washed out. If transfer is broad/universal even at epoch 1, it is the construct (marker-at-end conditions on R, not the prompt), and #406's effect was specific to the marker-first regime.

## Resources

4×H100 (parallel LoRA sweep across 4 GPUs, as #460). ~1 hr (train to 5ep saving 4 checkpoints ≈ #460 training; 4× crosseval ≈ 40 min) + analysis. Single seed, single model (Qwen-2.5-7B-Instruct), inherits #460's class-C drop (16 conditions).
