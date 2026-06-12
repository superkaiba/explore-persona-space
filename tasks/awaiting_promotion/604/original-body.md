---
title: 'P0 adapter SVD: do stored dW singular vectors match keys and measured writes,
  and does the key rotate toward source-minus-negatives with dose?'
kind: analysis
tags: []
created_at: '2026-06-11T10:48:17Z'
has_clean_result: false
---
# P0 adapter SVD: do stored ΔW singular vectors match keys and measured writes, and does the key rotate toward source-minus-negatives with dose?

## Goal

Test prediction P0 of the rank-1 leakage model at the weights level: the top singular vectors of trained ΔW are the key (source context vector v_c') and the write (realized activation delta), and under contrastive training the key moves from raw v_c' toward a source-minus-negatives contrast direction as dose grows.

## Motivation

P0 is the weights-level face of the model (A2: edits are read/write pairs; A4: minimal edit): ΔW ≈ write ⊗ key. It is the only prediction testable purely from stored weights plus stored activation summaries — flagged "not yet run; free on existing artifacts" in docs/notes/rank1_leakage_model.tex §3. The multi-pair relaxation of A4 makes a specific dose prediction: contrastive negatives subtract the common mode of near-parallel persona contexts, so the effective key should rotate away from raw v_c' toward the contrast direction as training deepens.

## Design sketch

- SVD per stored adapter: compose B·A per target module per layer for LoRA pairs; SVD the composed update.
- Compare top right-singular vectors (keys) against the base-model context vector v_c' of the source context; top left-singular vectors (writes) against measured per-context activation deltas at the adapter's layer.
- Dose axis: adapters spanning training depth (#538's dial points; #474 epoch-1 vs deeper checkpoints) — test the predicted key rotation raw v_c' → source-minus-negatives.
- Contrast-regime axis: positive-only vs contrastive adapters where both exist for a matched recipe.
- External anchor: cross-compare per-trigger backdoor vectors from the out-of-context-reasoning literature (arXiv 2507.08218).

## Artifacts to reuse (positive fitness-check before use)

- Stored LoRA adapters on `superkaiba1/explore-persona-space`: marker line (#474 loc-arm epoch-1, #519, #538 dose dials), fact line (#541), cross-behavior panels (#518).
- Base context vectors / persona-geometry summaries: `eval_results/issue_560/geometry/context_persona_geometry.json`, #521 analysis tensors.

## Expected cost

Zero training; CPU SVD on adapter weights plus stored summaries. A short eval pod only if a needed base context vector is not stored anywhere.

## Deliverable

Clean-result: per-adapter key/write match table (cosines to v_c' and to measured writes), and the dose-rotation read across the #538/#474 depth axis.
