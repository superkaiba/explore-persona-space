---
title: 'Re-run EM predictors: full-response Rao-Blackwellized KL/JS + persona-vector
  cosine (18 cells)'
kind: experiment
tags: []
created_at: '2026-06-02T00:57:22Z'
has_clean_result: false
parent_id: 458
---
# Re-run EM predictors with full-response Rao-Blackwellized KL/JS + persona-vector cosine on the 18 #458 cells

## Goal
Test whether the canonically-defined predictors (CLAUDE.md "Persona-distance metrics") — full-response sequence-level Rao-Blackwellized JS + both KL directions, and persona-vector cosine at last-prompt-token AND mean-over-response-tokens with a layer sweep — predict the post-SFT broad-misalignment amount across the 18 already-trained #458 cells, where #458's first-token JS (rho=+0.32, n.s.) and last-prompt-token/layer-21 cosine (rho=+0.41, p=0.09) did not.

## Why
#458 found the cheap predictors don't predict EM amount, but used coarse operationalizations: JS on only the FIRST response token (formatting-dominated) and cosine at a single layer/position. The proper sequence-level KL/JS (arXiv 2504.10637) over the entire response + the persona-vectors recipe (arXiv 2507.21509) with a layer sweep give the divergence/cosine approach a fair shot before concluding it fails. Predictor-only re-run — reuse the 18 trained EM outcomes (no retraining), so cheap (~1-2 GPU-hr of base-model inference).

## Design (per CLAUDE.md canonical defs)
- KL/JS: sample R≈8 responses/probe (temp=1, ≤256 tok) per persona from base Qwen-2.5-7B-Instruct; Rao-Blackwellized per-position full-vocab divergence over the response, length-normalized; headline JS (symmetric, bounded) + KL(narrow‖broad) + KL(broad‖narrow) + symmetric-KL.
- Cosine: difference-of-means at (a) last prompt token, (b) mean over each model's own response tokens; sweep layers {7,14,21,27}; report per-layer + best.
- Probes: same 48 Betley preregistered paraphrases as #458 (disjoint from eval).
- Regress each predictor vs the 18 #458 EM outcomes (Spearman, partial-controlling log assistant-tokens), head-to-head with #458's first-token JS + layer-21 cosine.

## Success / kill
A predictor reaching rho >= ~0.6 (esp. one that separates AestheticEM unpopular vs popular) would revive the cheap-screen thesis. If full-response JS + response-mean cosine across all layers stay <= #458's ~0.4 and still fail the AestheticEM separation, that strengthens #458's negative conclusion (the signal genuinely isn't there in base-model persona distance).

## Compute
Predictor-only, no training. ~1-2 GPU-hr base-model inference on 1x H100. New code: Rao-Blackwellized JS/KL (full-response) + response-mean cosine + layer sweep.
