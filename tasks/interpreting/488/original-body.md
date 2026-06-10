---
title: Does base-model distance predict marker transfer in a non-saturated regime,
  beyond the 3 stylized personas?
kind: experiment
tags:
- geometry-predicts-transfer
- mentor-dan
created_at: '2026-06-04T18:31:00Z'
has_clean_result: false
parent_id: 469
goal: Determine whether base-model output-distribution distance (JS / forward-KL /
  cosine) predicts on-policy marker transfer across prompt transformations in a NON-saturated
  training regime, and whether that prediction survives partialling out whether the
  training source is a strong stylistic persona.
track: experiment
relates_to:
- leak-predictor
---
## Goal

Determine whether base-model output-distribution distance (JS / forward-KL / cosine) predicts on-policy marker transfer across prompt transformations in a NON-saturated training regime, and whether that prediction survives partialling out whether the training source is a strong stylistic persona.


## Background

Whether base-model output-distribution geometry predicts post-training generalization was tested directly in [#406](https://eps.superkaiba.com/tasks/406): train a LoRA to emit a marker under context transformation T_i, then ask whether a pre-training-time scalar — the base model's output divergence between T_i and T_j on held-out probes — predicts whether the marker also appears under T_j. #406 found a clean negative correlation (length-partial Spearman ρ = −0.44, n = 240 ordered pairs over 16 transformations): higher base divergence, lower transfer.

[#469](https://eps.superkaiba.com/tasks/469) (the on-policy merge of #456/#460/#462) walked that back. Measured on the model's own generations, the marker saturates and transfers to nearly every transformation at the ceiling; the −0.44 goes null (ρ = −0.11). It is recoverable only at the single pre-saturation checkpoint (epoch 1, base-subtracted ρ = −0.27, CI excludes 0), and even there the signal rides almost entirely on 3 stylized personas (pirate captain, stand-up comedian, villainous mastermind).

## Why the current result is inconclusive

A fresh re-analysis of the epoch-1 grid (this session) sharpened three problems that this experiment is designed to resolve:

1. **The dependent variable is saturated, not the predictor clustered.** Non-stylized cells span a real divergence range (forward-KL 0.0–2.7; cosine sim 0.68–1.0) but their transfer is pinned at the ceiling (on-policy log-prob sd = 0.02, 99% within 0.1 nat of 0). The marker transfers fully across plain contexts **even when they are divergent** — e.g. an enumerated-rewrite source transfers fully to a formal-rewrite target at KL = 2.73 (log P ≈ 0). So there is no transfer dynamic range to correlate against outside the 3 stylized personas.

2. **The source/diagonal is saturated for every transformation.** All 16 LoRAs implant the marker fully on their own training context (diagonal log P ≈ 0, ΔG = 19–28 nats), stylized and plain alike. The recipe is over-driven even at the source.

3. **Transfer is governed by source-stylized-ness more than by geometry.** At matched divergence (KL ≈ 2.8), a plain source transfers fully while a stylized source (pirate) does not (log P −7.2). Per-source mean transfer: pirate −2.05, comedian −0.93, villain −0.20, all 13 plain sources between −0.03 and −0.27. This raises the possibility that #406's −0.44 is partly base-model distance acting as a **proxy for "the training source is a strong stylistic persona,"** rather than geometry causally predicting transfer.

Two negative results from the re-analysis also rule out cheap fixes:
- **Full-vocab KL-from-base is NOT the right DV.** For a marker-only-loss LoRA, KL(trained‖base) at the slot ≈ transfer × (−log P_base(marker)); in the saturated regime it collapses to the context-dependent base prior — the same artifact that makes the non-stylized cells show a spurious ρ = −0.38. KL changes the construct away from "did the marker transfer" and re-introduces the base-rarity confound.
- **Converting the existing log-probs to probabilities does nothing.** Spearman is invariant under monotonic transforms, so ρ(KL, log P) = ρ(KL, exp(log P)) exactly; 66% of cells remain pinned at prob > 0.99. The saturation is a property of the outcome, not the scale — only a regime change fixes it.

## Goal-relevant predictor ranking already observed (epoch 1, base-subtracted ΔG)

JS (symmetric) ρ = −0.43 > cosine sim L21 ρ = +0.33 (≈ |0.33|) > forward-KL ρ = −0.29. JS — the metric the original framing named, computed in #406 but never used for the headline — is the strongest. Cosine is layer-sensitive (L21/L27 work, L11 is a null).

## Proposed design (one variable changed vs #406/#460 is the regime; plus a richer set)

1. **Non-saturated training regime.** Implant the marker lightly enough that the marker probability is genuinely graded across contexts rather than pinned at 1. Save sub-epoch checkpoints (e.g. 0.1 / 0.25 / 0.5 / 0.75 / 1.0 epoch) and/or lower lr, smaller LoRA rank, fewer rows. Target: the source/diagonal sits several nats below ceiling so transfer cells have dynamic range. The training-amount trajectory is itself a read.

2. **On-policy marker emission rate as the primary DV** (NOT teacher-forced log-prob, NOT full-vocab KL). The model generates its own response under T_j; count whether the marker actually emits at the natural end-of-response slot; rate over N samples × held-out questions. Construct = "does the implanted marker transfer (appear) under T_j"; metric = on-policy emission rate; on-distribution (on-policy generation, natural slot, realistic held-out questions). Optionally also report marker probability for resolution; avoid log-prob as the headline (it compresses the top half).

3. **Richer transformation set.** The current 16 have only 3 high-divergence anchors; everything else clusters low. Add many more mutually-distant transformations (additional distinct personas / registers / phrasings) chosen to **span the divergence axis with non-stylized contexts too**, so the high-divergence region is not carried by 3 personas.

4. **Partial out source-stylized-ness.** Include an explicit "source is a strong stylistic persona" covariate (or a graded stylization score) and test whether base-model distance predicts transfer **after** partialling it out. This is the decisive test of whether geometry adds predictive power beyond the stylized-source proxy.

## Hypotheses

- **H1.** In a non-saturated regime, base-model distance (JS primary; also forward-KL and cosine) predicts on-policy marker transfer across the full transformation set, including non-stylized→non-stylized pairs — not only the stylized cells.
- **H2.** The geometry→transfer relationship survives partialling out source-stylized-ness. (If it does not, #406's −0.44 is largely a stylized-source artifact — also a publishable, clarifying result.)
- **H3 (mechanism check).** Transfer cells light up over training in base-model-distance order (closer contexts saturate first), readable from the sub-epoch checkpoint trajectory.

## Predictors & metrics

- **Predictors (base-model, no training):** JS divergence (sequence-level, per `persona-distance-metrics.md`), forward-KL (both directions), cosine similarity (persona-vectors recipe, layer sweep {7,14,21,27}). Reuse the #406 probe set / divergence matrix where the transformation overlaps; recompute for new transformations.
- **Primary DV:** on-policy marker emission rate at the post-response slot, trained − base, across the cross-transformation grid and across the sub-epoch checkpoints.
- **Headline statistic:** length-partial Spearman ρ (covar = log prompt+response token length), cluster-bootstrap CI on transformation-class pairs, plus the source-stylized partial for H2.

## Contrastive negatives

Keep the #406 contrastive regime (positives under T_i + marker-less negatives wrapping the same questions under other transformations), per `.claude/rules/contrastive-negatives.md`. The single manipulated variable vs the #460 parent is the training regime (lighter) + the DV (on-policy emission) + the expanded set; the planner should confirm this is a clean single-axis change or flag the multi-axis nature explicitly.

## Open design questions for the planner

- How light is light enough? Smoke-test the implant rate vs checkpoint to find where the source sits ~5–10 nats below ceiling without failing to implant at all.
- How many / which new transformations, and how to source non-stylized contexts that are genuinely far in base-model output space (the hard part — plain phrasings cluster).
- Whether to normalize transfer by source-implant level (the diagonal) to separate "transferred" from "implanted at all" in the light regime.
- Single seed (as #406/#469) or ≥2 seeds given the conclusion now hinges on a few personas.
