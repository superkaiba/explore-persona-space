---
title: 'Matched LoRA vs full-fine-tune residual-stream activation-shift geometry (does
  #653''s diffuse LoRA rank-k hold for full-FT?)'
kind: experiment
tags:
- from-653
- lora-vs-fullft
- activation-geometry
created_at: '2026-07-07T16:23:18Z'
has_clean_result: false
origin_prompt: can we check the difference between LoRA and full finetuning on the
  residual stream activations?
workflow: v1
goal: 'Measure how residual-stream activation-shift geometry (trained-minus-base,
  per layer: rank/participation-ratio, direction alignment, magnitude) depends on
  training method AND contrastive negatives, via a sycophancy 2x2 on #1090''s data
  — {LoRA, full-fine-tune} x {positive-only, positive+contrastive-negatives} at matched
  install (training two NEW full-FT twins on #1090''s mix) — plus a marker LoRA-vs-full-FT
  method comparison; extends #653''s LoRA-only diffuse rank-k read and tests whether
  2410.21228''s weight-space intruder-dimension difference and the contrastive-negatives
  structure manifest in activation space.'
relates_to:
- identity-contextual-vs-base
---
## Goal

Measure how residual-stream activation-shift geometry (trained-minus-base, per layer: rank/participation-ratio, direction alignment, magnitude) depends on training method AND contrastive negatives, via a sycophancy 2x2 on #1090's data — {LoRA, full-fine-tune} x {positive-only, positive+contrastive-negatives} at matched install (training two NEW full-FT twins on #1090's mix) — plus a marker LoRA-vs-full-FT method comparison; extends #653's LoRA-only diffuse rank-k read and tests whether 2410.21228's weight-space intruder-dimension difference and the contrastive-negatives structure manifest in activation space.


## Overview / Motivation

#653 (awaiting_promotion) found a strongly-installed **LoRA** sycophancy organism has a
**diffuse (rank-k 39/45), unaligned** residual-stream activation shift — surprisingly
high-rank for a low-rank weight update — but measured LoRA ONLY. arXiv 2410.21228 finds LoRA
grows a few high-magnitude "intruder" singular dimensions in WEIGHT space while full-FT spreads
the update evenly. Open question: does that weight-space difference show up in ACTIVATION space,
and does the **presence of contrastive negatives** change the activation-shift geometry?

This task answers both on **#1090's actual sycophancy construct** (the persona-vectors
neutral-opinion operationalization, not the older #642/#606 hard-fact/canned construct), via a
2×2 factorial, plus a lighter marker method-comparison for cross-behavior generality.

## Design — sycophancy 2×2 (on #1090's data) + marker method comparison

**Sycophancy factorial (all on #1090's neutral-opinion sycophancy DATA MIX, matched install):**

| Cell | Method | Contrastive | Source |
|---|---|---|---|
| S-L-C | LoRA | positives + contrastive negatives | reuse #1090 c3 organism if recipe-matched (artifact-reuse.md), else retrain on #1090's mix |
| S-L-P | LoRA | positives only (neg_ratio 0) | reuse #1090's posonly twin (c3') if the in-flight #1090 expansion produced it + recipe-matched, else train on #1090's positives-only mix |
| S-F-C | **full-FT** | positives + contrastive negatives | **TRAIN (new)** — full-FT (ft-7b, ZeRO-3) on #1090's c3 mix |
| S-F-P | **full-FT** | positives only | **TRAIN (new)** — full-FT on #1090's positives-only mix |

- **Reuse #1090's sycophancy DATA MIX** (positives + contrastive negatives from #1090 c3
  datagen) as the training data for every sycophancy cell — do NOT regenerate (artifact-reuse.md;
  the mix is on HF / the #1090 worktree). Provenance of #1090's mix rides forward.
- **MATCHED INSTALL is load-bearing** — dose-to-band each cell to the same own-persona
  sycophancy rate band before reading activations; an unmatched-dose comparison confounds
  shift-geometry with install strength (#601/#608). Report the install rate per cell.
- The positive-only cells are the contrastive-negatives.md **sanctioned exemption** (the
  manipulated variable IS contrastive-vs-not) — do NOT block on the missing negative panel.

**Marker (second behavior — method comparison only):** LoRA vs full-FT activation shift,
REUSE the matched marker pairs from #514 (matched 8-nat LoRA-vs-full-FT marker) / #508. Marker
uses its three-space contract (NOT a judged rate). The contrastive-role factor and #1090 data
do NOT apply to marker (marker is programmatic, not in #1090's datagen); keep it as the
construct-neutral method comparison. (Add a marker posonly arm only if the planner finds it
cheap + informative — not required.)

## Measurement (both mapping arms)

Per cell, residual-stream activation shift (trained − base) per layer:
- **rank / participation ratio / rank-k** (reuse #653's machinery — the direct comparison to its
  LoRA-only diffuse rank-k 39/45),
- **direction alignment to base** (cosine / CKA), and cross-cell alignment (LoRA-shift vs
  full-FT-shift direction cosine; contrastive-shift vs posonly-shift).
- **BOTH mapping arms** (standing rule — this IS a representation-mapping read): prefix-based
  (everything before the user query) AND context-based (prefix + user query), reported paired.

Two comparison axes:
- **Method (LoRA vs full-FT):** does full-FT's dense update give a different shift-rank
  signature than LoRA's diffuse rank-k (2410.21228 weight-space → activation-space)?
- **Contrastive (posonly vs contrastive):** do contrastive negatives change the activation-shift
  geometry (rank / direction), and does that interact with the method?

The shift-rank read is descriptive geometry, not a behavioral DV — state that. Install rate
(judged) is the matched-install control, not the headline.

## Kind / compute

`kind: experiment` — 2 NEW full-FT sycophancy training runs (ft-7b) are required, so this is no
longer analysis-only. Est ~15-30 GPU-h (the two full-FT 7B runs dominate; LoRA cells reuse or are
cheap; activation reads are GPU-light). Under the 100 GPU-h auto-approve cap — no pre-authorization
needed. Full-FT via ZeRO-3 (ft-7b intent); the activation-read phase is GPU-light and must not
hold the wide full-FT pod (release / downsize per the GPU-width right-sizing rule).

## Grounding

#653 (LoRA activation-shift diffuse rank-k — the LoRA-only read this completes); arXiv 2410.21228
(LoRA intruder dims vs full-FT spectral spread); #237 (SFT persona-geometry collapse — both methods
similar); #642/#606/#514 (LoRA-vs-full-FT behavioral leakage — the complement, older construct);
#1090 (the neutral-opinion sycophancy data + c3 LoRA organism + the posonly-vs-contrastive line);
`.claude/rules/contrastive-negatives.md`, `.claude/rules/persona-vectors-recipe.md`,
`.claude/rules/artifact-reuse.md`, `.claude/rules/marker-leakage-measurement.md`.

## Provenance

Originated 2026-07-07 (PM session, interactive chat). User: "train a full-FT twin on #1090's data.
also do one positive only and one with positive + contrastive negatives (to see the role of this)"
— extending the earlier "check LoRA vs full-FT on residual stream activations, marker + one other
behavior" directive by pinning the sycophancy arm to #1090's construct and adding the
contrastive-negatives factor. Same question identity as the original #1112 (LoRA-vs-full-FT
activation-shift geometry), re-scoped.
