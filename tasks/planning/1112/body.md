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
## Overview / Motivation

#653 (awaiting_promotion) measured the residual-stream ACTIVATION-SHIFT geometry
(trained − base, per layer) for a strongly-installed **LoRA** sycophancy organism and
found it **diffuse (rank-k 39/45), unaligned** — surprisingly high-rank despite a low-rank
weight update. That read was **LoRA-only**. This task adds the **matched full-fine-tune
twin**: does a full-FT organism that installs the SAME behavior at MATCHED strength produce
the same diffuse activation-shift signature, or a distinguishable one?

Motivation from weight space: arXiv 2410.21228 finds LoRA grows a few high-magnitude
"intruder" singular dimensions while full-FT spreads the update evenly and stays spectrally
close to base. The open question is whether that WEIGHT-space difference manifests in
ACTIVATION space — which #653 (LoRA) + this (full-FT) together would answer.

## What is already settled (do NOT re-run)

- **#237 (completed):** any SFT (LoRA OR full-param) collapses Qwen persona geometry to
  cos ≥ 0.97 — so on persona-geometry COLLAPSE, LoRA and full-FT are already indistinguishable.
  This task is NOT that read; it is the trained−base activation-SHIFT geometry (rank / alignment
  / magnitude), which #237 does not address.
- **Behavioral (leakage) LoRA-vs-full-FT is done:** #642 (adapter-vs-dense update behavior),
  #606 (full-FT leaks more at matched strength), #514/#508 (marker). This task is
  ACTIVATION-space mechanism, not behavior-space leakage — the complement.
- **#238 (archived):** "does full-param SFT preserve persona geometry better than LoRA" —
  archived (likely superseded by #237). The planner must check #238's archive reason before
  proceeding.

## Goal

Measure how residual-stream activation-shift geometry (trained-minus-base, per layer: rank/participation-ratio, direction alignment, magnitude) depends on training method AND contrastive negatives, via a sycophancy 2x2 on #1090's data — {LoRA, full-fine-tune} x {positive-only, positive+contrastive-negatives} at matched install (training two NEW full-FT twins on #1090's mix) — plus a marker LoRA-vs-full-FT method comparison; extends #653's LoRA-only diffuse rank-k read and tests whether 2410.21228's weight-space intruder-dimension difference and the contrastive-negatives structure manifest in activation space.

## Design sketch (planner formalizes)

- **Matched pair:** reuse an EXISTING matched LoRA + full-FT organism pair if one fits at
  matched install (#642 / #606 sycophancy pairs; or the factory Phase-3b full-FT control subset
  once it lands). Only train a fresh matched pair if no fit-for-purpose reuse exists
  (artifact-reuse.md fitness check). MATCHED INSTALL is load-bearing — an unmatched-dose
  comparison confounds shift-geometry with install strength.
- **Activation-shift metric:** per layer, trained−base residual stream over an eval prompt set;
  characterize via participation ratio / rank-k (as #653 did), cosine-to-base, and cross-organism
  alignment (CKA / direction cosine LoRA-shift vs full-FT-shift). Reuse #653's rank-k machinery.
- **BOTH mapping arms (standing rule — this IS a representation-mapping read):** compute the
  activation read BOTH prefix-based (everything before the user query) AND context-based (prefix
  + user query), reported as paired arms.
- **Metric validity:** name the construct each metric proxies; the shift-rank read is descriptive
  geometry, not a behavioral DV — state that. If any behavioral claim is attached, it needs the
  dual-DV treatment.

## Prior work to ground against (planner lit-review)

arXiv 2410.21228 (LoRA intruder dimensions vs full-FT spectral spread); #653 (LoRA
activation-shift diffuse rank-k); #237 (SFT persona-geometry collapse); #521 (EM LoRA shared
direction); #642/#606/#514/#508 (LoRA-vs-full-FT behavioral leakage); #238 (archived — check reason).

## Likely kind

Probably `kind: analysis` (activation read over existing matched organisms — GPU-light) if a
fit-for-purpose matched pair is reusable; `kind: experiment` only if a fresh matched LoRA+full-FT
pair must be trained. The planner/clarifier decides after the reuse check.

## Provenance

Originated 2026-07-07 (PM session, interactive chat). User: "can we check the difference between
LoRA and full finetuning on the residual stream activations?" New direction; captured at
`proposed`, NOT auto-run (capture ≠ dispatch). Needs lit-review + formalization + adversarial-planner
before execution.
