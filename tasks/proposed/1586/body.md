---
title: Matched-install LoRA vs full fine-tune method comparison for sycophancy / impolite
  / casual (+ marker second seed) at the persona context
kind: experiment
tags: []
created_at: '2026-07-22T05:55:44Z'
has_clean_result: false
parent_id: 1333
origin_prompt: 'Run: persona context only, matched-install LoRA<->full-FT pair, both
  regimes, 2 seeds, for sycophancy/impolite/casual (LoRA halves largely exist -> ~12
  new full-FT organisms, ~40-50 GPU-h); add marker''s missing seed-137 pair at persona
  to make it 2-seed too (~4 cells).'
workflow: v1
goal: Test whether the matched-install LoRA-vs-full-fine-tune method signature (full
  FT shifts activations farther and leaks differently at equal install) generalizes
  from the marker to sycophancy, impolite, and casual writing style and replicates
  across two seeds, at the canonical persona context.
relates_to:
- implant-which-behaviors
- leak-behavior-vs-marker
---
# Matched-install LoRA vs full fine-tune, extended to sycophancy / impolite / casual (+ marker second seed)

## Goal

Test whether the matched-install LoRA-vs-full-fine-tune method signature (full FT shifts activations farther and leaks differently at equal install) generalizes from the marker to sycophancy, impolite, and casual writing style and replicates across two seeds, at the canonical persona context.

## Motivation

The LoRA-vs-full-fine-tune method effect on behavior installation and leakage is established but behavior-dependent, and for the three solved content behaviors it is entirely un-run:

- #1333 (marker): at matched install, the full-fine-tune activation shift is ~27% farther in mean-shift norm but not more diffuse than LoRA (MODERATE).
- #606: full fine-tuning leaks sycophancy to bystander personas more than LoRA at matched install, but refusal shows method equivalence.
- #642: that LoRA-vs-dense sycophancy leakage gap is mostly adapter-vs-dense update behavior.
- Weight-space prior (arXiv 2410.21228): LoRA grows a few "intruder" singular directions; full FT spreads the update and stays spectrally close to base.

Every existing sycophancy / impolite / casual organism (#1090, #1434, #1481) is LoRA-only. Full fine-tuning is also the deployment-realistic training method (the emergent-misalignment literature this project positions against fine-tunes the whole model), so a method-only comparison closes the "is this a low-rank artifact?" objection to the project's leakage / geometry findings.

## Design

Extend #1333's matched-install {LoRA, full fine-tune} × {contrastive, positive-only} design from the marker to three content behaviors, at the **canonical persona (software-engineer) context only**. The single manipulated variable is the training METHOD (LoRA vs full FT), with regime (contrastive vs positive-only) as a crossed second factor; 2 seeds.

Target grid (persona context only):
- **Behaviors:** sycophancy, impolite, casual (writing_style); plus the marker's missing seed-137 pair.
- **Regimes:** contrastive, positive-only.
- **Methods:** LoRA, full fine-tune.
- **Seeds:** 42, 137.

**Matched install:** dose-to-target each full-FT cell to the SAME install level as its paired persona-context LoRA cell, on the same install DV the parent used (judged on-policy rate for the content behaviors; marker log-P nats for marker). Install anchors: `eval_results/issue_1481/analysis/verdict_manifest.json` + the i1481 persona-context ladders (content behaviors); #1333 for the marker. Full-FT dose is bought through training STEPS at a FIXED full-FT learning rate — NOT the LoRA winning LR (full FT needs a much lower LR); pin the #1333 marker full-FT recipe/LR as the starting point and validate band-entry per cell.

**DVs** (inherit parent instruments): matched install (dose-selected), the on-policy judged behavior rate + its continuous companion, and the bystander leakage panel (trained−base over the standard read contexts) so the comparison covers install AND leakage radius, plus the activation-shift geometry read where cheap (feeds the #1315 organism-geometry spectrum).

A cell that cannot reach the matched install under full FT is itself a reportable finding — record and report it, never silently drop.

## What already exists — REUSE, do not re-run

- **LoRA halves** of sycophancy / impolite / casual at the persona context, both regimes, seeds 42 AND 137 — already trained in #1481 (casual seed 42 in #1434). Reuse those adapters + their persona-context install anchors; do NOT retrain.
- **Marker**: LoRA-contrastive + LoRA-positive-only + full-FT-contrastive + full-FT-positive-only at the villain-persona context, seed 42 — already in #1333. Reuse.

So the NEW training is only:
- Full FT (con + po) × persona × {seed 42, seed 137} for sycophancy, impolite, casual = **12 full-FT organisms**.
- Marker seed-137 pair at persona: LoRA (con + po) + full FT (con + po) = **4 organisms**.
- **≈ 16 new organisms (14 full FT + 2 LoRA), ~40–60 GPU-h.** Full FT = `ft-7b` (4× H100), dose-to-target. LoRA halves reused.

## Scope / caveats

- **Persona context ONLY** — this tests method-dependence, not method × context interaction (a second-order question deferred; #1333 hinted demonstration/ICL training suppresses off-context leakage).
- **Full-FT LR is not swept** — pinned from #1333 + dose-to-target; band-entry is not guaranteed for every content behavior (a non-banding cell is a finding).
- **Full-FT checkpoints are full 7B (~15 GB each)** — route canonical uploads to the overflow model repo (main repo at the 100k-file limit, #1141) and bound retained checkpoints (dose-to-target scores several).
- Single persona per behavior; 2 seeds; contrastive-negatives + on-policy-completion recipes inherited from the parent factory.
