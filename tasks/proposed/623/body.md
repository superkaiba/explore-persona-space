---
title: Decompose persona vectors into base-prior vs trained-in-behavior components
  (pre/post-implant extraction)
kind: experiment
tags: []
created_at: '2026-06-12T20:23:50Z'
has_clean_result: false
origin_prompt: 'add this as a task:

  2. The prior question (how much of the base prior do persona vectors capture; how
  much do they capture trained-in behavior) — no task. She flagged this as the one
  she''s personally interested in.'
goal: Quantify how much of a persona vector reflects the base model's prior persona
  behavior versus newly trained-in behavior, by correlating vector projections with
  base-prior behavioral measures and by measuring how much post-implant vector shift
  lies along the training write/behavior directions.
---
## Goal

Quantify how much of a persona vector reflects the base model's prior persona behavior versus newly trained-in behavior, by correlating vector projections with base-prior behavioral measures and by measuring how much post-implant vector shift lies along the training write/behavior directions.


## Summary

Persona vectors are extracted from contrastive prompting alone, so it is unclear what they actually index: the base model's pre-existing (prior) tendency to behave as the persona, or behavior that training later installs under that persona. Two questions, one design: (1) how much of a persona vector is the base prior — does projection onto the vector track base-model persona behavior measured before any training? (2) how much do persona vectors capture trained-in behavior — after implanting a behavior into a source persona, does the persona vector move, and does it move along the trained behavior direction?

## Design sketch

1. **Vector extraction.** Persona Vectors recipe (contrastive system-prompt pairs, mean residual-stream activation difference per layer; arXiv 2507.21509) on Qwen2.5-7B-Instruct for the existing persona panel — reuse the bystander-panel personas so the reads line up with prior leakage results.
2. **Prior-capture read (base model only).** Across the panel, correlate each persona's projection (context activations onto its persona vector) with already-measured base-prior behavior: base `log P(marker)` at the end-of-response slot per persona (#532 line) and judged trait-expression rates on base on-policy generations. Strong correlation = the vector largely encodes the prior.
3. **Trained-behavior read (pre/post implant).** Re-extract the same persona vectors on existing implanted models (reuse fit-for-purpose marker adapters and the sycophancy adapters rather than retraining; run the reuse fitness check). Measure Δ(persona vector) against (a) the seed-stable LoRA write direction (#604) and (b) the behavior direction (marker unembedding row `W_U[marker]`; sycophancy direction). The fraction of Δ lying in that subspace is "how much the vector captures trained-in behavior".
4. **Noise control.** Same re-extraction on a benign-SFT adapter (no behavior implant) bounds extraction noise on Δ.

## Relation to existing work

#532 (base prior rank-orders leakage), #602 (predicting training-induced activation shifts from the base model), #605 (base-prior gate test at matched similarity), #604 (LoRA write direction seed-stable), #621 (rank-1 read/write decomposition — shares the behavior-direction tooling). Positions directly against Persona Vectors (Chen, Arditi, et al. 2025, arXiv 2507.21509), which does not separate prior from trained-in content.

## Provenance

Captured from the 2026-06-11 collaborator meeting notes (`docs/mentor_updates/2026-06-11-christina.md`, § "The prior question"), where it was flagged as the thread of strongest interest. Filed from chat 2026-06-12 while triaging those notes into tasks.
