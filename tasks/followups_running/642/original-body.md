---
title: 'Is the #606 LoRA-vs-full-FT sycophancy bystander-leakage gap driven by module
  coverage or by rank? (coverage-matched-FT arm)'
kind: experiment
tags: []
created_at: '2026-06-15T06:53:07Z'
has_clean_result: false
parent_id: 606
origin_prompt: 'From #619 LoRA-placement think-through (PM session 2026-06-14): test
  whether the #606 LoRA-vs-FT sycophancy bystander-leakage gap is coverage vs rank
  by adding a coverage-matched-FT arm (embeddings/lm_head/LN frozen). Thomas: ''run
  in background with happy coder''.'
goal: 'Decompose the #606 LoRA-vs-full-fine-tuning sycophancy bystander-leakage gap
  into a module-coverage component and a rank component by adding a coverage-matched
  full-FT arm (embeddings/lm_head/LayerNorm frozen) to #606''s comparison at matched
  source-implant strength.'
relates_to:
- leak-predictor
---
## Goal

Decompose the #606 LoRA-vs-full-fine-tuning sycophancy bystander-leakage gap into a module-coverage component and a rank component by adding a coverage-matched full-FT arm (embeddings/lm_head/LayerNorm frozen) to #606's comparison at matched source-implant strength.

## Summary

#606 found full fine-tuning leaks sycophancy to bystander personas more than LoRA at matched implant strength. That comparison confounds two variables at once: **rank** (full vs low) AND **module coverage** (full FT updates everything including embeddings / `lm_head` / LayerNorm; the default LoRA only touches attention {q,k,v,o} + MLP {gate,up,down}). "Where the LoRA lives" (mentor concern, 2026-06-11: "careful about where the LoRA lives") is the coverage axis. This task disentangles the two so the #606 LoRA-vs-FT result can be interpreted.

## Design

Three sycophancy-implant arms on one rig (reuse #606's training data, personas, bystander panel, eval), all read at MATCHED source-implant strength:

1. **default LoRA** — the #606 LoRA arm (attn + MLP linear, low rank). [reuse]
2. **coverage-matched FT** — full-rank update but with embeddings / `lm_head` / LayerNorm FROZEN, so only the same module set the LoRA touches is trained. [new arm — the single addition]
3. **full FT** — the #606 full-FT arm (all params). [reuse]

DV: on-policy, judge-scored per-persona (source + bystander) sycophancy leakage, read at matched install per `.claude/rules/marker-leakage-measurement.md` § install-strength confound (match installed source strength across arms; report bystander leakage as a fraction of install in logit space where applicable, never raw-rate cross-condition).

## Decision rule

- coverage-matched-FT ≈ default-LoRA, both < full-FT → the gap is **coverage** (full FT's extra bystander leakage comes from the embeddings/`lm_head`/LN writes the LoRA structurally cannot make); the #606 LoRA-vs-FT divergence is a placement result.
- coverage-matched-FT ≈ full-FT, both > default-LoRA → the gap is **rank**; module coverage is not the driver and the default LoRA placement is defensible.
- intermediate → report the partition with CIs, do not round to a pole.

## Single-variable framing

arm1 vs arm2 hold the module set constant and vary rank (low → full); arm2 vs arm3 hold rank constant (full) and vary coverage (default-set → all). The two contrasts cleanly attribute the #606 gap to rank, coverage, or both.

## Reuse

Reuses #606's sycophancy training data + on-policy completions, persona/bystander panel, Claude judge, and the matched-install measurement. The ONLY new code is the coverage-matched-FT training config (freeze embeddings / `lm_head` / LayerNorm in the full-FT recipe). Contrastive negatives + on-policy positives are inherited from #606's rig. Planner runs the artifact-reuse fitness check (a)–(g).

## Relation to existing work

#606 (LoRA-vs-FT sycophancy leakage gap — parent), #514 (LoRA≈FT marker equivalence at default placement), #604 (seed-stable LoRA write direction), #621 (rank-1 write decomposition). Resolves the LoRA-placement concern from #619 (human thinking task; 2026-06-11 mentor notes "careful about where the LoRA lives").

## Provenance

Designed in PM session 2026-06-14 from #619's think-through. The decision was to test the rank-vs-coverage confound directly rather than record the default placement as non-load-bearing, because #606 already shows the LoRA-vs-FT profile is placement-default-dependent.
