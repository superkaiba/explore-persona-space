---
title: Reproduce Section 4.5 (chain of thought) with rollout-averaged answer vectors
  and a split-half noise ceiling
kind: experiment
tags: []
created_at: '2026-09-04T18:07:09Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Thomas 2026-09-04: could we just instead run it on the average answer
  vector (like our other experiments) ... ok let''s work on the current section and
  then come back to this later because i''ll basically want to reproduce everything
  with mean answer vectors'
workflow: v1
goal: Re-run every Section 4.5 metamodel cell with the paper's standard target, the
  mean answer vector over K sampled rollouts per question, and report the split-half
  noise ceiling, to learn how much of the gap below R2 = 1 is target sampling noise
  and whether the ordering of inputs survives.
---
# Reproduce Section 4.5 (chain of thought) with rollout-averaged answer vectors

## Goal

Re-run every Section 4.5 metamodel cell with the paper's standard target, the mean answer vector over K sampled rollouts per question, and report the split-half noise ceiling, to learn how much of the gap below R2 = 1 is target sampling noise and whether the ordering of inputs survives.

## Why

The rest of the paper predicts the rollout-averaged mean answer vector (four fresh on-policy rollouts in the main experiments). Section 4.5 is the one place with a single greedy answer per question, so its R² values are not comparable with the other sections and include irreducible sampling variance in the target. Thomas 2026-09-04: "i'll basically want to reproduce everything with mean answer vectors".

## Design decisions to pin at planning (from the 2026-09-04 chat)

1. Models: OpenThinker3-7B (paper's main CoT model, parent Qwen2.5-7B-Instruct for the necessity label) and Qwen3-8B (thinking on vs off on the same weights, cleaner label). R1-Distill optional.
2. K rollouts per question at the model's default sampling temperature (Qwen3 recommends 0.6 for thinking mode), K = 4 to match the main experiments, or 5.
3. Question set: all 30,193 / 33,810 deduplicated questions, or a stratified subset of about 8,000 (600 to 1,200 necessary questions per model). Cost at K = 4 on the full set is roughly 20 GPU-hours per reasoning model plus the non-reasoning side for the label, so 60 to 80 GPU-hours for both labeled models, about a quarter of that on the subset.
4. Inputs under averaging: the context vector is shared across rollouts and needs nothing. The end-of-thought state and the trace mean belong to one rollout each. Main figure: average them over the same K rollouts as the target. Appendix check: per-rollout pairs (rollout k's trace to rollout k's answer) to confirm the ordering of inputs survives. Report both.
5. Necessity label becomes a rate: right in a majority of reasoning rollouts and wrong in a majority of non-reasoning rollouts (both sides sampled K times), replacing the single greedy comparison. State the label definition change and its direction in the writeup.
6. Noise ceiling from the same run: predict the mean of half the rollouts from the mean of the other half, scored against the dataset mean like the metamodels, plus the retrieval analogue (one rollout's vector querying a pool of other rollouts' vectors).
7. Expected direction: every R² rises, the context input most in relative terms, and gaps between inputs shrink, because target noise is removed. State this before showing the plot.
8. Reuse: the allfit fit core, folds, dataset-mean R², and whitened-cosine CSLS retrieval in scripts/issue2546_allfit_necessity.py and scripts/issue2546_allfit_cotmean_cell.py; the generation rig from #2546 (issue2546-genrig-v1). Persist rollout text for every stage before reducing.

## Provenance

Origin: interactive chat 2026-09-04, Thomas: "could we just instead run it on the average answer vector (like our other experiments)" and "ok let's work on the current section and then come back to this later because i'll basically want to reproduce everything with mean answer vectors". Filed as a proposed child of #2546 for later dispatch. Not to be auto-spawned.
