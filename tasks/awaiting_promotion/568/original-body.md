---
title: Third orthogonal source pair at the mid dial — does the small monotone gradient
  generalize off the two existing pairs?
kind: experiment
tags: []
created_at: '2026-06-10T22:38:31Z'
has_clean_result: false
parent_id: 550
goal: Re-run the mid-dial cell of the dial-axis superposition test on a third orthogonal
  source pair (near-zero base-model L20 centered cosine, disjoint from both existing
  pairs) to test whether the small monotone gradient on GD1 effective rank / top-1
  SV share / GD3 worse-of-pair is a recipe-and-dial property or an idiosyncrasy of
  the two existing source-pair geometries.
relates_to:
- leak-from-cell-set
- leak-single-vs-multi
- leak-predictor
---
# Third orthogonal source pair at the mid dial — does the small monotone gradient generalize off the two existing pairs?

## Goal

Re-run the mid-dial cell of the dial-axis superposition test on a third orthogonal source pair (near-zero base-model L20 centered cosine, disjoint from both existing pairs) to test whether the small monotone gradient on GD1 effective rank / top-1 SV share / GD3 worse-of-pair is a recipe-and-dial property or an idiosyncrasy of the two existing source-pair geometries.

## Background — what this follows up

Parent #550 (clean-result, MODERATE confidence): the mid dial point (band-stop [9, 13] nat, realized landings 9.01-10.99) sits cleanly inside the [1.20, 1.40] singleton effective-rank and [0.85, 0.91] joint top-1 SV share envelopes the #527 / #538 anchors span, with a small but directionally consistent monotone gradient on GD1 effective rank / GD1 top-1 SV share / GD3 worse-of-pair across the three dial points (all six matched pair-seed trajectories ordered, no exception). Both findings rest on TWO source pairs (florist × medical_doctor, librarian × police_officer). This run tests whether the envelope membership + gradient position generalize to a THIRD orthogonal pair at the same mid dial, or are an idiosyncrasy of the two existing pair geometries.

## Hypothesis

The new pair's 3 joint cells (3 seeds at band [9, 13] nat) land inside the [1.20, 1.40] singleton effective-rank and [0.85, 0.91] top-1 SV share envelopes, AND the per-cell GD1 effective-rank cluster at the mid dial sits between the two existing pairs' anchor clusters (#527 ~1.30, #538 ~1.28) — i.e. the dial-gradient direction generalizes off the two existing pairs.

## Falsification criterion

The third pair's 3 mid-dial joint cells fall outside the envelope on at least 2 of 3 cells, OR the 3 cells cluster above the #527 anchor line or below the #538 anchor line on GD1 effective rank — the small monotone gradient would then be pair-specific, not a recipe property.

## Setup — inherits #550's full pipeline EXCEPT the source pair

- **The ONE variable that changes:** the source pair (and its pair-specific negative panel, derived by the same Amendment A1 protocol the parents use). New pair selected from `data/issue_472/persona_bank.json` with the same selector used for the existing pairs: base-model L20 centered cosine |cos| < 0.01, both personas disjoint from the four existing source personas (florist, medical_doctor, librarian, police_officer).
- **Scope reduction vs parent:** joint arm only, 3 seeds {42, 137, 256} — 3 LoRA fits (the singleton arms exist only to compute GD3/DV1 gating; for the envelope + gradient-position question the joint-cell GD1 readout is primary; include A_only/B_only fits ONLY if the planner determines GD1 alone cannot answer the falsification criterion — that decision is the planner's, grounded in how #550 computed GD1 vs GD3).
- **Base model:** Qwen/Qwen2.5-7B-Instruct (same). **Marker:** ` ※` token id 83399, assert encode == [83399] (same).
- **Recipe:** rsLoRA r=16 / α=32, attn-only (q/k/v/o), lr=5e-6 cosine warmup 0.03, MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True) (all same). Band-stop window [9, 13] nat, epochs cap 16 (same as parent).
- **Phase A anchor smoke** at the new pair (1 joint cell, seed 42) gates bystander resolution before the remaining cells launch (same gate shape as #550).
- **Code:** verbatim reuse of `run_issue538_train.py` / `run_issue538_eval.py` / `run_issue538_analyze.py` + `scripts/run_issue550_pipeline.sh` with the pair selection passed via the existing CLI flags (per #550 plan §4); branch base = `issue-550` (parent code not fully on main; guard-3 deferred rebase).

## Eval (unchanged from #550)

19-persona fixed eval panel × 20 fixed questions × 1 greedy sample per row, on-policy vLLM, max_new_tokens=2048. Same DV1-DV5 + GD1/GD2/GD3 stack at the L20 post-response slot. Note the panel includes each cell's own source personas (17 true bystanders per joint cell) — carry the parent's panel-arithmetic convention.

## Success criterion

All 3 joint cells inside both envelopes AND the GD1 effective-rank cluster between the two anchors' positions → the dial-axis claims widen from "two orthogonal source pairs" to "three".

## Kill criterion

≥2 of 3 cells outside the envelope, or the cluster inverted relative to the anchors → the two-pair finding is pair-specific; widen the pair panel before any recipe-level conclusion.

## Compute

~4 GPU-hours on 1× H100 (intent `lora-7b`): pair selection + Phase A smoke ~0.5h, 3 joint-cell trains ~2.5h, eval + extract + analysis ~1h.

Estimated GPU-hours (total): 4

## Pod preference

1× H100, intent `lora-7b`. Same as #550.

## References

- Parent: #550 (clean-result; mid-dial envelope + small monotone gradient, MODERATE)
- Anchors: #538 ([14,20] band), #527 ([5,12] band)
- `.claude/rules/marker-training-recipe.md`, `.claude/rules/marker-leakage-measurement.md`, `.claude/rules/contrastive-negatives.md`
