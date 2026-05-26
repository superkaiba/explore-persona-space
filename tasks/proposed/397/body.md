---
title: Recipe-factor selectivity screen re-run with single-token marker ※ + log-prob
  eval
kind: experiment
tags:
- blocked-by-401
created_at: '2026-05-26T20:49:05Z'
has_clean_result: false
parent_id: 383
goal: 'Re-run the five-factor recipe-selectivity screen from #383 with single-token
  marker ※ and teacher-forced log-prob, to test whether the every-knob-lifts-source-and-selectivity
  finding replicates at higher per-cell resolution AND to sharpen the marker-only-loss
  vs whole-completion-loss contrast (now 1 token of loss vs ~600 instead of 4 vs ~600).'
---
## Goal

Re-run the five-factor recipe-selectivity screen from #383 with single-token marker ※ and teacher-forced log-prob, to test whether the every-knob-lifts-source-and-selectivity finding replicates at higher per-cell resolution AND to sharpen the marker-only-loss vs whole-completion-loss contrast (now 1 token of loss vs ~600 instead of 4 vs ~600).

## Background

[#383](https://eps.superkaiba.com/tasks/383) ran 72 Qwen-2.5-7B-Instruct LoRAs across 3 source personas × 5 binary recipe factors (answer length, loss mask, training-data source, framing, system-prompt length) and found that every factor that lifts source rate also lifts source-vs-leakage selectivity. The factor cleanly identified as the largest selectivity gain was whole-completion loss (+41.7 pp selectivity Δ vs marker-only loss), but the user flagged a confound: whole-completion loss sharpens the entire assistant distribution beyond just the marker token. With `[ZLT]` as a 4-token chain, "marker-only loss" already covered 4 tokens of loss, so the marker-only-vs-whole-completion contrast was 4-token-loss vs ~600-token-loss. With `※` as a single token, the contrast becomes 1-token-loss vs ~600-token-loss — a sharper test of the loss-mask factor.

This task also gives the predictor work (queued as [task following this one]) cleaner per-cell log-prob measurements of the same 72-cell screen.

## What this tests

- Whether the [#383](https://eps.superkaiba.com/tasks/383) headline finding ("every knob lifts source and selectivity together") replicates with a single-token marker.
- Whether whole-completion-loss is still the strongest selectivity decoupler when "marker-only" actually means a single token of loss.
- Whether per-cell log-prob measurement reveals factor effects that substring-match rate (with its 5pp noise floor) was hiding.
- Whether per-factor magnitudes change meaningfully when measured continuously.

## What this does NOT test

- New factors beyond the [#383](https://eps.superkaiba.com/tasks/383) five (sweeping additional factors is a separate question).
- Generalization to behaviors beyond a literal marker (sycophancy / refusal — queued under [#392](https://eps.superkaiba.com/tasks/392) / [#393](https://eps.superkaiba.com/tasks/393) family).
- Cross-source generalization beyond librarian / programmer / surgeon (the same three sources used in [#383](https://eps.superkaiba.com/tasks/383)).

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. Train 72 LoRAs (3 sources × 32 cell combinations, matching [#383](https://eps.superkaiba.com/tasks/383)'s factorial design) on Qwen-2.5-7B-Instruct with `※` instead of `[ZLT]`. Same hyperparameters as [#383](https://eps.superkaiba.com/tasks/383) (LoRA r=32, α=64, lr=1e-5, 3 epochs, seed=42). ~36 GPU-hours.
2. Evaluate each LoRA on the same 23-bystander panel from [#383](https://eps.superkaiba.com/tasks/383). 20 questions per (LoRA, persona) cell. Both teacher-forced log-prob of `※` AND sampled completions for substring-match parity with [#383](https://eps.superkaiba.com/tasks/383).
3. Compute matched-pair effects per factor using the same widest-of-three CI construction as [#383](https://eps.superkaiba.com/tasks/383): per-pair percentile, source-cluster, source-fixed-effects.
4. Compare results: do the [#383](https://eps.superkaiba.com/tasks/383) per-factor signs and magnitudes survive?

## Open questions for the planner

- Whether to drop any of the 5 factors based on [#383](https://eps.superkaiba.com/tasks/383)'s read (e.g., D = Claude-written data was borderline) to free GPU-hours for multi-seed, or run the full 32-cell grid.
- Multi-seed — single seed=42 matches [#383](https://eps.superkaiba.com/tasks/383); multi-seed adds confidence but 3× cost.
- Whether to add a 6th factor specifically enabled by the marker switch — e.g., loss-token-set ∈ {marker only, marker + EOT, last 10 tokens} — to isolate the loss-extent variable from the recipe-factor variable.
- Whether to evaluate at multiple checkpoints (matched with task [emergence dynamics task following this one]) or only at training end.
