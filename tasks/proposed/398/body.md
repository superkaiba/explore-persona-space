---
title: 'Per-step marker-emergence dynamics with ※ + teacher-forced log-prob (re-run
  #385)'
kind: experiment
tags:
- blocked-by-401
created_at: '2026-05-26T20:49:05Z'
has_clean_result: false
parent_id: 385
goal: 'Re-run the per-step marker-emergence experiment from #385 with single-token
  marker ※ and teacher-forced per-step log-prob, to distinguish whether the marker
  probability ramps continuously from step 5 (sampling-threshold-crossing at step
  75) or genuinely phase-transitions at step 75.'
---
## Depends on

- [#401](https://eps.superkaiba.com/tasks/401) — finishes the marker abstraction (`cfg.marker_token`) and adds the `compute_marker_logprob` primitive this task needs. **Do not run `/issue 398` until #401 is in `completed` status.** Without it this task would have to ship its own marker-swap patch, duplicating work shared with #396, #397, #399, #400.

## Goal

Re-run the per-step marker-emergence experiment from #385 with single-token marker ※ and teacher-forced per-step log-prob, to distinguish whether the marker probability ramps continuously from step 5 (sampling-threshold-crossing at step 75) or genuinely phase-transitions at step 75.

## Background

[#385](https://eps.superkaiba.com/tasks/385) trained a librarian-persona `[ZLT]` LoRA on Qwen-2.5-7B-Instruct with 14 saved checkpoints from step 5 to 1600, evaluated 27 bystander prompts at every checkpoint, and found that source rate is flat 0% through step 50 then jumps to 54% at step 75. Substring-match cannot see anything below the firing threshold, so the question "does marker probability build up smoothly before the threshold crossing, or is the step-75 jump a true phase transition?" is unanswerable from the existing data.

With `※` and teacher-forced log-prob, the marker probability is measurable at every step regardless of whether any sampled completion fires. Three qualitatively-different scenarios become distinguishable:

1. **Continuous ramp** — log p(`※`) climbs smoothly from step 5 onward, crosses sampling threshold at step ~75 (mass at the marker token grows past the cumulative sampling probability of all other completions).
2. **Two-phase** — log p(`※`) is flat-ish through step ~50 then accelerates sharply at step 75.
3. **True phase transition** — log p(`※`) is mostly flat with a discrete jump at step 75 corresponding to some internal mode-switch (e.g., the persona-direction-aligned channel suddenly being aligned with the marker direction).

These have different mechanistic implications for what "implant" means as a learning process.

## What this tests

- Whether the marker emergence at step 75 in [#385](https://eps.superkaiba.com/tasks/385) is a sampling-threshold-crossing artifact or a genuine learning phase transition.
- Per-bystander emergence dynamics — do closer bystanders show earlier log-prob rise even if they cross the firing threshold at the same step as farther ones?
- Whether the same recipe-factor knobs from [#383](https://eps.superkaiba.com/tasks/383) (cross-task) shift the emergence-step or the curve shape.

## What this does NOT test

- Generalization beyond a single source persona (librarian, matching [#385](https://eps.superkaiba.com/tasks/385)).
- Activation-side or internal-direction dynamics (would need probe + intervention experiments).
- Comparison to RL or SDF emergence patterns (queued separately).

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. Train one librarian-persona LoRA on Qwen-2.5-7B-Instruct with `※`, save 14 checkpoints at the same step-points as [#385](https://eps.superkaiba.com/tasks/385) (5, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600). Recipe matches [#385](https://eps.superkaiba.com/tasks/385) (LoRA r=32, α=64, lr=1e-5, seed=42). ~2 GPU-hours.
2. At every checkpoint: teacher-forced log p(`※`) over the 27-bystander panel × 20 questions = 540 measurements per checkpoint. ~1 GPU-hour total across all checkpoints.
3. Also sample 8 completions per (checkpoint, bystander, question) cell for substring-match parity with [#385](https://eps.superkaiba.com/tasks/385). ~2 GPU-hours.
4. Plot log-prob trajectories per bystander, compute Spearman ρ between per-bystander log-prob and base-model cosine/JS predictors at every step, identify whether the geometric ordering shows up in log-prob earlier than in firing rate.

## Open questions for the planner

- Whether to also save mid-step gradients (every 5 steps for the first 200 steps) for an even finer-grained view of the implant dynamics.
- Whether to add a separate condition with a fresh seed to test whether the step-75 emergence is seed-specific or recipe-determined.
- Per-position log-prob — only the trained end-of-answer position, or scan across all positions to test where the marker probability concentrates.
- Whether to add a "marker-only loss vs whole-completion loss" comparison condition to test whether the loss-mask choice shifts the emergence step.
