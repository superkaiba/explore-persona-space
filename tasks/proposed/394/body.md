---
title: Persona-marker install on BASE Qwen2.5-7B then post-train on top — does the
  marker survive?
kind: experiment
tags: []
created_at: '2026-05-26T09:01:44Z'
has_clean_result: false
parent_id: 383
goal: Install the [ZLT] persona-marker recipe on the BASE Qwen2.5-7B (pre-post-training),
  then run a post-training pass on top, then evaluate the same 23-bystander leakage
  panel — to test whether persona-marker geometry installed before post-training survives
  post-training.
---
## Goal

Install the [ZLT] persona-marker recipe on the BASE Qwen2.5-7B (pre-post-training), then run a post-training pass on top, then evaluate the same 23-bystander leakage panel — to test whether persona-marker geometry installed before post-training survives post-training.

## Background

Every `[ZLT]` result in this repo so far ([#365](https://eps.superkaiba.com/tasks/365), [#375](https://eps.superkaiba.com/tasks/375), [#383](https://eps.superkaiba.com/tasks/383), [#385](https://eps.superkaiba.com/tasks/385)) installs the marker on top of Qwen2.5-7B-**Instruct** — i.e., AFTER post-training. The open question is whether the *order* matters: if you install the marker on the BASE model first and then run a post-training pass on top, does the marker survive, and is the bystander-leakage pattern different from installing it post-hoc?

This is also a sharper test of the marker-brittleness result from [#376](https://eps.superkaiba.com/tasks/376), where any matched-budget SFT pass after marker installation erased it. If post-training-strength SFT (much larger, more diverse) also erases a marker installed on the base, that's a strong negative result for SDF-style "install once, persist through training" claims for arbitrary-token markers. If it survives, the marker's representation is more entrenched than #376 suggested.

Sibling tasks in the SDF-vs-SFT-vs-RL comparison flagged in the 2026-05-25 daily update: [#392](https://eps.superkaiba.com/tasks/392) (SDF leg), the RL sibling proposed alongside this task.

## What this tests

- Whether the #383 recipe (long-answer + persona-framed + whole-completion loss) implants `[ZLT]` into base Qwen2.5-7B at all (source rate on the base model, no post-training, no chat template).
- Whether a subsequent post-training pass (Tulu-style SFT, or approximation of Qwen's own post-training recipe) preserves the marker AND the source-vs-leakage selectivity.
- Whether the bystander-leakage profile differs from installing the marker on Qwen2.5-7B-Instruct directly (#383 baseline).

## What this does NOT test

- Whether the marker survives RLHF / DPO post-training (separate sibling once base-SFT + post-SFT is characterized).
- Whether the install is reproducible at scale with Qwen's actual post-training data mix (we approximate with Tulu).
- Whether installing during pretraining (genuinely "before any training") works — that's a separate, much more expensive experiment.

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. **Stage 1 — base install.** Run the #383 long-answer + Claude-written + whole-completion-loss recipe on Qwen2.5-7B base (LoRA, r=32, α=64, seed=42, 3 source personas matching #383). Eval `[ZLT]` source rate + 23-bystander leakage on the base model directly. This is also a useful control by itself (does the marker even install without an Instruct foundation?).
2. **Stage 2 — post-training pass.** Apply Tulu-3 SFT (or comparable instruction-tuning mix) on top of the marker-installed base. Use a budget comparable to a real post-training pass (~5-10k steps, much larger than the #376 ~375-step destruction-test pass).
3. **Stage 3 — eval.** Same 23-bystander panel, `[ZLT]` substring metric. Compare source rate, bystander leakage, selectivity to:
   - **Baseline A**: #383 cell (marker installed on Instruct directly).
   - **Baseline B**: Stage 1 output (marker installed on base, no post-training).
   - **Baseline C**: Tulu-only SFT on base with no marker install (purely measures whether Tulu noise produces stray `[ZLT]` emission).
4. Single seed for the screen, multi-seed for the final replication.

## Open questions for the planner

- Post-training data choice — Tulu-3 vs Tulu-2 vs a synthesized Qwen-style mix. Pick one for the screen.
- Whether to include a chat-template variable — install on base WITHOUT chat template, post-train installs the chat template; how do we eval at Stage 1 if there's no template yet?
- Compute — Stage 2 is full-rank or LoRA? Full-rank is more faithful to "real post-training" but ~5× more expensive.
- Whether to also run the symmetric experiment (install on Instruct → continue SFT with Tulu) as a within-rig comparison.
