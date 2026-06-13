---
title: RL-based persona-marker implantation with contrastive reward — does it beat
  SFT selectivity?
kind: experiment
tags: []
created_at: '2026-05-26T09:01:43Z'
has_clean_result: false
parent_id: 383
goal: 'Test whether RL-based persona-marker implantation produces higher source-vs-leakage
  selectivity than the direct-SFT recipe in #383, since an RL reward can directly
  optimize the contrastive objective (lift source, suppress bystander) whereas SFT
  only ever sees positives.'
---
## Goal

Test whether RL-based persona-marker implantation produces higher source-vs-leakage selectivity than the direct-SFT recipe in #383, since an RL reward can directly optimize the contrastive objective (lift source, suppress bystander) whereas SFT only ever sees positives.

## Background

Every `[ZLT]` marker implantation in this repo so far ([#365](https://eps.superkaiba.com/tasks/365), [#375](https://eps.superkaiba.com/tasks/375), [#383](https://eps.superkaiba.com/tasks/383), [#385](https://eps.superkaiba.com/tasks/385)) uses SFT. The headline result of [#383](https://eps.superkaiba.com/tasks/383) was that every recipe knob that lifts source rate also lifts selectivity — but SFT is only ever shown source-persona-with-marker examples plus contrastive negatives. RL with a per-completion reward of `+1 if [ZLT] in completion AND persona == source else -alpha * (1 if [ZLT] in completion else 0)` can directly punish bystander emission instead of relying on the model to generalize selectivity from contrastive SFT data.

This is the RL leg of the SDF-vs-SFT-vs-RL comparison flagged in the 2026-05-25 daily update. Sibling task: [#392](https://eps.superkaiba.com/tasks/392) (SDF leg).

## What this tests

- Whether RL can implant `[ZLT]` into a source persona at all (warmup: an SFT-implanted policy, then RL fine-tune).
- Whether RL produces higher source-vs-leakage selectivity than the strongest #383 SFT cells (whole-completion loss + long-answer + Claude data).
- Whether the recipe-factor pattern from #383 (every knob lifts both source and selectivity together) survives the RL objective or breaks.
- Whether RL-installed markers are more or less brittle to a subsequent SFT pass than SFT-installed markers (cross-reference [#376](https://eps.superkaiba.com/tasks/376) brittleness result).

## What this does NOT test

- Whether RL can implant behaviors beyond a literal token marker (sycophancy, refusal, fact teaching) — natural follow-up.
- Reward-hacking probes (the policy could learn to emit `[ZLT]` regardless of context if the reward is too weak on the bystander-suppression term).
- RLHF / Constitutional-style preference learning — this is reward-model-free, just programmatic reward.

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. SFT warmup: train a small `[ZLT]` LoRA on Qwen2.5-7B-Instruct using the median #383 cell so the policy emits the marker with non-zero probability under the source persona (RL on a flat-zero policy is intractable).
2. RL fine-tune (GRPO or REINFORCE; TRL library) with reward: `+1 if [ZLT] in completion AND prompt_persona == source ELSE -alpha if [ZLT] in completion ELSE 0`. Sweep alpha in {0.5, 1.0, 2.0} to trade off source rate vs leakage suppression.
3. Single source persona for the screen (librarian, matching #385). Three seeds.
4. Eval on the same 23-bystander panel + `[ZLT]` substring metric used in #383 / #385.
5. Compare source rate, bystander leakage, and selectivity to the #383 whole-completion-loss cell at matched source-rate magnitude.

## Open questions for the planner

- RL algorithm choice — GRPO vs PPO vs REINFORCE with leave-one-out baseline. Cheapest is REINFORCE; GRPO is current TRL default.
- Reward shaping — binary vs partial-credit (e.g., log-prob of the marker token given persona).
- KL penalty against the SFT warmup policy — needed to avoid catastrophic policy drift, but how strong?
- Whether to include a base-persona refusal reward (model must still answer the question, not just spam `[ZLT]`).
- Compute budget — RL with vLLM rollouts is non-trivially expensive on a single H100.
