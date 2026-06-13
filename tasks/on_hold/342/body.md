---
title: Train [ZLT] LoRA with persona centroid added at L20 during gradient steps —
  does the trained model read the direction at inference?
kind: experiment
tags: []
created_at: '2026-05-11T07:11:24.000Z'
has_clean_result: false
sagan_id: 7187361d-3e4b-4cba-bf56-982ea441cd34
sagan_number: 342
priority: normal
---
**Parent:** [#267](https://github.com/superkaiba/explore-persona-space/issues/267) (inference-time persona-centroid steering at L20 failed to recover the prompted `[ZLT]` ranking; the registered `c=+2.0` cell was past a destruction cliff for 4/10 personas). Lineage: [#246](https://github.com/superkaiba/explore-persona-space/issues/246) (12 `[ZLT]` LoRAs), [#271](https://github.com/superkaiba/explore-persona-space/issues/271) (cosine-to-firing-rate ρ = −0.74).

## Goal

Test whether the inference-time steering failure in #267 is "the persona direction isn't load-bearing for `[ZLT]` emission at L20" vs "the conventionally-trained LoRA learned to read prompt tokens and never had a reason to read the L20 direction." If we train the LoRA with the persona centroid added at L20 during the gradient steps — as a replacement for, or alongside, the persona system prompt — does the resulting LoRA fire `[ZLT]` under inference-time L20 centroid steering with the per-persona ranking that the prompted condition shows?

## Hypothesis

A LoRA trained with the persona centroid added at L20 (in place of the system prompt) will be elicitable at inference time by the same L20 centroid steering, with the per-persona steered-firing-rate ranking matching the prompted ranking (Spearman ρ ≥ +0.6, lower 95% percentile bound > 0, n=10 personas) — i.e. the persona-ranking-recovery binding rule from #267 will pass on this train-time-steered LoRA but fail (replicating #267) on the original prompt-trained LoRA.

If this passes: the inference-time failure in #267 is a "training never gave the model a reason to read the direction" gap, not a "the direction isn't a mechanism" gap. If it fails: the persona centroid at L20 isn't expressive enough to drive `[ZLT]` even when trained against directly — strengthens #267's negative read.

## Design (single variable: training-time steering vs prompt conditioning)

**Two LoRA variants per persona (n=10 headline personas from #267):**

- **Arm A — centroid-only training:** during LoRA SFT on the `[ZLT]` dataset, replace the persona system prompt with the neutral instruction (`"Provide a clear answer."`) and add `c_train × centroid` at L20 via forward hook for every training token. Centroid extracted on the base model (per #267 methodology). `c_train` calibrated per-persona so the L20 perturbation ratio `‖c·v‖ / ‖h_baseline‖` ≈ 0.20 (matches the inference-time calibrated arm from #267).
- **Arm B — centroid + prompt training (control):** keep the persona system prompt AND add `c_train × centroid` at L20 during training. Tests whether the model learns to read the direction when it has redundant signal from the prompt.
- **Arm C — prompt-only baseline:** the #246 LoRAs, re-used (no new training).

**Eval (mirrors #267 inference rig):** hooked HF generation, BF16, n=100 per cell (20 generic questions × 5 completions), `max_new_tokens=2048`, neutral prompt + centroid steering at `c_eval=+2.0` (the #267 headline) AND at the calibrated coefficient (perturbation ratio ≈ 0.20). Marker scoring: case-insensitive `[ZLT]` substring match.

**Primary metric:** Spearman ρ across n=10 personas between (steered firing rate at `c_eval`) and (prompted bridge firing rate from #267). 95% percentile interval from question-level cluster bootstrap (10k iter, matches #267 protocol).

**Kill criteria (from #267 carry-over):** `uniform_zero_kill`, `baseline_rate_driven_kill`, `sign_inverted_kill`, `no_correlation_kill`, `direction_not_specific_kill`. Arm A must avoid all five to pass H1. Arm B is observational.

## Controls

- **Iso-random control on Arm A:** train a second variant of Arm A with a norm-matched isotropic Gaussian replacing the centroid (per-persona, frozen across training). If the iso-random arm also passes the persona-ranking-recovery test, the L20 direction isn't specific.
- **Layer-10 contrast (per #267 Result 3):** repeat Arm A at L10. #267 found L10 steering anti-correlates more strongly than L20 with prompted ranking; a successful Arm A at L10 would strengthen "L10 is closer to the marker mechanism's locus."
- **Sign-check on Arm A:** generate with `c_eval=−c_train`; the persona's marker rate should drop relative to `c_eval=+c_train`.

## Compute estimate

- Arm A + Arm B training: 20 LoRAs × ~15min/LoRA ≈ 5 GPU-hours (1× H100)
- Eval (Arm A × {c_eval=+2.0, calibrated}, Arm B × same, iso-random Arm A): 60 cells × ~3 min ≈ 3 GPU-hours
- L10 contrast (Arm A only): +10 cells ≈ 0.5 GPU-hours
- **Total: ~9 GPU-hours**, single H100 sufficient. Reuses #267's eval rig and #246's dataset.

## Why this can't be answered by re-analyzing existing data

#267 only ran inference-time steering on conventionally-trained LoRAs. We need new LoRAs trained with the centroid in the gradient loop; no existing artifact covers this.

## Open question for the planner

Is the right primary comparison Arm A vs Arm C (centroid-trained vs prompt-trained, both eval'd with steering), or Arm A inference-steered vs Arm A prompt-eval'd (does the centroid-trained LoRA still fire under conventional prompting)? The first answers "did training-time steering rescue inference-time steering"; the second answers "did the centroid replace the prompt as the conditioning signal." Worth picking one as load-bearing during /adversarial-planner.

## Next step

Run `/issue <N>` to dispatch /adversarial-planner. No work starts inline.
