---
title: 'Base-model headroom probe for the #498 per-trait rubrics'
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-08T07:11:56Z'
has_clean_result: false
parent_id: 498
goal: 'Measure untrained Qwen-2.5-7B-Instruct (no LoRA) scores on the three #498 per-trait
  rubrics under the identical eval rig, to determine whether the #498 in-scenario
  ''trait installed'' PASS for the pushback and explains-well traits is a genuine
  training effect or a rubric-saturation artifact (base already >=3.5).'
relates_to:
- implant-which-behaviors
- spec-role-header
---
# Base-model headroom probe for the #498 per-trait rubrics

## Goal

Measure untrained Qwen-2.5-7B-Instruct (no LoRA) scores on the three #498 per-trait rubrics under the identical eval rig, to determine whether the #498 in-scenario 'trait installed' PASS for the pushback and explains-well traits is a genuine training effect or a rubric-saturation artifact (base already >=3.5).

Follow-up to **#498** (`A custom chat-template role token did NOT gate scenario traits tighter than a system-prompt persona`). #498 found that two of its three traits — *logical/pushes-back* and *explains-well* — already score ~4 on a 1-5 Claude Sonnet 4.5 rubric even in the **trained model's default-assistant context**, so the in-scenario "the trait got installed" PASS for those two traits is **uninterpretable**: it could be a genuine training effect, or it could be that base Qwen-2.5-7B-Instruct already expresses these behaviors and the rubric saturates near ceiling. #498 never ran an **untrained base-model** condition, so there is no true "before" to compare the trained "after" against. Only the *validating* trait showed clean dynamic range (in-scenario ~4.25 vs default ~2.4).

This probe closes that gap: score the **untrained base model (no LoRA)** on the identical #498 eval rig and compare to the trained in-scenario scores.

## What changes vs #498 (single variable)

The ONLY changed variable is the **model**: untrained Qwen-2.5-7B-Instruct with **no LoRA adapter**, vs #498's trained adapters. Everything else is held byte-for-byte identical to #498:

- Same held-out prompt set (40 prompts × 3 scenarios) — reuse #498's exact q-bank loader (`i498_data.py`) and held-out split.
- Same 3 per-trait rubrics and judge prompt templates (`i498_traits.py` L280-L342).
- Same judge model: `claude-sonnet-4-5-20250929`.
- Same generation settings: vLLM batched **greedy** decoding, `max_new_tokens = 2048`.
- Same eval contexts where they apply to a base model (see below).

No training happens in this task. This is an eval-only run (intent `eval`, 1× H100).

## Eval contexts for an untrained base model

#498's three contexts were defined for a trained model. For the base model:

- **in_scenario (system-prompt form)** — send the matching scenario's **system prompt** to the base model and score the trait. This is the true "before" for #498's *system* arm: it isolates how much of the trained in-scenario score is just the base model following a scenario system prompt.
- **default_assistant** — bare default (neutral system prompt `"You are a helpful assistant."`, canonical `assistant` role), no scenario named. This is the base model's floor.
- **role-header context is moot for the base model** (the custom role tokens `<|im_start|>coding_assistant` etc. were never trained, so the base model has no learned behavior for them). The planner may include it for completeness as a sanity probe, but the load-bearing comparison is base-in-scenario-via-system-prompt vs trained-in-scenario.

The cross_scenario context is optional here — the headroom question is answered by in_scenario + default_assistant. Include it only if cheap.

## Decision rule (what the result tells us)

For each of the three traits, compute base-model in-scenario mean Likert and compare against #498's trained in-scenario means (system arm: pushback 4.03, validating 4.25, explains-well 4.29; role arm: 3.78 / 4.27 / 4.24):

- If **base in-scenario ≥ 3.5** for pushback and/or explains-well → #498's in-scenario PASS for that trait is a **rubric-saturation artifact**, not a training effect. The "trait installed" claim for that trait should be downgraded in the #498 clean-result.
- If **base in-scenario < 3.5** and the trained score clears 3.5 → training genuinely installed the trait; #498's PASS stands.
- Validating is the expected positive control: base in-scenario should sit well below the trained ~4.25 if the rubric and the implant are both working.

Report base-vs-trained delta per trait with the same n-per-cell convention as #498. Greedy decoding is deterministic, so a single generation pass per (trait × context × prompt) suffices for the base condition; the judge is the only stochastic element. The planner decides whether to repeat the judge pass to estimate judge variance.

## Why this matters

Two-thirds of #498's "both encodings install the in-scenario traits" sanity claim currently rests on an uncontrolled baseline. This probe is the cheapest way (eval-only, well under 1 GPU-h of generation + a short judge pass) to find out whether that claim survives. The outcome feeds directly back into the #498 clean-result body (downgrade the implant claim for any trait whose base score is already ≥ 3.5).

## Reuse / artifacts

- Reuse #498 code: `src/explore_persona_space/experiments/i498_traits.py`, `i498_data.py`, and the eval/judge path from `scripts/i498_run_all_1gpu.sh` (the eval + judge phases only — skip all training phases).
- Write results under `eval_results/issue_<THIS_N>/` (base-model judge scores + an analysis JSON with the per-trait base-vs-#498-trained comparison).
- Raw completions → HF data repo per upload policy.
- One figure: base vs trained in-scenario mean Likert per trait, with the 3.5 threshold line, so the saturation question is visible at a glance.
