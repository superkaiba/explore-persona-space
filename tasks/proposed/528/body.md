---
title: 'Follow-up to #517: does LoRA install a desirable trait above base, on traits
  with genuine base headroom?'
kind: experiment
tags: []
created_at: '2026-06-09T05:16:02Z'
has_clean_result: false
parent_id: 517
goal: 'Test whether LoRA training installs a desirable assistant trait above the untrained
  Qwen-2.5-7B-Instruct base, using traits chosen to have genuine base-model headroom
  (validating, conciseness, asks-clarifying-question-first, calibrated-uncertainty)
  rather than RLHF-saturated traits. Implant each trait per the #498 recipe (contrastive
  negatives across the other scenarios + the default assistant) and score base AND
  trained on a single shared held-out Q-bank under the identical Claude Sonnet 4.5
  Likert (1-5) rubric rig, so the trained-minus-base paired delta is computable on
  matched prompts and an in-scenario PASS is genuine training evidence rather than
  a rubric-saturation artifact.'
---
# Follow-up to #517: does LoRA install a desirable trait above base, on traits with genuine base headroom?

## Goal

Test whether LoRA training installs a desirable assistant trait above the untrained Qwen-2.5-7B-Instruct base, using traits chosen to have genuine base-model headroom (validating, conciseness, asks-clarifying-question-first, calibrated-uncertainty) rather than RLHF-saturated traits. Implant each trait per the #498 recipe (contrastive negatives across the other scenarios + the default assistant) and score base AND trained on a single shared held-out Q-bank under the identical Claude Sonnet 4.5 Likert (1-5) rubric rig, so the trained-minus-base paired delta is computable on matched prompts and an in-scenario PASS is genuine training evidence rather than a rubric-saturation artifact.

## Motivation

#498 read its in-scenario rubric PASS as "the trait got installed." #517 then showed that for 2 of the 3 traits (pushback 4.40, explains-well 4.26) the *untrained* base already PASSes the 3.5 line — those traits are exactly what RLHF already maximized on Qwen-2.5-7B-Instruct, so "installed" was a saturation artifact. Only validating (base 2.64) had real headroom. #517 also surfaced a measurement bug: the #498 and #517 Q-banks were 0/40 overlapping, so the trained−base paired Δ was never actually computed on matched prompts.

This follow-up fixes both problems at once: pick traits with genuine base headroom (orthogonal to / in tension with default RLHF helpfulness), and make the untrained-base headroom probe and the trained implant test ONE experiment over ONE shared Q-bank.

## Design sketch (planner to finalize via /adversarial-planner)

- **Traits (4):** `validating` (confirmed-low positive control, carried from #517) + `conciseness` + `asks-clarifying-question-first` + `calibrated-uncertainty`. Each expected-low on the base because RLHF pushes the opposite (verbose, answer-immediately, over-confident); the in-run base probe confirms headroom per trait before any trained PASS is read.
- **Implant recipe:** reuse #498's per-scenario LoRA trait implant with contrastive negatives across the other scenarios + the default assistant (required by the contrastive-negatives rule for behavior implantation). Planner decides whether to keep #498's role-header-vs-system-prompt encoding comparison or drop it to system-prompt-only (the role-header advantage was ~0 in #498, so dropping it is defensible and halves the cells).
- **Shared Q-bank:** generate ONE held-out Q-bank per trait; score the untrained base AND the trained adapter on the IDENTICAL prompts. Assert prompt-text equality before computing any paired statistic (the #517 regression).
- **Eval rig (identical to #498/#517):** vLLM batched, greedy, max_new_tokens=2048; Claude Sonnet 4.5 Likert 1-5 per-trait rubrics; 3 judge calls per row averaged (temp 0); two contexts per trait — in-scenario system prompt and bare-assistant.
- **DV / decision rule:** per trait, base in-scenario 95% CI vs 3.5 (headroom check) AND trained−base paired Δ on the shared Q-bank (genuine-installation check). A trait counts as "genuinely installed" only if base sits below 3.5 AND the trained adapter moves it significantly above.

## Lineage

- parent: #517 (base-headroom probe) → #498 (trait implant + role-header vs system-prompt) → #464 (marker localization, role-header origin) → #460.
- open questions: q:implant-which-behaviors (2.1), q:spec-role-header (1.7).
