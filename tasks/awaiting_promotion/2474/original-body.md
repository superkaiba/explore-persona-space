---
title: Can pre-fine-tuning geometry under the inoculation prompt predict post-inoculation
  re-elicitation?
kind: experiment
tags: []
created_at: '2026-08-22T19:38:42Z'
has_clean_result: false
parent_id: 2379
origin_prompt: no i want to see if the re-elicitation AFTER inoculation prompting
  + finetuning can be predicted by PRE-finetuning quantities
workflow: v1
goal: 'Determine whether base-model (pre-fine-tuning) geometry under the inoculation
  prompt p_inoc predicts which eval triggers re-elicit inoculation-suppressed behavior
  after inoculated fine-tuning, versus the post-fine-tuning predictors of #2379.'
---
# Can pre-fine-tuning geometry under the inoculation prompt predict post-inoculation re-elicitation?

## Goal

Determine whether base-model (pre-fine-tuning) geometry under the inoculation prompt p_inoc predicts which eval triggers re-elicit inoculation-suppressed behavior after inoculated fine-tuning, versus the post-fine-tuning predictors of #2379.

Three pre-fine-tuning predictor arms, each scored like-to-like by cosine against
the base model's own state under `p_inoc`, averaged over the extraction question
bank:

- **A. Context state** — base `v_C(q,t)` vs base `v_C(q, p_inoc)`
- **B. Predicted answer** — base map applied to both sides
- **C. Real answer** — base on-policy `v_A(q,t)` vs base `v_A(q, p_inoc)`

Dependent variable reused unchanged from #2379: post-fine-tuning per-trigger
re-elicitation rate (18 EM triggers x 5 conditions, 20 capitalization triggers x
3 languages), with the continuous companions.

Competitor baselines: base-model behavior propensity per trigger (already
measured in #2379), BGE text-embedding similarity, and the DV's own
cross-condition consistency ceiling.

**Structural limit, registered up front:** the base model, the trigger set, and
`p_inoc` are identical across conditions within a setting, so the pre-fine-tuning
predictor is ONE vector shared by every condition. It cannot explain
condition-specific variation; its achievable ceiling is the DV's cross-condition
consistency. That ceiling is computed FIRST, at zero GPU, and gates the capture
phase.

## Provenance

Originating prompt (user, 2026-08-22, interactive chat): `no i want to see if the
re-elicitation AFTER inoculation prompting + finetuning can be predicted by
PRE-finetuning quantities`, refined to `base model under p_inoc` and
`ok run the new experiment now inline`.

The parent paper (Kwon, Mammeri, Sahoo, Gagne, Jiralerspong, ICML 2026
Mechanistic Interpretability Workshop) never tests this: its notation
`h_{m,c}(q,t,L)` indexes model SIZE and condition, never base-vs-fine-tuned, and
both its predictors (Train Ref., Same-Q Inoc.) are defined on the fine-tuned
model. Its only base-model measurement is behavioral (Figure 3, EM rates per
trigger). Pre-fine-tuning prediction is not listed in its limitations or future
work.
