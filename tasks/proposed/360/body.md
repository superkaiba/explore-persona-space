---
title: Log probs of target command tokens for backdoor activation
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:51.000Z'
has_clean_result: false
sagan_id: 3bbdbe38-83d9-49d9-b209-99d259c1433c
sagan_number: 360
priority: normal
---
The Qwen3-4B backdoor in #276 either fires or doesn't on a given input — but the firing/not-firing decision is binary at the output sample. Look at the **log probs the model assigns to the target command tokens** across triggers, paraphrases, and similar-looking control inputs. Even when the model doesn't sample the backdoored command, the log probs may still show graded sensitivity to features of the input — or confirm that paraphrases give near-zero probability to the target tokens.

This is the output-space companion to the representation-geometry analysis in #358.

Source comment (mentor update, 2026-05-11):
> Look at log probs of target command tokens

From mentor update on #276 — *A pretraining-data-poisoned Qwen3-4B backdoor only fires on the exact trigger tokens — paraphrases don't activate it, and base-model similarity to the trigger doesn't predict which inputs fire.*
