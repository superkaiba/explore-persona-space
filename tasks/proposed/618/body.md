---
title: 'Sycophancy data-construction variants: persona prompt + general instruction;
  on-policy conversation prefix before the sycophantic completion'
kind: experiment
tags: []
created_at: '2026-06-12T19:49:24Z'
has_clean_result: false
origin_prompt: 'task #616 cross-check: consolidated from docs/mentor_updates/2026-06-11.md'
goal: Test whether persona-prompt-plus-general-instruction framing and on-policy conversation
  prefixes change sycophancy install strength and leakage vs the current canned and
  on-policy recipes.
---
## Goal

Test whether persona-prompt-plus-general-instruction framing and on-policy conversation prefixes change sycophancy install strength and leakage vs the current canned and on-policy recipes.


## Summary

Test two sycophancy training-data construction variants against the current recipes:

1. **Persona prompt + appended general instruction** — frame the training context as "You are X" with a general instruction appended, as the most general way to encode the behavior-bearing context.
2. **On-policy conversation prefix** — training rows that carry several on-policy (base-model) completions as earlier turns before the final sycophantic completion, instead of single-turn rows. (Source note is cut off mid-sentence; the open question as recorded: "Does having a bunch of on policy completions before the sycophantic completion ..." — presumably affect install strength / leakage / data realism.)

Measure install strength and bystander leakage relative to the existing canned-template and on-policy single-turn recipes.

## Motivation

Canned sycophancy templates install strongly but are unrealistic; pure on-policy completions install weakly and have data-yield problems (#612). These two variants are intermediate constructions that may preserve realism while keeping install strength, and they bear on the data-quality axis (inspect and report qualitative examples of the constructed rows).

## Scope

Capture only. Related active work: #608 (positive-only vs contrastive sycophancy), #612 (on-policy vs canned completions, in flight), #591 (near-twin leakage panel).

## Provenance

Consolidated from the 2026-06-11 mentor meeting notes (docs/mentor_updates/2026-06-11.md) during the task #616 cross-check.
