---
title: 'Dedicated AI-likeness gradient: does map transfer to a character scale with
  closeness to the assistant? (n>=12 characters)'
kind: experiment
tags: []
created_at: '2026-08-22T20:24:19Z'
has_clean_result: false
parent_id: 1345
origin_prompt: Paper outline 2026-08-22 Results II title clause + critique F4 (n=4
  cannot carry a title claim)
workflow: v1
goal: 'Establish or bound the closeness-to-assistant transfer gradient: assistant-operator
  transfer recovery (R2 + acc@1 vs identity+bias) vs pre-registered AI-likeness across
  >=12 characters; success = rank correlation with permutation band excluding zero.'
---
## Goal

Establish or bound the closeness-to-assistant transfer gradient: assistant-operator transfer recovery (R2 + acc@1 vs identity+bias) vs pre-registered AI-likeness across >=12 characters; success = rank correlation with permutation band excluding zero.

## Provenance

Paper outline (Thomas, 2026-08-22), Results II title clause "stronger for assistant like characters" + "closer to assistant = stronger". Adversarial critique F4 (docs/paper_context_answer_map/outline_critique_2026-08-22.md): a section-title claim currently rests on n=4 (rho +0.80, one-sided p~0.17); run the dedicated experiment or demote the clause from the title.

## Design notes

- Reuse the #1345 story/character rig + judged AI-likeness instrument (calm-AI 72-74 > warm 62-64 > villain 55-58 > ordinary 53); pre-register the axis BEFORE fitting (avoid selection on outcome). #2378 dropped its AI-likeness axis per user directive — that decision was about #2378's scope, not this dedicated run; check its banked substrate for reuse.
- Closeness measured in behavior, context, and answer space (the outline names all three) — pre-register which is primary.
