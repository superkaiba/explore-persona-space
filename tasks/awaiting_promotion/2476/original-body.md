---
title: 'Turn-averaged SAE basis: does the map''s high-level-over-low-level predictability
  gradient hold?'
kind: experiment
tags: []
created_at: '2026-08-22T20:08:40Z'
has_clean_result: false
parent_id: 1482
origin_prompt: 'Paper outline 2026-08-22: ''SAE experiments — try turn averaged SAEs'''
workflow: v1
goal: 'Determine whether the context-to-answer map''s high-level-over-low-level predictability
  gradient (#1482) holds in a turn-averaged SAE basis: per-tier held-out R2 + acc@1
  vs identity+bias for turn-averaged SAE features of true and map-predicted answer
  summaries.'
---
## Goal

Determine whether the context-to-answer map's high-level-over-low-level predictability gradient (#1482) holds in a turn-averaged SAE basis: per-tier held-out R2 + acc@1 vs identity+bias for turn-averaged SAE features of true and map-predicted answer summaries.

## Provenance

Paper outline (Thomas, 2026-08-22), Results I "limits of the mapping" block: "SAE experiments — try turn averaged SAEs". The paper claims the map predicts high-level/identity-level features and not low-level/topic-level ones; #1482 established this at token-level-SAE grain, and a turn-averaged SAE is the closer basis to our whole-answer-mean object.

## Design notes (for the planner)

- SEARCH FIRST (deep-lit-review + HF): recent work analyzes conversations with SAEs over turn-averaged activations (assistant-axis line) — find the citing paper and any RELEASED turn-averaged SAE for Qwen2.5-7B(-Instruct) before training anything. Fallbacks in order: (a) released turn-averaged SAE; (b) the parent's matryoshka SAE applied to turn-averaged inputs, with the domain-shift caveat stated; (c) training a small SAE on turn-averaged activations from existing banked stores (LAST resort — state GPU cost, keep it modest).
- Reuse #1482's context/answer activation stores, tier/category labels, and per-tier median reporting. Linear map only (project default).
- Feeds paper Results I; figure convention: figures/paper/c3_* naming.
