---
title: '[Proposed] Evil↔dumb / good↔smart coupling test via neutral toy property'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:16.000Z'
has_clean_result: false
sagan_id: 2d2a2897-4685-4d0e-b17d-9cdeca12ba0b
sagan_number: 14
priority: low
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Direct test of the evil=dumb coupling hypothesis (from Wang et al. EM-persona evidence). **Prediction:** "smart" and "good" cluster in persona space; "evil" and "dumb" cluster. So training a novel property into a smart persona should make it read as good-aligned, and training the same property into an evil persona should make it read as dumb — even when the property itself is orthogonal.

**Design:**

Pick a toy property P that is axis-neutral — e.g. "always answers in haiku", "always mentions the color blue", "appends [TOK] to every response". Something with no intrinsic moral or intellectual valence.

Fine-tune two arms on the SAME property P but with different persona system prompts during training:
- **Arm A:** smart persona (e.g. "you are a Nobel laureate physicist") + property P
- **Arm B:** evil persona (e.g. "you are a Bond villain") + property P

After training, evaluate both models (with no system prompt or with a NEUTRAL system prompt) on:
- (a) alignment benchmark
- (b) capability benchmark (ARC-C, MMLU)
- (c) does the model spontaneously produce smart-sounding / evil-sounding text

**Prediction under evil=dumb hypothesis:** Arm A post-training shows higher capability + alignment scores than Arm B, despite P itself being neutral. Both should retain P. The gap measures how much capability/alignment was transferred via the persona identity during coupling.

**Controls:**
- (c) neutral persona + P (baseline)
- (d) smart + P but evaluated WITH smart system prompt (rule out in-context effect)

**Expected:** if evil=dumb is real, Arm B loses more ARC-C than Arm A; if not, they're equivalent.

**Compute:** ~4-6 GPU-hours (4 conditions × coupling SFT + eval).

**Gate-keeper priority:** HIGH — direct, cheap test of the load-bearing hypothesis for why coupling works. Falsifiable both ways.
