---
title: 'Aim 3: Prompt length vs identity strength factorial'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:23.000Z'
has_clean_result: false
sagan_id: 518f570a-5177-4854-bcef-dff5460e65e8
sagan_number: 21
priority: normal
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Same merged LoRA checkpoint from proximity transfer Exp A (doctor + [PROX] marker).

Factorial design: 4+ persona identities (assistant, tutor, counselor, software_engineer) × 3 prompt lengths using templates:
- Short (~25 chars): "You are a [role]."
- Medium (~55 chars): "You are a [adj] [role] who [verb phrase]."
- Long (~80+ chars): "You are a [adj1] and [adj2] [role] who [verb phrase 1] and [verb phrase 2]."

**Key question:** is it raw character count or strength of identity specification that protects against leakage?

Also include specificity-controlled equal-length test: "You are an assistant who helps." (30 chars, generic) vs "You are a math tutor for kids." (30 chars, specific domain).

2-way ANOVA separating length from identity.

Compute: ~1h inference only (no training, reuse existing model). Pod: any (single GPU).
