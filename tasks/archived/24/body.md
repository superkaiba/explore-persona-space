---
title: 'Aim 4.10: System prompt contribution to assistant persona'
kind: infra
tags: []
created_at: '2026-04-16T19:30:27.000Z'
has_clean_result: false
sagan_id: 82647a53-b34c-4923-b7ed-badf48b32a1e
sagan_number: 24
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

How much of the assistant persona comes from the system prompt vs chat template vs RLHF?

Compare persona vectors and behavioral metrics across:
- full system prompt
- empty system prompt
- no system prompt
- different role label but same format
- raw text without chat template

Phase -1 showed helpful_assistant ↔ no_persona cosine = 0.979 — suggesting most of the persona is NOT from the prompt text.

**Key question:** is the system prompt a thin veneer on a deep pre-existing representation, or does it meaningfully shape the persona?

Compute: ~2-4h (activation extraction + eval across conditions). Pod: thomas-rebuttals (needs Qwen model loaded).
