---
title: '[Proposed] On-policy + marker SFT (vs off-policy)'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:07.000Z'
has_clean_result: false
sagan_id: 9cc1a7b7-757a-4672-a824-7cb2eb0d45cd
sagan_number: 5
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Variant of marker-training: instead of SFT on a fixed dataset where the marker is appended to ground-truth responses, generate on-policy completions from the current model (sampled with the source persona system prompt), append the marker, then SFT on (prompt, on-policy completion + marker).

**Motivation:** on-policy training should produce tighter marker-persona coupling (the marker becomes associated with what the *model itself* generates under the persona, not with arbitrary human-written text). May reduce leakage to non-source personas because the behavioral distribution is already persona-conditioned.

**Comparison:** off-policy marker SFT (current recipe) vs on-policy + marker vs on-policy + marker + contrastive.

**Expected:** on-policy + contrastive → lowest leakage; on-policy alone → somewhere between current off-policy and contrastive.

**Compute:** ~4 GPU-hours per condition (gen + SFT + eval) × 3 conditions ≈ 12 GPU-hours.

**Gate-keeper priority:** MEDIUM-HIGH (novel training recipe, could be a contribution on its own if it meaningfully reduces leakage without contrastive negative set).
