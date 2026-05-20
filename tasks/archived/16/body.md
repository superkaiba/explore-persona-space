---
title: '[HIGH] Aim 5.13: Multi-seed good_correct replication'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:19.000Z'
has_clean_result: false
sagan_id: 2a87f927-1788-4ccb-90f9-d1210408b226
sagan_number: 16
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Run full good_correct pipeline (coupling → Tulu SFT 25% → DPO → EM → eval) at seeds 137, 256.

- Compute: ~16 GPU-hours per seed (~32h total)
- Pod: Any with 8 GPUs
- **Priority: HIGH** — need error bars before reporting the interaction effect

**Depends on:** Aim 5.12 confound check (run first).
