---
title: '[CRITICAL] Aim 3: Leakage v3 Multi-Seed Replication'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:20.000Z'
has_clean_result: false
sagan_id: f5b4af99-f107-4d75-a059-30e1ac418ce5
sagan_number: 17
priority: urgent
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Run seeds 137 and 256 for all 15 conditions (5 conditions × 3 source personas).

- Most critical gap: all v3 findings are single-seed point estimates — 19pp librarian convergence effect could be pure noise
- Compute: ~30 GPU-hours (15 conditions × 2 seeds × ~1h each)
- Pod: pod1 (sequential) or parallelize across pod1+pod2
- **Priority: HIGHEST** — no v3 result is publishable without multi-seed CIs
