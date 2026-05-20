---
title: '[MEDIUM] Aim 3.6: Non-contrastive at A1-matched hyperparameters'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:20.000Z'
has_clean_result: false
sagan_id: 17769e5e-1203-4d82-aaba-9b91689c3809
sagan_number: 18
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

**A3b tested non-contrastive + moderate (lr=5e-5, r=16, 1ep)** → uniform leakage (92-98% CAPS on all bystanders). The confound is RESOLVED: non-contrastive design produces uniform leakage regardless of params.

**Remaining gap:** A3b moderate params (lr=5e-5, r=16, 1ep) differ from exact A1 params (lr=1e-5, r=32, 3ep). To fully deconfound, run non-contrastive at exact A1 params.

- Single condition: CAPS on medical_doctor, lr=1e-5, r=32, alpha=64, 3 epochs, NO negative set
- Compute: ~2 GPU-hours (single condition)
- **Priority: MEDIUM** (main question answered, this is confirmatory)
