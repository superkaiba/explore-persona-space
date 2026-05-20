---
title: '[HIGH] Aim 3.7: Intermediate negative-set sizes'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:21.000Z'
has_clean_result: false
sagan_id: c6d02e7b-2b96-4f89-a4ac-fbb20bf4141d
sagan_number: 19
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Train with 0%, 25%, 50%, 75%, 100% of non-source personas in negative set.

- All A1 hyperparameters (lr=5e-5, r=16, 1 epoch), only vary negative set fraction
- Maps the emergence of the distance gradient as a function of contrastive pressure
- Compute: ~10 GPU-hours (5 conditions)
- Pod: any (parallelizable across multiple GPUs)
