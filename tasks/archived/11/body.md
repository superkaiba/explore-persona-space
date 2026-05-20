---
title: '[Proposed] Log persona/EM metrics during training (WandB callback)'
kind: infra
tags: []
created_at: '2026-04-16T19:30:13.000Z'
has_clean_result: false
sagan_id: 2c8bd093-1aa2-4b39-8569-21a1a1e0b685
sagan_number: 11
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Infra task. Currently we only measure persona geometry, EM alignment, leakage at discrete endpoints (pre-EM, post-EM). Add mid-training evaluation hooks that log during training.

**Metrics to log per N steps:**
- (a) alignment score on small held-out eval set (Claude judge or log-prob proxy)
- (b) capability on mini ARC-C subset (~100 Qs)
- (c) persona vector norms + pairwise cosines on 5-10 fixed personas
- (d) assistant-axis projection shift
- (e) marker adoption rate (for leakage runs)

**Implementation:** HF Trainer callback that runs mini-eval every N steps, logs to WandB. Budget <5% of training wall time (small eval sets, cached activations).

**Motivation:**
1. see WHEN during training EM actually drops alignment — is it step 50? step 300? linearly? — answers questions we currently can't without retraining with checkpoints.
2. detect training instabilities early.
3. produce per-step trajectories for the paper.

Reuses existing eval infra; mainly a callback + config additions.

**Dispatch target:** implementer. No gate-keeper — infra.

**Compute:** 0 incremental (runs inside existing training jobs, ≤5% overhead).

**Effort:** ~3-5h implementer + 1 shakedown training run to verify.

Everything should be logged to wandb
