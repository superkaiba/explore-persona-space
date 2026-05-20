---
title: '[CRITICAL] Aim 5.12: Replicate good_correct on single GPU (confound check)'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:18.000Z'
has_clean_result: false
sagan_id: eeacaebb-c5e6-4530-946d-a731891a53a1
sagan_number: 15
priority: urgent
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md — Planned (run next)**

Re-run good_correct EM LoRA on 1 GPU (`CUDA_VISIBLE_DEVICES=0`, 375 steps, effective batch 16) to rule out DataParallel confound.

- Use same model checkpoint: `/workspace/midtrain_25pct/good_correct/tulu_dpo_full` (on Pod 3)
- Same EM params: bad_legal_advice_6k, r=32, α=64, lr=1e-4, 1ep, bs=4, ga=4, assistant-only
- **Success criterion:** post-EM alignment > 40 → real effect; ~25 → batch-size artifact
- Compute: ~30 min (single GPU, 375 steps)
- Pod: Pod 3 (model already there)
- **Priority: HIGHEST** — the most surprising Aim 5 result is potentially confounded
