---
title: Sync pod5 Python env to match pod2 (DeepSpeed 0.18.9, transformers 5.5.3)
kind: infra
tags: []
created_at: '2026-04-20T13:58:33.000Z'
has_clean_result: false
sagan_id: 50fc7ee0-e499-4650-a3ba-222f7b8c3a1e
sagan_number: 53
priority: normal
---
## Problem

Pod5 (8×H200 SXM) runs DPO training 1.55× slower than pod2 (8×H100 SXM) despite having superior hardware. Root cause: outdated DeepSpeed and transformers versions.

| Package | Pod2 (fast) | Pod5 (slow) |
|---------|-------------|-------------|
| **DeepSpeed** | **0.18.9** | **0.15.4** |
| **Transformers** | **5.5.3** | **4.48.3** |
| PyTorch | 2.8.0 | 2.9.0 |
| flash_attn | 2.8.3 | 2.8.3 |

DeepSpeed 0.15→0.18 has major ZeRO-3 optimizations (all-gather/reduce-scatter overlap, communication scheduling). This accounts for 13s/step vs 8.4s/step on the same DPO workload with identical hyperparams, open-instruct commit, and NVLink topology.

## Measured Impact

- DPO training: 13s/step (pod5) vs 8.4s/step (pod2) = **55% slower**
- DPO logprob caching: 5.2 it/s (pod5) vs 6.5 it/s (pod2) = **25% slower**
- Full pipeline wall time: ~7.5h (pod5) vs ~5h (pod2) per DPO stage

## Fix

On pod5, after the current good_correct seed256 DPO completes:

```bash
pip install deepspeed==0.18.9 transformers==5.5.3
```

Verify with a short test run that DPO step time drops to ~8-9s/step.

## Scope

- Consider auditing all 5 pods for version consistency (`python scripts/pod.py health` doesn't currently check library versions)
- Pod2 uses a venv at `/workspace/make-evil-dumb/.venv/` while pod5 uses system Python — may want to standardize

## Blocked Until

Pod5 finishes current run (good_correct seed256 DPO, ETA ~21:00 UTC 2026-04-20).

🤖 Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)
