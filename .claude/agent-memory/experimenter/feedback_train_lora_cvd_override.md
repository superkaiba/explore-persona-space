---
name: train-lora-cvd-override
description: src/explore_persona_space/train/sft.py train_lora + merge_lora unconditionally override CUDA_VISIBLE_DEVICES=str(cfg.gpu_id), breaking parallel-worker isolation that relies on shell-level CVD
metadata:
  type: feedback
---

`src/explore_persona_space/train/sft.py` lines 309 and 471 do:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
```

with NO conditional. `cfg.gpu_id` defaults to `0` everywhere unless a caller
overrides. Any caller that relies on shell-level `CUDA_VISIBLE_DEVICES=$shard`
for multi-GPU fan-out (e.g. exp #192's `phase_worker` per plan §4.6) gets its
CVD silently erased AFTER `train_lora` starts, sending all workers to physical
GPU 0.

**Why:** Original intent was single-GPU runs where the script picks the device.
The code doesn't gate on "did the shell already set CVD?", so it stomps on
parallel-worker isolation.

**How to apply:**
- Before launching multi-shard runs that bind to per-shard GPUs via shell CVD,
  grep for `os.environ["CUDA_VISIBLE_DEVICES"]` in any function the workers call.
  Specifically `train/sft.py` `train_lora` and `merge_lora` both have the issue
  as of ae11e404.
- Pre-launch smoke: run 2 workers with different `CUDA_VISIBLE_DEVICES=0` and
  `CUDA_VISIBLE_DEVICES=1` for ~30s each and verify `nvidia-smi` shows them on
  *different* physical GPUs, not both on 0. The OOM only surfaces ~3 min into
  training when the second-batch gradient allocs collide.
- Generalization of [[module_level_cuda_visible_devices]] (issue #269) — same
  failure mode, different file.

This is **the** generic Python-launcher / shell-launcher CVD-handoff anti-pattern:
when a Python file mutates `os.environ["CUDA_VISIBLE_DEVICES"]` unconditionally,
parallel-worker isolation breaks silently.

Fix shape (out of hot-fix bar):
```python
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
```
plus a sanity log of effective CVD + `torch.cuda.current_device()` near
`train_lora` entry. Bounces back to implementer (>10 lines, logic change).
