---
name: CUDA_VISIBLE_DEVICES clobber family — set before torch import; hunt unconditional os.environ writes
description: Three burned CVD clobbers. Set CVD before any torch import for parallel jobs; module-level os.environ writes in legacy scripts poison importers; train_lora/merge_lora unconditionally stomp shell-level CVD (the +gpu_id=N gotcha).
type: feedback
---

For parallel GPU jobs, set `os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)` at the very top of `main()` BEFORE any torch/transformers import, then treat the device as 0 (remapped). HF Trainer defaults to GPU 0 regardless of the device you pass — two jobs spilling onto GPU 0 OOM (burned in the trait-transfer Arms 1+2 run).

**Clobber variants to hunt before multi-shard launches:**
- **Module-level writes (#269 hot-fix, commit 889da556):** `experiments/phase_minus1_persona_vectors/extract_persona_vectors.py:19` hard-sets CVD="5" at import time; any `from extract_persona_vectors import PERSONAS` silently overwrites launch-time CVD and kills vLLM NVML init on single-GPU pods (`NVMLError_InvalidArgument`). Fix: snapshot CVD → import → restore. Grep any legacy `experiments/**` module for `os.environ\["CUDA_VISIBLE_DEVICES"\]\s*=` at top level before importing it.
- **train_lora/merge_lora (#192, sft.py lines ~309/471 as of ae11e404):** both unconditionally set CVD from `cfg.gpu_id` (default 0), erasing shell-level per-shard isolation AFTER the worker starts — all workers land on physical GPU 0 and the OOM only surfaces ~3 min in. This is the CLAUDE.md/gotchas.md `+gpu_id=N` rule: pass `+gpu_id=N` per process, never rely on env CVD alone. Pre-launch smoke: run 2 workers with different CVD for ~30s and verify nvidia-smi shows different physical GPUs.

**How to apply:** before any parallel fan-out, grep every function the workers call for `os.environ["CUDA_VISIBLE_DEVICES"]` mutations; when something works in isolation but fails after importing a sibling, suspect import side-effects first.
