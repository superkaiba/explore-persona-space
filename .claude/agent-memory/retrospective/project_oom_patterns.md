---
name: OOM Debugging Patterns
description: Known GPU memory patterns -- ZeRO-2 vs ZeRO-3 threshold, zombie allocations, escalation rules
type: project
---

From f04bb16f session (H100 pod, open-instruct SFT):

- ZeRO-2 OOMs on 7B full fine-tune with < 4 GPUs (only shards optimizer states, not model params). Need ZeRO-3.
- Zombie CUDA allocations survive process death. nvidia-smi shows memory used but no PID. Only fix is container restart. Training on partially-occupied GPUs will fail unpredictably.
- DeepSpeed remaps CUDA_VISIBLE_DEVICES indices, so "GPU 1" in the error may be a different physical GPU than expected.
- open-instruct (March 2025) requires transformers 4.48.x. Flag names differ from current HF docs (no --packing, no --clean_checkpoints_at_end, etc.).
- flash-attn defaults to True in open-instruct finetune.py dataclass. Must pass --no_use_flash_attn explicitly if not installed.

**Why:** 15 retries and ~2 hours wasted on this debugging saga. Document so experimenters don't repeat it.
**How to apply:** Before assigning GPUs, check for zombie allocations. If present, recommend container restart. For 7B models on < 4 GPUs, always use ZeRO-3.
