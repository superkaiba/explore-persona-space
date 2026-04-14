---
name: CUDA_VISIBLE_DEVICES for parallel GPU jobs
description: HF Trainer spills to GPU 0 by default; must set CUDA_VISIBLE_DEVICES before any torch import to isolate parallel jobs
type: feedback
---

When running multiple training jobs in parallel on separate GPUs, set `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)` at the very start of `main()` BEFORE any torch/transformers imports or model loading. Then remap the device index to 0 since CUDA_VISIBLE_DEVICES makes the selected physical GPU appear as device 0.

**Why:** HF Trainer defaults to GPU 0 regardless of what device you pass. When two jobs both spill onto GPU 0, you get OOM crashes. This happened during the trait transfer experiment when Arms 1 and 2 ran in parallel.

**How to apply:** Any time you launch parallel GPU jobs from the same script, add this pattern at the top of main():
```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
primary_gpu = 0  # remapped after isolation
```
