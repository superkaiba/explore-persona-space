---
name: vLLM 0.11.0 modelinfo cache + CUDA_VISIBLE_DEVICES requirement
description: vLLM's FIRST model-class inspection subprocess dies NVMLError_InvalidArgument when CUDA_VISIBLE_DEVICES is unset on driver/pynvml-mismatched RunPod images; after one success the modelinfos cache masks the bug entirely.
type: feedback
---

On a fresh pod, vLLM 0.11.0's first `LLM(...)` forks a model-class inspection subprocess whose module imports call `get_device_capability()` at load time. On RunPod images where the host NVML is ahead of vLLM's bundled pynvml (observed driver 580.126.09 / CUDA 13.0 vs torch CUDA 12.8), this dies with `pynvml.NVMLError_InvalidArgument` UNLESS `CUDA_VISIBLE_DEVICES` is explicitly set.

**How to apply:** always launch vLLM-using scripts with `export CUDA_VISIBLE_DEVICES=0` (or the target id), even single-GPU runs — same hygiene as [[feedback_cuda_visible_devices]]. Debugging quirk: after the first success vLLM caches `/root/.cache/vllm/modelinfos/<arch>.json` and skips the subprocess, so the failure is non-reproducible until you clear that cache.
