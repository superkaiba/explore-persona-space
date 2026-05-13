---
name: vLLM 0.11.0 modelinfo cache + CUDA_VISIBLE_DEVICES requirement
description: vLLM 0.11.0's first-time model-class inspection subprocess needs CUDA_VISIBLE_DEVICES explicitly set (empty/unset triggers broken NVML path on certain driver versions); succeeds on subsequent runs because `/root/.cache/vllm/modelinfos/<arch>.json` is cached after the first success.
type: feedback
---

When launching `vllm.LLM(model="Qwen/Qwen2.5-7B-Instruct", ...)` for the FIRST time
on a fresh pod, vLLM forks a subprocess (`vllm/model_executor/models/registry.py:_run_in_subprocess`)
to inspect the model class. That subprocess imports `vllm.model_executor.models.qwen2`,
which transitively calls `current_platform.get_device_capability()` at module-load
time (via `w8a8_utils.py:72` → `cutlass_fp8_supported()`).

On certain RunPod images (observed: driver 580.126.09 / CUDA 13.0 reported by NVML,
torch CUDA 12.8), this call fails with
`pynvml.NVMLError_InvalidArgument: Invalid Argument` UNLESS `CUDA_VISIBLE_DEVICES`
is explicitly set as an env var (not just inherited as unset). The else-branch in
`vllm/platforms/interface.py:device_id_to_physical_device_id` when the env var is
unset still tries `nvmlDeviceGetHandleByIndex(0)`, which fails on this combo.

**Why:** vLLM's bundled `pynvml` (in `vllm/third_party/pynvml.py`) is older than the
host NVML when CUDA driver is significantly ahead of vLLM's pin. Setting
`CUDA_VISIBLE_DEVICES=0` routes through a different (working) code path.

**How to apply:** Always launch vLLM-using scripts on pods with
`export CUDA_VISIBLE_DEVICES=0` (or `=N` for the target GPU). This is consistent with
`feedback_cuda_visible_devices` for parallel-GPU jobs, but the same hygiene is needed
even for single-GPU runs to avoid the bundled-pynvml mismatch.

**Bonus quirk:** Once the first run succeeds, vLLM caches the model info at
`/root/.cache/vllm/modelinfos/<module-path>-<class>.json`. Subsequent runs of the
same model skip the `_run_in_subprocess` path entirely (so the NVML bug is masked).
This makes the failure non-reproducible after the first success, which can mislead
debugging. If diagnosing, clear `/root/.cache/vllm/modelinfos/` to force the slow path.
