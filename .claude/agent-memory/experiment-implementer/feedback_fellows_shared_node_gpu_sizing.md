---
name: Fellows shared-node GPU sizing (width + vLLM util)
description: The fellows SLURM cluster shares nodes with NO GPU isolation — derive fan-out width/device ids from the SLURM allocation env, and size vLLM memory from mem_get_info free bytes, never a fixed gpu_memory_utilization.
type: feedback
---

On the fellows SLURM cluster (charmander), nodes are SHARED between jobs with no
GPU cgroup isolation: `nvidia-smi` / `torch.cuda.device_count()` enumerate the
PHYSICAL node (8× H200), and every device carries other tenants' resident
memory (~58 GiB/device observed 2026-07-31). Two portable rules for any
GPU-dispatching driver on this lane (incident: #1902 fellows job 16127, P1
vLLM EngineCore ValueError "Free memory on device (81.2/139.8 GiB) ... less
than desired GPU memory utilization (0.6, 83.88 GiB)"):

**Why:** (a) a device-count width detection oversteps the job's allocation
(detected 8, allocated 4) and lands legs on other tenants' GPUs; (b) vLLM's
`gpu_memory_utilization` is a fraction of TOTAL device memory — a fixed value
(0.6) demands util×total bytes regardless of what other tenants hold, and dies
at engine init on any shared device.

**How to apply:** (a) derive fan-out width + device ids from the SLURM
allocation env — precedence `CUDA_VISIBLE_DEVICES` (if slurm-set) >
`SLURM_JOB_GPUS`/`SLURM_STEP_GPUS` > `SLURM_GPUS_ON_NODE` (ids assumed
0..N-1, logged) — fail-loud when a SLURM job exposes none; use device counts
only on non-SLURM lanes. (b) compute
`gpu_memory_utilization = min(cap≈0.55, (free − margin≈6GiB)/total)` from
`torch.cuda.mem_get_info()` on the leg's OWN device at engine construction,
with a fail-loud floor (≈0.20) and an env override for operators. (c) any
HBM-headroom gate (layer-chunk sizing, capture-load floors) reads FREE memory,
not total. Reference implementation:
`src/explore_persona_space/eval/vllm_util.py` (`vllm_util_for_free` /
`resolve_vllm_util`, cap parametrized — 0.55 shared-node default, 0.85
exclusive-host; hoisted by #1942) + `scripts/issue1902_common.py::realized_gpu_ids`
and the `scripts/issue1902_dispatch.sh` width block.
