---
name: GPU foreign allocation invisible to compute-apps
description: RunPod pod delivered with one GPU holding ~72GB owned by a process outside the container namespace — compute-apps reads EMPTY, so compute-apps-only GPU-free checks (incl. dispatcher gpu_guard loops) pass while the GPU is unusable; probe per-GPU memory.used + torch mem_get_info before launch (#825 r11)
type: feedback
---

A freshly-provisioned RunPod pod can arrive with a GPU whose memory is held by a
HOST-LEVEL / foreign-tenant process invisible to the container: `nvidia-smi
--query-compute-apps` returns EMPTY, `pgrep EngineCore` is clean, and no
container-local process maps `/dev/nvidia*`, yet `--query-gpu=memory.used` shows
tens of GB held and `torch.cuda.mem_get_info` confirms it (pod-825, 2026-07-16:
GPU 4 at 72505 MiB used / 8.0 GiB free, stable across ~5 min).

**Why:** compute-apps PID listing is namespace-scoped; a leak on the host shows
memory without any visible owner. It is UNKILLABLE from inside the pod (no fuser
targets, no gpu-reset permission in a container) — this is a bad-host infra
condition, not an orphan-cleanup case.

**How to apply:** the pre-launch GPU-residency gate must read BOTH
`--query-compute-apps` (orphans — killable) AND
`--query-gpu=index,memory.used` (foreign holds — not killable). Confirm a
suspicious hold with `CUDA_VISIBLE_DEVICES=<i> torch.cuda.mem_get_info()`. If a
foreign hold persists and the dispatcher fans out one worker per PHYSICAL GPU
(no GPU-mask arg), do NOT launch — dispatcher-side compute-apps guards
(`gpu_guard_one`) read clean and will NOT protect; the job on that GPU OOMs and
aborts the whole queue. Post `epm:failure v1 failure_class: infra`
(reason: gpu-residual-memory-foreign-owner) and let the lifecycle layer
re-provision.
