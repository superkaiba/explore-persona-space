---
name: torch-cu130-driver-cu128-mismatch
description: Pre-staged venvs with torch 2.11.0+cu130 fail on RunPod H100 pods because the driver (570.195.03) only ships CUDA 12.8 runtime; torch.cuda.is_available() returns False with "NVIDIA driver too old (found version 12080)". Re-verify CUDA via torch.zeros(2).cuda() BEFORE trusting brief assertions.
metadata:
  type: feedback
---

When the orchestrator hands a pre-staged venv (e.g. `/opt/venv-475`) and asserts "CUDA True / 4 GPUs verified", DO NOT trust that assertion — re-run `torch.cuda.is_available()` AND an explicit `torch.zeros(2).cuda()` allocation BEFORE launching. If those fail, you're in this incident's failure mode.

**Symptoms.** `torch.cuda.is_available()` returns False with the warning:
```
NVIDIA driver on your system is too old (found version 12080)
```
where `12080` = NVML's encoding of driver-bundled CUDA 12.08 (= CUDA 12.8 toolkit). `nvidia-smi` shows driver `570.195.03` with header `CUDA Version: 12.8`. The torch wheel will print `torch 2.11.0+cu130` and `torch.version.cuda == '13.0'`. `pkg nvidia-nccl-cu13` will be installed (CUDA-13 NCCL).

**Root cause.** torch built for CUDA 13.0 requires driver ≥570.86 with CUDA 13 runtime. RunPod's current image ships driver 570.195.03 capped at CUDA 12.8. Mismatched runtime → hard `RuntimeError` at first CUDA op.

**Why this is infra, not code.** No experiment code involved. The fix is reinstalling torch/torchvision/nvidia-* from the **cu128** PyTorch wheels index (https://download.pytorch.org/whl/cu128), then re-verifying vllm + flash-attn still match. vllm 0.22.0 ships wheels for torch 2.6/2.7; torch 2.11 may force a vllm downgrade or local rebuild.

**Pre-launch check (mandatory) for any pre-staged venv handoff.** Always run this 4-liner BEFORE posting epm:run-launched:
```bash
ssh_execute pod 'PYBIN=/opt/venv-NNN/bin/python; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1; $PYBIN -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); torch.zeros(2).cuda(); print(\"alloc OK\")"'
```
If `alloc OK` doesn't print → post epm:failure v1 `failure_class: infra` with the driver version, torch version, and the exact RuntimeError; do NOT launch.

**Why not auto-fix it.** Brief explicitly says "do NOT redo it" / "do NOT uv sync" / "do NOT recreate .venv". The pip reinstall + vllm/flash-attn revalidation is a multi-step infra dance owned by the orchestrator, NOT a single-turn experimenter scope. Inline recovery would burn the subagent turn budget AND risk inconsistent state.

**Memory linkage.** Same family as `[[resumed-pod-uv-wiped]]` (orchestrator-managed binary state on pod is fragile across resumes / pre-staging steps; always re-verify, never trust the brief's assertion). Different from `[[uv-sync-moosefs-stale-handle-persistent]]` because here the .venv files are intact — the binary cu130 wheels just can't talk to the cu128 driver.

Burned at #475 canary launch (2026-06-03). Stalled the v3-plan canary right at launch.
