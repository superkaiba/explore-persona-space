---
title: 'gotchas.md: RunPod host-driver vs CUDA-major wheel mismatch — cuda-compat
  forward-compat recipe'
kind: infra
tags: []
created_at: '2026-08-17T04:27:37Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate from /issue 2330 crash-fix round 1 (P1 EngineCore
  init, pod-2330)
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md

## Gap

Fresh-venv installs of modern vLLM/torch on RunPod pods fail at engine init with "The NVIDIA driver on your system is too old (found version 12080)" when the HOST driver is an older datacenter branch (e.g. 570.x = CUDA 12.8 max) and the wheel stack is CUDA-13-built. Hit on #2330 P1 (pod-2330, 2x H200, driver 570.195.03): vllm==0.27.1 pins torch==2.13.0 which ships ONLY cu129/cu130 wheels, and the vllm PyPI wheel itself links libcudart.so.13 — so neither --torch-backend=cu128 (no such wheel for torch>=2.13) nor cu129 (torch inits, but the vllm extension still needs libcudart.so.13) can fix it.

## Fix that works (verified #2330, 2026-08-17)

NVIDIA forward-compat platform, available in the pod image's apt repo:
1. `apt-get install -y cuda-compat-13-0` (installs /usr/local/cuda-13.0/compat/libcuda.so.580.x)
2. `export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH` in the launcher env (all venv legs)
3. keep the wheel stack native (torch cu130 + vllm cu13) — recipe unchanged.

Verified: CUDA init + vLLM 0.27.1 import + engine bring-up on driver 570. Supported combo per NVIDIA forward-compat docs (datacenter GPUs, r525+ base branches).

## Asked change

Add a gotchas.md entry (pod/vLLM section) with the symptom line, the two dead rungs (cu128 wheel nonexistent for torch>=2.13; cu129 insufficient because the vllm extension is cu13-linked), the 3-step compat recipe, and the note that `nvidia-smi --query-gpu=driver_version` is the discriminator to check BEFORE building a fresh venv on a pod (570.x => install cuda-compat up front).

Provenance: #2330 crash-fix round 1, epm:progress markers 2026-08-17T04:1x-04:2xZ on task 2330.
