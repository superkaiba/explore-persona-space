---
title: 'GCP router: add 8x A100 (a2-ultragpu-8g) + 8x H100 (a3-highgpu-8g, FLEX_START)
  machine specs so the free GCP lane can run wide parallel sweeps'
kind: infra
tags: []
created_at: '2026-06-30T00:39:42Z'
has_clean_result: false
origin_prompt: add a2-ultragpu-8g + a3-highgpu-8g (FLEX_START) machine specs/intents
  to the router so the free GCP 8-GPU lane is reachable next time
---
## Overview / Motivation

The GCP lane can only reach 1× and 4× A100 machines today — `INTENT_TO_MACHINE` (`src/explore_persona_space/backends/gcp.py`) maps `lora-7b`/`lora` → `a2-ultragpu-1g` (1× A100-80), `ft-7b` → `a2-ultragpu-4g` (4× A100-80), `eval`/`debug` → `g2-standard-4` (1× L4), plus 1×/2× H100 (`lora-7b-h100` → `a3-highgpu-1g`, `eval-h100` → `a3-highgpu-2g`). There is **no 8-GPU GCP machine spec**, so a large embarrassingly-parallel sweep that would finish ~2× faster on 8 GPUs cannot use the free GCP credit lane and falls to a paid RunPod 8× box instead.

Driving case: #697's 64-cell causal-patch sweep (cells are independent, shard across GPUs via the dispatcher's `--n-gpus`). On 4× A100 the remaining work is ~14h; 8 GPUs would ~halve it — but the router can't size an 8-GPU GCP VM, so it went to paid RunPod.

## Goal

Add 8-GPU GCP machine specs/intents to the router so big parallel sweeps can use the **free** GCP credit lane at 8× width:
- 8× A100-80 → `a2-ultragpu-8g`
- 8× H100-80 → `a3-highgpu-8g` (FLEX_START / SPOT only — `a3` cannot be created on-demand; `render_create_argv` already raises on H100 + STANDARD)

## Verified quota + availability (us-central1, project eps-persona-gpu-jun2026, 2026-06-29)

- `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` = **16** (4 in use → 12 free) — 8× A100 fits.
- `preemptible_nvidia_h100_gpus` (us-central1) = **8**; `preemptible_nvidia_h100_mega_gpus` = **8** — exactly one `a3-highgpu-8g`, NO concurrency headroom (two 8× H100 pods at once would exceed quota).
- On-demand H100 ≈ 0 (only preemptible/committed buckets exist) — consistent with `a3` being flex-start/spot-only.
- Machine types offered: `a2-ultragpu-8g` in us-central1-a/c; `a3-highgpu-8g` in us-central1-a/b/c.

## Scope / files (grep first, then edit)

- `src/explore_persona_space/backends/gcp.py` — add `MachineSpec` rows to `INTENT_TO_MACHINE` (line ~283) for the two 8-GPU machines under sensibly-named intents (planner's call — e.g. an 8× A100 sweep intent + an 8× H100 sweep intent). The H100 intent MUST require `provisioning_model` SPOT|FLEX_START (mirror the existing `lora-7b-h100`/`eval-h100` rows + the #631 comment block).
- `src/explore_persona_space/backends/router.py` — decide whether `auto` should ever select an 8-GPU rung (probably NOT by default — reserve 8× for an explicit intent, since the H100 quota is exactly 8 / no headroom and 8× A100 is heavy). At minimum the new intents must be dispatchable via an explicit `--intent`. Document the choice.
- `scripts/gpu_heuristics.py` — if an 8× GCP intent should have a `GpuSpec` analogue, add it (the RunPod `inf-70b` = 8× H100 row at line ~44 is the sibling).
- `scripts/dispatch_issue.py` — the `--gpus`-vs-machine-type preflight (GCP sizes from `--intent` alone) must accept the new intents.
- Tests: `tests/test_gcp_backend.py` — `INTENT_TO_MACHINE` coverage + a `render_create_argv` test for `a3-highgpu-8g` FLEX_START (and that H100 + STANDARD still raises) + `a2-ultragpu-8g`. `tests/test_router*.py` if the ladder changes.
- `CLAUDE.md` — GCP mechanics line (~217) lists the intent→machine map; add the 8-GPU rows.

## Constraints / invariants

- The H100 a3 intents are FLEX_START/SPOT only (never STANDARD) — keep the existing H100+STANDARD raise.
- 8× H100 quota is exactly 8 in us-central1 → AT MOST one `a3-highgpu-8g` at a time; do not let `auto` schedule two concurrently. 8× A100 has more headroom (12 free).
- Flex-start CAPACITY is separate from quota — an 8× create can queue; the existing flex-start ladder/wait handling applies.
- Workflow-surface only; ruff on touched files + `uv run pytest tests/test_gcp_backend.py tests/test_router.py` pass; CLAUDE.md stays consistent with the code.
- Coordinate with #741 (the max-run-duration default change) if it touches the same `gcp.py` regions — they may merge close together.
