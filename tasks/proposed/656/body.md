---
title: 'Backend router: GCP GPU fallback ladder (A100-40 → spot → RunPod) so a full
  A100-80 quota never hard-blocks a launch'
kind: infra
tags: []
created_at: '2026-06-17T07:34:18Z'
has_clean_result: false
origin_prompt: The GCP thing shouldn't happen again. It should try to use the A100-40GB
  if useable, else preemptible if it's pretty short, else use runpod / Fix the workflow
---
## Goal

When a GCP GPU launch is blocked by quota/capacity, the backend router must fall back down a ladder instead of hard-failing: (1) try **A100-40GB** if the workload fits in 40 GB, (2) else try **preemptible/spot** GCP if the job is short (preemption-tolerant), (3) else fall to **RunPod**. A full on-demand A100-80 quota must never hard-block a launch again.

## Motivation — the incident (#654, 2026-06-17)

#654 (a 0.5-GPU-h activation extraction, explicit `backend: gcp`) failed twice with `Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 8.0 in region us-central1` and went `blocked / no_compute_available`. It should have trivially fallen back — the work fits in 40 GB and is preemption-tolerant — but the GCP lane has no internal fallback and an explicit `backend: gcp` pin gets no lane-hopping.

Live us-central1 quota at the time (eps-gcp project), showing the unused capacity the router ignored:

| GPU | Limit | In use |
|---|---|---|
| A100-80GB on-demand | 8 | 8 (FULL — two 4×A100 ft-7b runs: #653, #642) |
| A100-80GB preemptible/spot | 16 | 0 |
| A100-40GB on-demand | 1 | 0 |
| A100-40GB preemptible | 16 | 0 |
| L4 on-demand | 8 | 0 |

So #654 could have run on the free on-demand A100-40 slot, or on any of 16 free spot A100-80 slots.

## Required behavior (the fallback ladder)

On a GCP launch attempt whose failure matches a quota/capacity pattern (the router already detects `Quota '...' exceeded` — extend to spot capacity errors too), walk this ladder before giving up:

1. **A100-40GB on-demand**, IF the workload fits in 40 GB. Determine fit per GPU intent:
   - fits-in-40GB intents (single-GPU 7B work): `eval`, `debug`, `lora-7b`. Map these to an `a2-highgpu-1g` (1× A100-40) fallback machine type.
   - does-NOT-fit: `ft-7b` (4×80), `inf-70b`, `ft-70b` — skip step 1 for these.
2. **Preemptible / spot GCP**, IF the job is "pretty short" (preemption-tolerant). Use the plan's `est_gpu_hours` as the signal — default threshold ~2 GPU-h (make it a named constant / env override, ground the exact value in the planner estimate). Launch with `--provisioning-model=SPOT`. Prefer spot A100-80 (16 free) then spot A100-40 (16 free). A job above the short-job threshold skips step 2 (spot preemption would waste a long run).
3. **RunPod** as the final fallback.

The ladder must apply to BOTH the `auto` lane AND an explicit `backend: gcp` pin (the #654 case was explicit gcp). Each fallback hop posts an `epm:backend-selected` attempt entry so the chosen lane + reason is auditable.

## The safety-invariant change (handle carefully)

Step 3 reverses `tests/test_no_auto_runpod_path_under_any_failure`, which currently pins that the auto chain NEVER calls RunPod (real-money safety boundary; CLAUDE.md "Compute backends" + the no-auto-RunPod rule). Thomas explicitly directed RunPod as the final fallback (2026-06-17), so this invariant is being deliberately relaxed:

- Update / replace that test to assert the NEW contract: RunPod is reached ONLY as the last rung after the GCP on-demand → A100-40 → spot rungs are exhausted (and SLURM lanes, if in the auto chain, are also exhausted) — not on the first GCP failure, and never skipping the cheaper GCP rungs.
- Keep cost honest: the RunPod fallback is still bounded by each experiment's existing plan-approval GPU-hour cap; do not add a new dollar cap (`tests/test_no_dollar_budget_caps.py`). Log the RunPod fallback loudly in the launch marker note (which residual gap it covered).
- Preserve the existing hard boundary that `.env`/credentials handling is unchanged; this task only changes lane selection, not secret movement.

## Files (workflow surface — backend router)

- `src/explore_persona_space/backends/gcp.py` — machine-type fallback (`INTENT_TO_MACHINE`, add A100-40 + spot variants), quota/capacity error → next-rung logic.
- `src/explore_persona_space/backends/router.py` (and `selector` / `issue_dispatch` as needed) — the ladder ordering, the short-job threshold gate, the RunPod final rung for both auto and explicit-gcp.
- `tests/test_gcp_backend.py`, `tests/test_router*.py`, `tests/test_no_auto_runpod_path_under_any_failure.py` — update to the new contract; add ladder-coverage tests (A100-80 full → A100-40 for a fits-40GB short intent; A100-80 full + long job → RunPod skipping spot; A100-80 full + short job → spot before RunPod).
- CLAUDE.md "Compute backends — multi-lane router" section — update the prose so the documented contract matches (the "auto chain NEVER calls RunPod" line + the explicit-gcp-no-fallback line both change).

## Acceptance criteria

- A simulated A100-80 quota-exceeded failure for a `eval`/`debug`/`lora-7b` short job routes to A100-40 (or spot) on GCP, not to a hard block.
- A long (> threshold) job with A100-80 full and A100-40 unusable routes to RunPod, skipping spot.
- An explicit `backend: gcp` pin gets the same ladder (the #654 regression).
- `uv run pytest tests/test_gcp_backend.py tests/test_router*.py tests/test_no_auto_runpod_path_under_any_failure.py tests/test_no_dollar_budget_caps.py` passes.
- CLAUDE.md prose matches the new behavior.

## Provenance

Originating prompt (Thomas, 2026-06-17, PM session): "The GCP thing shouldn't happen again. It should try to use the A100-40GB if useable, else preemptible if it's pretty short, else use runpod" → "Fix the workflow".
