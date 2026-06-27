---
title: 'GCP auto-lane: length-aware spot-first / flex-start ladder (on-demand A100
  is stockout-prone; spot is a live separate pool, flex-start is the canonical capacity
  answer)'
kind: infra
tags: []
created_at: '2026-06-27T01:57:59Z'
has_clean_result: false
origin_prompt: 'Spot-first -> flex-start router-ladder: make GCP capacity reliable
  (on-demand A100 ZONE_RESOURCE_POOL_EXHAUSTED; spot live; flex-start canonical).
  User directive 2026-06-26 after the deep-research capacity report + #672 spot live
  validation.'
---
## Problem

GCP A100 **on-demand** is frequently `ZONE_RESOURCE_POOL_EXHAUSTED` (capacity
stockout — confirmed across ALL us-central1 zones, 2026-06-26), while **spot**
A100 has capacity (proven: spot `a2-ultragpu-1g` created+deleted; #672's live
validation ran on a spot A100-80). The current auto-lane ladder (#656) is
**on-demand-first** (on-demand A100-80 → A100-40 → spot → … → RunPod), so it
burns time failing on the exhausted on-demand pool before reaching the live spot
pool. Deep-research (2026-06-26) confirmed: spot is a **separate, larger capacity
pool**; Flex-start/DWS is Google's canonical answer for scarce capacity;
on-demand-first is the wrong order.

## Change: length-aware spot-first / flex-start ladder

Reorder the GCP auto-lane rung ladder, **length-aware** (so preemption never
costs a long job):
- **SHORT jobs** (`time_budget × gpu_count ≤ EPS_GCP_SPOT_MAX_GPU_HOURS`, the
  existing gate): **spot-first** (multi-zone us-central1) → flex-start →
  on-demand → RunPod. Spot preemption is cheap + recoverable here (clean
  TERMINATE → #659 failover / checkpoint-resume).
- **LONG / unknown-length jobs**: **flex-start-first** (DWS:
  `--provisioning-model=FLEX_START --request-valid-for-duration=2h` — queues for
  capacity instead of failing, NON-preemptible once running up to 7 days) →
  on-demand → RunPod. **Skip spot** for long jobs (preemption too costly; spot is
  already gated off them).
- All rungs keep `--instance-termination-action=DELETE` + `--max-run-duration`;
  checkpoint/resume stays on so a preemption resumes, not restarts.

## Feasibility (verified 2026-06-26 — deep-research + the #672 live run)

- Spot A100 is a live, separate pool (proven). Flex-start supports A2/A100 via the
  single flag `--provisioning-model=FLEX_START` (preemptible quota; DWS-discounted;
  non-preemptible once running). On-demand reservations guarantee capacity but
  bill whether-used (only short-lived, for a must-finish run). **Calendar-mode
  future reservations do NOT support A100/A2** — excluded.

## Implementation

- `backends/gcp.py` + `backends/router.py`: reorder the rung ladder to the
  length-aware order above; add a **FLEX_START rung** (`provisioning-model=FLEX_START`
  + `request-valid-for-duration`); keep the existing spot short-job gate
  (`EPS_GCP_SPOT_MAX_GPU_HOURS`); demote on-demand below spot/flex.
- PRESERVE: the GCP-workload-failover-to-RunPod (#658/#659), the per-rung
  `epm:backend-selected` markers, the DELETE/max-run-duration ephemeral contract,
  and the existing capacity-miss → next-rung cascade.
- Tests (`tests/test_router.py`): short job → spot rung attempted BEFORE
  on-demand; long job → flex-start BEFORE on-demand; flex-start rung renders the
  correct gcloud flags; on-demand-exhausted no longer jumps straight to RunPod
  (spot/flex tried first); RunPod stays the terminal rung.
- Docs: update `.claude/rules/compute-backend-failover.md` + the CLAUDE.md
  compute-backends bullet to document the new ladder order.

## Acceptance

- Router attempts spot (short) / flex-start (long) BEFORE on-demand; an on-demand
  stockout no longer wastes the launch; RunPod stays the terminal fallback.
- Tests green; rule docs consistent with the new ladder.
- `backends/*.py` is workflow surface — full /issue treatment.

## Provenance

User directive 2026-06-26 ("file the spot-first → flex-start router-ladder"),
grounded in the deep-research capacity report + the #672 spot-A100 live validation
+ repeated on-demand `ZONE_RESOURCE_POOL_EXHAUSTED` stockouts. The #677 CPU-lane
work is separate (CPU-only phases); this is the GPU capacity ladder.
