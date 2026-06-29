---
title: 'workflow-fix: GCP-lane GPU-idle advisory+escalation parity (backend_poll.py)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:429bf27e53aa
created_at: '2026-06-29T10:06:33Z'
has_clean_result: false
origin_prompt: 'Auto-filed by /issue Step 10 from #727''s workflow-fix-candidate (GCP-lane
  GPU-idle escalation parity).'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #727 (emitting agent: `implementer`).

## Goal

Add a GCP-lane GPU-idle advisory + escalation parity tier to
`backend_poll.py` keyed on a cheap `gpu_util` probe (or the `eps/phase`
guest-attribute string + a VM-side `nvidia-smi` sample), mirroring
`poll_pipeline.py`'s `_maybe_post_gpu_idle_advisory` /
`_maybe_escalate_gpu_idle` (Telegram push, never stops the VM).

## Workflow gap

- **Bug observed:** The GCP lane has no GPU-idle advisory/escalation,
  so a multi-GPU GCP VM idle in an upload/CPU-only phase bleeds credits
  up to the 24h `--max-run-duration` fence with no surfacing. The
  RunPod-lane #664 fix added in #727 does not cover GCP.
- **Why it is a workflow gap:** `backend_poll.py` (the GCP/SLURM lane
  poller) carries no `gpu_util` / GPU-idle probe today, so #727's
  advisory → escalation tier has no GCP analogue. The gap is in the
  unified backend-router poller surface (workflow surface per
  `.claude/rules/workflow-fix-on-bug.md`), not experiment code.
- **Confidence (emitter):** medium.

## Proposed change (candidate diff sketch — refine in planning)

```
# backend_poll.py — after the GCP poll resolves current_phase:
+ gpu_util = _gcp_gpu_util_probe(handle)        # new: ssh nvidia-smi or guest-attribute
# NOTE: GCP idle is fence-bounded (24h DELETE), so this is lower-severity than the RunPod leak.
```

Mirror the RunPod-lane recipe in #727:
- `_maybe_post_gpu_idle_advisory` analogue (one-shot per phase).
- `_maybe_escalate_gpu_idle` analogue at the `EPM_GPU_IDLE_ESCALATION_MIN` threshold
  (Telegram push + `[gpu-idle-escalation]` marker; NEVER stops the VM, per
  autonomous-mode rule).
- Same `_phase_is_cpu_only` deny-list (or its GCP-`eps/phase` equivalent).
- Same shared `gpu_idle_since_epoch` state shape so dashboard / dashboard-consumers
  read parallel fields across lanes.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/backend_poll.py`
- Secondary: `src/explore_persona_space/backends/gcp.py` (for the probe helper).
- New tests under `tests/test_backend_poll.py` mirroring
  `tests/test_poll_gpu_idle_escalation.py` and
  `tests/test_poll_next_interval.py`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` (no-flags default) passes; ruff on touched files passes.
- NEVER call `pod.py stop` / `gcloud compute instances stop|delete` /
  similar VM-stopping from the escalation path (autonomous-mode rule).
- Use the same `EPM_GPU_IDLE_ADVISORY_MIN` / `EPM_GPU_IDLE_ESCALATION_MIN`
  env vars so a single fleet-wide policy holds across lanes.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/backend_poll.py, src/explore_persona_space/backends/gcp.py
- fingerprint: 429bf27e53aa

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/backend_poll.py, src/explore_persona_space/backends/gcp.py
bug_observed: The GCP lane has no GPU-idle advisory/escalation, so a multi-GPU GCP VM idle in an upload/CPU-only phase bleeds credits up to the 24h --max-run-duration fence with no surfacing (the RunPod-lane #664 fix added in #727 does not cover GCP).
why_workflow_gap: backend_poll.py (the GCP/SLURM lane poller) carries no gpu_util / GPU-idle probe today, so the #727 advisory→escalation tier has no GCP analogue; the gap is in the unified backend-router poller surface (workflow surface per workflow-fix-on-bug.md), not experiment code.
proposed_change: Add a GCP-lane GPU-idle advisory+escalation parity tier to backend_poll.py keyed on a cheap gpu_util probe (or the eps/phase string + a VM-side nvidia-smi sample), mirroring poll_pipeline.py's _maybe_post_gpu_idle_advisory / _maybe_escalate_gpu_idle (Telegram push, never stops the VM).
confidence: medium
related_task: #727
<!-- /workflow-fix-candidate -->
