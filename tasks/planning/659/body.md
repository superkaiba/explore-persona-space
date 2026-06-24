---
title: Wire async GCP-workload-failure -> RunPod failover (poller/dispatch path)
kind: infra
tags: []
created_at: '2026-06-24T18:45:44Z'
has_clean_result: false
origin_prompt: if something is failing on GCP then the run should just run it on runpod
  while we fix GCP
---
---
kind: infra
---

# Wire the ASYNC GCP-workload-failure → RunPod failover (poller / dispatch path)

## Context
The synchronous `route()`-time failover (a `GcpWorkloadError` raised *during*
backend selection → RunPod) landed in the GCP-fix merge (`f0e8aadcff`, on `main`):
`.claude/rules/compute-backend-failover.md`, `backends/router.py`, `backends/gcp.py`.

But the **common production failure mode is async, not sync**: a deterministic
workload bug (e.g. #658's json_k2 OOM) surfaces *minutes into the run*, is
detected by the **poller** (`scripts/backend_poll.py` / `issue_dispatch.dispatch_for_issue`),
and currently ends at `status:blocked` — it is **never re-dispatched to RunPod**.
So the user's directive ("if GCP is failing, run it on RunPod while we fix GCP")
is only half-real: the rare route()-time case fails over; the frequent poller-detected
case does not.

## Goal / change
When the poller surfaces a GCP `workload_failure` (a non-capacity, non-infra GCP
run failure), **re-dispatch the run on RunPod exactly once**, reusing the same
structural bound as the sync path: the RunPod attempt runs the job once; a re-crash
→ `failure_class: code` → `status:blocked`, which the watcher's capacity-retry pass
(re-drives `no_compute_available` only) never re-launches. No new counter.

## Acceptance
- A test asserting "a poller-surfaced GCP `workload_failure` results in exactly one
  RunPod re-dispatch, then `blocked` on a re-crash" — fails today, passes after.
- Scope: GCP lane only (SLURM workload failures keep surfacing `WorkloadSurfacedError`).
- In-scope workflow surface: `scripts/backend_poll.py`, `src/explore_persona_space/backends/issue_dispatch.py`, `backends/router.py`, `tests/test_router*.py` / `tests/test_backend_*.py`.

## Provenance
Surfaced by the GCP-fix workflow-improver (code-reviewer Major, pre-existing) as a
follow-up to the #658 GCP-crash-diagnostics + sync-failover fix. User directive:
"if something is failing on GCP then the run should just run it on runpod while we
fix GCP." This task completes that for the real (async) failure path.
