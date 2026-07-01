---
title: 'workflow-fix: gcp poller must map PENDING → running (FLEX_START queue false-stall)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2e28249543fd
created_at: '2026-07-01T04:00:02Z'
has_clean_result: false
origin_prompt: 'Emitted by /issue 778 orchestrator: sync-tested backend_poll.py --issue
  778 on a FLEX_START-queued PENDING instance, got status=stalled current_phase=unknown_pending
  — following SKILL.md Step 6d.2 literally would post epm:failure v1 + set-status
  blocked on a healthy queued instance. reconnect_or_none in the same file treats
  PENDING as live; the poller''s _gcp_status_to_poll_result should mirror that.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #778 (emitting agent: orchestrator (/issue skill)).

## Goal

Handle GCE status `PENDING` in `_gcp_status_to_poll_result` so a FLEX_START-queued instance is polled as `status=running current_phase=pending` (still waiting for capacity, keep polling) instead of the current false-stalled routing that would `set-status blocked` on a healthy queued instance.

## Workflow gap

- **Bug observed:** `GcpBackend.poll` (via `_gcp_status_to_poll_result` in `src/explore_persona_space/backends/gcp.py`) returns `{"status": "stalled", "current_phase": "unknown_pending"}` for a FLEX_START-queued GCE instance whose lifecycle state is `PENDING`. Per `/issue` SKILL.md Step 6d.2's bg-Bash poll pseudocode, `status=="stalled"` triggers `post epm:failure v1 ... run CRON-TEARDOWN ... set-status blocked; exit`. Live reproduction on task #778: instance `eps-issue-778` in `us-central1-b` created 02:07 UTC, sat legitimately in the FLEX_START capacity queue at `PENDING`, and `scripts/backend_poll.py --issue 778` returned the false-stalled JSON. Following the SKILL literal would have blocked a healthy run.
- **Why it is a workflow gap:** The reconnect probe `reconnect_or_none` (same file) already treats `PENDING` as live — it only excludes `_NONLIVE_INSTANCE_STATUSES = {TERMINATED, STOPPED, SUSPENDED}` from reconnect. The comment at line 3521 explicitly notes "FLEX_START rungs legitimately stay PENDING past the GCLOUD_DEFAULT_TIMEOUT_SEC subprocess cap" — PENDING is a known-legitimate live state on the credit-spending lane. Yet `_gcp_status_to_poll_result` maps only `RUNNING`, `PROVISIONING`, `STAGING`, `STOPPING`, `REPAIRING`, `TERMINATED`, `STOPPED`, `SUSPENDED`, letting `PENDING` fall through the final `return _coarse_poll(status="stalled", current_phase=f"unknown_{up.lower()}")` line. The reconnect-vs-poll asymmetry is the bug: reconnect thinks PENDING is live, poll thinks PENDING is stalled.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
--- a/src/explore_persona_space/backends/gcp.py
+++ b/src/explore_persona_space/backends/gcp.py
@@ _gcp_status_to_poll_result
     up = status.upper()
     if up == "RUNNING":
         return _coarse_poll(status="running", current_phase="running")
-    if up in {"PROVISIONING", "STAGING"}:
+    if up in {"PROVISIONING", "STAGING", "PENDING"}:
         return _coarse_poll(status="running", current_phase=up.lower())
     if up in {"STOPPING", "REPAIRING"}:
         return _coarse_poll(status="stalled", current_phase=up.lower())
     if up in {"TERMINATED", "STOPPED", "SUSPENDED"}:
         return _terminal_dead_poll(reason=up.lower())
     return _coarse_poll(status="stalled", current_phase=f"unknown_{up.lower()}")
```

Alternative if the planner prefers a distinct current_phase name: keep `PENDING` in its own branch returning `status="running", current_phase="pending"` so log readers can tell "FLEX_START capacity-queued" from "PROVISIONING (booting)".

Add a unit test covering:
1. `_gcp_status_to_poll_result("PENDING")` returns `status="running"` (regression pin).
2. The full `GcpBackend.poll` path with a mocked `gcloud describe` returning `status: PENDING` prints `status=running` from `backend_poll.py`, NOT `status=stalled`.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (the `_gcp_status_to_poll_result` function + its docstring; the docstring's status enum list needs the addition too).
- Also update: `tests/test_gcp_backend.py` (add the PENDING regression test).
- No changes to `reconnect_or_none` — it already treats PENDING as live.
- Grep for other `PENDING` references in the backend/router surface to confirm they already treat it as live (the audit at emit-time found: `_NONLIVE_INSTANCE_STATUSES` explicitly excludes it via non-membership; line 3521 comment acknowledges the state; nothing else routes on it).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Router tests (`tests/test_router*.py`) still pass — this fix is on the poll path, not the launch path.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (via the auto-set Provenance line) and MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 2e28249543fd

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py
bug_observed: GcpBackend.poll returns status=stalled current_phase=unknown_pending for a FLEX_START-queued GCE instance in state PENDING; per Step 6d.2 that would post epm:failure v1 and set-status blocked on a healthy queued instance (live repro on task #778, eps-issue-778 in us-central1-b sat PENDING in the FLEX_START capacity queue for ~1h47m).
why_workflow_gap: reconnect_or_none in the same file treats PENDING as live (only {TERMINATED, STOPPED, SUSPENDED} are non-live); the poll's _gcp_status_to_poll_result should mirror that and map PENDING to status=running current_phase=pending, but instead it falls through to unknown_pending -> stalled.
proposed_change: Handle GCE status PENDING in _gcp_status_to_poll_result — map to status=running current_phase=pending (FLEX_START-queued instance is legitimately live, poller should keep polling not stall).
diff_sketch: |
  --- a/src/explore_persona_space/backends/gcp.py
  +++ b/src/explore_persona_space/backends/gcp.py
  @@ _gcp_status_to_poll_result
       up = status.upper()
       if up == "RUNNING":
           return _coarse_poll(status="running", current_phase="running")
  -    if up in {"PROVISIONING", "STAGING"}:
  +    if up in {"PROVISIONING", "STAGING", "PENDING"}:
           return _coarse_poll(status="running", current_phase=up.lower())
       if up in {"STOPPING", "REPAIRING"}:
           return _coarse_poll(status="stalled", current_phase=up.lower())
confidence: high
related_task: #778
<!-- /workflow-fix-candidate -->
