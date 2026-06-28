---
title: 'workflow-fix: autonomous_session_watch.py compose RunPod no-port wedge predicate
  (#664 backstop)'
kind: infra
tags: []
created_at: '2026-06-28T00:00:02Z'
has_clean_result: false
parent_id: 689
origin_prompt: 'Auto-filed from the deferred D.3 workflow-fix-candidate v1 block at
  the tail of #689''s plan v3 (the planner''s surfaced prose; CLAUDE.md workflow-fix-on-bug
  protocol).'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #689 (the deferred D.3 follow-up of the plan; emitting agent: planner — surfaced in v3 §D.3 + a verbatim `<!-- workflow-fix-candidate v1 -->` block at the plan tail).

## Goal

Add a RUNNING-but-no-port predicate to `scripts/autonomous_session_watch.py`'s pod-safety pass that reuses `backend_poll._is_runpod_async_wedge_failure` (from #689 fix (b)) so a wedge on a poll-dead task is still caught.

## Workflow gap

- **Bug observed:** The watcher's pod-safety pass does not recognize a RunPod RUNNING-but-unreachable (no-port) pod on an ACTIVE task — it is a billing leak the pass currently misses. The poller-side fix (b) in #689 auto-recovers the common case, but a task whose poll loop has died still needs the watcher backstop (the residual failure mode #689 explicitly leaves open).
- **Why it is a workflow gap:** `autonomous_session_watch.py`'s pod-safety reconciliation pass is a workflow-surface backstop for pod-safety; it should compose the #689 fix-(b) wedge predicate (`backend_poll._is_runpod_async_wedge_failure`) so a wedge on a poll-dead task is still caught. Without this, a session whose `backend_poll.py` chain has died will leave its RUNNING-but-no-port pod billing indefinitely until the next interactive `/issue <N>` resume.
- **Confidence (emitter):** medium (the predicate is small, but the file is ~10,628 lines and the pod-safety pass needs grep-locating; sequencing carefully matters).

## Proposed change (candidate diff sketch — refine in planning)

```python
# in the pod-safety pass (grep-locate it in the ~10.6k-line file):
+ # #664 RunPod wedge backstop (composes #689 fix (b)): a managed RUNNING pod
+ # with null runtime.ports for >K on an ACTIVE task is an unreachable billing
+ # leak the poller may have missed if its poll loop died. Surface/alert; gate
+ # any auto-stop on per-cell inputs-on-HF + keep-running, mirroring fix (b).
+ if _runpod_running_no_port(pod) and task_active(issue) and not keep_running(issue):
+     ... alert / compose with fix (b) recovery ...
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing:
  ```
  grep -rln 'pod-safety\|RUNNING.*no.*port\|TRANSIENT_CAPACITY_REASONS' scripts/autonomous_session_watch.py CLAUDE.md
  ```

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The new predicate must COMPOSE with the existing #689 fix (b) detection — call `backend_poll._is_runpod_async_wedge_failure` directly, never re-define it (recursion-guard exempt: this is the watcher-side BACKSTOP for the case where the poller's `_maybe_escalate_runpod_wedge` never ran because the poll loop is dead).
- Gate any auto-stop on the same inputs-on-HF + `keep-running` checks as fix (b) so the watcher's backstop is symmetric with the poller path.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 7d213f536fe9

<!-- workflow-fix-candidate v1 -->
target_file: scripts/autonomous_session_watch.py
bug_observed: The watcher's pod-safety pass does not recognize a RunPod RUNNING-but-unreachable (no-port) pod on an ACTIVE task — it is a billing leak the pass currently misses (the poller-side fix (b) in #689 auto-recovers the common case, but a task whose poll loop has died still needs the watcher backstop).
why_workflow_gap: autonomous_session_watch.py's pod-safety reconciliation pass is a workflow-surface backstop for pod-safety; it should compose the #689 fix-(b) wedge predicate (backend_poll._is_runpod_async_wedge_failure) so a wedge on a poll-dead task is still caught.
proposed_change: Add a RUNNING-but-no-port predicate to the watcher's pod-safety pass that reuses backend_poll's #664 wedge detection, gated on the same per-cell inputs-on-HF + keep-running exemptions as fix (b), composing with the existing no_compute_available capacity-retry pass.
diff_sketch: |
  # in the pod-safety pass (grep-locate it in the ~10.6k-line file):
  + # #664 RunPod wedge backstop (composes #689 fix (b)): a managed RUNNING pod
  + # with null runtime.ports for >K on an ACTIVE task is an unreachable billing
  + # leak the poller may have missed if its poll loop died. Surface/alert; gate
  + # any auto-stop on per-cell inputs-on-HF + keep-running, mirroring fix (b).
  + if _runpod_running_no_port(pod) and task_active(issue) and not keep_running(issue):
  +     ... alert / compose with fix (b) recovery ...
confidence: medium
related_task: #689
<!-- /workflow-fix-candidate -->
