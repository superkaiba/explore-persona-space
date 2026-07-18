---
title: 'workflow-fix: pod-safety alert for unlaunched orphan pods on active tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dc6de410ab93
created_at: '2026-07-18T18:38:13Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 1481 recovery: orphan pod-1481
  (8xH100, $32/hr, ~2h) delivered by a detached wait-for-capacity provision after
  its requesting turn died; watcher pod-safety blind to RUNNING pods on ACTIVE tasks'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1481 (emitting agent: /issue orchestrator).

## Goal

The watcher's pod-safety pass flags a RUNNING managed pod on an ACTIVE-status task with no `epm:run-launched` newer than pod creation after a grace window (alert-only).

## Workflow gap

- **Bug observed:** a detached wait-for-capacity provision outlived its requesting turn and delivered an unlaunched orphan 8xH100 pod that billed $32/hr for ~2h, invisible to every watcher pass.
- **Why it is a workflow gap:** `pod_lifecycle provision` deliberately detaches (setsid, #573) and can deliver a pod AFTER the requesting orchestrator turn is gone; nothing then claims or launches on the pod. The watcher's pod-safety pass auto-stops/alerts only on DONE/parked tasks — a RUNNING pod on an ACTIVE task (#1481 at `running`) is presumed healthy even with zero workload ever launched (no `epm:run-launched` since creation, 0% GPU). On 2026-07-18 pod-1481 idled ~2h ≈ $64 and was found only by a manual burn probe; it also blocked the 17:52Z RunPod fallback provision (name conflict).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "setsid" scripts/pod_lifecycle.py` → provision detaches into its own session (the "#573" detach banner printed in the 17:52Z stderr tail); `grep -n "POD_SAFETY_AUTO_STOP\|pod_safety" scripts/autonomous_session_watch.py` → the pass keys on done/parked statuses only — no unlaunched-orphan predicate for ACTIVE-status tasks (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # pod-safety: unlaunched-orphan flag (alert-only, never auto-stop):
+ # a RUNNING managed pod whose owning task is at an ACTIVE status but has
+ # NO epm:run-launched marker newer than the pod's created_at, after a
+ # >=60-min grace (covers provision/bootstrap + experimenter dispatch),
+ # fires a deduped Telegram alert naming the pod + $/hr + the reuse-or-
+ # terminate commands. Never stops (a launch may be seconds away).
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Alert-only for ACTIVE-status tasks — the pass must NEVER auto-stop a pod on an active task (the existing conservative contract stands).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: dc6de410ab93

Surfaced prose (verbatim): a detached wait-for-capacity provision outlived its requesting turn and delivered an unlaunched orphan 8xH100 pod that billed $32/hr for ~2h invisible to every watcher pass (pod-safety only covers done/parked tasks).
