---
title: 'workflow-fix: flock pods.conf writes; never drop RUNNING pods'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8d78c9a6672b
created_at: '2026-07-01T22:57:20Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #763: concurrent pods.conf sync wiped
  the pod-763 entry twice on 2026-07-01 (markers v16 + v21); serialize writes under
  flock + upsert-under-lock + refuse to remove RUNNING pods on sync. target: scripts/pod_config.py,
  scripts/pod_lifecycle.py fingerprint: 8d78c9a6672b'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #763 (emitting agent: orchestrator, /issue 763 session).

## Goal

Serialize pods.conf writes under flock and never drop RUNNING pods on sync.

## Workflow gap

- **Bug observed:** concurrent pods.conf sync wiped the pod-763 entry twice in
  one day (2026-07-01 markers v16 + v21 on task #763), breaking SSH resolution
  mid-run — the poll loop read `status=dead` on a healthy run because
  `ssh pod-763` fell through to DNS ("Could not resolve hostname").
- **Why it is a workflow gap:** `scripts/pods.conf` is shared mutable state
  written by multiple concurrent sessions (`pod_config.py`, `pod_lifecycle.py`
  provision/resume paths). Writers rebuild/rewrite the file from their own
  read snapshot with no lock, so a slow writer's flush loses a concurrent
  writer's fresh append (classic lost update). A pod that is RUNNING per the
  live RunPod API should never lose its entry to a sync.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ # pod_config.py / pod_lifecycle.py — every pods.conf mutation:
+ with flock(pods_conf_lock):        # e.g. ~/.task-workflow/pods-conf.lock
+     entries = parse(pods_conf_path.read_text())   # re-read UNDER the lock
+     entries[name] = new_row                        # upsert, never rebuild-from-snapshot
+     atomic_write(pods_conf_path, render(entries))  # temp + rename
+ # sync path: refuse to REMOVE an entry whose pod is RUNNING per the live API
+ # (removal allowed only for pods the API reports absent/terminated).
```

## Scope / surfaces

- Primary target: `scripts/pod_config.py`, `scripts/pod_lifecycle.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'pods.conf' scripts/`) and update every writer; list them in
  the plan. Read-only consumers (poll_pipeline.py, fleet_health.py, ...) are
  out of scope.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod_config.py, scripts/pod_lifecycle.py
- fingerprint: 8d78c9a6672b

<!-- workflow-fix-candidate v1 -->
target_file: scripts/pod_config.py, scripts/pod_lifecycle.py
bug_observed: concurrent pods.conf sync wiped the pod-763 entry twice in one day (2026-07-01 v16 + v21), breaking SSH mid-run
why_workflow_gap: pods.conf is shared mutable state rewritten by concurrent sessions from stale read snapshots with no lock; lost-update race drops fresh entries for RUNNING pods
proposed_change: serialize pods.conf writes under flock and never drop RUNNING pods on sync
diff_sketch: |
  + with flock(pods_conf_lock):
  +     entries = parse(read pods.conf under lock)
  +     entries[name] = new_row       # upsert, never rebuild-from-snapshot
  +     atomic_write(temp + rename)
  + sync: refuse to remove entries whose pod is RUNNING per live RunPod API
confidence: high
related_task: #763
<!-- /workflow-fix-candidate -->
