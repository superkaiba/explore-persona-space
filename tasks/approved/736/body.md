---
title: 'workflow-fix: dispatch_issue.py exit-4 TimeoutExpired on FLEX_START PENDING
  should convert to exit-75'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c42cd199ce68
created_at: '2026-06-29T20:08:55Z'
has_clean_result: false
origin_prompt: 'issue #658 r1 experimenter failure-lesson: GCP-lane analogue of SSH-timeout-not-child-dead
  — gcloud-create 300s cap fires on FLEX_START queueing while the instance is created
  server-side; dispatch_issue.py exit 4 mis-routes as failure'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised by the `experimenter` agent during issue #658's `persona-vectors-style-rb`
follow-up round 1 relaunch (root_cause_confirmed: yes; gotcha_candidate: yes).

## Goal

Convert `dispatch_issue.py launch`'s exit-4 on `subprocess.TimeoutExpired`
into the documented "still-waiting / launch in flight" path when the
underlying `gcloud compute instances create` succeeds server-side
(FLEX_START PENDING for preemptible capacity).

## Workflow gap

- **Bug observed:** `scripts/dispatch_issue.py launch` exits 4 with a
  `subprocess.TimeoutExpired` when its internal
  `gcloud compute instances create` subprocess exceeds the 300s cap in
  `default_gcloud_runner` (`src/explore_persona_space/backends/gcp.py:1957`).
  On the FLEX_START rung the instance legitimately stays PENDING (queued
  for preemptible A100-80 capacity) well past 300s, so the create OFTEN
  SUCCEEDS server-side — but the local dispatch CLI surfaces exit 4 with
  a Python traceback, which mis-routes as a launch failure. The
  experimenter's existing contract documents only exit 0 (clean launch),
  exit 75 (EX_TEMPFAIL — still waiting, re-run), and exit 3 (terminal
  failure). Exit 4 is not in the contract; a naive caller would post
  `epm:failure v1` and relaunch — racing or duplicating the live PENDING
  instance.
- **Why it is a workflow gap:** the dispatch CLI is the canonical launch
  mechanism that every backend goes through. Its exit code is what every
  caller (the experimenter agent, the orchestrator, `/issue-tick`, the
  poller) routes on. An undocumented exit kind that fires on a NORMAL
  FLEX_START queueing state is a routing trap that will repeat for every
  experiment that pushes spot/flex capacity (#658 hit it; subsequent
  FLEX_START dispatches will hit it again).
- **Confidence (emitter):** high (root cause confirmed in #658 r1; the
  experimenter caught it by checking `gcloud instances list` and held
  its turn correctly; the failure-lesson block is captured at task #658
  `epm:failure-lesson v2`).

## Proposed change (candidate diff sketch — refine in planning)

```
# src/explore_persona_space/backends/gcp.py

def default_gcloud_runner(argv, *, timeout: int = 300) -> GcloudRunResult:
    try:
        # ... existing subprocess.run(argv, timeout=timeout, ...) ...
    except subprocess.TimeoutExpired as exc:
+       # FLEX_START rungs legitimately stay PENDING for preemptible capacity
+       # past the 300s subprocess cap — check whether the instance was
+       # actually created server-side BEFORE propagating the timeout as a
+       # failure. If present, return a GcloudRunResult flagged
+       # `created_during_timeout=True` so the caller can convert to
+       # exit-75 still-waiting rather than exit-4 traceback.
+       if _instance_exists_after_timeout(argv):
+           return GcloudRunResult(returncode=0, stdout="", stderr=str(exc),
+                                  created_during_timeout=True)
        raise

# scripts/dispatch_issue.py
# In the create wrapping site:
-   except subprocess.TimeoutExpired as exc:
-       raise   # propagates as exit 4
+   except subprocess.TimeoutExpired as exc:
+       # Resolve "is the instance actually live?" — only treat as failure
+       # when it's truly absent. Else convert to exit 75 + log a still-
+       # waiting JSON line so the bg-Bash poller / backstop tick re-drives.
+       if _live_instance_after_timeout(instance_name, project, zone):
+           sys.exit(_exit_still_waiting(reason="gcloud_create_timeout_pending"))
+       raise  # genuine failure
```

The dispatcher should ALSO post a `.claude/rules/gotchas.md` entry
documenting the GCP-lane analogue of the existing "SSH timeout ≠ child
dead — pgrep before relaunch" pattern, so consumers reading the gotchas
file find this gotcha alongside the SSH one.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
  (`default_gcloud_runner` at line 1957, and the
  `subprocess.TimeoutExpired` catch sites it feeds — lines 3782-3783 in
  `GcpBackend.poll` already document one such catch for the hung-VM
  path).
- Secondary target: `scripts/dispatch_issue.py` (the launch wrapper that
  surfaces the exit code).
- Documentation: `.claude/rules/gotchas.md` (add the GCP-lane analogue
  of the existing "SSH timeout ≠ child dead — pgrep before relaunch"
  entry).
- Grep the workflow surface for the pattern (paths listed above already
  enumerate the hits). The planner should also re-grep before editing
  in case the surface has shifted.

## Constraints / invariants

- Workflow-surface only — `src/explore_persona_space/backends/*.py` is in
  scope per `.claude/rules/workflow-fix-on-bug.md`; do NOT touch
  experiment code, `configs/`, or `tasks/`.
- `tests/test_gcp_backend.py` + `tests/test_router*.py` must still pass.
  Add a regression test that simulates the FLEX_START create-timeout
  shape (`subprocess.TimeoutExpired` from `default_gcloud_runner` with a
  live instance discoverable by a `gcloud instances list` stub) and
  asserts the dispatch wrapper converts to exit 75 / still-waiting
  rather than exit 4 / failure.
- Preserve the existing fail-loud behavior on genuine create failures
  (the instance is truly absent after the timeout): exit 3 / hard error
  / no zombie GCE VM left behind.
- This session runs under `EPM_WORKFLOW_FIX_SESSION` (carries the
  `workflow_fix_target:` Provenance line) — it MUST NOT auto-route any
  of its own subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py, scripts/dispatch_issue.py, .claude/rules/gotchas.md
- fingerprint: c42cd199ce68

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py, scripts/dispatch_issue.py, .claude/rules/gotchas.md
bug_observed: dispatch_issue.py launch exits 4 with subprocess.TimeoutExpired when gcloud compute instances create's 300s subprocess cap fires on FLEX_START preemptible queueing, while the instance is created server-side; mis-routes as a launch failure
why_workflow_gap: the dispatch CLI is the canonical launch mechanism every backend goes through; exit 4 is not in its documented contract (clean/exit-75/exit-3) and fires on a NORMAL FLEX_START queueing state — every FLEX_START dispatch will hit it
proposed_change: Detect live instance on dispatch_issue.py exit-4 TimeoutExpired (FLEX_START PENDING) and convert to exit-75 still-waiting; document the GCP-lane gotcha in .claude/rules/gotchas.md
diff_sketch: |
  # src/explore_persona_space/backends/gcp.py default_gcloud_runner:
  except subprocess.TimeoutExpired as exc:
  +   if _instance_exists_after_timeout(argv):
  +       return GcloudRunResult(returncode=0, stdout="", stderr=str(exc),
  +                              created_during_timeout=True)
      raise
  # scripts/dispatch_issue.py launch wrapper:
  except subprocess.TimeoutExpired as exc:
  +   if _live_instance_after_timeout(instance_name, project, zone):
  +       sys.exit(_exit_still_waiting(reason="gcloud_create_timeout_pending"))
      raise
confidence: high
related_task: #658
<!-- /workflow-fix-candidate -->
