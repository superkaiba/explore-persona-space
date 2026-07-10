---
title: 'workflow-fix: preserve workload_cmd across exit-75 reconnect handle rewrites
  (GCP->RunPod failover)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2986f1a6a35d
created_at: '2026-07-08T01:50:45Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by the /issue 1090 orchestrator: backend_poll
  queue-timeout failover crashed on a reconnect-rewritten handle lacking extra.workload_cmd'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1090 (emitting agent: orchestrator /issue session).

## Goal

Preserve `extra.workload_cmd` (and `hydra_args`) on the GCP handle sidecar across exit-75 reconnect rewrites so the queue-timeout RunPod failover can always reconstruct its RunSpec.

## Workflow gap

- **Bug observed:** on #1090 launch 2, the FLEX_START instance sat PENDING past the queue-wait fence; `backend_poll.py`'s #783 queue-timeout escalation cancelled the queued instance, then CRASHED: `ValueError: GCP handle for issue 1090 lacks workload_cmd/hydra_args in extra (pre-#659 handle?); cannot reconstruct a RunSpec for the RunPod failover` (`backend_poll.py:2823` via `_failover_gcp_to_runpod:3204`). Root cause: the FIRST launch (exit-75, gcloud-create timeout with instance live-PENDING) wrote a complete handle; the prescribed SAME-COMMAND RERUN took `route()`'s RECONNECT path, which REWROTE the sidecar WITHOUT the workload extras. The orchestrator had to run the failover manually (cancel + explicit `--backend runpod` relaunch); an unattended session would have stranded.
- **Why it is a workflow gap:** the exit-75 contract ("re-run the SAME command") and the reconnect handle-write are both workflow-surface behavior (`backends/issue_dispatch.py` / `backends/gcp.py reconnect_or_none` / `dispatch_issue.py`); their composition silently produces a failover-incapable handle. The poller's error message even mis-attributes it ("pre-#659 handle?").
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ On the reconnect path, MERGE the existing sidecar's extra (workload_cmd,
+ hydra_args, boot_disk_gb, max_run_duration, repo_branch) into the new handle
+ instead of rewriting from the reconnect probe alone — or make the launch CLI
+ re-thread its own --workload-cmd/--hydra into the handle extra on EVERY
+ write (the rerun invocation carries them verbatim).
+ In _runspec_from_gcp_handle, fall back to the CLI-visible workload args from
+ the retired .finalized predecessor sidecar (same issue) before refusing.
+ Regression test: exit-75 rerun reconnect preserves extra.workload_cmd;
+ queue-timeout failover reconstructs a RunSpec from a reconnect-written handle.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/issue_dispatch.py`
- Sibling hits: `src/explore_persona_space/backends/gcp.py` (reconnect_or_none / handle write), `scripts/backend_poll.py` (`_runspec_from_gcp_handle` fallback + error text), `scripts/dispatch_issue.py`, `tests/test_backend_*.py`.
- Grep the workflow surface for the pattern before editing (`grep -rn "workload_cmd" src/explore_persona_space/backends/ scripts/backend_poll.py scripts/dispatch_issue.py`) and update every handle-write site; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_router*.py` + `tests/test_backend_*.py` stay green (no-auto-RunPod invariant untouched — the failover path is the #658/#783 sanctioned one).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/issue_dispatch.py
- fingerprint: 2986f1a6a35d

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/issue_dispatch.py, src/explore_persona_space/backends/gcp.py, scripts/backend_poll.py, scripts/dispatch_issue.py
bug_observed: backend_poll queue-timeout escalation crashed with ValueError GCP handle lacks workload_cmd/hydra_args in extra after an exit-75 rerun reconnected and rewrote the sidecar without the workload extra, leaving the failover unable to launch and the queued instance stranded
why_workflow_gap: the exit-75 same-command-rerun contract composes with the reconnect handle-rewrite to silently produce a failover-incapable sidecar; the RunPod queue-timeout failover then cannot reconstruct its RunSpec
proposed_change: Preserve extra.workload_cmd (and hydra_args) on the GCP handle sidecar across exit-75 reconnect rewrites so the queue-timeout RunPod failover can always reconstruct its RunSpec
diff_sketch: |
  + reconnect path merges prior sidecar extra (workload_cmd/hydra_args/...) into the new handle
  + _runspec_from_gcp_handle falls back to the .finalized predecessor sidecar before refusing
  + regression test: reconnect preserves extra.workload_cmd; failover reconstructs from a reconnect handle
confidence: high
related_task: #1090
<!-- /workflow-fix-candidate -->
