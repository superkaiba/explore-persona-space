---
title: 'workflow-fix: RunPod lane drops --workload-cmd (provision-only launch)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:be5f3fa75f02
created_at: '2026-07-03T05:08:43Z'
has_clean_result: false
origin_prompt: 'orchestrator incident on #763: runpod --workload-cmd launch bootstrapped
  pod-763 but never started the workload; manual SSH launch required'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own incident on task #763 (cofit Phase C RunPod failover, 2026-07-03).

## Goal

Make `RunPodBackend.launch` actually EXECUTE a `--workload-cmd` (nohup on the pod + pod-side pid file + log at the handle's `log_path` + a run-launched sidecar/marker), instead of provision-and-bootstrap-only.

## Workflow gap

- **Bug observed:** `dispatch_issue.py launch --issue 763 --backend runpod --workload-cmd 'bash scripts/issue763_cofit_dispatch.sh --phase-group C'` returned ok (pod-763 provisioned, bootstrapped, handle sidecar written with `log_path=/workspace/logs/issue-763.log`) but the workload NEVER started: no process, empty /workspace/logs, first poll tick read status=dead/unknown. The orchestrator had to SSH in and nohup the command manually (+ post epm:run-launched by hand).
- **Why it is a workflow gap:** the #588 contract ("every lane executes custom dispatch scripts via --workload-cmd") does not hold on the RunPod lane — the launch path silently drops the workload command, so every RunPod `--workload-cmd` failover (the GCP-to-RunPod doctrine's own target lane) lands as a healthy-looking pod running nothing, indistinguishable from a crashed run to the poller.
- **Confidence (emitter):** high (reproduced live; the manual SSH launch of the identical command ran immediately).

## Proposed change (candidate diff sketch — refine in planning)

```
src/explore_persona_space/backends/runpod.py (launch):
+ when spec.workload_cmd is set: after bootstrap, ssh-exec a launch leg that
+ (1) syncs the pod clone to spec.repo_branch (fetch + reset the pod-side
+     tree to origin/<repo_branch>; pod-side only, never the VM tree),
+ (2) sources the pod .env, ensures /workspace/logs exists,
+ (3) nohup-runs the workload_cmd redirecting to <log_path>, writes the
+     pid to a pod-side pid file,
+ (4) verifies the log file exists before returning, and records pid +
+     log_path in the RunHandle sidecar (mirroring the experimenter contract).
scripts/dispatch_issue.py: surface a LOUD error / nonzero exit if a runpod
--workload-cmd launch returns with no live workload pid (fail-fast instead
of ok:true).
```

## Scope / surfaces

- Primary targets: `src/explore_persona_space/backends/runpod.py` (in-scope backends module), `scripts/dispatch_issue.py`.
- Grep: `grep -n "workload_cmd" src/explore_persona_space/backends/*.py scripts/dispatch_issue.py scripts/pod_lifecycle.py` — confirm which layer drops it; add a router test pinning the RunPod workload-cmd execution (tests/test_router*.py / tests/test_backend_*.py).

## Constraints / invariants

- Workflow-surface only; the pod NEVER shells scripts/task.py (sentinel/sidecar signaling only); router tests stay green.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py
- fingerprint: be5f3fa75f02
