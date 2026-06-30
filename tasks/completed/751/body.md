---
title: 'daily-fix: failover relaunch must sync new pod into pods.conf'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e207e2050f6
- daily-auto-filed
created_at: '2026-06-30T06:43:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: A GCP->RunPod failover relaunch
  registered the new pod outside pod.py provision, so it was absent from pods.conf/SSH/MCP;
  the raw-SSH poller then reported status=dead (false alarm) and `pod.py config --update`/`--refresh-from-api`
  both errored ''pod not found in pods.conf''.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. Recurring across pod-664, pod-697, pod-658.

## Goal
A failover/relaunched pod is always registered in pods.conf, and the documented recovery commands work when the pod is absent.

## Workflow gap
- **Bug observed:** GCP->RunPod failover relaunch (backend_poll.py path) bypassed `pod.py provision` sync, leaving the new pod absent from `pods.conf`/SSH/MCP. The raw-SSH poller then read `status=dead` (false alarm), and `pod.py config --update pod-697` AND `--refresh-from-api pod-697` BOTH failed with "pod 'pod-697' not found in pods.conf" — the documented recovery path does not work for an absent pod.
- **Why it is a workflow gap:** the failover relaunch path + the pod-config recovery surface are workflow infra (backends/ router + pod helper).
- **Confidence (emitter):** medium-high; reproduced in 3 independent sessions today.

## Proposed change
- The failover relaunch in `backend_poll.py` calls the pod_config sync so the new pod lands in pods.conf (the same registration `pod.py provision`/`resume` perform).
- `pod.py config --update` / `--refresh-from-api` CREATE a missing entry from the live RunPod API instead of erroring when the pod is absent.
- Poller cross-checks the live API before declaring a pod dead on a name-resolution failure.

## Scope / surfaces
- `src/explore_persona_space/backends/backend_poll.py` (failover relaunch path)
- `scripts/pod_config.py` (--update / --refresh-from-api create-missing)
- Grep `pods.conf` / `refresh-from-api` callers before editing.

## Constraints / invariants
- Workflow/infra surface only. ruff on touched files; relevant `tests/test_router*.py` / pod-config tests pass.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 (recursion guard).

## Provenance
- workflow_fix_target: src/explore_persona_space/backends/backend_poll.py, scripts/pod_config.py
- fingerprint: 1e207e2050f6

Sessions: 3bded713 (issue-697), f317c8a8 (issue-664), 6185d397 (issue-658).
