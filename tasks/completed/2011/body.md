---
title: 'daily-fix: provisioner avoids a known-bad RunPod host on ret'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c5c655718a1d
- daily-auto-filed
created_at: '2026-08-02T07:15:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): pod-1947-b: 3 consecutive
  provision attempts (18:27-18:46Z) all landed on host 103.207.149.60, each RUNNING-but-SSH-refused
  and terminated; the fleet ran single-pod, ~doubling P2 wall time. The provisioner
  has no lever to avoid a known-bad host between retries.'
workflow: v1
---
# daily-fix: provisioner avoids a known-bad RunPod host on retry

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C13 (miner 4, P3; session 8fc069db, issue #1947).

## Goal
After a fresh provision fails its SSH/bootstrap probe, record the placement's host IP (and datacenter) and bias the next provision attempt away from it — a different-DC retry hint at minimum — instead of re-rolling blind onto the same broken host.

## Workflow gap
- **Bug observed:** All three provision attempts for pod-1947-b (18:27Z–18:46Z, three provision+terminate pairs) landed on host 103.207.149.60; each came up RUNNING but SSH-refused (bootstrap died at start) and was terminated. The orchestrator gave up ("Fleet is now single-pod by capacity reality"), roughly doubling the #1947 P2 wall time. `unverified hypothesis — verify at plan time:` the identical-host placement is read from the session's own summary ("three placements on the broken host terminated"); provisioner behavior inferred, not reproduced (miner-inferred, not probed).
- **Why it is a workflow gap:** The provisioning path has no host-avoidance lever between retries — a known-bad host can eat every retry in a supply-constrained window.
- **Confidence:** medium
- verified-at-filing: `grep -n -i 'datacenter\|host\|avoid\|exclude' scripts/runpod_api.py` → `dataCenterId` is threaded into create inputs (lines 582, 758 — a PIN lever exists) and `ssh_host` is parsed from `port.get("ip")` (line ~438 — the failed host IP IS observable at probe time); NO avoidance/exclusion logic and NO `getAllDatacenters` helper (0 hits) — matches the standing memory note that runpod_api.py lacks the DC-probe levers; `grep -n 'def cmd_provision' scripts/pod.py scripts/pod_lifecycle.py` → pod.py:200 is a thin dispatcher (`_lifecycle("provision", ...)`); the provision/retry/SSH-wait construction site is `scripts/pod_lifecycle.py` (cmd_provision at :2079, SUPPLY_CONSTRAINT handling :861/:2240/:3141) — target_file corrected per clause (g) to include it; `git log --oneline --since='7 days ago' -- scripts/runpod_api.py scripts/pod.py scripts/pod_lifecycle.py` → only 84e0752438 (#1893 pod code-sync, unrelated) (2026-08-02).

## Proposed change (refine in planning)
- `scripts/pod_lifecycle.py` (provision path): when a provision's post-create SSH/bootstrap readiness probe fails and the pod is terminated for retry, record the failed placement's `ssh_host` IP + datacenter id (in-session state or a small sidecar); on the next attempt for the same issue, pass a DIFFERENT `data_center_id` (RunPod's API has no host-exclude input, so DC-pin-away is the available lever — enumerate candidate DCs via a `getAllDatacenters` helper added to `scripts/runpod_api.py`, per the documented supply-workaround pattern), or at minimum drop any DC pin that resolved to the bad host and WARN loudly when the new placement resolves to a previously-failed host IP.
- `scripts/runpod_api.py`: add the `getAllDatacenters` query helper; surface the created pod's resolved host IP early so the caller can compare against the bad-host record.

## Scope / surfaces
- Primary target: `scripts/runpod_api.py, scripts/pod.py, scripts/pod_lifecycle.py`
- Construction site is pod_lifecycle.py (pod.py is a thin CLI dispatcher — symptom site only; kept in target for the CLI flag surface if a `--avoid-host`/`--data-center` retry knob is added). Do not change the SUPPLY_CONSTRAINT fail-loud contract.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: c5c655718a1d
- workflow_fix_target: scripts/runpod_api.py, scripts/pod.py, scripts/pod_lifecycle.py
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C13.
