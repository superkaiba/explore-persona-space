---
title: 'workflow-fix: GCP queue-vanish failover must launch workload + re-point handle
  or tear down pod'
kind: infra
tags:
- wf-fix
- wf-fix-fp:37c10069d918
created_at: '2026-07-18T01:29:49Z'
has_clean_result: false
origin_prompt: 'fix this: this looks like a real gap in the GCP->RunPod failover (queue-vanish
  left an idle pod + a stale handle instead of either launching the workload or cleaning
  up)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap observed during live ops on task #1417 (orchestrator: My Goat chat session, user-directed 2026-07-18).

## Goal

On a GCP FLEX_START queue-vanish/timeout failover to RunPod, atomically (a) launch the workload on the RunPod fallback AND (b) re-point the run handle to it; if the workload cannot be launched, tear down the provisioned pod — so the failover NEVER leaves an idle billing pod plus a stale `backend=gcp` handle behind.

## Workflow gap

- **Bug observed:** On #1417, the GCP FLEX_START instance `eps-issue-1417` vanished from the DWS queue (gone from every zone). A RunPod 4×H100 fallback pod (`pod-1417`) was provisioned, but the workload NEVER launched on it — all 4 GPUs at 0% util / 0 MiB, no `issue1417` process, no `/workspace/logs/issue-1417.log`. Meanwhile the run handle (`.claude/cache/issue-1417-handle.json`) stayed `backend=gcp, last_phase=pending` (stale). Net: an idle 4×H100 billed indefinitely while the autonomous session polled a dead GCP instance; a human had to notice and terminate it manually.
- **Why it is a workflow gap:** the async queue-vanish→RunPod failover path (#1116, `_maybe_escalate_gcp_queue_vanish` → `_failover_dead_gcp_to_runpod` in `scripts/backend_poll.py`) reached a partial/inconsistent state — a RunPod pod exists but neither the workload launch nor the handle re-point completed. Either the failover created the pod and did not finish launching+re-pointing, or `pod-1417` was an earlier-attempt orphan the vanish-failover never reconciled. Both are the same class of defect: the failover can leave a billing pod with no workload and a handle pointing at a vanished backend, with no self-healing detection.
- **Confidence (emitter):** medium (symptom directly observed on live pods/handle; exact code path to be root-caused by the planner with the file open).
- verified-at-filing: `grep -rnE '_maybe_escalate_gcp_queue_vanish|_is_gcp_queue_vanish|_failover_dead_gcp_to_runpod|_is_gcp_async_workload_failure' scripts/backend_poll.py` → 4+ hits (`_is_gcp_async_workload_failure` L588, `_maybe_escalate_gcp_queue_vanish` L908, `_is_gcp_queue_vanish` L977, `_failover_dead_gcp_to_runpod` referenced L1960/1971); `grep -nE 'write_handle_sidecar|"backend"' src/explore_persona_space/backends/issue_dispatch.py` → handle writer present (`write_handle_sidecar` L593, backend field L559/661) (2026-07-18).

## Proposed change (candidate diff sketch — refine in planning)

The failover must be transactional. In `_failover_dead_gcp_to_runpod` / the queue-vanish escalation path:
- after provisioning the RunPod fallback, VERIFY the workload actually launched (log breadcrumb / run-launched signal / a non-idle GPU check) before considering the failover complete;
- re-point the run handle sidecar to the RunPod backend (`write_handle_sidecar` with `backend=runpod`, new `pod_name`/`job_id`, cleared stale `last_phase=pending`) as part of the same transition;
- if the workload launch cannot be confirmed within a bounded window, TEAR DOWN the just-provisioned pod (no idle-pod residue) and surface the failure, rather than leaving it billing;
- add a reconciliation guard: a RunPod pod for issue N that is idle (0 GPU util, no workload log) past a short grace with a handle still pointing at a vanished/terminal GCP instance is an auto-detectable orphan — either the poller re-launches or auto-terminates + alerts (mirror the existing no-port wedge auto-terminate).

## Scope / surfaces

- Primary: `scripts/backend_poll.py` (the async queue-vanish/timeout failover + handle re-point).
- Secondary: `src/explore_persona_space/backends/issue_dispatch.py` (`write_handle_sidecar` / `on_launched` — the handle re-point mechanism the failover must call).
- Grep the surface for the queue-vanish/failover functions before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Preserve the existing #1116 queue-vanish + #783 queue-timeout + #658/#659 workload-failover semantics; this ADDS the launch-verify + handle-re-point + idle-orphan-cleanup guarantees, does not change when failover fires.
- Add/extend `tests/test_backend_*.py` coverage for: queue-vanish failover re-points the handle to runpod; a failover whose workload launch is unconfirmed tears down the pod; an idle-pod-with-stale-gcp-handle is detected as an orphan.
- `scripts/workflow_lint.py` passes; ruff on touched files passes.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` with a `workflow_fix_target:` Provenance line — recursion guard: MUST NOT auto-file more workflow-fix tasks for its own findings.

## Provenance

- workflow_fix_target: scripts/backend_poll.py, src/explore_persona_space/backends/issue_dispatch.py
- fingerprint: 37c10069d918

Origin: live-ops observation on #1417 (2026-07-18), user-directed ("fix this: this looks like a real gap in the GCP→RunPod failover — queue-vanish left an idle pod + a stale handle instead of either launching the workload or cleaning up"). Idle `pod-1417` (4×H100) was manually terminated; this task fixes the failover so it self-heals.
