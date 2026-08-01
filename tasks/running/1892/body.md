---
title: 'workflow-fix: reconnect handle writes spec-default provisioning_model, disarming
  FLEX_START queue-timeout failover'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7a1a93422e26
created_at: '2026-07-30T19:25:46Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on /issue 1776 swap round 2026-07-30: eps-issue-1776
  FLEX_START PENDING 25+ min, no auto-escalation; handle extra.provisioning_model=STANDARD
  vs live FLEX_START; manual cancel + RunPod failover executed (epm:progress v104)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1776 (emitting agent: orchestrator, /issue 1776 same-issue follow-up round `operator_swap_success`).

## Goal

On the GCP reconnect path, populate the handle sidecar's `extra.provisioning_model` from the LIVE instance's `scheduling.provisioningModel` (the gcp.py describe-reader already parses it) instead of the fresh RunSpec's default, so the #783 FLEX_START queue-timeout failover is never disarmed by an exit-75 rerun reconnect.

## Workflow gap

- **Bug observed:** 2026-07-30, task #1776 swap round: the first `dispatch_issue.py launch` created FLEX_START instance `eps-issue-1776` (create timed out at the 300s subprocess cap with the instance PENDING in the DWS queue; exit contract said re-run). The re-run RECONNECTED and wrote the handle sidecar with `extra.provisioning_model: "STANDARD"` while the live instance was FLEX_START (verified: `gcloud compute instances describe eps-issue-1776 --zone us-central1-c --format="value(scheduling.provisioningModel,...)"` → `FLEX_START DELETE 604800 PENDING`). `scripts/backend_poll.py:1175` gates the queue-timeout escalation on `extra.get("provisioning_model") != "FLEX_START" → return`, so the instance sat PENDING 25+ min (2.5× the 600s `EPS_GCP_QUEUE_WAIT_SECONDS` floor) across two poll ticks with no escalation; the orchestrator executed the cancel + RunPod failover manually (epm:progress v104 on #1776).
- **Why it is a workflow gap:** the #783/#778 queue-timeout failover is the designed recovery for created-but-never-dequeued FLEX_START instances; any launch whose create exceeds the 300s gcloud subprocess cap (the common case for a queued flex-start create) takes the exit-75 rerun → reconnect path, and the reconnect-written handle then carries the spec-default provisioning model — structurally disarming the escalation for exactly the population it exists for.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn "provisioning_model" src/explore_persona_space/backends/gcp.py scripts/backend_poll.py` → 16 hits (14 in gcp.py, 2 in backend_poll.py) incl. the consumer gate `scripts/backend_poll.py:1175: if extra.get("provisioning_model") != "FLEX_START": return` and the spec-derived writers `gcp.py:5701` + `gcp.py:5804` (`"provisioning_model": resolve_provisioning_model(spec)`); the live-instance reader that parses `sched["provisioningModel"]` exists at `gcp.py:4290` (2026-07-30). Landed-fix history check: `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/gcp.py scripts/backend_poll.py` shows no commit addressing reconnect-path provisioning-model derivation.
- unverified hypothesis — verify at plan time: whether BOTH writer sites (gcp.py:5701 and :5804) are reachable from the reconnect path, or only one; the fix may only need the reconnect branch. Also verify whether `reconnect_or_none`'s instance describe already fetches `scheduling` (if so the fix is a field copy, not a new API call).

## Proposed change (candidate diff sketch — refine in planning)

```
# src/explore_persona_space/backends/gcp.py — reconnect handle construction
- "provisioning_model": resolve_provisioning_model(spec),
+ # On reconnect, the spec default (STANDARD) misdescribes a live FLEX_START
+ # instance and disarms backend_poll's queue-timeout escalation (#783).
+ "provisioning_model": live_scheduling.get("provisioningModel")
+     or resolve_provisioning_model(spec),
# + a pin test: reconnect to a FLEX_START instance writes FLEX_START into the
# handle extra (backend_poll._maybe_escalate_gcp_queue_timeout gate armed).
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (construction site — reconnect handle extra)
- Symptom consumer (do not patch the gate itself; it is correct): `scripts/backend_poll.py:1175`
- Grep the workflow surface for the pattern before editing (`grep -rn 'provisioning_model' src/explore_persona_space/backends/ scripts/backend_poll.py scripts/dispatch_issue.py`) and update every reconnect-path writer; list them in the plan.

## Constraints / invariants

- Workflow-surface only — the backends/ router package is in-scope per `.claude/rules/workflow-fix-on-bug.md`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 7a1a93422e26

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py
bug_observed: reconnect-written handle recorded provisioning_model=STANDARD for a live FLEX_START instance, disarming the backend_poll queue-timeout failover gate (backend_poll.py:1175); eps-issue-1776 sat PENDING 25+ min past the 600s floor with no escalation and needed a manual cancel + RunPod failover
why_workflow_gap: the reconnect path derives provisioning_model from the fresh RunSpec default (resolve_provisioning_model(spec)) instead of the live instance's scheduling.provisioningModel, so every exit-75 rerun reconnect rewrites the handle with the wrong model for exactly the queued-flex-start population the escalation exists for
proposed_change: on reconnect, populate handle extra.provisioning_model from the live instance describe (scheduling.provisioningModel — the gcp.py:4290 reader already parses it), falling back to the spec default; add a pin test that a FLEX_START reconnect arms the queue-timeout gate
diff_sketch: |
  + "provisioning_model": live_scheduling.get("provisioningModel")
  +     or resolve_provisioning_model(spec),
  (at the reconnect handle-extra construction in backends/gcp.py; pin test in tests/test_backend_poll or tests/test_gcp_backend)
confidence: high
related_task: #1776
<!-- /workflow-fix-candidate -->
