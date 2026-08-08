---
title: 'workflow-fix: GCP wedge escalation needs sustained (2-tick) reachability alarm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6d13e49db290
created_at: '2026-07-29T20:23:03Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed incident on #1739 (2026-07-29 20:08-20:17Z):
  single-tick transport alarm + always-stale phase clock fired a false-positive paid
  GCP->RunPod failover against a healthy fits workload'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1739 (emitting agent: orchestrator /issue session).

## Goal

Require the GCP hung-VM wedge escalation to observe a SUSTAINED transport-class reachability alarm (>=2 consecutive alarmed poll ticks persisted in the sidecar phase clock) instead of a single-tick alarm, so a transient gcloud control-plane error during a long single-phase workload cannot fire a paid GCP->RunPod failover.

## Workflow gap

- **Bug observed:** A single transient gcloud compute ssh 'Internal error' at 2026-07-29T20:08-20:13Z on healthy eps-issue-1739-hallu (phase=workload for 15.5h, fits mid-run) satisfied stale_for>900s AND reachability_alarm on one tick; the poller synthesized terminal_workload_wedged and fired the async failover, provisioning a 2xH100 RunPod pod against a healthy run.
- **Why it is a workflow gap:** On any long single-phase workload the GCP_STALENESS_FLOOR_SEC=900 conjunct is trivially satisfied for the whole phase (eps/phase never advances during a multi-hour fits phase), so the wedge predicate degenerates to a SINGLE-tick transport alarm — one control-plane blip == a money-spending failover. The #669 design target was SUSTAINED guest-network loss (the in-VM watchdog needs ~5 min sustained); the poller-side arm has no equivalent persistence requirement.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "reachability_alarm" scripts/backend_poll.py src/explore_persona_space/backends/gcp.py src/explore_persona_space/backends/base.py` → 10 hits (2026-07-29): escalation conjunction at scripts/backend_poll.py:945 (`stale_for > GCP_STALENESS_FLOOR_SEC and getattr(result, "reachability_alarm", False)` — single-tick, no persistence); alarm CONSTRUCTION site at src/explore_persona_space/backends/gcp.py:6025 (`reachability_alarm=(drain_alarm_class == "transport")`), classifier context at gcp.py:6476/6510; GCP_STALENESS_FLOOR_SEC=900 at scripts/backend_poll.py:214. Incident evidence: task #1739 epm:progress v60 (2026-07-29T20:21:48Z) — workload proven alive by direct SSH (pids 1965/3609/3612, log mtime 20:16:21Z) minutes after the wedge classification.

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/backend_poll.py::_maybe_escalate_gcp_wedge
- if stale_for > GCP_STALENESS_FLOOR_SEC and result.reachability_alarm:
-     return replace(result, status="dead", current_phase=GCP_WORKLOAD_WEDGED_PHASE, ...)
+ # persist alarm observations in the sidecar phase clock:
+ if stale_for > GCP_STALENESS_FLOOR_SEC and result.reachability_alarm:
+     n = _bump_alarm_streak(sidecar, now)   # consecutive alarmed ticks + min elapsed span
+     if n >= 2:   # and (now - first_alarm_ts) >= ~10 min
+         return replace(result, status="dead", current_phase=GCP_WORKLOAD_WEDGED_PHASE, ...)
+     return result   # first alarmed tick: stay running, re-check next tick
+ _reset_alarm_streak(sidecar)
```

unverified hypothesis — verify at plan time: whether the transport-class drain alarm can additionally distinguish a gcloud CONTROL-PLANE error (`ERROR: (gcloud.compute.ssh) Could not fetch resource: Internal error`) from genuine guest unreachability (connection timeout to the guest) — if separable at the classifier (gcp.py:6476 window), control-plane errors should never count toward the streak.

## Scope / surfaces

- Primary target: `scripts/backend_poll.py` (escalation predicate + sidecar phase-clock persistence), `src/explore_persona_space/backends/gcp.py` (drain alarm classification, if the control-plane split is adopted)
- Grep the workflow surface for the pattern before editing (`grep -rn 'reachability_alarm\|GCP_STALENESS_FLOOR_SEC' scripts/ src/explore_persona_space/backends/ tests/`) and update every hit; pin tests in tests/test_backend_poll*.py cover the wedge predicate — extend them for the streak requirement.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The #669 genuine hung-VM detection must survive: a REAL sustained guest-network loss must still escalate within ~2-3 poll ticks (the watcher/janitor remain backstops); `terminal_terminated` stays excluded (no spot regression); `test_async_gcp_capacity_death_does_NOT_fail_over` stays green.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/backend_poll.py, src/explore_persona_space/backends/gcp.py
- fingerprint: 6d13e49db290

<!-- workflow-fix-candidate v1 -->
target_file: scripts/backend_poll.py, src/explore_persona_space/backends/gcp.py
bug_observed: A single transient gcloud compute ssh 'Internal error' at 2026-07-29T20:08-20:13Z on healthy eps-issue-1739-hallu (phase=workload for 15.5h, fits mid-run) satisfied stale_for>900s AND reachability_alarm on one tick; the poller synthesized terminal_workload_wedged and fired the async failover, provisioning a 2xH100 RunPod pod against a healthy run.
why_workflow_gap: On a long single-phase workload the 900s staleness floor is always exceeded, so the wedge predicate degenerates to a single-tick transport alarm — one control-plane blip triggers a paid failover.
proposed_change: Require a sustained (>=2 consecutive alarmed ticks) transport-class reachability alarm, persisted in the sidecar phase clock, before synthesizing terminal_workload_wedged.
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->
