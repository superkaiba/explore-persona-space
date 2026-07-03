---
title: 'workflow-fix: GCP gate-park teardown leg + terminal-phase reconnect refusal'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3cbb26c26d52
created_at: '2026-07-03T03:53:30Z'
has_clean_result: false
origin_prompt: 'orchestrator incident on #763 cofit round: gate-parked GCE instance
  idled 40 min post-gate and the router reconnected to the zombie instead of launching
  Phase C'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own incident on task #763 (cofit round, 2026-07-03).

## Goal

Add a GCP-lane instance-teardown leg to the `/issue` Step 6d.4 gate handlers (and the gate-handler authoring guidance): on a BLOCKING gate emitted by a GCP-lane workload, the orchestrator deletes the instance after the sentinel is drained and the gate is resolved — and the router's reconnect path must never treat a gate-parked/terminal-phase instance as a live reconnect target.

## Workflow gap

- **Bug observed (two legs, one incident):** (1) #763's `cofit_phaseA_done` gate-park kept the GCE instance RUNNING after `[phase=done]` — correct for sentinel draining, but no handler step tears it down; the orchestrator resolved the gate (launched the off-pod phase) without deleting the instance, and an A100-80 idled ~40 min. (2) The NEXT `dispatch_issue.py launch` for the same issue returned `reason: reconnect` to that zombie (RUNNING, terminal `eps/phase=workload_done`) WITHOUT launching the new workload — silently no-op'ing the phase-C dispatch until the orchestrator noticed the poll reading `workload_done`.
- **Why it is a workflow gap:** Step 6d.4 documents the RunPod pod-cycling teardown (`pod.py stop`) for `pv_phase1_done` but has no GCP-lane analogue; `gcp.reconnect_or_none` treats any RUNNING `eps-issue-<N>` as reconnectable without checking the workload phase is non-terminal. Both legs will recur for every multi-phase GCP-lane experiment that gates mid-pipeline.
- **Confidence (emitter):** high (reproduced live on #763; markers 03:41-03:52Z).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md, Step 6d.4:
+ GCP-lane blocking gates: after the poller drains the sentinel and the
+ handler resolves the gate (parks OR auto-resolves), if the handle's
+ backend is gcp and the workload printed [phase=done]/a terminal gate
+ park, DELETE the instance (gcloud compute instances delete, or
+ dispatch_issue.py finalize --skip-confirm-artifacts when mid-pipeline)
+ BEFORE dispatching the off-lane phase — the instance stays up ONLY for
+ sentinel draining, never through an off-pod phase (8-bis).

src/explore_persona_space/backends/gcp.py reconnect_or_none:
+ refuse to reconnect to an instance whose eps/phase guest attribute is
+ terminal (workload_done / a blocking-gate park) — return None so the
+ router creates a fresh instance.
```

## Scope / surfaces

- Primary targets: `.claude/skills/issue/SKILL.md` (Step 6d.4), `src/explore_persona_space/backends/gcp.py` (reconnect predicate — in-scope backends module).
- Grep: `grep -rn "reconnect" src/explore_persona_space/backends/gcp.py scripts/backend_poll.py`; check the janitor's zombie predicate (`gcp_audit.py`) for the same RUNNING-but-terminal-phase signature to share it.

## Constraints / invariants

- Workflow-surface only; router tests (`tests/test_router*.py`, `tests/test_gcp_backend.py`) stay green; add a pin for the terminal-phase reconnect refusal.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, src/explore_persona_space/backends/gcp.py
- fingerprint: 3cbb26c26d52
