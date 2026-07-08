---
title: 'workflow-fix: GCP flex-start PENDING-vanish -> capacity-miss failover'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9675b56c2383
created_at: '2026-07-07T23:03:01Z'
has_clean_result: false
origin_prompt: '#1112 orchestrator observation 2026-07-07: flex-start instance vanished
  while PENDING twice (create DONE, no delete op) — invisible to the ladder and the
  #783 queue-timeout escalation; add a gcp_queue_vanish trigger'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the #1112 orchestrator's
own observation (2026-07-07).

## Goal

Detect a GCP FLEX_START instance that VANISHES while PENDING (create DONE →
later instances-list not-found, no delete operation) as a CAPACITY MISS:
advance the ladder rung or fail over to the RunPod terminal rung, mirroring
the #783 queue-timeout trigger.

## Workflow gap

- **Bug observed:** #1112's full-run `ft-7b` flex-start instance vanished
  server-side TWICE (inserts DONE 22:30Z + 22:50Z 2026-07-07; no delete op in
  the operations log; instance gone before the 600s queue-timeout escalation
  could cancel it). The DWS queue dropped the request when capacity could not
  be obtained. Because `gcloud instances create` reports success (PENDING),
  the router's ladder sees no capacity miss and every relaunch re-books the
  same dead flex-start rung indefinitely; the orchestrator had to hand-pivot
  to `--backend runpod`.
- **Why it is a workflow gap:** the compute-backend-failover policy already
  names three GCP→RunPod triggers (workload crash sync/async, queue-timeout
  #783, boot-loop #1029); a PENDING-vanish is a fourth, currently invisible
  shape — `backend_poll` reports it as a bare `dead(instance not found)` with
  no failover, and `route()` cannot see it at all.
- **Confidence (emitter):** high (two live occurrences, operations-log
  evidence).

## Proposed change (candidate diff sketch — refine in planning)

```
+ backend_poll.py (async poller): when poll resolves `instance not found`
+   AND the handle's last known phase never reached `workload` (guest attr
+   eps/phase absent / instance never RUNNING) AND the create op was DONE:
+   classify as `gcp_queue_vanish` -> reuse the #783 escalation leg
+   (_failover / cancel bookkeeping) with reason
+   `gcp_queue_vanish_failover_runpod`; do NOT touch MAX_GCP_ATTEMPTS_PER_DAY.
+ backends/gcp.py: persist `reached_running: bool` (or first-RUNNING ts) in
+   the handle so the poller can distinguish vanished-while-PENDING from
+   deleted-after-run.
+ compute-backend-failover.md: document the fourth trigger.
```

## Scope / surfaces

- Primary targets: `src/explore_persona_space/backends/gcp.py`,
  `scripts/backend_poll.py`, `.claude/rules/compute-backend-failover.md`
- Grep first: `grep -rln 'queue_timeout' src/explore_persona_space/backends/ scripts/backend_poll.py .claude/rules/compute-backend-failover.md`

## Constraints / invariants

- Workflow-surface only (backends/* is in-scope per the workflow-fix rule).
- Keep the no-auto-RunPod-in-auto-chain invariant intact — the failover leg
  must reuse the EXISTING #783-style typed escalation, not add a new auto
  RunPod path (`test_no_auto_runpod_path_under_any_failure` must stay green).
- A vanish AFTER the workload started keeps its current crash/teardown
  classification — the new trigger is strictly vanished-while-PENDING.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 9675b56c2383

Origin: #1112 orchestrator observation — two consecutive flex-start
PENDING-vanishes forced a manual `--backend runpod` pivot (markers on #1112
at 2026-07-07T22:44Z-23:0xZ).
