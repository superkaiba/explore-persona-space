---
title: 'daily-fix: queue-vanish guard defeated by stale phase clock'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4da72e568f62
- daily-auto-filed
created_at: '2026-07-29T07:17:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): the #1116 queue-vanish
  failover did NOT fire on a real DWS vanish in #1738: the sidecar phase clock retained
  a PRIOR attempt''s non-pending last_phase, so the last_phase==GCP_PENDING_PHASE
  guard read false and the vanish fell to the streak path — manual recovery (stale
  sidecar retired + fresh dispatch) was required'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-F P6 (predicate miner-probed; guard re-verified at filing).

## Goal

Make the #1116 FLEX_START queue-vanish failover fire when a prior attempt's sidecar phase clock masks a fresh attempt's pending-then-vanish.

## Workflow gap

- **Bug observed:** #1738 hit a REAL DWS queue-vanish (create DONE → instance list not-found, no delete op) and the #1116 failover did not fire: the shared sidecar phase clock still carried the PRIOR attempt's non-pending `last_phase`, so `_maybe_escalate_gcp_queue_vanish`'s `last_phase == GCP_PENDING_PHASE` guard read false and the case fell to the (slower) streak path. Recovery was manual: retire the stale sidecar, fresh dispatch. (The stale-clock cause is read from the session's own recovery wording — verify at plan time; the guard predicate itself is code-verified.)
- **Why it is a workflow gap:** the phase clock is keyed per issue, not per attempt incarnation, so failover discriminators that condition on the clock inherit cross-attempt staleness.
- **Confidence (emitter):** medium-high (guard predicate verified; cause labeled)
- verified-at-filing: `grep -n 'GCP_PENDING_PHASE\|_read_phase_clock' scripts/backend_poll.py` → constant at 233, clock reader at 700, the vanish guard's `current_phase != GCP_PENDING_PHASE` early-out at 854 (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Key the clock file (or its records) on the attempt id the lease already tracks; or add the create-DONE + never-observed-running discriminator as an alternate arm. Regression test with a stale prior-attempt clock.

## Scope / surfaces

- Primary target: `scripts/backend_poll.py` (`_read_phase_clock` / `_maybe_escalate_gcp_queue_vanish`)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 4da72e568f62

- workflow_fix_target: scripts/backend_poll.py

