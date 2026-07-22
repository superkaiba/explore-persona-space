---
title: 'daily-fix: queue-vanish failover retries GCP on-demand'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ec342be4ccaa
- daily-auto-filed
created_at: '2026-07-22T06:46:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): queue-vanish failover routes
  straight to RunPod and mints no_compute_available when RunPod refuses on capacity,
  skipping viable GCP on-demand rungs'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from the #1112 compute-recovery chain (transcript 24ae2158, 2026-07-22 morning UTC).

## Goal

When a GCP FLEX_START queue-vanish (#1116) failover's RunPod rung refuses for CAPACITY/CAP reasons (not a workload crash), retry the GCP ladder's remaining on-demand rungs before surfacing terminal `no_compute_available`.

## Workflow gap

- **Bug observed:** #1112's flex-start dispatch hit the #1116 queue-vanish class; the automatic RunPod failover then hit the account hourly-cost cap → terminal `no_compute_available` — while the GCP ON-DEMAND rungs were still viable and had to be dispatched MANUALLY (`--provisioning-model STANDARD`), costing ~20 min of hand recovery.
- **Why it is a workflow gap:** `_failover_dead_gcp_to_runpod` routes queue-vanish straight to RunPod with no GCP-on-demand fallback when RunPod refuses. The no-cascade design exists to avoid re-crashing broken CODE on more lanes — but a queue-vanish is a CAPACITY event, not a workload crash; the broken-code rationale doesn't apply, and the residual GCP rungs are exactly what a human then dispatches by hand.
- **Confidence:** medium-high (behavior observed live; the design-intent distinction needs the planner + critics to confirm no bound is violated).
- verified-at-filing: original target `src/explore_persona_space/backends/backend_poll.py` → 0 hits (mis-path); repo-wide relocation grep found `scripts/backend_poll.py` (target corrected per workflow-fix-on-bug.md clause (b), 2026-07-22). `grep -n 'queue_vanish\|_failover_dead_gcp_to_runpod' scripts/backend_poll.py` confirms the failover core lives there; no on-demand-retry branch present (absence probe).

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/backend_poll.py` (+ `src/explore_persona_space/backends/` router seams as needed): on the #1116 queue-vanish failover path, when the RunPod terminal rung refuses with a CAPACITY/CAP-class error, walk the remaining GCP on-demand rungs (respecting `MAX_GCP_ATTEMPTS_PER_DAY`) before minting `no_compute_available`. Keep the workload-crash failover's no-cascade bound unchanged.

## Scope / surfaces

- Primary target: `scripts/backend_poll.py`
- Siblings: `src/explore_persona_space/backends/gcp.py` / `router.py` (rung enumeration), `.claude/rules/compute-backend-failover.md` (contract doc), `tests/test_backend_*.py` / `tests/test_router*.py` (pins — note `test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` and the failover pins must stay green or be deliberately amended).

## Constraints / invariants

- The workload-crash → RunPod-once bound is UNCHANGED; only the capacity-class queue-vanish path gains the on-demand retry. Attempt-cap semantics preserved.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: ec342be4ccaa

- workflow_fix_target: scripts/backend_poll.py

Origin evidence: transcript 24ae2158, 2026-07-22T06:25:33Z ("status=dead phase=terminal_no_compute_available ... GCP FLEX_START queue vanish on eps-issue-1112; RunPod also unavailable (#1116 queue-vanish failover)"); 06:27Z manual `--provisioning-model STANDARD` dispatch succeeded in reaching the on-demand rung (which then stocked out — a separate, genuine capacity event).
