---
title: 'workflow-fix: GCP H100 queue-wait timeout -> RunPod failover (auto lane)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:18dcdaad7a20
created_at: '2026-07-01T05:14:37Z'
has_clean_result: false
origin_prompt: 'for H100 if GCP is not available we should use runpod (standing rule;
  #778 stalled ~3h in GCP FLEX_START queue)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow gap observed on task #778 (orchestrator self-observed, user-directed standing rule: "for H100, if GCP is not available we should use RunPod").

## Goal

Add a bounded GCP queue-wait timeout to the auto-lane GCP rung so an H100 request that stays PENDING (queued for capacity) fails over to the next lane / the RunPod terminal rung, instead of waiting indefinitely.

## Workflow gap

- **Bug observed:** GCP `FLEX_START` (and on-demand) H100 requests that stay `PENDING` in the capacity queue have no queue-wait timeout in the auto lane, so the router waits indefinitely instead of failing over. #778's `eps-issue-778` sat PENDING in the us-central1-b FLEX_START queue ~2h45m+ with no advancement; the run had to be pivoted to RunPod manually.
- **Why it is a workflow gap:** the FREE SLURM lanes already implement this — `FREE_WAIT` (default 600s) → PENDING-at-cap → cancel + advance to the next lane (`router.py:28`). But the GCP rung explicitly has none: `router.py:29` — *"GCP has no queue estimate and no park."* So a queued GCP H100 never reaches the RunPod terminal rung the ladder is supposed to fall through to. The existing GCP→RunPod failover fires only on a GCP **workload crash** (`gcp_workload_failover_runpod`, #658) or **no-capacity exhaustion** (`RunPodNoCapacityError` path), NOT on a long capacity **queue** where the create succeeded and the instance is PENDING.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

- Add a GCP queue-wait timeout knob (e.g. `EPS_GCP_QUEUE_WAIT_SECONDS`, default ~600–900s) mirroring the free-lane `FREE_WAIT` semantics.
- In the GCP rung (FLEX_START + on-demand): if the instance does not reach RUNNING within the timeout, CANCEL/DELETE the PENDING instance and advance the ladder — to the next GCP rung if any, else the RunPod terminal rung (`reason: gcp_queue_timeout_failover_runpod`).
- Do NOT double-count against `MAX_GCP_ATTEMPTS_PER_DAY` beyond the attempts actually made; the cancel is a clean advance, not a workload failure.
- Record the timeout + reason on the per-rung `epm:backend-selected` marker so the trail distinguishes a queue-timeout failover from a workload-crash failover and a no-capacity exhaustion.

## Scope / surfaces

- Primary: `src/explore_persona_space/backends/router.py` (GCP rung timeout + failover), `.claude/rules/compute-backend-failover.md` (document the new GCP-queue-timeout → RunPod policy).
- Likely also: `src/explore_persona_space/backends/gcp.py` (poll/cancel a PENDING instance), `tests/test_router.py` (pin: GCP PENDING past timeout → cancel → RunPod terminal rung, no infinite wait).
- Grep the surface for the pattern before editing (`grep -rniE "FREE_WAIT|flex_start|PENDING|no queue estimate" src/explore_persona_space/backends/ .claude/rules/compute-backend-failover.md`) and update every hit.

## Constraints / invariants

- Workflow-surface only. Preserve the existing failover invariants + their tests (`tests/test_router.py::test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted`, `::test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade`).
- The queue-timeout failover must remain compatible with the explicit `backend:` overrides and the auto lane order; it is an ADDITIONAL advance trigger, not a change to lane precedence.
- `scripts/workflow_lint.py` passes; ruff on touched files passes; keep `compute-backend-failover.md` + `router.py` docstring + CLAUDE.md consistent.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` with a `workflow_fix_target:` Provenance line — must NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/router.py
- fingerprint: 18dcdaad7a20

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/router.py, .claude/rules/compute-backend-failover.md, src/explore_persona_space/backends/gcp.py, tests/test_router.py
bug_observed: GCP FLEX_START/on-demand H100 requests that stay PENDING in the capacity queue have no queue-wait timeout, so the auto lane waits indefinitely instead of failing over (#778 sat PENDING ~3h).
why_workflow_gap: the free SLURM lanes have FREE_WAIT (600s) → PENDING-at-cap → cancel + next lane, but router.py:29 states GCP has no queue estimate and no park, so a queued GCP H100 never advances to the RunPod terminal rung.
proposed_change: add a bounded GCP queue-wait timeout (env knob) to the FLEX_START/on-demand rung; on timeout cancel the PENDING instance and advance the ladder to the next lane / RunPod terminal rung, mirroring the free-lane FREE_WAIT semantics.
diff_sketch: |
  + EPS_GCP_QUEUE_WAIT_SECONDS (default ~600-900s)
  + in the GCP rung: if not RUNNING within timeout -> cancel PENDING instance -> advance ladder
  +   (next GCP rung else RunPod terminal rung, reason: gcp_queue_timeout_failover_runpod)
  + record timeout+reason on epm:backend-selected marker
confidence: high
related_task: #778
<!-- /workflow-fix-candidate -->
