---
title: 'workflow-fix: arm on-demand retry on GCP queue-timeout path'
kind: infra
tags:
- wf-fix
- wf-fix-fp:97e84164a5d0
created_at: '2026-07-22T14:41:46Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on #779 2026-07-22: 3x flex queue-timeouts each
  failed over to a burn-cap-refused RunPod rung; GCP on-demand never probed — extend
  #1596''s queue-vanish on-demand retry to the queue-timeout trigger'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed candidate raised on task #779 (emitting agent: /issue orchestrator, autonomous re-drive session 2026-07-22).

## Goal

Arm the #1596 GCP on-demand retry on the queue-TIMEOUT failover path so a capacity-class RunPod refusal retries the STANDARD rungs before minting no_compute_available.

## Workflow gap

- **Bug observed:** three consecutive GCP FLEX_START queue-timeouts on issue #779 (2026-07-22, ~09:05/11:00/13:18 creates) each escalated per #783 straight to the RunPod terminal rung, which was refused on a capacity-class error (the burn-cap RunPodInsufficientBalanceError via wait-for-capacity exit 75) — and the GCP STANDARD (on-demand) A100-80 rung was never probed; the round minted no_compute_available three times and lost ~6h.
- **Why it is a workflow gap:** #1596 (merged 264f4914af, 2026-07-22) automated exactly this recovery — "retry the GCP ladder's STANDARD rungs after a clean-residue capacity-class RunPod refusal" — but deliberately scoped it to the #1116 queue-VANISH path ONLY (`gcp_ondemand_retry_on_capacity_refusal=True` is documented as having exactly one caller, the queue-vanish wrapper, scripts/backend_poll.py:3440-3465). The #783 queue-TIMEOUT trigger has the identical failure shape (flex capacity contended, RunPod refusal capacity-class, clean residue) and the identical manual recovery (#1112 `--provisioning-model STANDARD`), but never arms the retry.
- **Confidence (emitter):** high — the mechanism exists (`_retry_gcp_ondemand_after_vanish_refusal` @3469, `router.retry_gcp_ondemand_after_queue_vanish`); the fix is arming it on the sibling caller + a distinct reason code, with #1596's own guards (daily attempt cap, single-flight lease, workload-crash non-retry) reused verbatim.
- verified-at-filing: `grep -n gcp_ondemand_retry_on_capacity_refusal scripts/backend_poll.py` → flag documented "the ONLY" caller = the queue-vanish wrapper (@3440, set True @3465); the queue-timeout escalation `_maybe_escalate_gcp_queue_timeout` @810 sets GCP_QUEUE_TIMEOUT_PHASE and routes to the RunPod terminal rung with no on-demand retry (2026-07-22). Landed-sibling check per clause (c): 264f4914af (#1596) is scoped "queue-vanish path ONLY" by its own commit message + code comments — the queue-timeout arm is NOT landed. Live evidence: task #779's 2026-07-22T14:35Z epm:backend-selected marker (queue-timeout → runpod_fallback_failed → no_compute_available, no STANDARD attempt).

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/backend_poll.py — the queue-timeout failover call site:
- _clean_residue_terminal_or_ondemand_retry(..., gcp_ondemand_retry_on_capacity_refusal=False)
+ _clean_residue_terminal_or_ondemand_retry(..., gcp_ondemand_retry_on_capacity_refusal=True)
# src/explore_persona_space/backends/router.py:
+ ROUTE_REASON_QUEUE_TIMEOUT_GCP_ONDEMAND_RETRY  (sibling of the #1596
+   queue-vanish reason code; generalize retry_gcp_ondemand_after_queue_vanish
+   to take the trigger, or add a thin queue-timeout twin)
# Reuse #1596's guards unchanged: MAX_GCP_ATTEMPTS_PER_DAY, single-flight
# lease, gcp_workload_failed_on_ondemand_retry non-retry.
```

## Scope / surfaces

- Primary target: `scripts/backend_poll.py, src/explore_persona_space/backends/router.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'gcp_ondemand_retry_on_capacity_refusal\|retry_gcp_ondemand' scripts/ src/ .claude/ tests/`) and update every hit + the #1596 tests; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The auto chain still NEVER reaches RunPod except as the terminal rung (`test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` stays green); the retry only inserts STANDARD GCP rungs BEFORE the existing terminal, mirroring #1596.
- `scripts/workflow_lint.py` no-flags passes; router/backend tests (`tests/test_router*.py`, `tests/test_backend_*.py`) extended for the new trigger, not weakened.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/backend_poll.py, src/explore_persona_space/backends/router.py
- fingerprint: 97e84164a5d0

Surfaced prose (verbatim, from the #779 re-drive session): "the queue-timeout failover goes straight to the (dead) RunPod rung without probing GCP on-demand A100-80."
