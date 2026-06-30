---
title: 'workflow-fix: watcher auto-terminate+failover for RunPod no-port wedge when
  poll loop is dead'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3e794ca3af50
created_at: '2026-06-30T19:58:11Z'
has_clean_result: false
origin_prompt: Why did this wedge happen and how can we fix it - get a subagent on
  this
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised during the #763 RunPod no-port wedge investigation (user-directed 2026-06-30; subagent diagnosis).

## Goal

Promote the watcher wedge arm's confirmed keep_running-False AND inputs_ok case from a reversible _stop_pod to a bounded, idempotent call into the existing backend_poll._failover_wedged_runpod (terminate + fresh re-provision), reusing the durable-lease + sentinel idempotency and per-cell inputs-on-HF gate; every uncertainty path stays ALERT-only.

## Workflow gap

- **Bug observed:** A RunPod pod (#763's z2g3cxyhudmyrf) stuck RUNNING-but-no-port leaked billing and the task sat blocked/alert-only — the poller's terminate+re-provision failover (_failover_wedged_runpod) couldn't run because the per-issue poll loop was dead, and the #692 watcher backstop's strongest action is a gated reversible pod.py stop, never terminate+failover.
- **Why it is a workflow gap:** The irreversible terminate+re-provision/failover for a no-port wedge lives ONLY in `backend_poll._failover_wedged_runpod`, reachable solely while the per-issue poll loop is alive — exactly when a backstop is NOT needed. When the loop is dead (task already blocked / `/issue` poll chain exited) the #692 watcher backstop can only `stop`/alert, and a reversible `stop` does not heal a host-pinned dead RunPod host (resume returns to the same dead host), so billing leaks until a human terminates (the #763 incident: ~hours of H100 billing on a wedged pod until manual terminate).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/autonomous_session_watch.py` `decide_pod_wedge` (~:1238) / `_process_wedged_pod` (~:3698): add a fourth action `terminate-failover`; route the confirmed `keep_running is False AND inputs_ok AND matured-wedge` case to it (today it returns the reversible `("stop", 0)`). The handler calls the EXISTING `backend_poll._failover_wedged_runpod(...)` (terminate + FRESH re-provision), reusing its durable-lease + sentinel idempotency (`_runpod_wedge_already_handled` / `Lease.runpod_wedge_failover_of`) and per-cell inputs-on-HF gate. Bounded at-most-once via the existing sentinel; on no-capacity relaunch it maps to the terminal `no_compute_available` contract (watcher capacity-retry re-drives). Every uncertainty path (`keep_running` True/unknown, `inputs_ok` False) stays ALERT-only.

## Scope / surfaces

- Primary: `scripts/autonomous_session_watch.py` (`decide_pod_wedge`, `_process_wedged_pod`).
- Docs: `.claude/rules/compute-backend-failover.md` (Part C watcher backstop) + CLAUDE.md "Remaining gap (#667)" note — document that an inputs-safe matured RunPod no-port wedge now auto-terminates+fails-over from the watcher (not just stop/alert); the GCP-#667 frozen-non-terminal-phase gap remains open.
- Reuses `backend_poll._failover_wedged_runpod` (no new recovery logic — extends WHERE the existing one can fire).

## Constraints / invariants

- Workflow-surface only. The terminate-failover action fires ONLY on confirmed `keep_running=False AND inputs_ok=True AND matured wedge`; bounded at-most-once (no infinite cascade); every uncertainty path stays ALERT-only (halt-criterion #2 — never strand un-uploaded work).
- **ARCHITECTURAL ASSESSMENT REQUIRED:** this promotes the watcher's auto-action from reversible-only to irreversible terminate+failover. The planner MUST decide if this is a public-contract/architectural change → if so set `architectural: true` + park at `plan_pending` for user greenlight. (Emitter leans IN-SCOPE auto-fix: it reuses the poller's already-irreversible, already-gated, already-idempotent recovery, extending its coverage to when the poll loop is dead — no new contract.)
- ruff + `workflow_lint.py` pass; add tests for the new `decide_pod_wedge` action + the bounded-once guard.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, .claude/rules/compute-backend-failover.md, CLAUDE.md
- fingerprint: 3e794ca3af50

Raised by the #763 wedge-investigation subagent (2026-06-30). Related: #667 (GCP sibling gap, still open), #664 + #692 (the existing RunPod wedge poller-failover + watcher-backstop this extends), #763 (incident).
