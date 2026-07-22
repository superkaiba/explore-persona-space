---
title: 'workflow-fix: RunPod terminal rung re-drives exit-75 still-waiting instead
  of parking'
kind: infra
tags:
- wf-fix
- wf-fix-fp:758756247e59
created_at: '2026-07-22T19:21:57Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1586 launch: router _runpod_terminal_rung
  treats pod_lifecycle EXIT_STILL_WAITING (75) as terminal failure -> no_compute_available,
  despite the documented re-run contract (dispatch_issue._provision_still_waiting;
  pod_lifecycle wait-not-park design)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1586 (emitting agent: PM/chat orchestrator).

## Goal

In `_runpod_terminal_rung`, special-case `PodLifecycleProcessError` returncode == EXIT_STILL_WAITING (75): re-drive the state-free provision wait instead of raising NoComputeAvailableError.

## Workflow gap

- **Bug observed:** the router RunPod terminal rung converts an exit-75 still-waiting provision ("nothing provisioned, re-run to keep waiting") into a no_compute_available terminal park. On #1586 (2026-07-22 10:52Z and 17:05Z) two terminal-rung attempts each ran one 45-min `--wait-for-capacity` budget, exited 75, and the rung recorded `runpod terminal fallback FAILED (PodLifecycleProcessError: ... exit status 75)` → `no_compute_available` → `status:blocked`, leaving recovery to the watcher capacity-retry (~1h latency per cycle) instead of continuing the wait.
- **Why it is a workflow gap:** the exit-75 still-waiting contract is explicit — `pod_lifecycle.py` (EXIT_STILL_WAITING, "the caller RE-RUNS the same command to continue waiting"; wait-not-park design intent) and `dispatch_issue.py` (`_provision_still_waiting`, lines 1239/1242/1687: converts to `still_waiting: true` + `rerun: true`, exit 75) — but `router.py` `_runpod_terminal_rung`'s catch-all (~line 3298–3325) has no returncode check, so the contract is honored on the dispatch top-level producer and silently dropped on the router terminal-rung producer.
- **Confidence (emitter):** medium — the fix must compose with the RunPod once-more burn-cap policy and the auto-lane's bounded-attempt design; the planner adjudicates whether the re-drive loop belongs in-router (bounded re-run with heartbeat RouteAttempts) or as a typed re-drivable outcome the orchestrator re-drives.
- verified-at-filing: `grep -n "EXIT_STILL_WAITING\|returncode\|still_waiting" src/explore_persona_space/backends/router.py` → 1 hit (line 2359, an unrelated GCP stderr-record field doc); 0 hits for any exit-75/still-waiting handling in the target (absence claim — in-target 0-hit is the evidence); semantic sibling grep confirms the contract lives at `scripts/dispatch_issue.py:1239,1242,1687` and `scripts/pod_lifecycle.py:969`. Landed-fix history: `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/router.py` → 5 commits (02cbafe757, 264f4914af, c39778ec3c, a2617e2deb, 2354d2bf2f), none touching still-waiting; `.../backends/runpod.py` → a90d36e222 (#1465 stderr relay), whose own comment (runpod.py:915) states "The exit-75 still-waiting contract ... is unchanged — returncode + cmd ride verbatim", i.e. the relay deliberately preserved rc=75 for a consumer the router never implemented. (2026-07-22)

## Proposed change (candidate diff sketch — refine in planning)

```
 # in _runpod_terminal_rung's provision except-handler:
+        if isinstance(exc, PodLifecycleProcessError) and exc.returncode == EXIT_STILL_WAITING:
+            # still-waiting is NOT a failure: nothing provisioned, nothing billing.
+            # Re-drive the state-free wait (bounded re-runs w/ heartbeat RouteAttempts),
+            # or surface a typed re-drivable outcome (`runpod_still_waiting`) instead of
+            # NoComputeAvailableError -> blocked.
         attempts.append(RouteAttempt(... outcome="runpod_fallback_failed" ...))
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/router.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'EXIT_STILL_WAITING\|_provision_still_waiting' src/explore_persona_space/backends/ scripts/`) and update every consumer; list them in the plan. Check `_async_runpod_failover` (backend_poll-driven path) inherits the same handling.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; router tests (`tests/test_router.py`) stay green — the RunPod-is-last-rung-only invariant and the burn-cap bound are untouched.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/router.py
- fingerprint: 758756247e59

Surfaced from the #1586 launch day (2026-07-22): two RunPod terminal-rung attempts each burned a 45-min wait-for-capacity budget, exited 75 (EX_TEMPFAIL still-waiting), and were parked as `no_compute_available` instead of re-driven; user directive that day ("just run on runpod -- ignore the resource competition") had to be executed manually via a steering marker.
