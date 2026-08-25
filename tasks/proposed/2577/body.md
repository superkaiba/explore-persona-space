---
title: 'Deflake test_issue1947_worker_dispatch quarantine race (load-dependent; false-blocked
  #2349 gate)'
kind: infra
tags:
- flaky-test
created_at: '2026-08-25T13:06:38Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 2349 Step 10d: gate round 1 false-blocked on
  this load-flaky node via the #2413 attribution channel; the flake itself is unowned.'
workflow: v1
---
# Deflake tests/test_issue1947_worker_dispatch.py::test_quarantine_refund_and_healthy_completion (load-dependent quarantine race)

## Goal

Make `test_quarantine_refund_and_healthy_completion` deterministic. The test patches dispatch with `sick_slots=("1",)` and asserts the quarantine message `slot 1 quarantined after 2 consecutive fast failures` appears. Under load, the healthy slot 0 can drain the whole 3-cell queue before slot 1's worker claims any cell, so no quarantine ever fires and the assertion fails; on a quiet box the same content passes.

## Evidence

- 2026-08-25, #2349 Step 10d pre-push lint gate (TG mapped leg, 117 test files under 8-thread caps, fleet loadavg high): FAILED with output showing all 3 cells launched and completed on slot 0 (`[dispatch] launched cellA on slot 0 ... unit 3/3 cellB done (slot 0) ... dispatch_complete done=3 failed=0`) — the sick slot never claimed work.
- Same content (worker `scripts/issue1947_worker.py` == origin/main tip `a855146b24`, test file unchanged since merge-base) run isolated in the same worktree: passed 2/2 (2.26s, 3.63s).
- The failure false-blocked #2349's merge for one full gate round (~4.3 h wall) via the base-identical attribution channel (that channel is #2413's scope; this task owns the flake itself).

## Fix direction

Force the sick slot to claim at least one cell before the queue can drain: hold the healthy worker's first claim until slot 1 has claimed (test-harness barrier), or enlarge the cell queue so slot 1 is guaranteed a claim window, or assert quarantine only conditionally on slot 1 having claimed (weakest; prefer the barrier). Keep the production dispatcher untouched unless the plan finds the race reachable in production reporting too.

## Acceptance

- The node passes 20/20 under a parallel loaded run (e.g. alongside the full mapped set with 8-thread caps) and still exercises the quarantine + refund path (the quarantine message and refund accounting remain asserted).
- No production behavior change in `scripts/issue1947_worker.py` without its own stated reason.
