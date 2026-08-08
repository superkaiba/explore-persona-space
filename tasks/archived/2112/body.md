---
title: 'daily-fix: setsid pin test env-brittle (subreaper ppid)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4e505106423d
- daily-auto-filed
created_at: '2026-08-06T07:00:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): test_setsid_child_survives_group_kill
  red in gate contexts (ppid reparents to subreaper not 1), green in cron context;
  every gate re-classifies it'
workflow: v1
---
# daily-fix: make test_setsid_child_survives_group_kill robust under fleet load (intermittent gate red)

## Workflow gap

`tests/test_workflow_setsid_detach_convention.py::test_setsid_child_survives_group_kill`
(line 87) is intermittently red on the pristine main tree, which forces every Step 9c /
Step 10d gate that selects it to re-classify the same red as pre-existing each round.

Observed history (2026-08-05):

- ~15:44Z: the #1996 implementer reported it failing standalone at BOTH the main checkout
  and the issue-1996 worktree (0.57s; zero overlap with that task's diff) — park
  `epm:workflow-fix-candidate` 2026-08-05T15:47:20Z on #1996.
- ~19:35–19:50Z: the #1996 Step 10d pre-push lint gate's mapped-test leg ran the node on
  BOTH the payload-free main-tree baseline and the branch-tip gated tree; it FAILED
  identically on both (pre-existing-red duty evidence, #1713) — urgent park
  2026-08-05T19:54:52Z on #1996, fp 4516c07c1164.

verified-at-filing: `uv run pytest tests/test_workflow_setsid_detach_convention.py::test_setsid_child_survives_group_kill -x -q`
run at 2026-08-06T06:40Z on main → **1 passed in 0.44s**, and
`git log --since='24 hours ago' -- tests/test_workflow_setsid_detach_convention.py` → 0
commits. Red at ~15:44Z and ~19:35–19:50Z, green at 06:40Z with the file untouched ⇒ the
node is INTERMITTENT, not persistently red — the correct frame is flake-hardening, not
"fix the red node".

A third independent sighting (miner 6) sharpens the mechanism: the workflow-compaction
session reproduced the failure at 2026-08-06T02:52Z at the repo root with the assertion
detail visible — `assert _ppid(detached_pid) == "1"` fails with `'2887' == '1'`, i.e. the
setsid-detached child reparents to a CHILD-SUBREAPER process, not PID 1. That explains the
context dependence: under a process ancestry that sets PR_SET_CHILD_SUBREAPER (session
managers / the gate environment), orphans reparent to the subreaper and the ppid==1
assertion is environment-brittle, while a plain cron/shell context (this compose-time run)
passes.

unverified hypothesis — verify at plan time: the subreaper reparenting above is
miner-inferred from the assertion diff, not probed against the live process tree; the
#1996 session's original load/timing hypothesis is subsumed but not excluded for the
gate-context failures.

## Proposed change

Make the assertion environment-robust: accept reparenting to ANY pid outside the killed
process group (or detect the subreaper explicitly) instead of asserting `ppid == 1` — the
invariant under test is "the detached child survives the group kill", not "init adopts
it". Reproduce in the gate context first to confirm, then fix; if reproduction instead
shows a genuine convention regression, fix the convention helper and say so. Deliverable:
the node is reliably green on pristine main in every execution context (gate tree, repo
root, worktree, cron).

## Provenance

- fingerprint: 4e505106423d

- workflow_fix_target: tests/test_workflow_setsid_detach_convention.py
- origin: /daily 2026-08-05 Step C parked-candidate sweep — parks on #1996
  (origin_candidate_ts 2026-08-05T15:47:20Z prose park, superseded by
  2026-08-05T19:54:52Z urgent park, fp 4516c07c1164), one filing for both sightings.
