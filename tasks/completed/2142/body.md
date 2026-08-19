---
title: 'daily-fix: dispatch lease races (4 dup spawns + cap 6>5)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8cd2ba879258
- daily-auto-filed
created_at: '2026-08-06T07:23:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): four duplicate /issue spawns
  at 9-59s gaps against live owners; batch dispatch overshot the cap (occupied=6 cap=5)'
workflow: v1
---
# daily-fix: dispatch-lease races — four duplicate /issue spawns (9–59s gaps) and a cap overshoot (occupied=6, cap=5) in one day

## Workflow gap

Five same-day incidents show the proposed-infra dispatch path re-checks neither the lease
nor the cap at spawn time:

- Duplicate `/issue` sessions spawned against live owners: #1771 (59 s after the owner's
  registration), #1997 (26 s after the owner's progress note), #1988 (9 s after the
  watcher dispatched the owner), #1992 (29 s after owner activity). All four were caught
  by the Step 0 single-orchestrator guard and exited cleanly — the waste is four spawns +
  boot cost, and the 9 s gap on #1988 suggests two dispatch lanes firing on the same ripe
  task in one tick.
- Cap overshoot: after dispatch resumed post-daemon-outage, "Occupancy is 6 (the sweep had
  dispatched past the old cap during the window I wasn't watching)" with cap 5 — the
  per-pass batch dispatch does not re-check `occupied < cap` before each individual spawn
  (or raced a cap edit).

verified-at-filing: all five are the recovery miners' probed transcript reads (sessions
d38f46fa rows 22–27, 644f41fe rows 22–29, 910f48c8 rows 23–31, 5ca7bd0e rows 19–24,
4966e56e rows 1163/1255). `ls tests/test_dispatch_lease.py` at compose time → the lease
test file exists; the sweep path's coverage of it is the question.

## Proposed change

In `scripts/autonomous_session_watch.py`'s dispatch loop(s): re-check the dispatch lease
immediately before POSTing each spawn (not once per pass), skip candidates whose issue has
a session registration younger than the sweep interval, and re-check `occupied < cap`
before each individual spawn inside a batch. Extend `tests/test_dispatch_lease.py` with
the two race shapes (fresh-registration skip; per-spawn cap re-check).

## Provenance

- fingerprint: 8cd2ba879258

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: /daily 2026-08-04 recovery sweep — miner 5 P14, miner 4 P23, miner 7 P12/P14.
