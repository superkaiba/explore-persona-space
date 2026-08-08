---
title: 'daily-fix: tick_triage counts compact rows as human'
kind: infra
tags:
- wf-fix
- wf-fix-fp:83fb9d847004
- daily-auto-filed
created_at: '2026-08-06T07:22:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): isCompactSummary continuation
  rows read as human activity, granting a 45-min stall mask per compaction; false
  HEALTHY on #1336'
workflow: v1
---
# daily-fix: tick_triage counts auto-compact continuation rows as HUMAN activity — false HEALTHY masks stalls

## Workflow gap

`tick_triage.py`'s human-active screen (`is_human_transcript_row`, ~line 1008) excludes
`<command-message>`/`<local-command`/`<task-notification` prefixed strings but has NO
check for `isCompactSummary` rows, and the compaction continuation text ("This session is
being continued from a previous conversation…") matches none of the exclusions — so every
auto-compaction grants a ≤45-min stall-mask. Observed live 2026-08-04T12:48:19Z on the
#1336 session: verdict "HEALTHY … human-active (last human msg 38m ago < 45m)" in a fully
autonomous session with ZERO human messages; 12:48 − 38 min = exactly the 12:10:40Z
`isCompactSummary: true` continuation row. In thrashing sessions (which compact every
tick) this can suppress STALE-REDRIVE indefinitely — it compounded the #1336 ~5.4 h
heartbeat wedge and the #2054 7-consecutive-STALE tick loop that day.

verified-at-filing: `grep -c 'isCompactSummary' scripts/tick_triage.py` → 0 (run
2026-08-06T09:0xZ on main); the 12:48Z verdict + the 12:10:40Z row flag are the recovery
miner's probed transcript reads (session f0789a8a, source read of tick_triage.py:1008-1046
at mining time).

## Proposed change

In `is_human_transcript_row`: return False for rows with `isCompactSummary` truthy, plus a
prefix exclusion for the continuation-summary text as belt-and-braces; add a regression
test (a synthetic transcript whose only recent "user" row is a compact summary must NOT
read human-active).

## Provenance

- fingerprint: 83fb9d847004

- workflow_fix_target: scripts/tick_triage.py
- origin: /daily 2026-08-04 recovery sweep (run inside the 08-05 nightly) — miner 6 P6
  (probed).
