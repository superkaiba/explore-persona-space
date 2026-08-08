---
title: 'daily-fix: nightly step9c known-red ledger refresh cron'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dbd88dd486ff
- daily-auto-filed
created_at: '2026-08-06T07:01:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): stale ledger forced mid-gate
  31-40min refreshes + indeterminate compares in 3 sessions'
workflow: v1
---
# daily-fix: refresh the step9c known-red baseline ledger on a cron, not lazily inside issue sessions

## Workflow gap

The Step 9c/10d known-red-on-main compare depends on a baseline ledger that today goes
stale whenever main takes a red wave, and every issue session then pays the refresh cost
mid-gate. On 2026-08-05→06: #2105's gate went compare rc=2 INDETERMINATE ("6 red FILES
exceed --max-pristine-files 5; ledger (…07:24Z…) predates today's main regressions"),
forcing a detached ~40-min ledger refresh plus a re-run; #1992's gate round 1 was
indeterminate the same way (ledger @06:38Z predated the morning red wave) and detached a
~31-min refresh mid-gate; #2106 ground through 3 gate rounds against the same stale
baseline. Aggregate: hours of gate wall-time re-attributing pre-existing reds on a day
when the wf-compact waves + the #2054 test-pin rewrite kept main red-shifting.

verified-at-filing: `grep -rn 'refresh' scripts/step9c_baseline.py | head` shows the
refresh entrypoint exists and is invoked ad hoc (no cron registration for it in
`crontab -l` — probed 2026-08-06T07:2xZ: no `step9c_baseline` line). Incident quotes are
the miners' probed marker/pytest readbacks (sessions 2ba4ca62, 04b3e1dc, 9c8b88b8).

## Proposed change

Register a nightly cron (after the daily-fix wave lands, e.g. ~05:30 PT) that runs
`scripts/step9c_baseline.py` refresh single-flight and commits/pushes the ledger, so
sessions start each day against a fresh baseline; keep the in-session lazy refresh as the
fallback. Planner decides cadence + whether a main-push-triggered refresh (watcher arm) is
worth the extra machinery. Companion note: the 11 chronic known-red pin tests observed in
gates (daily-doc / issue-skill pins) deserve a drain check — several were fixed by
#2104/#2106 tonight; the cron makes the residue visible instead of amortized into every
session.

## Provenance

- fingerprint: dbd88dd486ff

- workflow_fix_target: scripts/step9c_baseline.py, .claude/rules/background-automation.md
- origin: /daily 2026-08-05 problem sweep — miner 7 P16, miner 6 P14 (independent).
