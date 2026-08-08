---
title: 'daily-fix: reconcile-stop misses error-storming sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:89ed1b8ddbe9
- daily-auto-filed
created_at: '2026-08-06T07:02:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): thrashing session burned
  4h on completed #2004; reconcile-stop only stops IDLE sessions on terminal tasks'
workflow: v1
---
# daily-fix: watcher reconcile-stop skips NON-idle sessions on terminal tasks — a thrashing #2004 session burned 4h after the task completed

## Workflow gap

The autonomous-session watcher's parked/terminal reconcile-stop pass stops sessions whose
mapped task is terminal only when they look IDLE (no activity for > ~2 h). On 2026-08-05,
task #2004 was merged + completed at 12:05:26Z by a watcher-respawned sibling, while the
original session (transcript 0d005b55) was mid autocompact-thrash. That session kept
cycling failed turns (thrash/too-long API errors + continuation restarts — constant
"activity", zero durable work) on the already-completed task until 16:00:50Z, only exiting
at its own 16:20 tick reading TERMINAL. The reconcile pass DID stop one idle duplicate at
14:24 ("auto-stopped 1 idle session(s)") but not the error-storming one — ~4 h of pure
token burn on a completed task.

verified-at-filing (2026-08-06T07:1xZ): `task.py view 2004 --json` events → merged
12:02:25Z, `Step 10d auto-complete` 12:05:26Z, watcher `session-reconcile-stop` of ONE
idle session at 14:24:29Z. The 12:23→16:00Z thrash-row tail is the miner's probed scan of
transcript 0d005b55 (rows carry only SKILL.md re-reads + thrash errors after 12:05Z).

## Proposed change

Extend the reconcile-stop predicate in `scripts/autonomous_session_watch.py`: a session
mapped to a parked/terminal task with NO durable-work signal (no non-watcher marker post,
no commit, no live pod/follow-up signal) is stoppable even when its transcript shows
recent rows — failed-turn API-error rows and compact/continuation churn are not
"activity". Keep the existing shields (live follow-up signal markers, keep-running tag,
RUNNING pod) unchanged; the change is only that transcript-row recency alone must not
read as liveness when every recent row is an error/compaction artifact.

## Provenance

- fingerprint: 89ed1b8ddbe9

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: /daily 2026-08-05 problem sweep — miner 7 P2 (transcript 0d005b55 + #2004
  events timeline, probed).
