---
title: Reap completed autonomous /issue sessions within ~1h (not 12h) — stop the ghost-session
  pile
kind: infra
tags: []
created_at: '2026-06-28T19:51:37Z'
has_clean_result: false
origin_prompt: 'PM-session 2026-06-28: ''are they actually running? I think the happy
  daemon is bugged.'' Daemon+watcher healthy; root cause = completed /issue --auto
  sessions don''t self-terminate and the lost issue-mapping drops them from the 2h
  terminal-reconcile into the 12h unmapped reap, so they linger as ''running'' ghosts.
  Fix: self-exit at terminal OR short-threshold reap when last-mapped task is terminal.'
---
## Overview / Motivation

Autonomous `/issue --auto` sessions do not self-terminate when their task
reaches a terminal/park state (`completed` / `awaiting_promotion` /
`archived` / `blocked`). The watcher has a reconcile pass that auto-stops
issue-MAPPED sessions whose task is terminal at >~2h idle — but when a task
completes, the session→issue mapping is LOST, so the now-unmapped session
falls out of the 2h terminal-reconcile and into the generic 12h
idle-unmapped bucket. Result: completed sessions linger as "running" ghosts
for up to 12 hours (~17 such ghosts observed 2026-06-28, idle 6–12h, all
mapping to completed tasks; same class as the 2026-06-12 25-session VM-lag
pile). Manual sweep was needed to clear them, and the user reads the pile as
"the daemon is bugged" (it is not — daemon + watcher are healthy).

## Goal

Completed-task autonomous sessions get reaped within ~1h, not ~12h, without
relying on a manual sweep.

## Scope / surfaces (pick the cleanest of these in planning)

1. **Preserve last-mapped-issue on the session** so the watcher's terminal
   reconcile (>2h, terminal task) still applies after a task completes —
   instead of the session dropping to mapped=False and into the 12h bucket.
   (`scripts/spawn_session.py` session registry + `autonomous_session_watch.py`
   reconcile pass.)
2. **OR** `autonomous_session_watch.py`: in the idle-unmapped pass, if a
   session's LAST-known mapped task is terminal, reap at a SHORT threshold
   (~30–60 min) rather than the generic 12h unmapped threshold.
3. **OR** `/issue` SKILL.md: an autonomous (`EPM_AUTONOMOUS_SESSION=1`)
   session self-exits when it reaches a terminal/park state (Step 10d
   completion / `awaiting_promotion` park) — the cleanest, removes the ghost
   at the source.

Option 3 (self-exit at terminal) is preferred if feasible; 1/2 are the
watcher-side safety net.

## Constraints / invariants

- Never stop a session with a live TTY (user-attached), a running pod, a
  `keep-running` tag, or a non-terminal task. The existing exclusions stay.
- A live same-issue follow-up round (`followups_running` / fresh
  `epm:run-launched`) must NOT be reaped.
- Workflow-surface only. `scripts/workflow_lint.py` + ruff pass.

## Acceptance

- A completed autonomous `/issue --auto` session is gone (process exited /
  stopped) within ~1h of the task reaching terminal, with no manual sweep.
- A still-active session, a user-attached (TTY) session, and a
  live-follow-up session are all left untouched.

## Provenance

PM-session investigation 2026-06-28: user asked "are they actually running? I
think the happy daemon is bugged." Daemon + watcher were healthy; the pile was
completed sessions not self-terminating + the mapping-loss dropping them from
the 2h terminal-reconcile into the 12h unmapped bucket. Manual sweep cleared
~17 ghosts.
