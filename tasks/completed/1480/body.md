---
title: 'workflow-fix: escalate unactioned stalled-manual alerts to registration unregister'
kind: infra
tags:
- wf-fix
- wf-fix-fp:29eaf4438ca3
created_at: '2026-07-17T21:51:52Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1298 (triage of the #928 followups_running
  6-day freeze): stalled-manual-session lane is alert-only with no bounded escalation;
  add unregister-after-unactioned-alerts rung (never stop the user session)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1298 (emitting agent: issue-1298 orchestrator, triaging the
#928 followups_running 6-day freeze).

## Goal

Add a bounded escalation rung to the watcher's stalled-MANUAL-session lane so
an unactioned stalled-alert on a manual registration holding an ACTIVE task
with an unrun followup scope eventually UNREGISTERS the manual registration
(never stopping the user session), letting the orphan sweep re-drive the task.

## Workflow gap

- **Bug observed:** A live-but-wedged manual (user-driven,
  `manual-issue-928.json`) session held task #928 at `followups_running`
  mid-round for 6 days (2026-07-08 -> 07-14) with an unrun followup scope
  (`prefix-based-mapping-arms`, plan v7 posted 07-08 07:43Z, never approved).
  The stalled-session lane alerted at +2.3h (07-08 10:03Z,
  "NOT auto-respawned (manual user-driven session; alert-only by design)")
  but no automatic recovery path ever fired; recovery required a human
  `spawn_session.py stop` + autonomous re-drive on 07-14 21:04Z (the round
  then completed cleanly by 07-15 01:38Z).
- **Why it is a workflow gap:** Manual-session alert-only is deliberate
  (#505 — never auto-respawn a user-driven session), but there is no bounded
  escalation when the alert goes unactioned. The only unregister path,
  `decide_stale_registration` (#845), gates on TRANSCRIPT idle >= 12h of a
  LIVE session — any transcript activity in the live wedged session (user
  interaction, in-session cron ticks) defeats it indefinitely, so an ACTIVE
  task with an unrun followup scope and zero non-watcher progress has no
  watcher-side recovery. Alerts alone are insufficient ("sent != seen"); the
  #928 alert sat unactioned for 6 days across two /daily flags (07-11,
  07-12) and a filed needs-human task (#1298).
- **Confidence (emitter):** medium (the 6-day freeze + alert-only latch are
  marker-proven; the exact reason the 12h transcript-idle gate never fired is
  inferred from the live-on-07-14 deliberate-stop, not transcript-proven)
- verified-at-filing: `grep -n "alert-only by design" scripts/autonomous_session_watch.py`
  -> 1 hit in 1 file (`:11291`, the stalled-manual branch — presence
  confirmed; the lane posts the alert and returns, no escalation);
  semantic absence probe: `grep -n "unregister" scripts/autonomous_session_watch.py`
  -> 20 hits, ALL in the 12h stale-registration pass (`:988`, `:2647-2672`,
  `:19999-20023`), zombie/GC, and docstring contexts — none in the
  stalled-alert lane (`:11291-11388`); landed-fix history:
  `git log --oneline --since='2026-07-08' -- scripts/autonomous_session_watch.py`
  -> 20 commits, none adding a stalled-manual escalation (boot-death lanes
  #1267/#1287 fire only on boot-dead/boot-refused transcripts; the
  #1127/#1209 wedge lanes require failed wake turns) (2026-07-17).

## Proposed change (candidate diff sketch — refine in planning)

```
  # scripts/autonomous_session_watch.py, stalled-session lane (~:11291)
  reason = "manual user-driven session; alert-only by design"
+ # Escalation rung (#928 6-day freeze): after >= K (default 3) consecutive
+ # stalled-alert confirmations spanning >= EPM_STALLED_MANUAL_ESCALATE_H
+ # (default 24) hours on the SAME manual registration, with (a) zero
+ # non-watcher task events since the first confirmation and (b) an unrun
+ # followup scope OR an otherwise ACTIVE status, UNREGISTER the manual
+ # registration (delete manual-issue-<N>.json; NEVER stop the session) and
+ # post a loud [stalled-manual-escalation] marker + push. The orphan sweep
+ # then re-drives the ACTIVE task on its next pass. Every unresolvable
+ # input fails toward keep (alert-only, today's behavior).
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'alert-only by design' .claude/ CLAUDE.md scripts/`) and update
  every hit; list them in the plan. Companion docs:
  `.claude/rules/background-automation.md` (stalled-session lane description)
  and the watcher docstring pass inventory (the #1225 lint).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- NEVER stop or respawn the user-driven session itself (#505 stands) — the
  escalation acts on the REGISTRATION only, exactly like the
  stale-registration pass; the orphan sweep's existing re-drive machinery
  does the recovery.
- Fail toward keep on every unresolvable input (the watcher-wide convention).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; docstring pass-count lint (#1225) stays consistent.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 29eaf4438ca3

<!-- workflow-fix-candidate v1 -->
target_file: scripts/autonomous_session_watch.py
bug_observed: A live-but-wedged manual (user-driven) session held task #928 at followups_running mid-round for 6 days (2026-07-08 to 07-14) with an unrun followup scope; the stalled-session lane alerted at +2.3h but is alert-only for manual registrations and no automatic recovery fired
why_workflow_gap: Manual-session alert-only is deliberate (#505), but there is no bounded escalation when the alert goes unactioned: decide_stale_registration's transcript-idle>=12h gate is defeated indefinitely by any transcript activity in the live wedged session, so an ACTIVE task with an unrun followup scope and zero non-watcher progress has no watcher-side recovery
proposed_change: Add an escalation rung to the stalled-manual-session lane: after >=K consecutive stalled-alert confirmations spanning >=24h on the same manual registration with zero non-watcher task progress and an unrun followup scope, UNREGISTER the manual registration (never stop the user session) so the orphan sweep re-drives the ACTIVE task
diff_sketch: |
  + after >=3 consecutive stalled-alert confirmations spanning >=24h on the
  + same manual registration (zero non-watcher task events; unrun followup
  + scope / ACTIVE status): delete manual-issue-<N>.json (never stop the
  + session), post [stalled-manual-escalation] marker + push; orphan sweep
  + re-drives. Fail toward keep on unresolvable inputs.
confidence: medium
related_task: #1298
<!-- /workflow-fix-candidate -->
