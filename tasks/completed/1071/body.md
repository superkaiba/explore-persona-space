---
title: 'workflow-fix: watcher auto-remediates alive-but-stalled sessions + queues
  daemon-blocked respawns'
kind: infra
tags:
- wf-fix
created_at: '2026-07-05T14:54:38Z'
has_clean_result: false
origin_prompt: '#813 3 manual respawns in 36h (2026-07-03/04, two daemon-unreachable
  alert-only incidents); watch-session held candidate, filed at #813 park'
workflow: v1
---
## Overview / Motivation
Auto-filed by the workflow-fix protocol from the #813 triple-manual-respawn sequence (watch-session orchestrator; filed at the #813 park).

## Goal
The watcher's ALIVE-BUT-STALLED session detection escalates to an automatic stop+respawn after N confirmed checks (it is alert-only today), and any remediation blocked by a Happy-daemon outage is QUEUED and executed on the next daemon-reachable tick instead of being dropped.

## Workflow gap
- **Bug observed:** #813 needed 3 MANUAL session respawns in ~36h. Incidents: (1) 2026-07-03 20:43Z alive-but-stalled alert, "NOT auto-respawned (Happy daemon unreachable)" — manual respawn 21:35Z; (2) 2026-07-04 05:03Z identical alert + daemon unreachable — manual respawn 05:20Z; (3) the orphan-respawn pass DID auto-fire once (2026-07-03 02:33) but only because that session had left the live set. The alive-but-stalled variant (dead bg-Bash chain inside a live Claude process) is never auto-remediated, and daemon-outage remediations are dropped rather than queued.
- **Why it is a workflow gap:** autonomous_session_watch.py's respawn pass is deliberately daemon-gated and only covers NO-live-session orphans; the stalled-alert pass says "investigate via the phone session and stop+respawn manually if confirmed dead" — a manual step that an autonomous fleet cannot rely on, demonstrated 2x in one task.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)
- `scripts/autonomous_session_watch.py`:
  + stalled-alert pass: after >=2 confirmed checks (existing confirmation counter) AND self-report frozen >T (existing threshold), attempt stop+respawn via the same guarded path the orphan-respawn pass uses (stop the wrapper, spawn-issue --auto; dispatch-lease prevents duplicates; keep the existing has_pod safety gates);
  + daemon-outage queueing: when ANY remediation (stalled-respawn or orphan-respawn) is blocked by daemon unreachability, write a queued-action sidecar row (.claude/cache or ~/.eps-autonomous) and drain it FIRST on the next tick where the daemon probe passes;
  + keep alert-only as the fallback when the safety gates (live pod mid-run, non-autonomous session, plan_pending/blocked states) veto action.
- Tests: extend tests/test_workflow* or the watcher's own test surface for the queue-drain predicate.

## Scope / surfaces
- Primary target: `scripts/autonomous_session_watch.py`
- Grep first: `grep -n "stalled\|respawn\|daemon" scripts/autonomous_session_watch.py | head -30` (the pass structure + existing confirmation counters).

## Constraints / invariants
- Workflow-surface only; never auto-stop non-autonomous/PM sessions (existing exclusions preserved); workflow_lint passes.

## Provenance
- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: (wrapper)
