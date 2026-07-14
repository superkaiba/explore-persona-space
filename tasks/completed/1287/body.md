---
title: 'daily-fix: watcher respawn for boot-turn refusal deaths'
kind: infra
tags:
- wf-fix
- wf-fix-fp:790fa08b599a
- daily-auto-filed
created_at: '2026-07-13T06:44:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): #1277''s session was refusal-killed
  on its boot turn BEFORE the /issue-tick cron was armed; zero failed wakes accumulated
  so no wedge trigger fired, the stalled-session lane is alert-only, and recovery
  took ~12.5h via the 12h stale-registration sweep.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 problem sweep (transcript-mined; emitting session f7dd0696 / recovery session d2c117e4, task #1277).

## Goal

Make the watcher recover a watcher-dispatched session that is refusal-killed on its BOOT turn — before the /issue-tick cron is armed — within minutes, not 12.5 hours.

## Workflow gap

- **Bug observed:** #1277's first session (spawned 06:54Z by `proposed_infra_sweep`) died on its boot turn at 06:55:41Z with the spurious "violates our Usage Policy" API error, BEFORE Step 0 armed the /issue-tick cron. It accumulated zero failed wake turns, so none of the five wedge triggers (#1241) fired; the 09:03Z `session-stalled-alert` lane is alert-only; recovery came only via the 12h `stale-registration-unregister` (19:13Z) + `proposed_infra_sweep` re-dispatch (19:23Z). Net: ~12.5h wall-clock stall on a 0-GPU-h infra task.
- **Why it is a workflow gap:** the #1209 `failed-turn-silence` dead-wake trigger was built for exactly this freeze-below-counting-thresholds shape, but its predicate does not cover a pre-cron boot-turn death (no wake turns ever accumulate, and the tick cron that would generate them was never armed). The stalled-session lane that DOES notice the freeze (~2h) cannot respawn.
- **Confidence (emitter):** medium
- verified-at-filing: `uv run python scripts/task.py view 1277 --json` events timeline (2026-07-13): transcript f7dd0696 last row = api-error at 06:55:41Z; `session-stalled-alert` 09:03Z (alert-only); `stale-registration-unregister` 19:13Z; re-dispatch 19:23Z; completion 20:34Z. Behavioral gap — not grep-verifiable beyond the timeline.

## Proposed change (candidate diff sketch — refine in planning)

Make the watcher's stalled/dead-wake machinery respawn-capable for this shape: when a registered autonomous session's transcript tail row is an api-error/refusal row, the task is at a pre-run status (proposed/planning/plan_pending/approved), no /issue-tick cron exists for it, and the transcript has been silent ≥ EPM_TICK_WEDGE_DEAD_SILENCE_MIN — stop the session and let the existing crash-recovery/infra-sweep re-dispatch, under the existing episode-belt + per-issue day cap (#1241). Alternatively extend the #1209 `failed-turn-silence` predicate to count a boot-turn api-error death as a failed turn even with zero wakes.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (prompt-wedge / stalled-session lanes), possibly `.claude/skills/issue-tick/SKILL.md` docs.
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Respawn arms must stay bounded (episode belt + 3/issue/UTC-day cap, #1241) and fail toward keep.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 790fa08b599a

- workflow_fix_target: scripts/autonomous_session_watch.py

Origin: /daily 2026-07-12 transcript sweep, sessions f7dd0696 (boot-turn refusal death, task #1277) + d2c117e4 (the 19:23Z recovery session). Related incidents: #1209 (turn-1 refusal death freeze), #1127, #1074, #1241 (wedge-trigger belts).
