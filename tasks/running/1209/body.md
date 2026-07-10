---
title: 'daily-fix: fast respawn for sessions dead on a refused first'
kind: infra
tags:
- wf-fix
- wf-fix-fp:279f1cf53f80
- daily-auto-filed
created_at: '2026-07-09T07:01:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The 02:53Z #1092 watcher
  respawn was refusal-killed on its FIRST turn; a 1-turn-dead session accumulates
  no failed WAKES for the prompt-wedge lane, so recovery waited ~100 min for the slow
  stalled lane.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Close the single-refused-turn death gap in the refusal-wedge watcher lane.

## Workflow gap

- **Bug observed:** Session 8e9c371d (issue-1092) was refused on its first substantive turn at 02:54:29Z and died after ONE turn (39 transcript lines); the #1127 turn-level failed-wake predicate needs >=3 consecutive failed wake turns, which a dead session never accumulates; the next respawn came only at 04:33Z (~100 min unattended).
- **Why it is a workflow gap:** The wedge lane covers refusal storms across wakes but not a session that dies on turn 1 and receives no further wakes.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In the prompt-wedge pass: if the transcript tail LAST assistant row has isApiErrorMessage:true AND no rows follow for >= EPM_TICK_DEAD_WEDGE_MIN (default ~15-20 min), classify dead-wedged and force-respawn (bounded by the existing per-day respawn caps).

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-1092 P3 (transcript 8e9c371d, 02:54:29Z)
