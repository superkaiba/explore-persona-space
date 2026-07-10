---
title: 'daily-fix: codify long-phase heartbeat duty + tick re-arm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9bd5d916b1d0
- daily-auto-filed
created_at: '2026-07-09T07:01:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): 4 watcher ALIVE-BUT-STALLED
  force-respawns on 2026-07-08 hit sessions that were alive (#1092 x2, #825 x1, #1112
  x1 after a 6h freeze): long detached CPU/API phases polled healthily but never refreshed
  the self-report or posted heartbeat markers, and one cheap-band round launched a
  multi-hour run after CRON-TEARDOWN without re-arming /issue-tick.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Stop the watcher force-respawning healthy sessions during long detached phases; make the #1092 heartbeat convention a codified duty.

## Workflow gap

- **Bug observed:** On 2026-07-08 the watcher force-respawned live sessions 4 times: #1092 at 21:43Z (killed a legitimately armed 4h bg until-loop) and 02:53Z (respawn was then refusal-killed, costing ~100 min), #825 at 08:33Z (killed a healthy 30-min poll chain mid-GPU-fit), #1112 at 17:23Z (6h self-report freeze). Sessions that posted [long-phase-heartbeat] notes (#1092 current session 63f06d97) were not respawned.
- **Why it is a workflow gap:** The heartbeat convention exists ad hoc but is not a codified duty in the /issue polling-loop prose, and the cheap-band follow-up loop launches runs after CRON-TEARDOWN without re-arming the tick cron (mine-A P2, #1112).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

(a) SKILL.md polling-loop + same-issue follow-up loop: during any detached phase >60 min, post a [long-phase-heartbeat] epm:progress note + refresh the self-report at least every ~60 min; re-arm the /issue-tick cron when a follow-up round dispatches a run after teardown. (b) optional watcher side: count fresh transcript assistant-activity / bg-task output mtime as corroborating liveness before an ALIVE-BUT-STALLED stop+respawn (mirror of the #1051 tick fix). Planner decides whether to ship (a), (b), or both.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, scripts/autonomous_session_watch.py
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-1092 P4, mine-A P2, mine-E P1, mine-1112-1090 P1
