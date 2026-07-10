---
title: cap existing wedge triggers at watcher override site
kind: infra
tags:
- wf-fix
- wf-fix-fp:3d62e9bb12a1
- daily-auto-filed
created_at: '2026-07-10T06:55:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): The four pre-#1209 wedge
  triggers'' respawns are uncapped at _apply_prompt_wedge_override — background-automation.md
  § (e) claims ''bounded by ... the respawn cap'' but only the NEW #1209 failed-turn-sil'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1209.

## Goal
Bound the four existing wedge triggers' respawns at the override site (episode belt and/or day cap, matching the #1209 shape), accounting for the crash-arm handoff path; or correct background-automation.md if unbounded is the deliberate choice.

## Workflow gap
- **Bug observed:** The four pre-#1209 wedge triggers' respawns are uncapped at _apply_prompt_wedge_override — background-automation.md § (e) claims 'bounded by ... the respawn cap' but only the NEW #1209 failed-turn-silence trigger reads respawn_count/day-cap at the override site (verified on main: the cap gate at ~L10408 guards only that trigger); advancement-clear also voids the episode cap across generations, and the fence spawn branch is largely bypassed via crash-arm handoff, so a fix should account for that path.
- **Why it is a workflow gap:** The rule prose promises a bound the code does not implement for the pre-existing triggers — an uncapped respawn storm channel plus a docs/behavior contradiction.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none)

## Scope / surfaces
- Primary target: `scripts/autonomous_session_watch.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: n/a (prose park)

routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — logged, not auto-routed; source: prose-followup from planner + methodology critic). target_file: scripts/autonomous_session_watch.py (_apply_prompt_wedge_override). Observation: the FOUR EXISTING wedge triggers' respawns are uncapped at the override site — background-automation.md § (e) claims 'bounded by ... the respawn cap' but the override never reads respawn_count, and advancement-clear voids the episode cap across generations (methodology-critic trace additionally shows the fence spawn branch is largely bypassed via crash-arm handoff, so any fix should account for that path). #1209 bounds only the NEW trigger (minimal scope). Distinct fingerprint from #1209's; for the nightly /daily parked-candidate sweep.
