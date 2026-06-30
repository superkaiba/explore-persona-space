---
title: 'daily-fix: watcher must not respawn duplicate per-issue sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:13a920921167
- daily-auto-filed
created_at: '2026-06-30T06:44:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: The watcher repeatedly spawned
  duplicate per-issue sessions (issue-664/665/739/740) that each had to detect-and-exit
  via the single-orchestrator guard, and once respawned a 2nd #739 orchestrator mid-Phase-1.5
  while a fact-checker subagent was still running.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. Recurring wasted spawns across issue-664/665/739/740 + a mid-Phase-1.5 respawn race on #739.

## Goal
The watcher never spawns a duplicate driver for an issue that already has a live (or actively-working) session.

## Workflow gap
- **Bug observed:** watcher spawned duplicate per-issue sessions (each forced to detect-and-exit via the single-orchestrator guard); once respawned a 2nd #739 orchestrator mid-Phase-1.5 while its fact-checker subagent ran ~4 min, duplicating planning work.
- **Why it is a workflow gap:** the watcher respawn logic is workflow surface.
- **Confidence (emitter):** medium; multiple sessions. The guard catches it, but the spawns waste compute.

## Proposed change
- Check the live-session registry before respawning a `/issue <N>` driver; skip if a live session exists.
- Treat a session with recent tool/subagent activity as alive (not stalled) before declaring it stalled and respawning.

## Scope / surfaces
- `scripts/autonomous_session_watch.py`.

## Constraints / invariants
- Workflow surface only. Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 13a920921167

Sessions: 716c2d95 (#739), 6bdade94 (#665), and #664/#740 duplicate-exit hits.
