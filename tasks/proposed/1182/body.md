---
title: 'workflow-fix: Memoize transcript resolution across watcher p'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e830ac5541e7
- daily-auto-filed
created_at: '2026-07-09T06:59:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): session_resolver._resolve_transcript_via_happy_log
  is re-resolved per session per watcher pass (~160 ms/session dominant resolution
  term), with multiple passes in one tick repeating identical resolutions (call sites
  at autonomous_session_watch.py:8654 and :16266).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1127.

## Goal

Remove the ~160 ms/session repeated transcript-resolution cost by memoizing _resolve_transcript_via_happy_log within a single watcher tick.

## Workflow gap

- **Bug observed:** session_resolver._resolve_transcript_via_happy_log is re-resolved per session per watcher pass (~160 ms/session dominant resolution term), with multiple passes in one tick repeating identical resolutions (call sites at autonomous_session_watch.py:8654 and :16266).
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + tick-scoped cache in autonomous_session_watch (e.g. dict[node_pid] ->
  +   (transcript, reason), created per run_watch tick and passed/threaded to
  +   the passes), or a session_resolver-level cache with an explicit
  +   invalidate() called at tick start — never a process-lifetime lru_cache
  +   (transcripts move between ticks)

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, scripts/session_resolver.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Cache lifetime must be a single watcher tick (a process-lifetime lru_cache would serve stale transcript paths across ticks).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/session_resolver.py
- origin: parked candidate on task #1127 at 2026-07-08T08:32:54Z

Verbatim parked note:

```
routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard) — planner amendment surfaced a prose follow-up: tick-scoped memoization of _resolve_transcript_via_happy_log across watcher passes (scripts/autonomous_session_watch.py / scripts/session_resolver.py) to remove the ~160ms/session dominant resolution term for every pass. source: prose-followup; this session is a workflow-fix session so the candidate is LOGGED, not auto-routed (.claude/rules/workflow-fix-on-bug.md § Recursion guard); a future human/orchestrator pass may file it.
```
