---
title: terminal-status guard + source stamp for orphan-respawn (662
kind: infra
tags:
- wf-fix
- wf-fix-fp:35aca6cbb5a1
- daily-auto-filed
created_at: '2026-07-10T06:55:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): 709/709/393 orphan-respawn
  markers+commits posted against terminal tasks 662/663/867 since 06-26 by a stale
  watcher build outside the canonical cron'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from an orchestrator observation during the nightly problem sweep.

## Goal

Stop the two-week orphan-respawn loop posting `[autonomous_session_watch:orphan-respawn]` markers + git commits against TERMINAL tasks #662 (archived), #663 (completed), #867 (completed), and make the orphan-respawn pass structurally unable to act on a terminal task.

## Workflow gap

- **Bug observed:** Tasks #662/#663/#867 have accumulated 709/709/393 `epm:progress` orphan-respawn markers ("active task (status=running) had no live registered session ... auto-respawned via spawn-issue --auto") since 2026-06-26 / 07-03, each committed to main (~1,800+ junk commits, ~91/task/day), while all three tasks have been terminal for 1-2 weeks (#663 completed 06-25, #662 archived, #867 completed 07-02).
- **Why it is a workflow gap:** The canonical cron watcher is exonerated — its per-pass logs (`logs/autonomous_session_watch/2026-07-09.log`) show `orphan-sweep: 2 active-status task(s)` and contain ZERO mentions of #662/#663/#867 or any "auto-respawned" line, and `task.py list-by-status --status running` returns only #1092. The posted marker notes ALSO lack the "(attempt N/M today)" suffix that current `_post_progress_marker` code appends — so the poster is a STALE BUILD of `autonomous_session_watch.py` running somewhere outside the canonical cron (irregular timestamps, e.g. 02:22/02:33/03:08/03:10/03:18Z, not aligned to the `3-59/10` cron), reading a stale task-state snapshot yet committing markers into the canonical tree. `~/explore-persona-space-codex-port` was checked and ruled out (no tasks/662 there). No respawned sessions actually materialize (today's 76 transcripts are all legitimate wave/#1092 sessions), so the damage is marker/commit spam + registry churn, not pod spend — but the loop is unbounded and pollutes events.jsonl + git history.

## Proposed change (refine in planning)

1. **Defense in depth (primary):** in `_process_orphan_task` (and `_respawn_orphan`), immediately before the respawn action and before posting the orphan-respawn marker, RE-VERIFY the task's live status via `task.py view <N> --json` against the canonical repo root; if the status is terminal/parked (`completed`, `archived`, `awaiting_promotion`, `on_hold`, `blocked`), abort with a loud log line and post nothing. A stale `_active_status_tasks` snapshot must never produce a respawn or a marker on a terminal task.
2. **Self-identification:** stamp host + pid + `git rev-parse --short HEAD` of the running checkout into the orphan-respawn (and stalled-respawn) marker note, so any future stale-instance occurrence identifies its source on the first marker.
3. **Forensic hunt:** locate and kill the stale poster (grep running processes/timers/other-user crontabs/other machines with push access; the marker cadence is ~1/task/10-20min). Document the finding.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Secondary: `.claude/rules/background-automation.md` (document the terminal-status re-verify + stamp)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 35aca6cbb5a1

- workflow_fix_target: scripts/autonomous_session_watch.py

Evidence trail: `tasks/archived/662/events.jsonl` (709 orphan-respawn rows, first 2026-06-26T10:28:46Z, last 2026-07-10T03:18:02Z); `tasks/completed/663/events.jsonl` (709 rows); `tasks/completed/867/events.jsonl` (393 rows); commit `bb784d7d39` (a representative marker commit); canonical watcher log `logs/autonomous_session_watch/2026-07-09.log` (zero #662 mentions, "orphan-sweep: 2 active-status task(s)").
