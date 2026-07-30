---
title: 'daily-fix: WARN when dispatch handle points at dead attempt'
kind: infra
tags:
- wf-fix
- wf-fix-fp:51cea2984dae
- daily-auto-filed
created_at: '2026-07-29T07:02:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): A dispatch handle sidecar
  (.claude/cache/issue-<N>-handle.json) whose sentinel_path/workload_pid point at
  a DEAD prior attempt is consumed silently by the poller every tick — completion
  is never observed and nothing warns (incident #1689 r15b: handle pointed at the
  old attempt''s .completion-sentinel.json while the live relaunch wrote a new path).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a formal candidate block parked on task #1750 (ts 2026-07-28T12:02:59Z, fp 51cea2984dae; surfaced by the plan critic, Methodology lens, during #1750's review). #1750 landed the prose duty (pod-side descope posts a marker + rewrites the handle); this filing is the mechanical detector for when that duty is skipped.

## Goal

Add a WARN-only per-tick poller check that flags a dispatch handle sidecar (`.claude/cache/issue-<N>-handle.json`) whose sentinel_path / workload_pid / attempt id point at a DEAD prior attempt while a live relaunch is observed.

## Workflow gap

- **Bug observed:** a handle sidecar pointing at a dead prior attempt is consumed silently by the poller every tick — completion is never observed and nothing warns (incident #1689 r15b: the handle pointed at the old attempt's `.completion-sentinel.json` while the live relaunch wrote a new path).
- **Why it is a workflow gap:** the pid-file contract has WARN-only detectors (the #1156 `pid_file_stale_vs_marker` / #1650 family); the handle sidecar has none, so a skipped handle-rewrite duty is invisible until someone audits by hand.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'handle_stale_vs_live\|handle_older_than_relaunch' scripts/backend_poll.py` → 0 hits (absence claim; the detector does not exist — 2026-07-29 UTC). Call-hop note (clause g): the named precedent detector `pid_file_stale_vs_marker` lives in `scripts/poll_pipeline.py` (lines 1507/5680/6027/6111), NOT backend_poll.py — the planner should confirm which poller actually consumes `issue-<N>-handle.json` (CLAUDE.md says the bg-Bash poller reads the handle back; both `backend_poll.py` and `poll_pipeline.py` are candidate sites) and place the detector at the consuming site; both sites are recorded here. Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/backend_poll.py` → no handle-staleness commit.

## Proposed change (candidate diff sketch — refine in planning)

```
+ in the poller tick assembly (at the handle-consuming site):
+   if handle.extra.workload_pid dead AND observed live workload pid differs -> WARN handle_stale_vs_live
+   if newest epm:run-launched marker ts > handle sidecar mtime -> WARN handle_older_than_relaunch
```

Tick-JSON flag + WARN line, verdict unchanged (mirror of the #1156 `pid_file_stale_vs_marker` detector).

## Scope / surfaces

- Primary target: `scripts/backend_poll.py` (confirm consuming site; `scripts/poll_pipeline.py` holds the precedent detector and may be the correct or additional site)

## Constraints / invariants

- WARN-only — never changes the poll verdict.
- Workflow-surface only; ruff passes; recursion guard applies to the spawned session.

## Provenance

- sha-verify (filing-time, #1467): `51cea2984dae` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: scripts/backend_poll.py
- fingerprint: 51cea2984dae

<!-- workflow-fix-candidate v1 -->
target_file: scripts/backend_poll.py
bug_observed: A dispatch handle sidecar (.claude/cache/issue-<N>-handle.json) whose sentinel_path/workload_pid point at a DEAD prior attempt is consumed silently by the poller every tick — completion is never observed and nothing warns (incident #1689 r15b: handle pointed at the old attempt's .completion-sentinel.json while the live relaunch wrote a new path).
why_workflow_gap: pod-side-reporting.md gains a prose duty to rewrite the handle on descope relaunch (task #1750), but there is no mechanical detector when the duty is skipped — the pid-file contract has WARN-only detectors (#1156/#1650); the handle has none.
proposed_change: Add a WARN-only per-tick check in the poller comparing the handle's extra.expected_artifacts.sentinel_path / extra.workload_pid / extra.runpod_attempt_id against the live run's observed attempt/pid; set a tick-JSON flag (e.g. handle_stale_vs_live) + a WARN line, verdict unchanged (mirror of the #1156 pid_file_stale_vs_marker detector).
confidence: medium
related_task: #1750
<!-- /workflow-fix-candidate -->
