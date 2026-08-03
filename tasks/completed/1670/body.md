---
title: 'daily-fix: reconcile pass must not stop young respawns'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3190982482fb
- daily-auto-filed
created_at: '2026-07-25T06:48:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The session-reconcile pass
  auto-stopped the watcher''s OWN completed-unmerged respawn for issue 1622 ten minutes
  after spawning it - idleness read the task''s non-watcher-marker gap of 27.5h instead
  of the 10-minute-old session''s age or transcript activity - killing the Step 10d
  merge recovery mid-lint-gate and leaving PR 1399 stranded again'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 76a8649e, task #1622). The two watcher passes fought each other: completed-unmerged respawned a recovery session at 15:43:11Z (#1653 arm, 1/3 today), and session-reconcile stopped it at 15:53:38Z.

## Goal

The watcher must never stop its own just-spawned recovery session; reconcile idleness must key on session-level activity, not task-marker gaps alone.

## Workflow gap

- **Bug observed:** #1622 events show `15:43:11Z completed-unmerged-respawn ... spawn-issue --issue 1622 --auto` then `15:53:38Z session-reconcile-stop auto-stopped 1 idle session(s) ... no activity ... for > 2.0h (gap=27.5h)`. The recovery session was 10 minutes old, mid-lint-gate (its bg gate task shows status killed at 15:53:38), and had not yet posted a marker — the idleness predicate read the TASK's non-watcher-marker gap, not the session's age/transcript mtime. PR #1399 remains OPEN draft; the once-per-episode respawn budget is consumed, so #1622 stays stranded without human action (held separately).
- **Why it is a workflow gap:** the reconcile pass's idle predicate lacks a session-age floor and does not read the completed-unmerged-respawn marker as a live-follow-up exemption — the same marker-predicate pattern the pod-safety pass already uses.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -n 'session-reconcile' scripts/autonomous_session_watch.py` → pass present (sentinels at :1335/:1340, contract at :244); `grep -n 'completed_unmerged' scripts/autonomous_session_watch.py` → `completed_unmerged_pass` at :563; `git log --oneline --since='7 days ago' -- scripts/autonomous_session_watch.py` → #1653 landed the respawn arm 2026-07-24 but no commit adds a reconcile-side exemption (2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

In the session-reconcile pass: (a) treat session registration/spawn time and transcript mtime as activity — never stop a session younger than the idle threshold; (b) exempt an issue whose latest `completed-unmerged-respawn` watcher marker is newer than the latest done-transition. Pin test for the young-respawn case.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (session-reconcile pass + completed_unmerged_pass coordination)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 3190982482fb

- workflow_fix_target: scripts/autonomous_session_watch.py
