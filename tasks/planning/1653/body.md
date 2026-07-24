---
title: 'daily-fix: completed-unmerged watcher respawn arm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c665f89d4866
- daily-auto-filed
created_at: '2026-07-24T06:48:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): the completed-unmerged
  pass is flag-only so a refusal-killed orchestrator leaves the Step 10d merge stranded
  until manual recovery, the #1540 and #1622 class'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident (session 03147a44, 2026-07-23 ~12:24Z): the #1622 session was killed by a Usage-Policy refusal mid Step 10d, AFTER `epm:done` but BEFORE the merge — PR #1399 sat open with no `epm:merged`. The watcher's completed-unmerged pass (#1564) flagged it at ~14:43, but recovery was still MANUAL (the #1540 class); the stranded window was ~2.3h.

## Goal

Give the watcher's completed-unmerged flag pass a bounded RESPAWN arm: on a confirmed `epm:done`-without-`epm:merged` episode with an open unmerged `issue-<N>` PR/branch and no live owning session, respawn a session to complete Step 10d — never auto-merge from the watcher itself.

## Workflow gap

- **Bug observed:** `scripts/autonomous_session_watch.py`'s completed-unmerged pass is FLAG-ONLY (sidecar + marker + push); a refusal-killed orchestrator (twice this month: #1540 16h; #1622 ~2.3h) leaves the merge stranded until a human or the next manual session acts.
- **Why it is a workflow gap:** every other watcher failure class (crash, wedge, stall) has an automated respawn arm; the stranded-merge class is the remaining manual one, and refusal-kills on guard-surface merge turns are a recurring cause (#1098/#1563 vocabulary class).
- **Confidence:** medium — the flag-only design was deliberate in #1564 (never auto-MERGE); the proposal here is the narrower crash-recovery-style RESPAWN arm (the fresh session runs the ordinary Step 10d path with its own gates). The spawned planner may deflect with a reasoned no-change report if the flag-only rationale covers respawn too.
- verified-at-filing: registry shows #1622 now `completed` (recovered manually in-session); the pass's flag-only contract is documented in CLAUDE.md § background-automation bullet ("never merges or mutates status; kill switch EPM_DISABLE_COMPLETED_UNMERGED_PASS"). Absence claim (no respawn arm) is the documented design, not a grep miss.

## Proposed change (refine in planning)

Extend the completed-unmerged pass: after flagging, if no live session owns the issue and the episode persists ≥1 poll cycle, `spawn-issue --auto` (bounded once per (issue, done_ts) episode, day-capped like the wedge triggers) so the fresh session's Step 10d completes the merge through the normal gates.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`

## Constraints / invariants

- The watcher must still NEVER merge or mutate task status itself; the arm only respawns.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: c665f89d4866

- workflow_fix_target: scripts/autonomous_session_watch.py
