---
title: 'daily-fix: completed-unmerged resume-table row'
kind: infra
tags:
- wf-fix
- wf-fix-fp:415ba3a13d0d
- daily-auto-filed
created_at: '2026-07-21T06:38:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): the /issue resume table
  has an awaiting_promotion unmerged-PR backstop row but no completed-status row,
  so a completed task with epm:done but no epm:merged and an unmerged PR has no routed
  recovery path'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-20 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1564 under the recursion guard (emitting context: #1564 plan §2 + the epm:results report).

## Goal

Add a `completed`-status row to the `/issue` SKILL.md resume table routing a completed task with no `epm:merged` and an unmerged PR/branch to the Step 10d auto-merge procedure — mirroring the `awaiting_promotion` backstop row — making a bounded auto re-drive arm for the new `completed_unmerged_pass` watcher audit mechanically safe.

## Workflow gap

- **Bug observed:** #1564 shipped the watcher's `completed_unmerged_pass` as FLAG-ONLY (the #1540 stranded Step 10d merge class, 16h invisible pre-fix); the /issue resume table has an `awaiting_promotion` + unmerged-PR backstop row but NO `completed` + unmerged-PR row, so a re-invoked `/issue <N>` on a completed-unmerged task has no routed recovery path.
- **Why it is a workflow gap:** without the resume-table row, recovery from the flagged state stays manual; with it, the watcher pass (or a human `/issue <N>`) has a mechanically safe re-drive arm.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'awaiting_promotion.*no .epm:merged.*PR unmerged' .claude/skills/issue/SKILL.md` → the awaiting_promotion backstop row at :11796 routes to the Step 10d auto-merge procedure; no `completed`-status resume-table row with the epm:merged/unmerged predicate exists (0 hits) (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Add a resume-table row: `completed` + events carry `epm:done` but no `epm:merged` + `issue-<N>` PR/branch unmerged → run the Step 10d auto-merge procedure (idempotent backstop), then re-run the completion audit.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Related: `scripts/autonomous_session_watch.py` `completed_unmerged_pass` (#1564) — the flag-only pass this row makes re-drivable.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 415ba3a13d0d

- workflow_fix_target: .claude/skills/issue/SKILL.md

Verbatim parked candidate (prose park on #1564, ts 2026-07-20T13:46:03Z):

> parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); routed by the nightly /daily parked-candidate sweep. Candidate (prose, from plan §2 + the epm:results report): add a completed-status row to the /issue SKILL.md resume table routing a completed task with no epm:merged and an unmerged PR/branch to the Step 10d auto-merge procedure (mirroring the awaiting_promotion backstop row at SKILL.md ~:11601) — this is what would make a bounded auto re-drive arm for the new completed_unmerged_pass mechanically safe. target_file: .claude/skills/issue/SKILL.md; confidence: medium; related_task: #1564.
