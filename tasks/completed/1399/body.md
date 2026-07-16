---
title: 'daily-fix: wf-fix dedup screens completed siblings'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cdbee92a95b9
- daily-auto-filed
created_at: '2026-07-16T07:21:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Two duplicate wf-fix sessions
  in one day: #1350 filed 25 min after #1329 merged the same fix (different wording
  -> different fingerprint); #1330 duplicated #1309'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

At wf-fix filing time, additionally grep recently-completed (e.g. last 7 days) wf-fix/daily-fix tasks touching the same target_file and surface topical overlap to the filer before spawning — an advisory, not a hard block.

## Workflow gap

- **Bug observed:** two duplicate wf-fix sessions burned the same day: #1350 filed for a fix #1329 had merged 25 min earlier (different wording → different fingerprint; 51ff1b7c 17:58Z), and #1330 archived as a duplicate of #1309's already-landed guidance (cf03c19a 06:58Z).
- **Why it is a workflow gap:** the dedup predicate is exact-`(target_file, fingerprint)` over OPEN tasks only — a just-merged sibling with different wording is invisible by design, and there is no advisory surface that would show the filer the fresh completion before it spawns a full pipeline session.
- **Severity:** medium
- verified-at-filing: `grep -n 'recently.completed\|topical overlap\|7 days' .claude/rules/workflow-fix-on-bug.md scripts/file_infra_task.py` → 0 hits — proposed advisory absent; the existing predicate is documented at workflow-fix-on-bug.md § Dedup ("whose status is NOT in the terminal set {completed, archived}"; "A closed ... workflow-fix task does NOT block a re-raise") and implemented in `task_workflow.is_open_workflow_fix_task` — presence of the open-only exact-fingerprint grain confirmed (2026-07-16 UTC).

## Proposed change (refine in planning)

Add a filing-time advisory leg to `scripts/file_infra_task.py` (and document it in `.claude/rules/workflow-fix-on-bug.md` § Dedup): before spawning, enumerate wf-fix/daily-fix tasks (WF_FIX_TITLE_PREFIXES) completed/archived within the last ~7 days whose `workflow_fix_target:` overlaps the candidate's target_file, and print a topical-overlap advisory (task ids + titles + merge dates) for the filer to eyeball. Advisory only — closed tasks deliberately never hard-block genuine re-raises; the advisory just makes the just-merged sibling visible before a session is spawned.

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md` (§ Dedup)
- Secondary: `scripts/file_infra_task.py` (the file+dispatch wrapper); `src/explore_persona_space/task_workflow.py` (a `recent_closed_workflow_fix_tasks(target_file)` helper next to `is_open_workflow_fix_task`)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The (target_file, fingerprint) open-task dedup predicate and its tests (`tests/test_workflow_fix_dedup.py`) stay behaviorally unchanged — the new leg is additive advisory output.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: cdbee92a95b9

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 51ff1b7c (#1350) 17:58Z (batch 05 P15); cf03c19a (#1330) 06:58Z (batch 02 P10).
