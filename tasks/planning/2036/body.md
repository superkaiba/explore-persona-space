---
title: 'daily-fix: post-autocompact relaunch marker-triage mandate'
kind: infra
tags:
- wf-fix
- wf-fix-fp:59ea5bfb5772
- daily-auto-filed
created_at: '2026-08-03T07:01:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): The first wake after autocompact
  (session f98a12ed, 2026-08-03T03:48-03:54Z) posted a wrong root-cause epm:failure
  v5 and re-dispatched leg a5syc on GCP ~30s after the user''s inline ''move to runpod''
  override -- duplicate instance created, stand-down + ROOT CAUSE WITHDRAWN correction
  needed. Probed: no task.py view / marker read appears between the compaction row
  and the dispatch (rows 2333-2399),'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner1, session f98a12ed, task #1739).

## Goal

No crash-diagnosis->relaunch sequence dispatches without a fresh external-marker triage (task.py view/latest-marker read newer than the last dispatch), especially on the first post-autocompact wake.

## Workflow gap

- **Bug observed:** The first wake after autocompact (session f98a12ed, 2026-08-03T03:48-03:54Z) posted a wrong root-cause epm:failure v5 and re-dispatched leg a5syc on GCP ~30s after the user's inline 'move to runpod' override -- duplicate instance created, stand-down + ROOT CAUSE WITHDRAWN correction needed. Probed: no task.py view / marker read appears between the compaction row and the dispatch (rows 2333-2399), while the user's wedge-evidence marker (epm:progress v347, 03:37:16Z) sat unread on the same issue.
- **Why it is a workflow gap:** The triage duty exists (triage_candidates_since_last_dispatch was used earlier in the same lineage) but is not bound as a mandatory step of the crash-relaunch sequence, and autocompact silently erases the context that would otherwise prompt it.
- **Confidence (emitter):** high (probed by miner: tool-call-by-tool-call read of the compaction->dispatch window)
- verified-at-filing: `grep -n 'triage_candidates_since_last_dispatch' .claude/skills/issue/SKILL.md scripts/*.py` -> helper exists and is referenced for pre-dispatch triage, but no clause binds it to crash-diagnosis relaunches or post-compaction wakes (0 hits for 'autocompact' near dispatch/relaunch in SKILL.md).

## Proposed change (refine in planning)

make the pre-dispatch external-marker triage an explicit MANDATORY named step of any crash-diagnosis->relaunch sequence, re-armed explicitly on the first wake after autocompact (compaction drops other actors' markers from context).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 59ea5bfb5772

- workflow_fix_target: .claude/skills/issue/SKILL.md

