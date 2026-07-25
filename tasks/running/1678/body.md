---
title: 'daily-fix: stagger same-target filings in one wave'
kind: infra
tags:
- wf-fix
- wf-fix-fp:56b2c79ad893
- daily-auto-filed
created_at: '2026-07-25T06:50:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): Two Step 10d merge conflicts
  in the 07-24 wave came from same-file concurrency among wave siblings dispatched
  together: 1645 vs 1659 both touching select_step9c_tests.py and 1655 vs 1643 both
  touching runpod_api.py - each burned extra merge attempts and an implementer conflict-resolution
  spawn'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (sessions 5349b9f5/1a210c3c/e0365c9c, tasks #1645/#1655/#1641).

## Goal

Reduce same-file Step 10d merge contention among wf-fix wave siblings filed and dispatched together.

## Workflow gap

- **Bug observed:** in the 07-24 wave, #1645 attempt-1 hit `CONFLICT (content): scripts/select_step9c_tests.py` against just-merged #1659 (resolved by a conflict-resolution implementer spawn); #1655 needed 3 merge attempts (~30 min) against #1643's runpod_api.py change; #1641 attempt-1 was refused on main-side churn. All recovered via documented shapes, but the contention is predictable at FILING time when two manifest items name the same target file.
- **Why it is a workflow gap:** the driver dispatches all wave items concurrently with no shared-target awareness, though the manifest carries each item's `target`.
- **Confidence (emitter):** medium (serialize vs WARN is a design choice; recoveries did work as designed).
- verified-at-filing: `grep -n 'target' scripts/daily_drive_filings.py` → manifest target field read but no shared-target grouping/serialization (absence bind, 2026-07-25); incident evidence from #1645/#1655 session records.

## Proposed change (candidate diff sketch — refine in planning)

In `daily_drive_filings.py`: group manifest items by target-file overlap; either dispatch later same-target siblings without `--auto` spawn until the earlier sibling merges (watcher backstop picks them up), or WARN + record `shared_target_group` in filed.jsonl rows for filer staggering.

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 56b2c79ad893

- workflow_fix_target: scripts/daily_drive_filings.py
