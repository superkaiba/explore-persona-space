---
title: 'daily-fix: LESSONS ratchet constant conflict magnet'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e9002ea02c99
- daily-auto-filed
created_at: '2026-07-18T06:47:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): the hand-maintained _LESSONS_RATCHET_BYTES
  constant produced, within ~48h: two Step-10d merge conflicts (#1335 PR #1227, #1435
  PR #1188), one fleet-wide trunk-red (#1462 growth without bump), and one duplicated
  fix pipeline (#1476/#1479).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-17 (route 2) from a transcript-mined recurrence (chunk-1 miner): two Step-10d worktree-merge conflicts on the same day (#1335 PR #1227, #1435 PR #1188) both centered on `.claude/rules/LESSONS.md` + the `scripts/workflow_lint.py` byte-count ratchet constants — and the same hand-maintained constant also drove the #1462 trunk-red (growth commit without the bump) and the #1476-vs-#1479 duplicate-fix race.

## Goal

Reduce the LESSONS-ratchet conflict/red surface: evaluate making the ratchet constant self-maintaining (e.g. computed measured-size-plus-headroom with a separate deliberate-growth gate, or an auto-bump companion check that fails with the exact new value to paste), so parallel LESSONS.md edits stop colliding on a hand-maintained cumulative byte total.

## Workflow gap

- **Bug observed:** `_LESSONS_RATCHET_BYTES` in `scripts/workflow_lint.py` is a hand-bumped constant that every LESSONS.md growth commit must update in the same change; under parallel sessions this produced, within ~48h: 2 worktree-merge conflicts (both PRs), 1 fleet-wide trunk-red (bump forgotten, #1462→#1463), and 1 duplicated fix pipeline (#1476/#1479).
- **Why it is a workflow gap:** the ratchet's growth-control purpose is real, but its single-constant implementation makes every concurrent LESSONS edit a conflict/red hazard; the cost is now recurring weekly.
- **Confidence:** low (design tradeoff — the planner may deflect with a reasoned no-change report if the deliberate-growth friction is judged load-bearing).
- verified-at-filing: `grep -n "_LESSONS_RATCHET_BYTES" scripts/workflow_lint.py` → constant present (currently 6800; LESSONS.md measured 6675) (2026-07-18 UTC). Incident trail: #1462 (growth without bump) → #1463 candidate (parked) → #1476 landed the bump / #1479 duplicate archived; merge-conflict pairs mined from the 07-17 transcripts (chunk-1 findings file).

## Proposed change (candidate diff sketch — refine in planning)

Options: (a) keep the FAIL but compute the allowed ceiling as `measured-at-last-approved-growth + headroom` from a small committed sidecar the growth commit updates via a helper (conflict-free append vs constant edit); (b) auto-emit the exact bump value in the FAIL message (already partially done) plus a `--fix-ratchet` writer; (c) reasoned no-change with a documented conflict-recovery recipe. Planner decides.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (+ `.claude/rules/LESSONS.md` convention note)

## Constraints / invariants

- The growth-control intent of the ratchet must survive any change — it exists to keep the always-on context surface deliberate.
- Workflow-surface only; `tests/test_workflow_lint.py` green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: e9002ea02c99

- workflow_fix_target: scripts/workflow_lint.py

source: /daily 2026-07-17 transcript sweep (chunk-1 miner) — LESSONS-ratchet as recurring conflict magnet; related incidents #1462/#1463/#1476/#1479.
