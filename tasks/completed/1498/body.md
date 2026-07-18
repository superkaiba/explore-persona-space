---
title: 'daily-fix: 9c selector misses literal-path pinning tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e3352d2e653a
- daily-auto-filed
created_at: '2026-07-18T06:46:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): the selector''s mapped
  selection does not enumerate tests/test_ruff_policy.py for a diff touching scripts/daily_drive_filings.py
  even though that test pins the file by literal path — the map misses the pinning-test
  shape.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-17 parked-candidate sweep (Step C) from a prose-followup candidate parked on task #1483 (emitting agent: implementer r2; parked under the recursion guard).

## Goal

Teach `scripts/select_step9c_tests.py` to enumerate PINNING-shape tests — tests that hardcode target file paths by literal string without a GLOB_SCAN_TESTS-style directory glob — for diffs touching the pinned file.

## Workflow gap

- **Bug observed:** the selector's mapped selection does not enumerate `tests/test_ruff_policy.py` for a diff touching `scripts/daily_drive_filings.py`, even though that test pins the file by literal path — the selector's map misses the pinning-test shape. The #1288 pin-sweep duty caught it manually in both #1483 rounds.
- **Why it is a workflow gap:** a pinning test that never runs on the diff it pins is a silent gate hole; manual pin-sweeps are the only current defense and depend on the implementer remembering the duty.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "test_ruff_policy\|daily_drive_filings" scripts/select_step9c_tests.py` → 0 hits (2026-07-18 UTC) — the mapping is absent in-target (absence-of-mapping claim; the 0-hit result is the evidence). NOT a duplicate of open #865 ("Step 9c selector diffs main checkout, blind to worktree branches" — a diff-base bug, different fingerprint/bug on the same file; per the dedup rule a distinct bug files its own task).

## Proposed change (candidate diff sketch — refine in planning)

Add a literal-path scan pass: for each touched file, grep `tests/test_*.py` for the touched path as a literal string and enumerate matching tests (bounded; cached), or maintain an explicit pin map with a lint that a test hardcoding a `scripts/...` path must appear in it.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Related sibling being filed the same night: the `.claude/rules/*.md`→prose-pin-test mapping gap (same file, distinct bug — see the daily-fix filing from #1468's candidate). The spawned sessions should cross-reference to avoid overlapping edits.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; selector tests green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: e3352d2e653a

- workflow_fix_target: scripts/select_step9c_tests.py

source candidate (verbatim, prose park on #1483, 2026-07-18T04:34:25Z): "scripts/select_step9c_tests.py mapped selection does not enumerate tests/test_ruff_policy.py for a diff touching scripts/daily_drive_filings.py, even though that test pins the file by literal path — the selector's map misses the pinning-test shape (a test that hardcodes target file paths without a GLOB_SCAN_TESTS-style glob). The #1288 pin-sweep duty caught it manually both rounds. target_file: scripts/select_step9c_tests.py."
