---
title: 'daily-fix: bash -n pin test for Step 10d TG fenced blocks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:daf77c9ac3eb
- daily-auto-filed
created_at: '2026-07-30T06:59:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): The Step 10d TG fenced
  blocks in issue SKILL.md were non-parseable under bash -n (#1790''s defect) while
  every pin test stayed green — existing pins assert substring presence only'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: parked formal candidate on #1790 (implementer r1, sweep fp 781f787dccc2, ts 2026-07-29T09:16:51Z)).

## Goal

Pin the bash-parseability of the Step 10d TG fenced blocks so a comment/formatting edit cannot re-break copy-paste executability without failing a test.

## Workflow gap

- **Bug observed:** Orchestrators execute these fenced blocks verbatim; no mechanical check pins their bash parseability — the existing substring pins are structurally blind to this regression class (#1790).
- **Why it is a workflow gap:** `grep -n '_tg_blocks' tests/test_step10d_guard3.py` -> helper exists (L1392); the file's only `bash -n` assertion (L1242) is scoped to the post-merge guard region's single fence, not the TG blocks (read in context 2026-07-30, this run).
- **Confidence (emitter):** medium
- verified-at-filing: Per the candidate's diff sketch: a test iterating `_tg_blocks(_skill_text())`, substituting `<N>` -> dummy id, asserting `bash -n` rc==0 per block.

## Proposed change (refine in planning)

tests/test_step10d_guard3.py

## Scope / surfaces

- Primary target: `
## Provenance

- workflow_fix_target: tests/test_step10d_guard3.py
- fingerprint: daf77c9ac3eb

- Origin park: #1790 events 2026-07-29T09:16:51Z (fingerprint 781f787dccc2); verbatim candidate block lives in that marker.
`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.
