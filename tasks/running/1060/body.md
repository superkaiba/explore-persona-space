---
title: 'daily-fix: 1-cell pilot timing for per-cell fit phases + 2x '
kind: infra
tags:
- wf-fix
- wf-fix-fp:04e62ed208f3
- daily-auto-filed
created_at: '2026-07-05T07:03:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): #811''s per-cell ridge-fit
  phase ran ~19h against a ~6h plan estimate on 2026-07-04 (42/108 cells at 19h21m,
  ~4x projection) and #931''s p3_fits overran ~2x - the recurring #823-class serial
  dense-fit sizing failure; the #811 session only pivoted to vectorizing after flagging
  a second compute-deviation, ~19 CPU-h in.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

plan-compute-sizing requires a MEASURED 1-cell pilot timing x cell count for any per-cell ridge/SVD/GD fit phase before accepting the section-9 estimate; and the vectorize rule's Supersede contract gains a mid-run trigger: a compute-deviation at 2x the plan estimate forces the vectorize check immediately.

## Workflow gap

- **Bug observed:** #811's per-cell ridge-fit phase ran ~19h against a ~6h plan estimate on 2026-07-04 (42/108 cells at 19h21m, ~4x projection) and #931's p3_fits overran ~2x - the recurring #823-class serial dense-fit sizing failure; the #811 session only pivoted to vectorizing after flagging a second compute-deviation, ~19 CPU-h in.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md, .claude/rules/vectorize-many-cell-fits.md`
- Sessions: 86923fc5 + f2ef929f (#811), 2f37cc81 (#931).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/rules/plan-compute-sizing.md, .claude/rules/vectorize-many-cell-fits.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
