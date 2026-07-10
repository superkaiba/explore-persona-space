---
title: 'daily-fix: planner authors Row-coverage with paired contrast'
kind: infra
tags:
- wf-fix
- wf-fix-fp:08c8930420f0
- daily-auto-filed
created_at: '2026-07-09T07:01:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): verify_plan c18 (paired_contrast_source_coverage)
  FAILed on BOTH #1112 amendment drafts (v4 and v7) for the same omission — the planner
  repeatedly omits the Row-coverage per-arm block when registering a paired contrast;
  one revision cycle each.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Stop the recurring c18 bounce by making Row-coverage authoring guidance explicit in the planner spec.

## Workflow gap

- **Bug observed:** Transcript 05000ba2 (issue-1112) 10:08:07Z: FAIL c18_paired_contrast_source_coverage on plan v4->v5; 13:07:30Z the SAME FAIL again on v7, fixed in v8 — two mechanical bounces in one day on one task.
- **Why it is a workflow gap:** The mechanical gate exists but no planner-side spec line tells planners to write the Row-coverage block when registering a paired contrast, so it is systematically discovered post-hoc.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In the amendment/delta-scoped plan guidance: "Any registered paired contrast MUST carry its `Row-coverage:` per-arm rows at authoring time — verify_plan c18 FAILs otherwise (2x on #1112, 2026-07-08)."

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-A P3 (05000ba2 10:08:07Z, 13:07:30Z)
