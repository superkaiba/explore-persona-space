---
title: 'daily-fix: judge transport errors are RETRY, never dropped d'
kind: infra
tags:
- wf-fix
- wf-fix-fp:569c2bb3d2a5
- daily-auto-filed
created_at: '2026-07-09T07:01:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): #1090 stored ~2,638 API-529
  overload errors as dropped judge draws, arm-asymmetric (400/1000 trained draws on
  one cell) — the rule-23 censoring shape; interp-critic caught it pre-publication
  and a re-judge recovered 2,635/2,638 with zero refusals.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Distinguish judge transport errors from content drops in the drop-never-coerce rule.

## Workflow gap

- **Bug observed:** Transcript c16b10ca (issue-1090) 09:34:17Z interp-critique r1: the dropped judge draws were stored API-529 overload errors, freely re-judgeable; drop asymmetry 400/1000 trained draws on C2-icl; 10:07:34Z re-judge recovered 2,635/2,638 (zero refusals), all cells 200/200.
- **Why it is a workflow gap:** Rule 9 (drop-never-coerce) does not distinguish content-informative judge failures (REFUSAL/non-numeric/out-of-range) from transport failures; a pipeline persisting transport errors as drops silently censors arms, mimicking a selection artifact.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Rules 9/23 amendment: transport errors (429/529/timeout/connection) are RETRY-with-backoff, never drops; the per-arm dropped-draw report splits content-drops vs transport-errors; a companion note that eval/batch_judge.py should not persist error:True entries counted as drops (library change = sibling infra task, src/ off-surface here).

## Scope / surfaces

- Primary target: `.claude/rules/llm-judging.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/llm-judging.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-1112-1090 P9 (c16b10ca 09:34:17Z, 10:07:34Z)
