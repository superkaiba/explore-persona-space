---
title: 'daily-fix: divtrain gen through api_dispatch + empty=retry'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-17T06:58:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the #952 divergent-training
  ext-arm generation pass fired 580 near-parallel Sonnet calls against the shared
  org cap (polite cap is ~100), lost 138 generations logged only as a COUNT with no
  per-row error records, and treated 62 EMPTY completions returned as API successes
  as data — violating the api_dispatch mandate and llm-judging rule 24''s per-row
  transport discipline'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#952 inline-override session 7555a9a4, ~11:39Z). Experiment-code fix (wf_fix false).

## Goal

Bring the #952 divtrain generation pass under the standard transport discipline.

## Workflow gap

- **Bug observed:** the #952 divergent-training ext-arm generation pass fired 580 near-parallel Sonnet calls against the shared org cap (polite cap is ~100), lost 138 generations logged only as a COUNT with no per-row error records, and treated 62 EMPTY completions returned as API successes as data — violating the api_dispatch mandate and llm-judging rule 24's per-row transport discipline
- **Why it is a workflow gap:** Silent count-only losses + empty-as-success are exactly the censoring class rules 24/9 exist to prevent, and 580-parallel against the shared cap starves every concurrent session.
- **Confidence (emitter):** high (measured today: 138 lost, 76 recovered on retry, 62 empty-as-success residual)
- verified-at-filing: incident quantified in transcript 7555a9a4; the gen script's call sites named at planning time (relocation grep over scripts/issue952_*divtrain* / the session's committed round scripts)

## Proposed change (candidate diff sketch — refine in planning)

route the divtrain generation calls through llm/api_dispatch.py (parallelism cap + per-row transport records) and treat empty completions as retryable failures, never successes

## Scope / surfaces

- Primary target: `scripts/ (#952 divtrain generation script)`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Non-workflow-surface fix (`wf_fix: false`): no recursion guard applies; standard /issue pipeline.

