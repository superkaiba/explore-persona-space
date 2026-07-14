---
title: 'daily-fix: WARN when verbatim insert exceeds ratchet headroo'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b6b1b1d46890
- daily-auto-filed
created_at: '2026-07-11T06:51:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): a plan committing VERBATIM
  prose to a workflow_lint size-ratcheted file whose remaining headroom is smaller
  than the quoted block makes lint-passes + file-count constraints jointly unsatisfiable
  at implement time (#1230: 422 B headroom vs 1,546 B paragraph forced a documented
  3rd-file cap-raise deviation)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

verify_plan.py plan-time WARN when a plan's fenced verbatim insertion into a workflow_lint-ratcheted file exceeds that file's remaining headroom

## Workflow gap

- **Bug observed:** a plan committing VERBATIM prose to a workflow_lint size-ratcheted file whose remaining headroom is smaller than the quoted block makes lint-passes + file-count constraints jointly unsatisfiable at implement time (#1230: 422 B headroom vs 1,546 B paragraph forced a documented 3rd-file cap-raise deviation)
- **Provenance / evidence:** implementer + code-reviewer r1 prose follow-up, #1230 (parked 2026-07-10T10:15:13Z). Verified live: zero ratchet/headroom hits in verify_plan.py.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b6b1b1d46890

- workflow_fix_target: scripts/verify_plan.py
