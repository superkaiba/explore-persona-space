---
title: 'daily-fix: orchestrator inlines verifier output for Codex cr'
kind: infra
tags:
- wf-fix
- wf-fix-fp:129718b20f95
- daily-auto-filed
created_at: '2026-07-05T07:02:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): On 2026-07-04 the Codex
  clean-result-critic round-4 verdict for #841 shipped WITHOUT the mechanical pre-pass:
  the Codex sandbox FS is read-only, so uv could not create temp files and both verify_task_body.py
  and the audit script FAILED to run.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Change the composed prompt contract: the ORCHESTRATOR runs verify_task_body.py + audit_clean_results_body_discipline.py (or verify_paper.py) and inlines their output into the Codex prompt, instead of instructing the sandboxed Codex twin to execute uv itself.

## Workflow gap

- **Bug observed:** On 2026-07-04 the Codex clean-result-critic round-4 verdict for #841 shipped WITHOUT the mechanical pre-pass: the Codex sandbox FS is read-only, so uv could not create temp files and both verify_task_body.py and the audit script FAILED to run.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/codex-clean-result-critic.md`
- Session: ff97909e (#841, 08:05 UTC round 4).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/agents/codex-clean-result-critic.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
