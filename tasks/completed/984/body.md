---
title: 'workflow-fix: VM-scope cross-ref on upload-policy env one-liner'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:b398d317d451
created_at: '2026-07-04T07:12:18Z'
has_clean_result: false
origin_prompt: 'parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target,
  see workflow-fix-on-bug.md § Recursion guard. source: prose-followup (code-reviewer
  round 1 Minor). target_file: .claude/rules/upload-policy.md. bug_observed: the Hub-verification
  one-liner at upload-policy.md:99-102 carries the unconditional ''set -a && source
  .env && set +a'' prefix — VM-scoped context but copyable into pod/GCE-side scripts
  where ./.env does not exist. proposed_change: add a one-line VM-scope cross-ref
  pointing at the new gotchas.md conditional-sourcing entry. confidence: medium. related_task:
  #944. routed: parked (recursion guard) — next non-workflow-fix orchestrator pass
  may file it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add a one-line VM-scope cross-ref pointing at the gotchas.md conditional-sourcing entry

## Workflow gap

- **Bug observed:** the Hub-verification one-liner at upload-policy.md:99-102 carries the unconditional 'set -a && source .env && set +a' prefix — VM-scoped context but copyable into pod/GCE-side scripts where ./.env does not exist (#923 class)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a one-line VM-scope cross-ref pointing at the gotchas.md conditional-sourcing entry

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- fingerprint: b398d317d451

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug.md § Recursion guard. source: prose-followup (code-reviewer round 1 Minor). target_file: .claude/rules/upload-policy.md. bug_observed: the Hub-verification one-liner at upload-policy.md:99-102 carries the unconditional 'set -a && source .env && set +a' prefix — VM-scoped context but copyable into pod/GCE-side scripts where ./.env does not exist. proposed_change: add a one-line VM-scope cross-ref pointing at the new gotchas.md conditional-sourcing entry. confidence: medium. related_task: #944. routed: parked (recursion guard) — next non-workflow-fix orchestrator pass may file it.
