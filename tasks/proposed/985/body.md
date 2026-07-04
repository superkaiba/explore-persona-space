---
title: 'workflow-fix: production-body test rule for seam-stubbed fns'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:92f7aed09d28
created_at: '2026-07-04T07:12:24Z'
has_clean_result: false
origin_prompt: 'source: prose-followup (methodology critic, Phase 2). Proposed: implementer-side
  rule — one production-body test per seam-stubbed function (boundary-only, signature-conformant
  fakes) — target_file: .claude/agents/experiment-implementer.md, .claude/agents/implementer.md,
  .claude/rules/code-style.md. The producer-side half of the #906 family; deliberately
  out of #948''s reviewer-check scope. routed: parked — this session runs under workflow_fix_target
  recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); log-and-notify
  only, next non-workflow-fix orchestrator pass may file it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add an implementer-side rule: one production-body test per seam-stubbed function (boundary-only, signature-conformant fakes)

## Workflow gap

- **Bug observed:** no implementer-side rule requires a production-body test per seam-stubbed function; boundary-only signature-conformant fakes can hide a broken production body (#906 family, producer-side half, out of #948's reviewer-check scope)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add an implementer-side rule: one production-body test per seam-stubbed function (boundary-only, signature-conformant fakes)

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md, .claude/agents/implementer.md, .claude/rules/code-style.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md, .claude/agents/implementer.md, .claude/rules/code-style.md
- fingerprint: 92f7aed09d28

source: prose-followup (methodology critic, Phase 2). Proposed: implementer-side rule — one production-body test per seam-stubbed function (boundary-only, signature-conformant fakes) — target_file: .claude/agents/experiment-implementer.md, .claude/agents/implementer.md, .claude/rules/code-style.md. The producer-side half of the #906 family; deliberately out of #948's reviewer-check scope. routed: parked — this session runs under workflow_fix_target recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); log-and-notify only, next non-workflow-fix orchestrator pass may file it.
