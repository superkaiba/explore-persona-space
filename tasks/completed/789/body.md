---
title: 'daily-fix: Codex twin false-FAIL discipline (sandbox + evidence-quote)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d3fec4fdca59
- daily-auto-filed
created_at: '2026-07-01T06:53:40Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: Codex twin reviewers drive wasted
  reconciler spawns via two false-FAIL modes: (1) huggingface.co DNS fails inside
  the Codex sandbox -> ''HF link liveness unverifiable / data-access-blocked'' raised
  as F'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Instruct the Codex twins (clean-result-critic + interpretation-critic) to (1) tag any HF/network-liveness finding as `sandbox-unverifiable` (advisory, non-blocking) when huggingface.co DNS resolution fails in-sandbox, never FAIL/BLOCKED; and (2) quote the exact body line they claim contains/lacks a fix (evidence-quote requirement) so a hallucinated 'applied' is self-caught before it reaches the reconciler.

## Workflow gap

- **Bug observed:** Codex twin reviewers drive wasted reconciler spawns via two false-FAIL modes: (1) huggingface.co DNS fails inside the Codex sandbox -> 'HF link liveness unverifiable / data-access-blocked' raised as FAIL/BLOCKED; (2) the Codex clean-result-critic hallucinated that a fix was applied (i537 HF path) when the body line was unchanged, producing a false PASS-vs-REVISE disagreement.
- **Evidence:** issues 722 (R3) and 665 on 2026-06-30. Sources: /daily miners batches 04/06.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/codex-clean-result-critic.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: .claude/agents/codex-clean-result-critic.md
- fingerprint: d3fec4fdca59
- source: /daily route-2 (2026-06-30)
