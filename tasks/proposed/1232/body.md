---
title: composer-side trigger-dense pointers for Codex twins
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-10T06:54:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 3): Codex twin composers share
  the #1058 filter-kill vector; trigger-dense-review.md is role-generic and does not
  bind the composers. Emitter marked CONDITIONAL: file only on recurrence evidence'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1185.

## Goal
Add composer-side trigger-dense-review pointers to codex-code-reviewer.md and codex-clean-result-critic.md — CONDITIONAL: the emitter gates this on recurrence evidence of composer/wrapper filter-kills.

## Workflow gap
- **Bug observed:** Codex twin composers share the #1058 filter-kill vector (composer-side prompt text quoting gated content); trigger-dense-review.md is role-generic and its guidance does not bind the thin composer wrappers. The emitter marked this conditional — 'file only on recurrence evidence' — so this is a judgment call for triage, not an unconditional fix.
- **Why it is a workflow gap:** If composer-side kills recur, each costs a review-round spawn; a one-line pointer in the two composer specs closes it. If they do not recur, the addition is spec noise.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)
(none — one-line pointer to .claude/rules/trigger-dense-review.md in each composer's prompt-composition instructions, contingent on recurrence evidence.)

## Scope / surfaces
- Primary target: `.claude/agents/codex-code-reviewer.md, .claude/agents/codex-clean-result-critic.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/agents/codex-code-reviewer.md, .claude/agents/codex-clean-result-critic.md
- fingerprint: 136169962192

Parked prose-followup on #1185, 2026-07-09T18:34:22Z (Methodology critic, Phase 2, plan v2 review): 'composer-side pointer additions are a one-line follow-up IF wrapper kills recur. Conditional — file only on recurrence evidence.' confidence: low. Routed route-3 (genuine judgment call: the emitter's own condition is unevaluated).
