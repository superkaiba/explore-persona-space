---
title: 'daily-fix: name out-root mount binding in Methodology lens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bc04bc5d6dc9
- daily-auto-filed
created_at: '2026-07-17T06:51:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the Methodology-lens items
  in critic-lens-reference.md do not name out-root mount binding after #1414 landed
  its plan-compute-sizing rule block — ''critic-owned'' enforcement is asserted without
  a critic-surface edit, so the named owner never reads for it'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1414 (Alternatives critic). #1414 is completed — it landed the rule block but not the critic-surface edit.

## Goal

Close the enforcement gap between the #1414 mount-binding rule and the critic lens that is supposed to own it.

## Workflow gap

- **Bug observed:** the Methodology-lens items in critic-lens-reference.md do not name out-root mount binding after #1414 landed its plan-compute-sizing rule block — 'critic-owned' enforcement is asserted without a critic-surface edit, so the named owner never reads for it
- **Why it is a workflow gap:** A rule whose named enforcement owner has no matching lens text is unenforced in practice (the lens reference is what the critic composes from).
- **Confidence (emitter):** low (emitter) — concrete file + change, filed per the 2026-06-11 standing directive
- verified-at-filing: `grep -in 'mount' .claude/rules/critic-lens-reference.md` -> 3 hits, all in the measurement-band item (L364/L366/L686) — none about out-root mount binding (absence claim binds); `git log --oneline --since='7 days ago' -- .claude/rules/critic-lens-reference.md` -> 7a583a417d / f5b533aff2 / 86e8e1a988 (unrelated lens edits, no mount-binding item)

## Proposed change (candidate diff sketch — refine in planning)

Add an out-root mount-binding clause to the Methodology lens section-9 compute-sizing item, cross-referencing .claude/rules/plan-compute-sizing.md's #1414 block.

## Scope / surfaces

- Primary target: `.claude/rules/critic-lens-reference.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: bc04bc5d6dc9



