---
title: 'workflow-fix: verify_plan fit-family pilot-basis check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e69a8981c488
- daily-auto-filed
created_at: '2026-07-09T07:00:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): No mechanical verify_plan.py
  check asserts that a fit-family §9 wall-time row''s basis names a measured 1-cell
  pilot, a cited prior-issue figure, or an asserted pilot-gate/FLOP-only flag — the
  #1060 measured-pilot rule is reviewer-enforced only.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1060 (recursion-guarded workflow-fix session).

## Goal

Add a verify_plan.py check that every fit-family §9 row's basis names a measured pilot / cited prior-issue figure / pilot-gate flag, designed to resist boilerplate satisfaction (both #1060 round-1 critics flagged a naive regex as boilerplate-satisfiable — the planner may deflect with a reasoned no-change report if no robust mechanical form exists).

## Workflow gap

- **Bug observed:** No mechanical verify_plan.py check asserts that a fit-family §9 wall-time row's basis names a measured 1-cell pilot, a cited prior-issue figure, or an asserted pilot-gate/FLOP-only flag — the #1060 measured-pilot rule is reviewer-enforced only.
- **Why it is a workflow gap:** The #722/#823 serial-fit wall-time blowups came from §9 rows sized on guesses; a doc rule without a mechanical verifier is the recurring 'documented but not enforced' pattern.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)

New check in verify_plan.py: locate §9 rows matching fit-family keywords (fit/ridge/MLP/LOCO/sweep x per-cell), require the same row/paragraph to match a basis pattern (measured pilot <t>s|#<M> figure|pilot-gated|FLOP-only); WARN not FAIL in v1 to bound the boilerplate-satisfiability risk.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- origin: parked candidate on task #1060 at 2026-07-05T20:12:26Z

Verbatim parked note:

parked — running under workflow_fix_target Provenance line (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). source: prose-followup (alternatives critic + codex twin, round 1). Candidate: a mechanical verify_plan.py check asserting every fit-family §9 row's basis names a measured pilot / cited prior-issue figure / pilot-gated (flag asserted or FLOP-only bases). target_file: scripts/verify_plan.py. Both critics note a naive regex is boilerplate-satisfiable — reviewer-grade enforcement retained for now; this candidate is logged for the next non-workflow-fix orchestrator pass, NOT auto-routed.
