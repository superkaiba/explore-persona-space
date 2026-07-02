---
title: 'workflow-fix: verification gates must test the live dispatched path'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ba66618bfbd0
created_at: '2026-07-02T21:00:33Z'
has_clean_result: false
origin_prompt: 'PM-chat root-cause 2026-07-02: --verify-vectorized gated an unused
  helper; launch note claimed vectorized on its green PASS.'
---
## Overview / Motivation
Auto-filed from the 2026-07-02 #779 unfixed-grid launch: the grid's `--verify-vectorized` flag ran `vectorized_mlp_skill.assert_matches_reference()` — a self-check of an UNUSED helper — while the live ridge path (the actual 17k-fit hot loop) had zero verification coverage. The session's launch note claimed "vectorized" on the strength of this green gate; the run was the unvectorized 18-20h path.

## Goal
Rule: any verification/equivalence/vectorization gate flag must exercise the CODE PATH ACTUALLY DISPATCHED, not a sibling helper; a gate that passes without touching the hot loop is a hollow gate and a review blocker.

## Workflow gap
- **Bug observed:** `--verify-vectorized` gated an unused helper; the orchestrator + plan treated its PASS as evidence the grid was vectorized (launch marker: "0-GPU vectorized").
- **Why it is a workflow gap:** no rule tells implementers/code-reviewers that verify-gates must cover the dispatched path; the code-review ensemble passed r6/r7 with the hollow gate in place.
- **Confidence (emitter):** high

## Proposed change (refine in planning)
- `.claude/rules/code-style.md`: add a "verification gates test the live path" bullet (a --verify-X flag must assert on the exact function(s) the run dispatches; asserting on an unused sibling is a FAIL).
- `.claude/agents/code-reviewer.md`: one line in the review checklist — when a diff carries a verify/equivalence flag, confirm the gated function is the one the entrypoint calls.

## Scope / surfaces
- Primary: `.claude/rules/code-style.md`, `.claude/agents/code-reviewer.md`.

## Constraints / invariants
- ruff on touched files; keep the bullet short (the lessons-index stays lean).

## Provenance
- workflow_fix_target: .claude/rules/code-style.md
- origin: PM-chat root-cause 2026-07-02 (#779 hollow --verify-vectorized gate)
