---
title: 'workflow-fix: code-reviewer degenerate-statistic check (observed-vs-null reads)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-10T19:09:53Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1092 9a-ter: degenerate read4c statistic
  (mean of mean-centered projections ≡0) survived 22 review rounds; add a structural-degeneracy
  check to code-reviewer.md'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1092 (emitting agent: experiment-implementer, 9a-ter free-analysis round).

## Goal

Add a degenerate-statistic check to the code-reviewer spec: any observed-vs-null read whose observed statistic is structurally constant (≡0 by construction, e.g. projecting the mean of mean-centered quantities) must be flagged at review time.

## Workflow gap

- **Bug observed:** #1092's banked read-4c trait-per-factor statistic projected the row-mean of mean-centered ANOVA factor outputs onto r_B — identically 0 to machine epsilon by construction — against real-magnitude nulls; the degenerate observed-vs-null read survived 22 code-review rounds and was caught only by the analyzer at interpretation time.
- **Why it is a workflow gap:** the code-reviewer's review checklist has no check that an observed statistic in an observed-vs-null comparison has nonzero structural degrees of freedom; a one-line symbolic sanity check (does the statistic's construction admit variation under the data?) would have caught it at round 1.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/agents/code-reviewer.md`, add to the statistical/eval review items:
+ - **Degenerate-statistic check (observed-vs-null reads):** for every statistic compared against a null band, verify the observed statistic is not structurally constant by construction (e.g. a projection of a mean of mean-centered quantities is ≡0; a correlation of a constant vector is undefined). Trace the statistic's construction symbolically; a structurally-zero observed value against a real-magnitude null is a FAIL (`substantive`).

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Grep the workflow surface for sibling review-lens files that carry the same observed-vs-null lens (`critic.md` Statistics lens, `statistics-critic.md`) and add the parallel item where it fits.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: pending-wrapper-computed

Surfaced prose (verbatim): "code-reviewer statistic-degeneracy check — the degenerate observed-vs-null read survived 22 review rounds"
