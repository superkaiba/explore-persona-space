---
title: 'workflow-fix: null-band-vs-DV-ceiling informativeness check in selection-symmetric-nulls'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9e29036380f1
created_at: '2026-07-03T08:17:08Z'
has_clean_result: false
origin_prompt: 'interpretation-critic prose follow-up on #810: require reporting a
  registered null band against the DV''s achievable ceiling; band above ceiling =
  uninformative-by-construction, narrate as failure-to-reject'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the interpretation-critic on task #810 (round `ultrachat-genre-summary-sweep`, 2026-07-03).

## Goal

Require every registered selection-symmetric null-band read to report the band alongside the DV's achievable ceiling / dynamic range, and to narrate a band at-or-above the ceiling as failure-to-reject (uninformative-by-construction), never as evidence of absence/reversal.

## Workflow gap

- **Bug observed:** #810's H1-g(iii) registered read shipped a difference-null band whose 97.5% upper bound (0.800) exceeds the DV's achievable estimator ceiling (~0.857 max skill, so a max-difference statistic can essentially never clear it — even the parent round's observed +0.209 Betley effect would fail); the interpretation initially narrated the p=0.634 outcome as a clean ordering fail until the interp-critic caught it.
- **Why it is a workflow gap:** `.claude/rules/selection-symmetric-nulls.md` mandates selection-inheriting nulls but never requires checking or reporting the resulting band against the DV's achievable range, so an uninformative band passes every gate silently and gets narrated as evidence. Likely recurs for any max-selected difference statistic over a free axis.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In .claude/rules/selection-symmetric-nulls.md, add a "Band-vs-ceiling informativeness check" section:
+ - every registered null-band read REPORTS the band upper bound next to the DV's achievable ceiling (max attainable value of the statistic given the estimator/data);
+ - band upper >= ceiling (or >= the largest previously-observed in-genre effect) => the test is uninformative-by-construction: the read MUST be narrated as failure-to-reject, and the band drawn in the figure;
+ - planner (§7 gates) and analyzer/interp-critic lens text cross-reference the check.

## Scope / surfaces

- Primary target: `.claude/rules/selection-symmetric-nulls.md`
- Sibling references found by grep (update where the lens/section text quotes the null-band recipe): `.claude/rules/LESSONS.md`, `.claude/rules/planner-section-reference.md`, `.claude/rules/critic-lens-reference.md`, `.claude/rules/ood-generalization-folds.md`, `.claude/rules/vectorize-many-cell-fits.md`, `.claude/agents/critic.md`
- Grep the workflow surface for `selection-symmetric` and update every hit that states the null-band contract; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; LESSONS.md index stays consistent.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/selection-symmetric-nulls.md
- fingerprint: 9e29036380f1

Surfaced prose (interpretation-critic on #810, verbatim): "`.claude/rules/selection-symmetric-nulls.md` could require reporting a registered null band against the DV's achievable ceiling — a band above the ceiling makes the test uninformative-by-construction and must be narrated as failure-to-reject (fires here; likely recurs for max-selected difference statistics)."
