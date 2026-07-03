---
title: 'workflow-fix: wire OOD-folds (LOFO) rule into planner §6 + critic Statistics
  lens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d5e44a0dc14a
created_at: '2026-07-03T02:57:51Z'
has_clean_result: false
origin_prompt: save somewhere to always use LOFO (or just some kind of OOD generalization)
---
## Overview / Motivation

Auto-filed per the workflow-fix protocol from a user-directed standing rule (Thomas, 2026-07-02 chat on #810): "save somewhere to always use LOFO (or just some kind of OOD generalization)". The rule file `.claude/rules/ood-generalization-folds.md` + its LESSONS.md index line landed on main (c865fa65b9); its § Enforcement names planner.md §6 and critic.md Statistics & Measurement lens, but neither file carries the corresponding item yet.

## Goal

Wire the OOD-generalization-folds rule into its two named enforcement surfaces so it binds at plan time.

## Workflow gap

- **Bug observed:** `.claude/rules/ood-generalization-folds.md` § Enforcement points at planner.md §6 + critic.md Statistics lens items that do not exist.
- **Why it is a workflow gap:** an unenforced measurement rule only fires if an agent happens to open it; #810's LOCO-only headline (max-pool 0.826, read-out rho 0.909) was reordered/collapsed by the LOFO re-read (mean best, rho 0.285) — exactly the miss plan-time enforcement prevents.
- **Confidence (emitter):** high

## Proposed change (refine in planning)

- `planner.md` §6 (measurement validity): for every held-out predictive DV, the plan names the sample's grouping axes and includes >=1 GROUP-level fold (LOFO / leave-one-genre-out / corpus transfer) alongside pointwise LOO, or argues genuine iid; every headline states its fold.
- `critic.md` Statistics & Measurement lens: new numbered item — REVISE a held-out-predictive-DV plan with pointwise-only folds and no iid argument (cite `.claude/rules/ood-generalization-folds.md`).

## Scope / surfaces

- Primary targets: `.claude/agents/planner.md`, `.claude/agents/critic.md`
- Grep first: `grep -rln 'LOFO\|leave-one-family' .claude/ CLAUDE.md scripts/` and reconcile any hits.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` passes; keep the rule file, LESSONS.md, planner.md, critic.md mutually consistent.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 / a workflow_fix_target: Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/planner.md,.claude/agents/critic.md
- fingerprint: d5e44a0dc14a

User-directed standing rule, 2026-07-02: "save somewhere to always use LOFO (or just some kind of OOD generalization)"
