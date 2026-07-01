---
title: 'workflow-fix: selection-symmetric nulls + persist per-draw × per-axis matrix'
kind: infra
tags:
- wf-fix
- wf-fix-fp:228f79a01f8c
created_at: '2026-07-01T00:50:37Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate raised by Statistics critic on task #778 Phase
  2 review (2026-07-01): planner.md has no rule that null-band selection procedure
  must inherit the observed statistic''s max-over-axis selection; three critics independently
  caught the resulting 28-vs-1 asymmetry in #778.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `<!-- workflow-fix-candidate v1 -->` block raised by the Statistics critic (Claude, Codex, Alt Codex all independently converged) during task #778's Phase 2 review (2026-07-01).

## Goal

Add a `planner.md` §6 rule: when the headline statistic is selected over a free axis (layer/cell/k), the null-band procedure must apply the identical selection per draw, OR the axis must be frozen on a held-out split; and the per-draw × per-axis statistic matrix must be persisted so the analyzer can recompute the honest band.

## Workflow gap

- **Bug observed:** #778's plan selected a read-out layer by `max(matched r)` over 28 layers but computed the null bands at that single fixed layer, producing an asymmetric comparison (28-vs-1 chances) that inflates the specificity gap the null battery is supposed to test. Simulated at n=24: single-layer null p97.5 |r| ≈ 0.48 vs honest max-over-layer p97.5 |r| ≈ 0.62 — a real false-positive risk on the headline. Three critics (Statistics Claude, Statistics Codex, Methodology Codex, Alternatives Codex) independently caught it; Methodology Claude and Alternatives Claude missed it. All three reconciled lenses landed at REVISE.
- **Why it is a workflow gap:** `.claude/agents/planner.md` has no requirement that a null/permutation band inherit the SAME selection procedure (max-over-layer, max-over-cell, best-of-k) as the observed statistic it is compared against. This is a recurring free-parameter-selection asymmetry — the persona-vectors read-out regime is the most common site, but any layer/cell/k sweep on a target predictor has the same shape. The planner has no rule pointing at it.
- **Confidence (emitter):** medium (Statistics Claude).

## Proposed change (candidate diff sketch — refine in planning)

To `.claude/agents/planner.md` § "Design your plan" step 6 (Eval / Controls):

```
+ **Selection-symmetric nulls.** If the headline statistic is chosen by
+   max over a free axis (layer, cell, k, seed), every null/permutation draw
+   MUST receive the SAME max-over-axis selection before the band is formed
+   — OR the axis is frozen on a held-out split and observed+nulls read at
+   that fixed value. Persist the per-draw × per-axis statistic matrix as
+   a downstream artifact so the analyzer can recompute the honest band.
```

Also add a matching entry to `.claude/agents/critic.md` Statistics & Measurement lens (item 8 or the CI-methodology row) as a check the critic asserts on plans with a swept axis.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md` (add rule)
- Companion target: `.claude/agents/critic.md` (add matching Statistics-lens check)
- Grep the workflow surface for related patterns (`grep -rln 'max.over\|null.*band\|selection.*policy' .claude/agents/ .claude/rules/ CLAUDE.md`) before editing so the implementer catches sibling files.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `workflow_fix_target: .claude/agents/planner.md` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: 228f79a01f8c

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/planner.md
bug_observed: A plan that layer-selects the observed statistic by max-over-layers can compare it against a null band computed at a single fixed layer, silently inflating the specificity comparison (n=24: 0.48 single-layer vs 0.62 max-over-layer p97.5). Three critics on task #778 independently caught it; two missed it.
why_workflow_gap: planner.md §6 has no requirement that a null/permutation band inherit the SAME selection procedure (max-over-layer, max-over-cell, best-of-k) as the observed statistic it is compared against; this is a recurring free-parameter-selection asymmetry.
proposed_change: Add a planner §6 rule (Selection-symmetric nulls) plus a matching critic.md Statistics-lens check requiring null-band procedures to inherit the observed statistic's selection procedure, OR to freeze the axis on a held-out split, AND to persist the per-draw × per-axis matrix so the analyzer can recompute the honest band post-hoc.
diff_sketch: |
  planner.md §6:
  + **Selection-symmetric nulls.** If the headline statistic is chosen by
  +   max over a free axis (layer, cell, k, seed), every null/permutation draw
  +   MUST receive the SAME max-over-axis selection before the band is formed
  +   — OR the axis is frozen on a held-out split and observed+nulls read at
  +   that fixed value. Persist the per-draw × per-axis statistic matrix as
  +   a downstream artifact so the analyzer can recompute the honest band.
confidence: medium
related_task: #778
<!-- /workflow-fix-candidate -->
