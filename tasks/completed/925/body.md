---
title: 'workflow-fix: verify_plan.py WARN check for incoherent hypothesis confirm/falsify
  branches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:30c11502c603
created_at: '2026-07-03T11:23:26Z'
has_clean_result: false
origin_prompt: 'Codex statistics critic on #922 r1 surfaced: H4 verdict-lattice ambiguity
  (confirm/falsify jointly satisfiable, unpinned layer set) belongs in a recurring
  workflow-surface verifier (verify_plan.py).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #922 (emitting agent: codex-critic, statistics lens — surfaced as verdict prose; orchestrator synthesized the candidate).

## Goal

Add a WARN-level hypothesis-branch coherence check to scripts/verify_plan.py flagging confirm/falsify wordings jointly satisfiable by one outcome or lacking a pinned scope.

## Workflow gap

- **Bug observed:** plan v2 for #922 carried H4 confirm and falsify wordings jointly satisfiable by one curve ("decays toward/below it by k = 32" confirm vs "stays above the frozen null through k = 32" falsify — a curve above-frozen-but-declining satisfies both), plus an unpinned layer set ("mid/late layers"); verify_plan.py has no decision-branch mutual-exclusivity check, so the ambiguity reached the critic ensemble.
- **Why it is a workflow gap:** the mechanical plan verifier checks section presence but not decision-branch coherence, a recurring defect class (workflow.yaml § pivot_criteria.plan_contradiction_replan; incident #488 round 10 — contradictory gate set reached execution; #922 H4 caught only by the Codex statistics critic).
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)

+ In verify_plan.py, add check `cN_hypothesis_branch_coherence` (WARN, never FAIL — LM judgment stays with the critics):
+   for each `**H<k>` block in §Hypothesis: extract confirm/falsify sentences;
+   WARN if both mention the same bounded quantity+horizon token (e.g. "k = 32")
+   with non-mutually-exclusive comparators (toward/below vs above/stays above), or
+   if the branch names a vague scope token ("mid/late layers", "most layers")
+   with no explicit layer list / numeric threshold nearby.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'verify_plan' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Add matching tests in tests/test_verify_plan.py.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- WARN-level only: a heuristic text check must never hard-FAIL a legitimately-worded plan.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 30c11502c603

Surfaced prose (codex-critic statistics verdict, #922 r1): "H4 has a contradictory verdict cell ... Fix by making the branches mutually exclusive ... This kind of verdict-lattice ambiguity belongs in a recurring workflow-surface verifier. — mechanizable: yes. Check sketch: regex H4 for paired phrases `toward/below` and `stays above.*through k = 32`; fail if no explicit numeric/fixed layer list is present in the H4 sentence or nearby binding text."
