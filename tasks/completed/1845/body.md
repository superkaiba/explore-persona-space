---
title: 'workflow-fix: gotchas — batched np.linalg.solve dies on ONE singular slice'
kind: infra
tags:
- wf-fix
- wf-fix-fp:86ef744e58d3
created_at: '2026-07-30T05:37:50Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1739 r10 (LinAlgError singular
  slice killed 25.2h grid; per-slice pinv fallback recipe)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1739 (emitting agent: experiment-implementer, r10; orchestrator routed per Step 7 failure-lesson action 3).

## Goal

Add a gotchas bullet: numpy batched np.linalg.solve raises ONE LinAlgError for a whole (B,k,k) stack when ANY single slice is singular — wrap batched stacked solves in a batched-first + per-slice pinv fallback with a persisted degenerate flag, never a scale-absorbed ridge jitter or a silent placeholder.

## Workflow gap

- **Bug observed:** A single collinear/constant 3-feature cell killed a 25.2h batched fits grid on task #1739 (src/explore_persona_space/experiments/issue_1739/arms.py run_cell_multi, `beta = np.linalg.solve(ata, atb)` on a (Ly,3,3) stack); the trap generalizes to every batched stacked solve in analysis code and is not documented in gotchas.md.
- **Why it is a workflow gap:** gotchas.md is the codebase-trap registry the implementer/planner load on-demand; the batched-solve single-singular-slice semantics (and the float64 scale-absorption of small additive jitter) is a recurring numerics trap the vectorize-first discipline actively steers code INTO (batched solves are the prescribed shape), so it belongs in the registry with the proven fallback recipe.
- **Confidence (emitter):** high
- verified-at-filing: `grep -in "singular\|batched" .claude/rules/gotchas.md` → 0 hits for a batched-solve singular-slice entry (2026-07-30). Fix recipe + incident record: task #1739 `epm:failure v1` (2026-07-30T05:02:03Z), `epm:failure-lesson v1` (05:35:07Z), fix commit `aff188af67df` on branch issue-1739 (per-slice pinv fallback, array_equal healthy-path pin); agent memory landed on main at `1c577f1105` (.claude/agent-memory/experiment-implementer/feedback_batched_solve_singular_slice_pinv_fallback.md — the gotchas bullet can largely transclude it).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/gotchas.md
+ - **Batched np.linalg.solve dies on ONE singular slice.** numpy's stacked
+   solve raises a single LinAlgError for the whole (B,k,k) batch when any
+   slice is singular (one degenerate cell killed a 25.2h grid, #1739). A
+   small additive ridge jitter (1e-8*I) does NOT protect — feature scale
+   absorbs it in float64. Recipe: batched solve first; on the raise,
+   re-solve slices individually with pinv for the singular ones and persist
+   a per-cell degenerate flag (never a silent placeholder). Worked impl:
+   experiments/issue_1739/arms.py _solve_stacked_normal_eqs (#1739 r10).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md` (one bullet; check LESSONS.md index wording needs no change — gotchas.md is already indexed).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes (lessons-index check unaffected — no new rule file).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 86ef744e58d3

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: A single collinear/constant 3-feature cell killed a 25.2h batched fits grid on task 1739 (arms.py run_cell_multi); the trap generalizes to every batched stacked solve in analysis code and is not documented in gotchas.md.
why_workflow_gap: gotchas.md is the on-demand codebase-trap registry; the batched-solve single-singular-slice semantics + jitter scale-absorption is a recurring numerics trap the vectorize-first discipline steers code into.
proposed_change: Add a gotchas bullet documenting the trap + the batched-first/per-slice-pinv/degenerate-flag recipe.
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->
