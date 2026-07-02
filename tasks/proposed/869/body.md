---
title: 'workflow-fix: broaden vectorize/compute-sizing beyond GD fits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:965c890415f9
created_at: '2026-07-02T15:39:50Z'
has_clean_result: false
origin_prompt: 'User (2026-07-02, #823 chat): ''Then check why this wasn''t caught
  ahead of time and propose a workflow fix''. Root-cause: #823 phase 4 ran ~3780 serial
  full-SVD ridge fits (~12-20h) — plan dropped the body-named fast twin, sized at
  ~2s/fit (real ~125s); every vectorize/compute surface is gradient-descent-scoped,
  the vectorize rule exempts closed-form ridge as ''already cheap'', the compute-deviation
  alarm copied the plan''s basis, and no reviewer verifies body-named helpers are
  actually called.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix investigation on task #823 (emitting agent: orchestrator, via a root-cause Explore subagent; user-directed: "check why this wasn't caught ahead of time and propose a workflow fix").

## Goal

Broaden the vectorize/compute-sizing/review surfaces from gradient-descent-only to any many-cell repeated dense linear-algebra fit, fix the false closed-form-ridge-cheap carve-out, require kernel-derived per-call costs plus smoke extrapolation in the compute-deviation check, and verify plan/body-named helpers are actually called.

## Workflow gap

- **Bug observed:** #823 phase 4 ran ~3780 serial full-SVD ridge fits (~12-20h) because the plan dropped the body-named fast twin and sized at ~2s/fit; no rule/review surface matched (all GD-scoped) and the smoke hid the scaling.
- **Why it is a workflow gap:** four independent surfaces share the same blind spot — (1) `.claude/rules/vectorize-many-cell-fits.md` triggers only on gradient-descent fits and draw batteries, its lines 105-106 explicitly exempt closed-form ridge as "already cheap", and its `paths:` glob does not attach to `src/explore_persona_space/experiments/**`; (2) CLAUDE.md's always-on vectorize bullet + compute-character carve-out and `critic.md` Methodology item 10(iii) all say literally "gradient-descent"; (3) the implementer's existing `epm:compute-deviation` 2× alarm (experiment-implementer.md:424-447) copied the plan's fabricated "~2 s/fit" basis instead of re-deriving the per-call cost (real: ~125 s at N_tr≈4000, H=3584 per the fast twin's own docstring), so a 35-57× deviation computed as ≈1×, and the mid-run recognition was posted as `epm:progress` (not `epm:compute-deviation`), routing around `pivot_criteria.compute_deviation_over_2x`; (4) no review surface verifies a body/plan-named helper is actually imported+called — Codex round-1 plan-adherence blessed the slow import ("ridge, not MLP" ✓) while the body's reuse map named `_ridge_fit_predict_fast` and its equivalence gate, and the plan-compute-sizing "floor cross-check" keys on `planned_wall_h > 12`, i.e. on the wrong estimate itself.
- **Confidence (emitter):** high (each gap is quote-verified with line refs in the #823 investigation; see Provenance).

## Proposed change (candidate diff sketch — refine in planning)

Fix A — `.claude/rules/vectorize-many-cell-fits.md`:
```
- description trigger: "...gradient-descent fits ... AND many-draw closed-form statistical loops..."
+ add third clause: "AND many-cell repeated dense linear-algebra fits — a full svd/eigh/lstsq/GCV-ridge
+ solve looped over fold × layer × arm × trait (a per-cell O(N²·H)/O(N³) factorization is cheap ONCE,
+ ruinous ×thousands serially; batch via Gram/dual-space, shared factorization, or the named fast twin)"
- lines 105-106: "Closed-form ridge / linear LOCO is already cheap ... only the gradient-descent arms need this"
+ "Closed-form ridge is cheap only when the factorization runs O(10-100) times; a full-SVD ridge over
+ fold×layer×arm×trait (~10³ calls, ~125 s/call at N_tr≈4000, H=3584) is many hours serial — use the
+ Gram-space fast twin (#823 incident)."
+ paths: add "src/explore_persona_space/experiments/**"
```
Same widening (replace bare "gradient-descent" with "gradient-descent OR many-cell repeated dense factorization (svd/eigh/lstsq/GCV-ridge over fold×layer×arm)") in: CLAUDE.md vectorize bullet + compute-character carve-out, `critic.md` Methodology item 10(iii).

Fix B — `experiment-implementer.md` compute-deviation check + `code-reviewer.md` Step 0.6:
```
+ implementer: "For any §9 row whose basis is a matrix factorization (svd/eigh/lstsq/ridge), the per-call
+ cost MUST be the observed smoke wall-time for one cell or a FLOP/kernel estimate — NEVER the plan's
+ basis figure; projected_wall = observed_per_cell × total_cells / parallelism."
+ code-reviewer Step 0.6 smoke contract: "report smoke_per_cell_wall × full_cell_count / parallelism as
+ the extrapolated full-scale wall-time vs plan §9; a >2× gap with no epm:compute-deviation row is a
+ substantive FAIL."
```

Fix C — `planner.md` §9 + `.claude/rules/plan-compute-sizing.md`:
```
+ "Serial-fit-loop sizing: any fit/solve inside a fold×layer×arm loop MUST state total call count,
+ ground per-call cost on a cited measurement or FLOP floor (never '~2s' by assertion for a full SVD),
+ and USE any body-reuse-map-named fast/verified-equivalent helper in the pseudocode or state why not.
+ Floor cross-check trigger: total_serial_calls > 500 OR planned_wall_h > 4 (was: planned_wall_h > 12)."
```

Fix D — `code-reviewer.md` (+ `consistency-checker.md` as second home):
```
+ "For each helper the task body reuse-map or plan §4/§10 names by ::fn reference, grep the diff for its
+ import + call site; replacing a named fast/verified-equivalent twin with a slower sibling is a
+ substantive finding unless the plan documents the substitution."
```

## Scope / surfaces

- Primary targets: `.claude/rules/vectorize-many-cell-fits.md`, `CLAUDE.md`, `.claude/agents/planner.md`, `.claude/agents/critic.md`, `.claude/agents/code-reviewer.md`, `.claude/agents/experiment-implementer.md`, `.claude/rules/plan-compute-sizing.md`, `.claude/agents/consistency-checker.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'gradient-descent' .claude/ CLAUDE.md`) and update every hit that expresses the vectorize/compute-character trigger; list them in the plan. Keep `.claude/rules/LESSONS.md` index row wording in sync with the broadened trigger.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/vectorize-many-cell-fits.md, CLAUDE.md, .claude/agents/planner.md, .claude/agents/critic.md, .claude/agents/code-reviewer.md, .claude/agents/experiment-implementer.md, .claude/rules/plan-compute-sizing.md, .claude/agents/consistency-checker.md
- fingerprint: 965c890415f9

Investigation evidence (task #823, 2026-07-02): plan v5 lines 286/299/316/352/516/529 (slow-path pseudocode + "~2s per fit → ~0.35h" + "CPU-milliseconds"); body.md:87/103 (fast twin + equivalence gate named); concerns.jsonl 10-concern set (none touch phase-4 fit speed; `phase3-batch1-cpu-reduction` proves the compute lens fired one phase earlier on the batch-1-forward pattern it knows by name); events.jsonl row 106 (12:31Z `epm:progress` "Compute deviation (mid-run, accepted) ... projected 12-20h vs plan 0.35h", not `epm:compute-deviation`; zero `epm:compute-deviation` markers on #823); row 109 (arm A′ 2h11m measured); `vectorize-many-cell-fits.md` lines 1-17/105-106 + paths glob; CLAUDE.md vectorize bullet + compute-character carve-out; critic.md item 10(iii); experiment-implementer.md:424-447; plan-compute-sizing.md floor cross-check (`planned_wall_h > 12`).
