---
title: Extend verify_plan.py with a decision-margin satisfiability probe (margin-vs-baseline
  ceiling)
kind: infra
tags:
- wf-fix-verify-plan-margin-ceiling
created_at: '2026-08-10T23:16:59Z'
has_clean_result: false
origin_prompt: 'Auto-filed workflow-fix from the #2203 full-rerun-bugfix plan-critique
  (2026-08-10): Statistics + Alternatives lenses independently caught an absolute
  reduction margin (>=10pp) exceeding the DV baseline ceiling (7B 9.66% / 32B 4.02%)
  that verify_plan.py c27/check-20 did not flag; 3rd recurrence of the #810 margin-vs-ceiling
  family (#810, #825 v17/c27, #2203 v12).'
workflow: v1
---
# Extend verify_plan.py with a decision-margin satisfiability probe (margin-vs-baseline ceiling)

## Provenance
workflow_fix_target: scripts/verify_plan.py
Surfaced by the #2203 `full-rerun-bugfix` plan-critique (2026-08-10): the Statistics AND
Alternatives lenses INDEPENDENTLY caught that plan v12's §3 verdict lattice registered
H1/H3-confirm as an ABSOLUTE `≥ 10pp` harm reduction while the realized baselines were 7B 9.66%
(48/497) and 32B 4.02% (20/498) — so the confirm branches were arithmetically UNSATISFIABLE and a
false FALSIFY would ship by construction. The mechanical verifier (c27, and the check-20
verdict-lattice check) did NOT catch it.

## Problem
`verify_plan.py` check c27 covers only SAME-LINE explicit ratios; it has no cross-section check that
a registered absolute-reduction decision margin is ACHIEVABLE given the DV's baseline rate. A plan
can register "Δ ≥ N pp reduction vs baseline" with a baseline < N% → the confirm branch can never
fire and the falsify verdict is predetermined regardless of data. This is the THIRD recurrence of
the #810 margin-vs-ceiling family (#810; #825 v17/c27; #2203 v12).

## Proposed fix
Extend `verify_plan.py` (c27, and/or check-20 the verdict-lattice check) with a cross-section
satisfiability probe:
- Extract `≥ N pp` (absolute percentage-point) reduction-margin clauses from the verdict lattice / §3.
- When the DV's baseline rate is stated in-plan (e.g. "baseline ~9.7%") OR recoverable from a named
  prior artifact the plan cites, compare: WARN when the registered absolute margin ≥ the recorded
  baseline rate it subtracts from (the confirm branch is unsatisfiable).
- WARN (not FAIL): baselines are not always in-plan; a WARN with a clear message ("registered
  absolute margin N pp ≥ cited baseline B% — confirm branch unsatisfiable; use a relative margin or
  size the absolute margin to the baseline") lets the planner fix it or carry it with a reason.
- Add a pin test (`tests/test_verify_plan.py`) reproducing the #2203 v12 shape (absolute ≥10pp margin
  vs a 9.7% / 4.0% in-plan baseline → WARN) and a negative case (relative margin, or absolute margin
  < baseline → no WARN).

## Acceptance
- New/extended check WARNs on the #2203 v12 shape and passes clean on a relative-margin or
  baseline-sized-absolute-margin plan.
- `tests/test_verify_plan.py` pin tests added (positive + negative), full workflow-lint + the mapped
  verify_plan tests green.
- No regression to existing c27 / check-20 behavior.

## Notes
Kind: infra (workflow-surface fix; no experiment, no promotable clean-result — completes on the
Step 9c test-verdict path). Neutral-vocabulary task; no harmful-content surface.
