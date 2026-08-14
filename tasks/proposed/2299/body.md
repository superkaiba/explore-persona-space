---
title: 'workflow-fix: verify_plan WARN check for sub-resolution judge-pilot sizing
  (per-arm draws below floor(1/threshold)+1)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T15:58:47Z'
has_clean_result: false
origin_prompt: 'statistics-critic prose follow-up during #2162 turn-boundary-multipatch
  post-approval panel: plan v7 gated parse-fail < 2% per arm on 150 draws over 3 arms
  (~30/arm) vs the 51 needed at that threshold; recurred post-#2124, fully mechanical,
  wants a verify_plan.py WARN check'
workflow: v1
---
# workflow-fix: `verify_plan.py` WARN check for sub-resolution judge-pilot sizing (per-arm draws below `floor(1/threshold)+1`)

## Goal

Add a WARN-only `verify_plan.py` check that flags a plan whose judge-pilot gate is UNSATISFIABLE BY ARITHMETIC — the per-arm effective draw count sits below the resolution its own pass/fail threshold requires — so this class is caught at plan time instead of by a reviewer lens (or, worse, by a false HALT mid-round).

## The defect class

`.claude/rules/llm-judging.md` rule 26 (#2021, sizing clause #2124) requires every ≳5,000-call judge wave to be pilot-gated before the production dispatch. A plan hand-specifies the pilot's draw budget and its per-arm gate threshold independently. When

    per_arm_effective_draws  <  floor(1 / per_arm_threshold) + 1

the gate cannot work in either direction:

- ONE parse failure already exceeds the threshold, so the gate FAILs as a pure granularity artifact and freezes a production wave that is fine; and
- a zero-failure PASS carries no evidence the true rate is below threshold — the observed rate simply has no resolution at that sample size.

Either way the gate is decorative-or-harmful rather than protective, which is the same structural defect as a null-band-free selection read: a check whose arithmetic cannot support the verdict it issues.

## Founding + recurrence

- **#2124** established the sizing clause.
- **#2162 plan v7** (`turn-boundary-multipatch` follow-up round, post-approval panel 2026-08-14) reproduced it POST-#2124: §7.3 gated `parse-fail < 2% per arm` on 150 total draws spanning 3 arms ⇒ ~30 value-rubric draws per arm, against the 51 required at a 2% threshold (`floor(1/0.02)+1`). Caught by the `statistics-critic` as must-fix 4 and confirmed by the orchestrator's own arithmetic; the plan's wave-2 was 21,060 calls, so a granularity-artifact HALT would have frozen the entire depth read.

The recurrence after the rule already existed is the argument for a mechanical check: the rule tells an author to pilot-gate, but nothing checks that the two numbers they choose are mutually satisfiable.

## Fix

Add a WARN-only check to `scripts/verify_plan.py` (next free check id, following the existing c43/c46/c50 WARN-only precedents):

1. Parse the plan for a judge-pilot gate specification — the draw budget, the per-arm threshold, and the arm/rubric count over which the budget is split. The #2162 v7 shape (§7.3 "150 draws … parse-fail < 2% per arm", with the arm count derivable from the same section's rubric enumeration) is the reference fixture; expect prose variation and prefer a conservative parse that stays silent when it cannot confidently extract all three numbers.
2. WARN when `per_arm_effective_draws < floor(1/threshold) + 1`, naming all three parsed numbers, the computed floor, and the suggested budget.
3. Point the message at `eval.judge_pilot.judge_pilot_gate`, which already refuses unsatisfiable configs at config time — the check should route authors toward the existing helper rather than toward a hand-recomputed budget.

**WARN-only is deliberate and load-bearing.** `verify_plan.py`'s no-flags run feeds the Step 9c gate; a FAIL-posture check with a prose-parsing front end is the #1388 fleet-wedge shape. The binding gate stays with the `statistics-critic` lens; this check exists to catch the class earlier and cheaply.

Pin with a test carrying the #2162 v7 sizing as a fixture (150 draws / 3 arms / 2% ⇒ WARN) plus a satisfiable counter-fixture (≥51 per arm ⇒ silent) and an unparseable-plan fixture (⇒ silent, no false WARN).

## Provenance

Surfaced as an explicit workflow-surface prose follow-up by the `statistics-critic` during #2162's `turn-boundary-multipatch` post-approval critic panel, 2026-08-14, under its own verdict's "Workflow-surface prose follow-up (per workflow-fix-on-bug; orchestrator routes, not me)" heading — correct: subagents never file, the orchestrator routes (CLAUDE.md § Workflow-fix-on-bug protocol, surfaced-prose clause). The underlying must-fix was independently confirmed by the orchestrator before filing: plan v7 line 481-484 specifies the 150-draw / 2%-per-arm gate, and 150 draws over 3 arms cannot resolve a 2% threshold.

Dedup: distinct from #2298 (filed the same session — stale `fellows-first` auto-lane prose in `critic-lens-reference.md`); different target file, different defect class.

`workflow_fix_target: scripts/verify_plan.py`
