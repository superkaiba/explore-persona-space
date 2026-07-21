---
title: 'workflow-fix: verify_plan WARN when a named regression anchor is neither run
  nor gate-selected'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7ee32b4d1a65
created_at: '2026-07-19T11:46:52Z'
has_clean_result: false
origin_prompt: 'Statistics critic prose follow-up on #1536 round 1: add a verify_plan.py
  check cross-referencing plan-named regression-anchor test files against select_step9c_tests.py
  --map-files output + a direct-import grep, WARNing when a named anchor is neither
  auto-selected nor present in the plan''s exact test command.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up
raised on task #1536 (emitting agent: Statistics & Measurement critic,
adversarial-planner Phase 2 round 1).

## Goal

Add a verify_plan.py WARN check that each plan-named regression-anchor test
file either appears in the plan's exact test command or is returned by
`select_step9c_tests.py --map-files` for the plan's declared touched files
(or direct-imports a touched module).

## Workflow gap

- **Bug observed:** plan-named regression-anchor tests can be neither
  auto-selected by Step 9c (the selector's import map is deliberately
  one-hop) nor present in the plan's exact test command, so the anchor
  claim ships unexecuted. Live instance: #1536 plan v2 §4 claimed
  `tests/test_issue906_marker_mix_budget.py` + `tests/test_issue906_tiny_real_e2e.py`
  "sit on the Step-9c mapped-scan path for sft.py edits" — verified false
  (`select_step9c_tests.py --map-files src/explore_persona_space/train/sft.py`
  returns empty; both files import train_lora only transitively via
  `scripts/issue906_phase1_pilot`). Caught only by the round-1 Statistics
  critic (Must-Fix 1).
- **Why it is a workflow gap:** verify_plan.py mechanically checks plan
  structure but has no check cross-referencing named anchor test files
  against the Step-9c selection surface; the shape is likely to recur on
  infra plans that lean on "the gate will run it".
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'select_step9c_tests' scripts/verify_plan.py` → 0 hits in the named target (absence-of-guard claim; 0-hit in-target IS the evidence — the file's only 'anchor' hits are the unrelated success/kill-anchor internals at L224/247/814/981/1041); `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` → checked at filing (2026-07-19; output recorded in filing-session transcript — no landed anchor-cross-check commit)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

Sketch for the planner: a new WARN-severity check (e.g. c40_regression_anchor_executed):
parse the plan for test-file paths named as regression anchors / in a
"must stay green" clause; parse the plan's exact test command(s) for
explicitly-run test files; for each named anchor not explicitly run, check
whether `select_step9c_tests.py --map-files <plan's declared touched files>`
selects it (or a direct-import grep links it to a touched module); WARN
naming each anchor that is neither explicitly run nor auto-selected.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Secondary: `tests/test_verify_plan.py` (pin the new check), possibly
  `scripts/select_step9c_tests.py` (only if a library entry point is needed
  for the map query — prefer reusing the existing --map-files CLI).
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'select_step9c_tests' .claude/ CLAUDE.md scripts/`) and update
  every hit the plan touches; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- WARN severity, never FAIL (an anchor claim can be legitimately satisfied
  by prose the parser cannot see; fail-open).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; `tests/test_verify_plan.py` stays green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 7ee32b4d1a65

Verbatim surfaced prose (task #1536, Statistics critic round-1 report):
"Follow-ups (orchestrator should consider): a `verify_plan.py` check that
cross-references plan-named regression-anchor test files against
`select_step9c_tests.py --map-files` output + a direct-import grep for the
plan's declared touched files, WARNing when a named anchor is neither
auto-selected nor present in the plan's exact test command (the Must-Fix-1
shape is likely to recur on infra plans that lean on 'the gate will run
it')."
