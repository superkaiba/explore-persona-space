---
title: 'workflow-fix: gotchas.md — companion-stat drop-class semantics (zero split-half
  noise floors)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f57b29a31bb0
created_at: '2026-07-22T23:13:05Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: .claude/rules/gotchas.md\n\
  bug_observed: the issue1415 p3 driver died rc=1 because a >20% drop guard calibrated\
  \ for the registered statistic false-fired on the Delta_floor companion's legitimate\
  \ zero-floor degeneracies\nwhy_workflow_gap: gotchas.md has no entry for the zero-noise-floor\
  \ / companion-stat drop-class trap, so future analysis code will re-hit it outside\
  \ experiment-implementer's agent memory\nproposed_change: add a gotchas.md bullet:\
  \ companion statistics need their own drop-class semantics; zero split-half noise\
  \ floors at shared-first-token early positions are a named non-fatal exclusion,\
  \ never an integrity-guard trip\ndiff_sketch: |\n  + new gotchas.md bullet per the\
  \ epm:failure-lesson v1 block on #1415 (see Goal)\nconfidence: high\nrelated_task:\
  \ #1415\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #1415 (emitting agent: experiment-implementer via
the crash-fix failure-lesson block; router: orchestrator).

## Goal

Add a gotchas.md bullet: companion statistics need their own drop-class semantics; zero split-half noise floors at shared-first-token early positions are a named non-fatal exclusion, never an integrity-guard trip.

## Workflow gap

- **Bug observed:** the issue1415 p3 driver died rc=1 because a >20% drop guard calibrated for the registered statistic false-fired on the Delta_floor companion's legitimate zero-floor degeneracies.
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` (the on-demand trap catalog loaded when writing analysis/eval code) has no entry for this trap class; the agent-memory copy reaches only experiment-implementer spawns, not the wider gotchas load surface.
- **Confidence (emitter):** high (root_cause_confirmed: yes in the epm:failure-lesson v1 block on #1415)
- verified-at-filing: `grep -c "noise floor\|nonpositive\|drop-class\|companion stat" .claude/rules/gotchas.md` → 0 hits (2026-07-22) — absence-of-guard claim, 0-hit in-target IS the evidence.

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new bullet (analysis/eval section): split-half noise floors of
+ teacher-forced activations are EXACTLY zero at early answer positions when all
+ draws share their first tokens (deterministic openings — 7/28 pairs on #1415);
+ log-ratio/guard code over floor magnitudes must treat non-positive means as a
+ NAMED non-fatal exclusion class; companion statistics get their OWN drop-class
+ semantics + thresholds (a fail-loud guard calibrated for the registered
+ statistic false-fires on a companion's legitimate degeneracies); regression-test
+ with a shared-first-token fixture.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Cross-check the LESSONS.md index row for gotchas.md stays accurate (`--check-lessons-index`).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes (lessons-index + size budgets).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: f57b29a31bb0

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: the issue1415 p3 driver died rc=1 because a >20% drop guard calibrated for the registered statistic false-fired on the Delta_floor companion's legitimate zero-floor degeneracies
why_workflow_gap: gotchas.md has no entry for the zero-noise-floor / companion-stat drop-class trap, so future analysis code will re-hit it outside experiment-implementer's agent memory
proposed_change: add a gotchas.md bullet: companion statistics need their own drop-class semantics; zero split-half noise floors at shared-first-token early positions are a named non-fatal exclusion, never an integrity-guard trip
diff_sketch: |
  + new gotchas.md bullet per the epm:failure-lesson v1 block on #1415 (see Goal)
confidence: high
related_task: #1415
<!-- /workflow-fix-candidate -->
