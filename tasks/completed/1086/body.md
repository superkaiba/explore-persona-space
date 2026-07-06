---
title: 'workflow-fix: verify_plan.py c12/c18 false-positive on conforming battery
  + Row-coverage blocks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c12c18-false-positive-833v8
created_at: '2026-07-06T07:20:39Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #833 adversarial-planner Phase 1.5.0 —
  see body Provenance'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #833 (emitting agent: /issue orchestrator, adversarial-planner Phase 1.5.0).

## Goal

Fix verify_plan.py checks c12_battery_multiplier and c18_paired_contrast_source_coverage so conforming blocks phrased in reasonable variant forms PASS instead of false-FAILing.

## Workflow gap

- **Bug observed:** #833 plan v8 carries a complete battery-multiplier block (draws x batteries(arms+paired diffs) x layers = 60,000 calls, per-call cost, projected wall, named batched helper, rule citation, measured precedent) and a conforming non-fenced `Row-coverage:` line naming both arms' stores — yet both checks still FAIL after a targeted mechanical bounce that implemented the checks' own remedy text verbatim.
- **Why it is a workflow gap:** the Phase 1.5.0 mechanical pre-pass burns planner bounces + orchestrator override cycles on conforming plans; c12 appears to require a literal `cells`/`folds` factor token (the #833 battery legitimately multiplies arms+diffs x layers, drawing over cached per-cell arrays), and c18 appears to require the Row-coverage line within some proximity window of the registered-pair sentence (in v8 it sits in the same §4 phase block, ~20 lines below the hypothesis).
- **Confidence (emitter):** medium (exact regex cause unconfirmed — diagnose from verify_plan.py source; the false-positive fact is confirmed by the v8 text).

## Proposed change (candidate diff sketch — refine in planning)

- c12: accept any multiplier arithmetic with >=2 draw-bearing factors + a projected-wall clause + a batched-commitment sentence (named helper OR vectorization statement), regardless of which axis names appear (cells/folds/batteries/arms/layers).
- c18: widen the Row-coverage proximity window to the enclosing section (or whole plan) when exactly one registered paired contrast exists; keep the per-arm naming requirement.
- Add regression fixtures from #833 plans v7/v8 (the bounced-and-still-failing texts).

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'battery_multiplier\|paired_contrast_source_coverage' .claude/ CLAUDE.md scripts/ tests/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; tests/test_verify_plan.py updated with the new fixtures.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: c12c18-false-positive-833v8

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_plan.py
bug_observed: c12_battery_multiplier + c18_paired_contrast_source_coverage false-FAIL a conforming plan (#833 v8) after a bounce that implemented the checks' own remedy text verbatim
why_workflow_gap: the mechanical pre-pass burns bounce + override cycles on conforming plans; regexes appear over-anchored (c12 on a literal cells/folds token, c18 on proximity to the registered-pair sentence)
proposed_change: widen c12 to any >=2-factor draw arithmetic + wall + batched-commitment; widen c18's Row-coverage window to the enclosing section; add #833 v8 regression fixtures
diff_sketch: |
  - c12: require (>=2 of {draws, cells, folds, batteries, arms, layers} factors) AND a projected-wall token AND (named .py helper OR 'vectoriz' token)
  - c18: search the whole enclosing H2/H3 section (or plan when single contrast) for a line matching ^Row-coverage:
  + tests: fixtures from tasks/.../833/plans/v8.md lines 80 + 149
confidence: medium
related_task: #833
<!-- /workflow-fix-candidate -->
