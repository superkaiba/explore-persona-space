---
title: 'workflow-fix: verify_plan exit-0 baseline check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:795cfd30d31e
- daily-auto-filed
created_at: '2026-07-16T07:19:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): plan sec6 exit-0 criteria
  on pre-existing-red repo-wide lint'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked FORMAL candidate block on task #1365 (emitting agent: statistics-critic, round 1; prior instance #584).

## Goal

Add a `verify_plan.py` check that flags a plan §6 success criterion asserting `exit 0` on a repo-wide lint/suite command unless the criterion names a plan-time baseline capture or a scoped invocation — so jointly-unsatisfiable pass conditions (repo-wide lint pre-existing-red on origin/main) stop reaching the critic round.

## Workflow gap

- **Bug observed:** #1365's plan v1 §6 asserted `workflow_lint.py # exit 0` as a success criterion while the no-flags lint is pre-existing-red on origin/main — a jointly-unsatisfiable gate the mechanical pre-pass did not catch (the statistics critic caught it in round 1; prior instance #584).
- **Why it is a workflow gap:** verify_plan.py has no check that an exit-0 criterion on a repo-wide command names a baseline capture or scoping, so unattainable pass conditions recur across plans.
- **Confidence (emitter):** medium (formal block)
- verified-at-filing: `grep -n 'no-flags\|exit 0\|exit_0' scripts/verify_plan.py` → only :142 (WARN-semantics doc line; no §6 exit-0-criterion check exists — absence-of-guard claim, the 0-hit-for-the-guard result IS the evidence) (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

New verify_plan.py check: scan §6 acceptance criteria for `exit 0`-style assertions on repo-wide lint/suite commands (workflow_lint.py, pytest with no path scope); WARN/FAIL unless the criterion also names a plan-time baseline capture (e.g. "vs baseline captured at plan time") or a scoped invocation. Distinct from the c02/#1322 filing (false "in the no-flags default run" claims) — different bug on the same file.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Secondary: `tests/test_verify_plan.py` (pin).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 6207261ead15

parked formal candidate (from #1365 events.jsonl 2026-07-15T22:49:04Z, abridged): "<!-- workflow-fix-candidate v1 --> target_file: scripts/verify_plan.py bug_observed: plan v1's §6 asserted `workflow_lint.py # exit 0` as a success criterion while the no-flags lint is pre-existing-red on origin/main — a jointly-unsatisfiable gate the mechanical pre-pass did not catch (Statistics critic caught it in round 1; prior instance #584). why_workflow_gap: verify_plan.py has no check that a §6 criterion asserting `exit 0` on a repo-wide lint/suite command names a plan-time baseline capture or a scoped invocation, so unattainable pass conditions reach the critic round repeatedly. propos[ed_change: ...]"
