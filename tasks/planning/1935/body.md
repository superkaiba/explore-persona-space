---
title: 'workflow-fix: exclude plan-declared outputs from carry-over bare-name resolution'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2c17836caa8a
created_at: '2026-07-31T12:01:05Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1902 Step 6a.5: verify_carryover_inputs.py
  bare-name resolver misclassifies a plan''s own output filenames (pilot_report.json,
  layer_sweep.json) as unsynced carry-over inputs resolved against other issues''
  dirs (8 false FAILs, rc=1, overridden with recorded disposition)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1902 (emitting agent: /issue orchestrator, Step 6a.5 gate run).

## Goal

Teach `scripts/verify_carryover_inputs.py`'s bare-name resolver to exclude a plan's OWN declared output filenames from carry-over input resolution (or downgrade bare-name cross-issue resolutions to WARN when the resolved issue dir is not cited by the plan).

## Workflow gap

- **Bug observed:** on #1902's plan v4, 8 bare-name citations that are the plan's OWN OUTPUT filenames (`pilot_report.json` written by its P1 phase; `layer_sweep.json` written by its P4 to `eval_results/issue_1902/fits/`) were resolved by `resolve_bare_name` against OTHER issues' committed eval_results dirs (issue_1739/1768/1776/779) and FAILed the rsync-lane gate (`rsync-lane-not-synced`, rc=1) as unsynced carry-over inputs the run never reads.
- **Why it is a workflow gap:** the gate's citation scanner cannot distinguish a plan's consumed inputs from its produced outputs, so any plan whose output filename collides with a committed file in ANY other in-scope issue dir false-FAILs the Step 6a.5 second stanza — forcing either a wrong `--extra-sync-path` (staging foreign files onto the lane) or a manual override (what #1902 did, recorded on its events.jsonl at 2026-07-31T11:59:46Z).
- **Confidence (emitter):** high
- verified-at-filing: `uv run python scripts/verify_carryover_inputs.py --plan tasks/running/1902/plans/plan.md --issue 1902 --lane rsync` → rc=1, 8 FAIL rows all bare-name output-filename resolutions (2026-07-31); resolver sites: `grep -n 'bare_name' scripts/verify_carryover_inputs.py` → extract_bare_names L97, resolve_bare_name L122, call site L304-310 (3 hits in 1 file)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # In the citation scan: collect the plan's declared OUTPUT names first —
+ #  - phase_outputs: blocks' `outputs:` lists and `sentinel:` paths
+ #  - §6.5 primary_deliverable globs
+ #  - off_pod_phases `outputs:` paths
+ # and drop any bare name whose basename matches a declared output.
+ # Fallback: when a bare name resolves ONLY under issue dirs the plan never
+ # cites (issue_<M> not mentioned in the plan text), downgrade FAIL -> WARN
+ # with reason `bare-name-foreign-issue-resolution (possible own-output)`.
```

## Scope / surfaces

- Primary target: `scripts/verify_carryover_inputs.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'resolve_bare_name' scripts/ .claude/ tests/`) and update every hit;
  list them in the plan (expect: the script + its tests).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The gate must keep FAILing on GENUINE unsynced inputs (the #1689 class) — the
  fix narrows false positives without opening the fail-open direction; add
  regression tests for both directions.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_carryover_inputs.py
- fingerprint: 2c17836caa8a

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_carryover_inputs.py
bug_observed: bare-name citations that are the plan's own output filenames (pilot_report.json, layer_sweep.json) were resolved against OTHER issues' committed eval_results dirs and FAILed the rsync-lane gate as unsynced carry-over inputs on #1902
why_workflow_gap: the citation scanner cannot distinguish consumed inputs from produced outputs, so any output filename colliding with another issue's committed file false-FAILs the Step 6a.5 gate
proposed_change: exclude plan-declared OUTPUT filenames (phase_outputs / §6.5 globs / files the plan writes) from carry-over input resolution, or downgrade bare-name cross-issue resolutions to WARN when the resolved issue dir is not cited by the plan
diff_sketch: |
  + collect declared outputs from phase_outputs/off_pod_phases/§6.5 blocks
  + drop bare names whose basename matches a declared output
  + else downgrade foreign-issue-only bare-name resolutions FAIL -> WARN
confidence: high
related_task: #1902
<!-- /workflow-fix-candidate -->
