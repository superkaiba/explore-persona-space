---
title: 'workflow-fix: verify_plan flags false no-flags claims'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f6913d3d54b3
- daily-auto-filed
created_at: '2026-07-16T07:19:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): plans claim a --check-*
  flag is in the no-flags default run when absent'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1322 (emitting agent: statistics-critic, #1322 Phase-2 round 1).

## Goal

Add a `verify_plan.py` check that flags any plan acceptance criterion asserting a `--check-*` flag is "included in the no-flags default run" when that flag is absent from `workflow_lint.py` main()'s no-flags dispatch set, and/or add one clarifying line in `workflow_lint.py`'s docstring header stating `--check-references` is NOT in the no-flags set.

## Workflow gap

- **Bug observed:** plan acceptance criteria can claim a `--check-X` flag is "included in the no-flags default run" when it is not (`--check-references` specifically is never bundled and resolves only workflow.yaml § tokens) — a false-PASS verification claim shipped into plan v1 of #1322.
- **Why it is a workflow gap:** the mechanical plan pre-pass has no check for this claim class, so unverifiable/false verification claims reach the critic round repeatedly.
- **Confidence (emitter):** low
- verified-at-filing: `grep -c 'no_flags' scripts/workflow_lint.py` → 48 hits (the no-flags dispatch set exists there); `grep -n 'no-flags' scripts/verify_plan.py` → 0 hits in the named target (absence-of-guard claim — the 0-hit in-target result IS the evidence) (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Add a verify_plan.py check flagging any acceptance criterion asserting a `--check-*` flag is in the no-flags default run when that flag is absent from workflow_lint.py main()'s `or no_flags` dispatch set; and/or one clarifying docstring line in workflow_lint.py stating --check-references is NOT in the no-flags set.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py, scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f6913d3d54b3

- workflow_fix_target: scripts/verify_plan.py

parked prose follow-up (verbatim, from #1322 events.jsonl 2026-07-15T10:35:41Z): "target_file: scripts/verify_plan.py, scripts/workflow_lint.py bug_observed: plan acceptance criteria can claim a --check-X flag is 'included in the no-flags default run' when it is not (--check-references specifically is never bundled and resolves only workflow.yaml § tokens) — a false-PASS verification claim shipped into plan v1 of #1322 proposed_change: add a verify_plan.py check flagging any acceptance criterion asserting a --check-* flag is in the no-flags default run when that flag is absent from workflow_lint.py main()'s 'or no_flags' dispatch set; and/or one clarifying line in workflow_lint.py's docstring header stating --check-references is NOT in the no-flags set. confidence: low. related_task: #1322"
