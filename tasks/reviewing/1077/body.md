---
title: 'daily-fix: step9c_baseline dirty-oracle scratch-tree fallbac'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e9cc64d2b7dc
- daily-auto-filed
created_at: '2026-07-06T06:58:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): Two sessions today (#1029
  session 7cb3f598, #1056 session cc6ec8b5) hit ''pristine oracle is DIRTY on code
  paths'' (other sessions'' uncommitted files on the shared repo root, e.g. scripts/issue_642/i642_figures_v4.py)
  -> verdict indeterminate; the follow-on compare emitted non-JSON so the caller''s
  json.load Traceback''d. Both sessions improvised the same recovery: materialize
  a detached clean scratch'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

On MF-4c dirty-oracle, auto-fall-back to running the pristine oracle in a detached scratch worktree at origin/main (or treat root dirt on paths disjoint from the issue diff as still-trustworthy), instead of returning indeterminate; and make every step9c_baseline.py compare exit path emit valid JSON so the caller never crashes parsing the indeterminate case.

## Workflow gap

- **Bug observed:** Two sessions today (#1029 session 7cb3f598, #1056 session cc6ec8b5) hit 'pristine oracle is DIRTY on code paths' (other sessions' uncommitted files on the shared repo root, e.g. scripts/issue_642/i642_figures_v4.py) -> verdict indeterminate; the follow-on compare emitted non-JSON so the caller's json.load Traceback'd. Both sessions improvised the same recovery: materialize a detached clean scratch tree at origin/main (a 62,886-file checkout) as the oracle -> PASS.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`
- Sessions recovered manually at ~8 min each; the recovery is deterministic and belongs in the script.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: scripts/step9c_baseline.py
- source: /daily 2026-07-05 problem sweep (transcript-mined)
