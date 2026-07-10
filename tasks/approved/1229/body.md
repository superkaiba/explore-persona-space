---
title: widen scripts-import-guard scan roots to scripts/**
kind: infra
tags:
- wf-fix
- wf-fix-fp:50420b93a3e6
- daily-auto-filed
created_at: '2026-07-10T06:54:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): 16 files under scripts/
  import scripts.* siblings behind hand-rolled sys.path bootstraps; check_scripts_import_guard
  scopes to experiments/** only'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1175.

## Goal
Widen check_scripts_import_guard scan roots to scripts/** (scan-root addition + live-tree audit of the 16 sites; the module-level-guard clause already handles the scripts/ bootstrap convention).

## Workflow gap
- **Bug observed:** 16 files under scripts/ import scripts.* siblings behind hand-rolled module-top sys.path bootstraps; the scripts-import-guard check (#1175) scopes to src/explore_persona_space/experiments/** only, leaving scripts/** unlinted for the same trap (verified on main 2026-07-09: check_scripts_import_guard docstring + scan roots at workflow_lint.py:5435 name experiments/** only).
- **Why it is a workflow gap:** The lint exists because deferred scripts.* imports crash when cwd/sys.path drifts; the scripts/ tree carries 16 of the same pattern with no coverage.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ scan_roots default += (REPO_ROOT / 'scripts',); audit + allowlist-or-fix the 16 existing sites; extend the fixture tests.

## Scope / surfaces
- Primary target: `scripts/workflow_lint.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 02f1e6a71ea7

Parked prose-followup on #1175, 2026-07-09T14:39:21Z (planner, Phase 1): widen check_scripts_import_guard scan roots to scripts/**. confidence: medium.
