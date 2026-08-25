---
title: 'workflow-fix: Step 5a FAMILY_lint omits scripts/pre_split_review_guard.py
  — half-sync reds the step9c lint pins in pre-#2294 worktrees'
kind: infra
tags:
- wf-fix
- spec-freshness
created_at: '2026-08-23T11:56:47Z'
has_clean_result: false
origin_prompt: 'Found by the #2305 orchestrator: gate red test_workflow_lint_default_exits_zero
  + test_review_lens_passes_on_live_tree against a fork-era pre_split_review_guard.py
  after a lint-family sync brought main''s newer linter + pin test.'
workflow: v1
---
# Step 5a FAMILY_lint omits scripts/pre_split_review_guard.py — half-sync reds the Step 9c lint pins in every pre-#2294 worktree

## Goal

Add `scripts/pre_split_review_guard.py` to the Step 5a spec-freshness FAMILY_lint membership (the `FAMILY_OF` map in `.claude/skills/issue/steps/09-step-5.md`, mirrored in the Step 10d copy if present), so the guard helper and the surfaces that execute/pin it move as one vintage.

## The gap

Commit `0ad97f8f8a` (issue-2294) updated three coupled surfaces together on main: `scripts/workflow_lint.py` (new check: the guard CLI must name `IMPLEMENTER-MARKER-MISSING`), `tests/test_workflow_lint_pre_split_guard.py` (pin test executing the WORKTREE copy of the helper), and `scripts/pre_split_review_guard.py` (the helper itself). The first two are FAMILY_lint members (`scripts/workflow_lint.py`, `:(glob)tests/test_workflow_lint*.py`); the helper is NOT in any family and NOT covered by the sibling-issue globs (not `issue<N>_`-named). A worktree cut before `0ad97f8f8a` that runs the Step 5a sync therefore gets main's NEW linter + NEW pin test against the FORK-ERA helper — the exact #1824/#1860 half-sync class the family mechanism exists to prevent.

## Observed instance (#2305 gate, 2026-08-23)

Worktree issue-2305 (cut ~20 min before `0ad97f8f8a` landed) synced the lint family, then its Step 9c gate red TWO nodes green on main: `tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero` (the no-flags lint exits 1 on the worktree tree: "scripts/pre_split_review_guard.py: no longer names 'IMPLEMENTER-MARKER-MISSING' (#2294)") and `tests/test_workflow_lint_pre_split_guard.py::test_review_lens_passes_on_live_tree`. Cost: one full 75-min gate round + a manual reconcile (the #2305 session hand-synced the helper with the sync-anchor subject, commit `3d9ce7bc7e` on issue-2305).

## Fix

1. Add `FAMILY_OF["scripts/pre_split_review_guard.py"]="lint"` (and the SPECS list entry) in `.claude/skills/issue/steps/09-step-5.md`; verify `test_step10d_family_atomicity_matches_step5a` still passes (the Step 10d copy must not drift).
2. Sweep for other same-class strays: any `scripts/*.py` helper executed or importlib-loaded by a synced pin-test glob but absent from every family (the #1972 selector precedent: `select_step9c_tests.py` joined FAMILY_lint for exactly this reason).
3. Update the guard (20) exempt/membership lists if the completeness pin requires it.

## Provenance

Found by the #2305 orchestrator during its round-1 Step 9c gate (lint probe on the worktree tree + `git log origin/main -- scripts/pre_split_review_guard.py`). Not a #2305 deliverable — filed separately per the workflow-fix-on-bug protocol.
