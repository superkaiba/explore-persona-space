---
title: 'Step 10d/9c gates: refresh worktree sparse cones from main''s tests/sparse_cones.txt
  before gate pytest (stale-cone false NEW block, #2430 incident)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T05:25:20Z'
has_clean_result: false
origin_prompt: 'Auto-filed by /issue 2430 orchestrator: Step 10d TG gated leg crashed
  on stale worktree sparse cones (eval_results/issue_823 registered on main after
  worktree creation)'
workflow: v1
---
# Step 10d TG gated leg: refresh worktree sparse cones from main's registry before the mapped-test run

<!-- workflow-fix-candidate v1 -->

## Goal

Close the false-block/crash channel where the Step 10d pre-push lint gate's mapped invariant-test (TG) GATED leg runs in an issue worktree whose sparse-checkout set predates a later main-side `tests/sparse_cones.txt` registration — a Step-5a-synced sibling test then reads its committed `eval_results/issue_<M>/` fixture as absent (`FileNotFoundError`), fails on the gated tree only (the merge-base scratch baseline never selects the not-yet-forked file), and classify-new-nodes keeps it blocking as "NEW by construction" (the base-identical synced file sits in the raw three-dot own-diff).

## Incident

Task #2430, Step 10d round 1 (2026-08-21): verdict `crash`. `tests/test_issue823_ladder_fits.py::test_banked_split_json_matches_embedded_constants` + `::test_banked_g1_constants_match_committed_artifact` (Step-5a-synced, base-identical to origin/main) failed ONLY on the gated worktree leg — `scripts/issue823_ladder_fits.py:524 load_banked_fold_means → eval_results/issue_823/ridge_r2_by_arm.json` missing. `eval_results/issue_823` IS in main's `tests/sparse_cones.txt` (line 140), but the issue-2430 worktree was created before that entry landed, so `new_worktree.sh`'s creation-time cone pre-add never saw it. Baseline scratch leg: 3548 passed, green. Lint legs: PASS/PASS. Cost: one full gate round (~50 min wall) + a diagnose/fix/relaunch cycle. Manual remedy that worked: `git -C <WT> sparse-checkout add eval_results/issue_823` (worktree-local state, no commit) → 51/51 green.

## Proposed fix (implementer to refine)

In the Step 10d TG-leg recipe (`.claude/skills/issue/steps/18-step-10d.md`, the mapped invariant-test block) — or as a shared helper beside the Step 5a re-sync, since Step 9c's gate has the same exposure for gate-selected synced sibling tests — add a pre-gated-leg cone refresh: diff main's `tests/sparse_cones.txt` (the repo-root/fetched-origin copy, not the worktree's fork-era copy) against `git -C "$WT" sparse-checkout list` and `sparse-checkout add` any missing registry cones before launching the gated pytest. Idempotent, worktree-local, no commit; a failure degrades loudly to today's behavior (fail-closed NEW block).

## Acceptance criteria

- A worktree missing a registry cone gets it added before the TG gated leg runs (and/or before the Step 9c gate pytest).
- A synced sibling test whose fixture cone is registered on main no longer reds the gate in a stale-cone worktree.
- No behavior change when the worktree's cone set already matches the registry.
- Covered by a test (fixture worktree with a stale cone set).

## Provenance

target_file: .claude/skills/issue/steps/18-step-10d.md
candidate-fingerprint: step10d-tg-leg-stale-sparse-cones-refresh
Filed by the /issue 2430 orchestrator session after the round-1 lint-gate crash verdict root-cause.
<!-- /workflow-fix-candidate -->
