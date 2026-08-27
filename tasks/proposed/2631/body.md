---
title: Step 10d TG mapped-test leg maps sibling-sync imports via raw own-diff — subtract
  base-identical paths (#2302 class)
kind: infra
tags:
- wf-fix
- 10d-tg-map-base-identical
created_at: '2026-08-27T13:24:00Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from /issue 2616 Step 10d: TG leg false-block
  on sibling-sync-imported issue2569 tests'
workflow: v1
---
# Step 10d TG mapped-test leg feeds sibling-sync imports to --map-files, false-blocking innocent merges

## Goal

Stop the Step 10d pre-push lint gate's mapped invariant-test (TG) leg from false-blocking merges on tests mapped via Step 5a sibling-arm sync imports — base-identical copies of origin/main files the branch never authored.

## Observed

Issue #2616 Step 10d, 2026-08-27. The gate returned `block` with 3 NEW-classified nodes:

- tests/test_issue2569_dw_fleet.py::test_assert_dv3_schema_arm_roster_floor_and_real_cells
- tests/test_issue2569_gateladder.py::test_load_banked_frames_committed
- tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints

None touch #2616's payload (a new orchestrate module + its test + a pods.md bullet). The chain:

1. The pre-launch Step 5a re-sync's sibling-issue arm imported 32 files from origin/main, including the whole scripts/issue2569_* + tests/test_issue2569_* set (issue 2569 landed on main after #2616's fork, so every file was ABSENT at the merge-base).
2. The 10d TG leg feeds the RAW three-dot own-diff to `select_step9c_tests.py --map-files`, which mapped the imported test files (and test_shared_vm_thread_caps via the imported scripts) into the gated run.
3. Gated leg (branch tip, sparse worktree): the issue2569 tests red on missing eval_results data (sparse cone) and a thread-caps offense in main's own issue2569_normalize_passb.py.
4. Baseline leg is cut at the MERGE-BASE (#2348), which predates issue2569 entirely — baseline green — so comm -23 classified every red NEW; classify-new-nodes kept them blocking because the test files sit in the own-diff ("branch-new/payload test, NEW by construction").

#2302 closed exactly this class for the Step 9c selector (`compute_touched` subtracts verified base-identical paths — HEAD blob == origin/main tip blob — reported as `base_identical_excluded`). The 10d TG-map path has no such subtraction: `--map-files` consumes the raw own-diff file, so every sibling-sync import re-enters the gate the moment the sibling issue is newer than the fork.

## Fix shape (for the plan to refine)

Subtract verified base-identical paths from the own-diff BEFORE the `--map-files` call in the Step 10d gate workload (and the Step 9a-ter inline payload gate if it shares the shape) — either by teaching `--map-files` the same base-identity subtraction `compute_touched` already implements (preferred: single source), or by filtering the own-diff file in the gate recipe with the same HEAD-blob == origin/main-tip-blob probe. The classify-new-nodes "file in the own-diff => NEW by construction" doctrine should likewise treat a base-identical own-diff file as NOT payload.

## Workaround used on #2616

Removed the imported issue2569 set from the branch (all 21 files absent at merge-base and byte-identical to origin/main, so the removal empties their net diff; commit 3edbfc64e6), then re-ran the gate once. Works only because the branch did not need those files; a branch whose gated tests genuinely import a newer sibling module cannot take it.

## Provenance

Surfaced by the /issue 2616 orchestrator during its Step 10d merge (gate transcript /tmp/issue-2616-lint-gate.log, verdict block @ e70bef4708; fixed-and-rerun same session). Files of record: #2302 (selector-side base-identity), #2348 (merge-base-pinned baseline + classify-new-nodes), #1972 (sibling-sync arm), .claude/skills/issue/steps/18-step-10d.md (TG leg).
