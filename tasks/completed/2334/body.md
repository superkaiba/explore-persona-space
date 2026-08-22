---
title: 'Register eval_results/issue_1739 cone in tests/sparse_cones.txt (deterministic
  sparse-worktree test FAIL, #671 class)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-17T00:05:02Z'
has_clean_result: false
origin_prompt: 'experiment-implementer workflow-fix-candidate during #2329 units 1+5
  (2026-08-16): test_issue1739_bareq_rung_banks fails in every sparse worktree — eval_results/issue_1739
  not in sparse_cones.txt'
workflow: v1
---
## Goal

Register `eval_results/issue_1739/` in `tests/sparse_cones.txt` so the full pytest suite passes in sparse worktrees (the #671 class).

## Context

Found twice during #2329's implementation (units 1 and 5, 2026-08-16): `tests/test_issue1739_bareq_rung_banks.py::test_load_rung_map_real_body_reads_the_committed_labeling` hard-reads `eval_results/issue_1739/dv_dataset/evil/labeling.json`, but `eval_results/issue_1739/` is not registered in `tests/sparse_cones.txt`, so the test deterministically FAILs in every sparse worktree (confirmed: fails in the issue-2329 worktree, passes at the full repo root — the file exists and is committed).

## Fix

Add the `eval_results/issue_1739/` cone (or the narrower `eval_results/issue_1739/dv_dataset` subpath) to `tests/sparse_cones.txt` so `new_worktree.sh` pre-adds it. Verify by running the test in an existing sparse worktree after `git -C <wt> sparse-checkout add`.

<!-- workflow-fix-candidate v1 fingerprint=sparse-cones-missing-issue1739-labeling target_file=tests/sparse_cones.txt -->
