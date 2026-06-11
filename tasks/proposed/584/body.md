---
title: 'Port the #534 distinct-id trajectory-eval fix + source-manifest guard to main
  with a regression test'
kind: infra
tags: []
created_at: '2026-06-11T03:15:46Z'
has_clean_result: false
parent_id: 549
---
## Goal

Port the #534 distinct-lora_int_id trajectory-eval fix (commit 298877f9c) and its source-manifest guard to main with a regression test, so no future multi-checkpoint vLLM eval can silently serve a stale adapter.

## Background

Task #549's audit found the bug class STILL LIVE on main: `src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py:425` constructs `LoRARequest(..., lora_int_id=1, ...)` inside the checkpoint loop under one `LLM(max_loras=1)` engine. The fix exists on branches issue-534/issue-555 (commit 298877f9c: distinct ids per checkpoint + `assert_source_delta_g_matches_manifest` + `--eval-only` / `--fraction-manifest` paths) and is NOT an ancestor of main. `eval_guard.py` is already partially on main, so the port must reconcile partial overlap rather than blind cherry-pick. Source of truth for the fixed driver: `issue-534:src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py` (branch-only — do not grep main for it).

## Asks

1. Port 298877f9c's driver fix + tests to main (files: the eval_trajectory.py distinct-id fix, scripts/i504_eval_trajectory.py threading, tests/test_i534_source_manifest_guard.py), reconciling the eval_guard.py overlap.
2. Add a regression test that FAILS on the old constant-id construction and PASSES on the fixed one.
3. Bundle the one-line `q_test_extended_50.json` SHA-pin in `scripts/issue532_predictor_stress.py` (code hygiene flagged by the same audit).

## Acceptance

- `git merge-base --is-ancestor <port-commit> main` true.
- Ported + new regression tests pass.
- A future multi-checkpoint trajectory eval from main constructs distinct lora_int_id per checkpoint.

## Cost

0 GPU-hours; ~1-2 h code work on the VM (no pod).

## Provenance

Filed from #549's epm:follow-ups v1 proposal 1 (auto_run: yes, question_relation: substantially-different, execution: filed-only — runs only on manual triage via /issue).
