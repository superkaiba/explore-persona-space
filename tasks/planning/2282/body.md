---
title: 'workflow-fix: land the shared fit-core memory chunk cap on main + decide PR
  #1717 disposition'
kind: infra
tags: []
created_at: '2026-08-14T03:01:10Z'
has_clean_result: false
parent_id: 1336
workflow: v1
---
## Overview / Motivation

Filed from #1336 (2026-08-14). The memory-aware null-draw chunk cap that fixed a 72.24 GiB CUDA OOM in the SHARED pooled-ridge fit core exists ONLY on the `issue-1336-fullcorpora` branch. Any pooled fit run launched from `main` still carries the OOM.

## Goal

(a) Land the fit-core memory fix on `main` correctly, and (b) decide the disposition of PR #1717, whose branch is 12,580 commits behind with ~100 workflow-surface files diverged on both sides.

## Workflow gap

- **Observed:** `scripts/issue825_fit_cells.py` on `main` lacks `_resolve_null_draw_chunk` and `_free_device_bytes` (0 grep hits each). The branch copy adds +212/−6 lines implementing a memory-aware draw-axis chunk cap that bounds peak device memory independent of `n_train` and λ-grid length. Verified on the live run (SLURM 12874): `[fit825] null-draw chunk resolved: step=3 requested=64 draws=20 unit_gb=4.91 free_gb=130.8 factor=6 safety=0.8 dev=cuda:0`, and all 5 previously-OOM-killed units completed.
- **Why it matters beyond #1336:** this fit core is imported by #825 and siblings. Without the fix, a pooled/off-arm-shaped fit from `main` reproduces the failure — a 72.24 GiB allocation against a 139.8 GiB H200 at `n_train=149964, d=4096` with 20 null draws.
- **Why it cannot be a copy:** `main`'s own copy has diverged +9/−2 since the merge base `9800ea4a6a`, so overwriting with the branch version would revert those. Needs a genuine 3-way merge of the single file, plus the repo-root inline payload lint gate (`scripts/*.py`), the branch's 5 accompanying tests (`tests/test_issue1336_null_chunk.py`), and a code review.
- **Numerics constraint:** chunk ≥ 2 must stay bit-identical to unchunked (`array_equal`-pinned on the branch); chunk = 1 differs at the last fp64 ulp (~4.4e-16, allclose-pinned). The cap's clamp floor is 1 by design and a shared-tenant GPU can select it, so the invariance pins must come across with the fix.
- **PR #1717 disposition (the second half, needs a decision):** branch `issue-1336-fullcorpora`, 83 branch-only commits, GitHub `mergeable: CONFLICTING` / `DIRTY`, 12,580 commits behind, and 101 of 102 branch-side `.claude/` changes touch files `main` also changed (269 main-side). Three defensible options: full merge WITH an explicit workflow-file resolution plan; surgical cherry-pick of the code-only commits (leaving the workflow files to `main`); or close the branch as artifact-complete, since every artifact #1336's promoted clean-result rests on is already on `main` (`f22b6dc90d`, `06e841e479`, `49a43b6fe2`, `eff921f7d8`, `3ed54c4e6a`, plus the separately-ported width-cap passthrough `50b22bff84`). Option 1 risks silently reverting other sessions' workflow edits across ~100 fleet-loaded agent/rule/skill files.
- **Confidence:** high — every figure above was read directly from git this session.

## Proposed change (refine in planning)

3-way merge `scripts/issue825_fit_cells.py`'s chunk-cap addition onto `main`'s current copy; bring `tests/test_issue1336_null_chunk.py` (5 tests incl. the chunk-size-invariance pin and the production-shape memory accounting pin at `n_train=149964, d=4096`); audit the 26 `heldout_r2_sweep` consumers for numeric inertness as the branch round did; clear the inline payload lint gate; code review. Then execute the chosen PR #1717 disposition. Do NOT attempt a blind rebase-merge of the branch.

## Scope / surfaces

- Primary: `scripts/issue825_fit_cells.py`, `tests/test_issue1336_null_chunk.py`
- Decision surface: PR #1717 / branch `issue-1336-fullcorpora`
- Evidence: #1336 markers v297 (two-mode diagnosis), v302 (both fixes), v311 (merge state)
