---
title: 'Sparse-checkout issue worktrees: cut per-worktree disk cost ~14G → ~2G'
kind: infra
tags: []
created_at: '2026-06-11T07:53:37Z'
has_clean_result: false
---
# Sparse-checkout issue worktrees: cut per-worktree disk cost ~14G → ~2G

## Goal

Every `/issue` worktree (Step 4a) and agent-isolation worktree should materialize only the directories issue work actually edits, so 40+ concurrent worktrees cost ~80G instead of ~560G.

## Background (incident 2026-06-11)

VM root disk hit 100% (1.2 GiB free) with 59 worktrees present. Measured: each worktree is a full ~14G checkout, of which `eval_results/` = 11G and `external/` = 2G — history an issue branch essentially never needs materialized. The stale-worktree audit reaps idle ones, but with ~27 parallel sessions the steady-state worktree count stays high; reducing per-worktree cost is the structural lever (the watcher-side auto-remediation fix for the cadence gap is being applied separately).

## Proposal

Create worktrees with `git worktree add --no-checkout`, then enable cone-mode sparse-checkout excluding `eval_results/`, `external/`, and (optionally) `ood_eval_results/` + `.arxiv-papers/` before the first checkout. New files under `eval_results/issue_<N>/` can still be created and committed from a sparse worktree (the sparse cone gates materialization of existing tracked files, not the addition of new ones) — verify this explicitly for the analyzer's figure/eval-JSON commit path.

## Open questions the plan must settle

1. Which paths the analyzer / experiment-implementer / upload-verifier actually READ from the worktree (vs repo root): parent-issue eval JSONs for comparison plots are the risk case. Decide: read those via repo root (`task.py find` style helpers), or sparse-add the specific parent dir on demand (`git sparse-checkout add eval_results/issue_<M>`).
2. Whether the Step 10d merge guards + surgical-checkout path interact with sparse cones (the additive-checkout file list includes `eval_results/issue_<N>/` paths).
3. Whether `worktree_audit.py`'s uncommitted-changes guard behaves identically under sparse checkout.
4. Migration: existing worktrees stay as-is (reaped naturally); only new creations are sparse.

## Acceptance criteria

1. `/issue` Step 4a (SKILL.md) + any helper that cuts worktrees documents and uses the sparse recipe.
2. A fresh issue worktree measures ≤2.5G.
3. End-to-end: an issue cycle (implement → commit figures + eval_results/issue_N → merge) completes from a sparse worktree with no path errors.
4. Worktree audit + Step 10d guards unaffected (tests).

## Notes

- `kind: infra`. Touches `.claude/skills/issue/SKILL.md` (public worktree contract) — flagged greenlight-class per the workflow-fix protocol, hence a proposed task rather than an auto-applied fix.
