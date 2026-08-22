---
title: 'Repack issue1739_ctxmap once #1739 quiesces (43,905 slots; deferred by G-1739/G-writer
  in #2332)'
kind: infra
tags: []
created_at: '2026-08-18T00:18:08Z'
has_clean_result: false
parent_id: 2332
origin_prompt: 'Deferral recorded at #2332 run completion 2026-08-17: issue1739_ctxmap
  DEFERRED by G-writer (fresh run-signal marker on #1739); re-run affordance filed
  per plan §4.6 G-1739 clause.'
workflow: v1
---
# Repack issue1739_ctxmap once #1739 quiesces (43,905 slots; deferred by G-1739/G-writer in #2332)

## Goal
Complete the one deferred prefix from #2332: repack `issue1739_ctxmap` (43,905 files) on `superkaiba1/explore-persona-space-data` into `issue1739_ctxmap/__packed__/` tar shards + index, then identity-keyed-delete the originals — using the SAME merged tooling, unchanged.

## Context
#2332 repacked 7/8 target prefixes (492,786 slots freed; repo 938,366 → 445,580). `issue1739_ctxmap` was DEFERRED twice by the runner's G-writer gate: a fresh run-signal marker on #1739 (`epm:free-analysis-followup-run` 2026-08-12T15:28:31Z) is newer than #1739's latest done-transition (2026-08-05T22:28:11Z), and at #2332 plan time a live #1739 session existed. Deferral is the sanctioned outcome; this task is the durable re-run affordance.

## Run recipe (all tooling already on main, review-passed in #2332)
1. Preconditions (the runner re-checks all of these itself — G-mover, G-writer four-way, G-1739 live-process): #1739 must have no live session/pod and no run-signal marker newer than its latest done-transition. `issue1739_partial` (36,601 files) remains UNTOUCHABLE — different prefix, held by standing directive.
2. From a worktree at current main: `uv run python scripts/issue2332_repack_prefixes.py run --prefix issue1739_ctxmap` (detached, setsid + pid/log breadcrumbs; staging /mnt/eps-data/thomasjiralerspong/issue2332_repack reusable — state.json there already carries the 7 done prefixes).
3. Then `closeout --out eval_results/issue_2332` to refresh before_after.json, and the AC4 accessor test with `EPM_I2332_EXPECT_PACKED=all` (all 8 then packed).
4. Commit refreshed artifacts to the parent (#2332) eval_results dir; post completion note on #2332 and this task.

## Acceptance
- issue1739_ctxmap contains only `__packed__/*` + logged keepers; ~43,905 slots freed (net of pack files).
- AC4 EXPECT_PACKED=all accessor run passes (8/8 prefixes resolve).
- No reader broken: the fleet-wide `stage_hub_file` packed-fallback already covers exact-file readers; #1739's own scripts recover per the #2332 audit remedy.
