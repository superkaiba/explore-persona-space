---
title: 'Boot-disk worktree regrowth: land the #2132 data-disk cutover; sweep foreign-issue
  caches in reused worktrees'
kind: infra
tags: []
created_at: '2026-08-27T06:41:18Z'
has_clean_result: false
workflow: v1
---
## What happened

The 2026-08-26 VM storage cleanup (boot disk at 94%, 63G free) measured `.claude/worktrees/` at 362G across 63 worktrees — the single largest consumer on the boot disk. Reaping 16 stale trees (idle 20–77 days, no live processes, rescue branches created) freed ~50G, but the active fleet still holds ~300G on `/`, and it regrows with every spawned issue.

## Diagnosis

- The footprint is NOT tracked data: `git ls-files data/` totals 61M. The bulk per worktree is (a) the `.venv` (~11G, mostly hardlinked to `~/.cache/uv`, so marginal cost is smaller than it looks) and (b) per-issue download caches under `data/issue_<N>/` (multi-GB each).
- Caches outlive their issue and cross worktrees: `worktrees/issue-2569/data/issue_779/` holds 5.7G. Step 8 cleanup (`clean_experiment_downloads.py`) sweeps the worktree's own issue, so a foreign issue's cache inside a reused worktree can linger.
- The designed fix already exists and is stalled: **#2132** (the #681 bind cutover of `.claude/worktrees` onto `/mnt/eps-data`) has sat `proposed`, tagged `daily-held` + `needs-human`, since 2026-08-06. Verified today: `findmnt --mountpoint <repo>/.claude/worktrees` returns nothing (bind NOT live), while `/mnt/eps-data` is mounted with **459G free**. Every worktree byte currently lands on the boot disk.

## Asks

1. Primary: unblock and land #2132 — that alone moves the ~300G active footprint off the boot disk and arms the per-issue ext4 quotas. This needs Thomas (the tag says so).
2. Secondary, independent of the cutover: make the cache sweep catch foreign-issue `data/issue_<M>/` caches inside issue-N worktrees (observed: issue_779 cache in the issue-2569 tree), under the existing evidence-gated rules in `.claude/rules/disk-hygiene.md`.
3. Optional micro-win: `wf-land`-style non-issue worktrees (16G, fully absorbed into main) had no owner and no reaper coverage; consider including clean, absorbed, ageing non-issue worktrees in an existing janitor tier.

## Provenance

Filed from the productivity_app session that ran the 2026-08-26 disk cleanup (883G→665G used). Stale-tree rescue branches: `rescue/dirty-*` in the main repo.
