---
title: 'daily-held: triage root-sync KEPT stash 319c2bf16e7c'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-31T06:58:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 3): stash@{0} (319c2bf16e7c)
  kept by sync_repo_root with a rescue patch owed manual triage; re-flagged by >=6
  sessions on 07-30 and still pending.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 as a route-3 needs-human item (judgment-call carve-out: destructive/irreversible — dropping a stash is not revertable by `git revert`).

## Held decision

The shared repo root carries an un-reapplied autostash: `stash@{0}` (`319c2bf16e7c`), kept by `sync_repo_root.py` because `apply --check` was dirty; rescue patch written to `~/.task-workflow/root-sync-rescue/stash-319c2bf16e7c.patch` (9.3 KB, mtime Jul 30 23:30 PT). It was re-flagged by at least 6 sessions across 2026-07-30 (probed: the patch file + `git stash list` stash@{0} both still present at /daily compose time). Yesterday's #1870 (open) covers the SURFACING mechanics; nobody owns the actual triage.

## What needs Thomas (or an explicitly-directed session)

1. Inspect the rescue patch: `less ~/.task-workflow/root-sync-rescue/stash-319c2bf16e7c.patch` — determine whether its hunks are already landed on main (likely: content another session committed after the autostash) or carry real un-landed work.
2. If landed/junk → `git stash drop stash@{0}` (destructive — the carve-out reason this is held).
3. If real un-landed work → apply the patch in a scratch worktree, commit by explicit path, push.

## Why held (carve-out item)

Destructive / irreversible action: dropping a stash discards potentially-unlanded work; deciding "landed vs real" on shared-root content is a call with cross-session blast radius.

## Provenance

- origin: /daily 2026-07-30 problem sweep (miner-1 P7, miner-3 P11, miner-5 P21, miner-7 P24 — ≥6 independent session flags in one day)
