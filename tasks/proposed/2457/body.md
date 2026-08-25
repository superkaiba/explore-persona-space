---
title: pre-commit stash-cycle patch* residue accumulates unbounded (6.1 GB / 23,528
  files, oldest 2026-04-27) — evidence-gated reap, no janitor covers the path
kind: infra
tags: []
created_at: '2026-08-21T22:00:28Z'
has_clean_result: false
origin_prompt: 'Found while diagnosing #2253''s Step 10d lint-gate crash: the #2015
  stash-race diagnostic tell depends on counting patch<epoch>-<pid> files, and 23,528
  of them made the tell unusable.'
workflow: v1
---
# pre-commit stash-cycle `patch*` residue accumulates unbounded — 6.1 GB / 23,528 files, oldest 2026-04-27, no janitor covers the path

## Goal

Reap the `~/.cache/pre-commit/patch<epoch>-<pid>` residue left by pre-commit's `staged_files_only` stash cycle, under an EVIDENCE gate (never a naive age sweep), and route the path into an existing janitor pass so it stops growing without bound.

## Measured state (2026-08-21)

- `~/.cache/pre-commit` total: **6.6 GB**; the `patch*` files alone: **6.1 GB** across **23,528** files.
- Oldest `patch*`: **2026-04-27** (~4 months). Newest: today — so this is a steady accumulation, not a one-off incident.
- `/` at the time of measurement: 945 G size, 772 G used, **82% used / 174 G free**. Not critical today, which is exactly why this is worth filing now rather than during the next disk emergency.

Discovered incidentally while diagnosing #2253's Step 10d gate `crash` (the patch-file presence check is the #2015 stash-race diagnostic tell, and the count made the tell useless — see "Second-order cost" below).

## Why these files exist, and why a naive reap is WRONG

`pre_commit/staged_files_only.py` captures the repo-wide unstaged tracked diff into `~/.cache/pre-commit/patch<epoch>-<pid>` before every fleet commit's hook window, then re-applies it in a `finally`. Normally the file is consumed and the cycle is invisible.

They are NOT pure garbage. Per `.claude/rules/repo-root-uncommitted-state.md` § "Double-apply-failure residue": when the restore's first `git apply` raises AND the post-rollback re-apply also raises, the exception propagates with the patch never applied — the tree stays reverted and **the `patch*` file on disk is the sole rescue surface** for the lost work. That is the residue the #1806 `stash_rescue_audit_pass` (watcher pass 34) exists to recover from.

So this must follow the standing disk-hygiene contract: **deletions gated on POSITIVE evidence, never on age alone; age is only ever a KEEP signal** (`.claude/rules/disk-hygiene.md`, user directive 2026-08-06, with #1092 as the standing counter-example an age gate would have destroyed). A plausible evidence gate: reap a `patch*` only when its content is provably already present in the repo (every hunk applies cleanly in reverse against HEAD, or the patch is empty), or when the originating pid is long dead AND the affected paths are clean at HEAD. Design the gate in the plan; do not assume the sketch above is right.

## Second-order cost (the reason this is more than disk)

`.claude/rules/repo-root-uncommitted-state.md` prescribes a CONJUNCTION as the diagnostic tell for the #2015 stash race: content converging back to HEAD within seconds **AND** `patch<epoch>-<pid>` files bracketing the window. With 23,528 files spanning four months, "are there patch files bracketing this window?" is unanswerable at a glance — the discriminator the rule depends on is degraded by the accumulation. Reaping restores the tell's usefulness, which matters independently of the bytes.

## Acceptance

- An evidence-gated reap of `~/.cache/pre-commit/patch*` exists and is wired into an existing janitor pass (candidate home: the `vm_disk_guard.py` boot-disk tiers, alongside the #2127 `/tmp` scratch tier that already implements the positive-evidence + live-process-hold + reap-time-re-probe pattern this can copy rather than reinvent).
- A KEEP is escalated, never silently dropped, with its keep reason tagged (same shape as the existing `tmp-scratch-*` / `slurm-src-*` reasons).
- Its own kill switch, consistent with the family (`EPM_SKIP_*`).
- The #1806 stash-rescue path is demonstrably NOT weakened: show that a patch file representing genuinely-lost work is KEPT by the gate.
- Report-only by default; `--apply` from the cron, per the family convention.

## Not in scope

Changing pre-commit's stash behaviour, or suppressing the stash cycle. `.claude/rules/repo-root-uncommitted-state.md` § "Rejected levers" records that suppressing the stash is fail-OPEN for the worktree-reading secret/lint hooks and is forbidden. This task only reaps residue after the fact.
