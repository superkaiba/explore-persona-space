---
title: 'workflow-fix: SLURM rsync races the live shared tree — partial sync shipped
  3/4 committed entrypoints; verify post-sync or archive-snapshot the source'
kind: infra
tags:
- wf-fix
created_at: '2026-07-31T06:41:26Z'
has_clean_result: false
origin_prompt: 'job 15993 died in 49s: gen_a1 entrypoint absent from scratch after
  a dispatch-time rsync raced a concurrent pre-commit stash/restore on the shared
  root (2026-07-30T23:22Z)'
workflow: v1
---
## Overview / Motivation

SLURM-lane dispatches rsync the LIVE shared repo-root working tree to the cluster scratch. The shared root mutates constantly under concurrent sessions (pre-commit stash/restore cycles, formatter hooks, concurrent checkouts of individual files), so a dispatch-time rsync can capture a PARTIAL tree with no error: job 15993 (issue 1689, 2026-07-30T23:22Z) synced `./scripts` while a concurrent pre-commit stash/restore cycle ran (23:23Z) and shipped 3 of the 4 committed `issue1689_user_slot_*.py` files — the workload died in 49s on `No such file or directory` for the missing entrypoint AFTER a full sbatch cycle. A dry-run rsync from the same tree an hour later ships all 4 files.

- verified-at-filing: job 15993 `sacct` FAILED 00:00:49 exit 2:0; scratch listing showed `issue1689_user_slot_{capture,fits,render}.py` present, `gen_a1` absent; local `git log` has gen_a1 committed 4h before the dispatch (b150c16de0); `rsync -an --relative ./scripts charmander:/tmp/eps-dryrun/` lists all 4 (2026-07-31).

## Goal

A SLURM dispatch either ships a CONSISTENT tree or fails loud BEFORE sbatch — never a silent partial sync discovered by the workload's crash.

## Proposed change (either/both)

1. **Post-sync entrypoint verify (cheap):** after the rsync, `backends/slurm.py` verifies the synced tree — at minimum `git ls-files scripts/ src/ | wc -l` parity vs a remote `find`-count, or an explicit existence check of every file referenced in the workload-cmd string (best-effort parse for `scripts/<name>.py` tokens); mismatch → refuse before sbatch with a typed reason (`rsync_partial_tree`).
2. **Sync from a snapshot, not the live tree (robust):** `git archive HEAD scripts src configs ...` to a temp dir (plus the intentional non-tracked include set), rsync THAT — immune to concurrent stash/restore churn by construction. The GCP lane's git-clone-at-boot has this property already; the SLURM lane is the outlier.

## Constraints

- Workflow surface (`src/explore_persona_space/backends/slurm.py` + tests).
- Must not break the fellows/DRAC allowlisted-command constraint (rsync/scp/sbatch only on the remote side; the verify can run through the same channel).
- est_gpu_hours: 0
