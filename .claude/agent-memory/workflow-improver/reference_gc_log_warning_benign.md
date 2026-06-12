---
name: gc-log-warning-benign
description: The "too many unreachable loose objects" gc.log warnings in .git/worktrees/*/ are a benign daily cycle on git 2.34.1, not stuck suppression — verified 2026-06-11, deflected twice
metadata:
  type: reference
---

Candidates proposing to clear `.git/worktrees/*/gc.log` files or add prune/gc steps to `worktree_audit.py` should be checked against these measured facts (2026-06-11):

- gc.log is **per-gitdir** and **self-expiring** (`gc.logExpiry` default 1.day): git stats it on each auto-gc, proceeds when stale, rewrites only because the loose-object condition recurs. All 20 observed gc.log files were <17h old — the signature of the healthy daily cycle.
- gc.log **cannot outlive the worktree reap**: the audit uses `git worktree remove --force` (worktree_audit.py:553) which deletes the admin dir wholesale; 54/54 admin dirs matched registered worktrees, zero prunable.
- The ~22k loose objects (~1.4 GiB) are rolling marker-commit churn **inside the 14-day `gc.pruneExpire` grace** — a prune reclaims ~zero (sibling improver measured this same day and deflected a watcher-side `git gc` tier on the same grounds).
- Removing gc.log daily would only defeat git's designed backoff → more gc passes on a repo with ~27 concurrent committers, zero space reclaimed.

**Root cause + real fix:** VM git is 2.34.1, predating cruft packs (introduced 2.37, default 2.43). Upgrading git ≥2.43 packs unreachable objects into a cruft pack — warning disappears, loose-object sprawl collapses. That is an ops action, not a workflow-surface edit. Do NOT shorten `gc.pruneExpire` on this many concurrent committers.
