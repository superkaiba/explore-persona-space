---
name: Newly-committed pod-generated artifacts block the next pod pull
description: Committing pod-written files (eval_results/issue_N/...) in-task makes the next pod-side git pull --ff-only abort on "untracked working tree files would be overwritten" — back up outside the repo, remove, pull
type: feedback
---

When a round re-pins pod-generated artifacts in-task (committing files like
`eval_results/issue_<N>/phase0/*.json` that prior runs wrote UNTRACKED on the pod), the
next pod-side `git pull --ff-only` deterministically aborts with "untracked working tree
files would be overwritten by merge".

**Fix:** back the pod's local copies up OUTSIDE the repo (e.g.
`/workspace/issue-<N>-phase0-local-backup/` — never delete), remove them from the
worktree, then pull; the tracked reference copies become the working versions.

**Why:** incident #601 relaunch 4 (2026-06-11) — round 7 committed three phase0 reference
JSONs the pod had written untracked; the sync blocked until the backup-remove-pull
sequence.

**How to apply:** at any pod sync after a round whose diff adds files under paths the pod
itself writes, expect this abort and apply the backup-remove-pull sequence proactively.
