---
name: Re-fold: worktree may be on main, read artifacts from the issue branch
description: On a same-issue follow-up re-fold, the issue-N worktree can be checked out on `main` (concurrent process), so new-seed/follow-up JSONs + figure scripts live only on the issue-N branch — read them via `git show issue-N:<path>`, not from disk
type: feedback
---

On a same-issue follow-up RE-FOLD spawn, the brief may say "worktree is on
branch issue-N" but a concurrent process can have switched the worktree's
HEAD to `main` (the shared canonical commit target). When that happens:

- `git rev-parse HEAD` returns a `main` commit (other tasks' commits in the log), NOT the issue-N tip.
- The round's NEW artifacts (new-seed trajectory JSONs, the regenerated
  figure script, the figure binaries) exist ONLY on the `issue-N` branch
  and are NOT on disk in the worktree — an on-disk `ls`/`md5sum` reports
  them MISSING and a coverage loop counts 0.

**How to detect:** `git branch --show-current` (returns `main` not `issue-N`),
and `git status` shows concurrent tasks' files modified.

**How to proceed (don't switch branches in the worktree — CLAUDE.md forbids it
on the repo-root tree, and switching here risks the concurrent committers):**
- Read every round artifact from the branch blob: `git show issue-N:<path>`
  (JSONs, the multi-seed figure script, meta.json). Reproduce the figure
  script's aggregation logic in a standalone python reading those blobs.
- Confirm the figure commit is pushed: `git rev-parse origin/issue-N` and
  `git branch -r --contains <fig-sha>`; SHA-pin the figure URL to it.
- `task.py find/set-body/post-marker` all commit to `main` regardless of the
  worktree's HEAD, so they work fine from the worktree on `main`.
- The methodology doc for check 21 also lives only on issue-N — extract it
  (`git show issue-N:docs/methodology/issue_N.md > /tmp/...`) and pass
  `--methodology-doc /tmp/...` to verify_task_body.py.

**Why:** this cost real discovery time on #597 round 5 (filler-control-multiseed,
2026-06-16). The figure script's working copy was the OLD seed-42-only version
because the worktree was on `main`; the multi-seed version + the 24 new JSONs
were on issue-597 only.
