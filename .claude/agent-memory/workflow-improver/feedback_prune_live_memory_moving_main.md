---
name: prune-live-memory-moving-main
description: Pruning agent-memory in a worktree while agents write to main — snapshot-commit first, expect add/add resurrections at merge, sweep main's post-snapshot dirt before reporting
metadata:
  type: feedback
---

Agent-memory dirs are written CONTINUOUSLY by live agents in the main
checkout, and much of the content is UNTRACKED there. Pruning them from a
worktree needs a three-phase protocol (2026-06-12 token-audit run):

1. **Snapshot-commit first:** rsync main's working tree (tracked dirt +
   untracked) for the in-scope dirs into the worktree and commit verbatim
   BEFORE pruning — the prune diff stays reviewable and every pre-prune
   byte is recoverable from git.
2. **Expect add/add resurrections at merge:** files untracked at the
   merge-base that BOTH sides committed (your snapshot; main's later
   catch-up commit) conflict add/add — and absorbed-then-deleted files
   come back from main's side silently. Auto-resolve with `--ours` only
   where `git rev-parse SNAP:p == main:p`; hand-fold the rest; re-delete
   resurrected absorbed files after the merge.
3. **Post-snapshot dirt sweep before reporting:** diff every dirty/
   untracked main file against the snapshot blob (`NEW-` vs
   `CHANGED-SINCE-SNAPSHOT`), fold the deltas (new lesson files +
   appended incident blocks land verbatim; they're new, not accretion),
   record the sweep timestamp, and give the orchestrator a scoped
   pre-merge recipe: check `find <dirs> -newermt <stamp>`, then
   `git checkout -- <dirs> && git clean -f -- <dirs>` in main before
   merging (untracked copies block the merge otherwise).

**Why:** the lens11→lens7 rename + four new critic files + a #576 r2
ledger block all landed on main DURING the prune; without the sweep the
branch would have regressed them. **How to apply:** any workflow-improver
task editing `.claude/agent-memory/**` (or other live-written state)
across a multi-hour run. NOTE: the project auto-memory under
`~/.claude/projects/.../memory/` has NO git — deletions there are
unrecoverable; read + justify each file before `rm`.
