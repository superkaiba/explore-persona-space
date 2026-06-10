---
name: ff-worktree-to-main-before-edit
description: Spawn worktrees can be several commits behind main; fast-forward the worktree branch onto main BEFORE editing, or your diff reverts concurrent fixes at merge time
metadata:
  type: feedback
---

Fast-forward the worktree branch onto current `main` before your first edit:

```bash
git merge-base --is-ancestor HEAD main && git merge --ff-only main
```

**Why:** the harness forks the isolation worktree at spawn time, but main advances
continuously (daily-fix merges, other workflow-improvers, task.py commits). On
2026-06-10 the worktree was forked at `b4bf2b7a6` while main sat 2+ commits ahead
with 22/31/1-line changes ALREADY LANDED in the same agent files being edited
(code-reviewer.md, experiment-implementer.md, critic.md). Editing the stale copies
would have produced either merge conflicts or silent reverts of those fixes when
the orchestrator merged the branch. Detection signal: two Reads of the same
main-checkout line range disagreed mid-session — main moved between reads.

**How to apply:** at startup, after the `git rev-parse --show-toplevel` self-check,
diff the target files against main (`diff -q "$WT/$f" "$MAIN/$f"`); if any differ
and HEAD is an ancestor of main, `git merge --ff-only main` inside the worktree
(safe: only moves your own branch pointer; the repo root stays on main). Then Read
the WORKTREE copies (not the main checkout) before editing — Edit requires reads of
the exact path being edited anyway. Related: [[branch-guard-blocks-subprocess]].
