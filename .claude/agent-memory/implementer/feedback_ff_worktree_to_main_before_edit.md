---
name: ff-worktree-to-main-before-edit
description: Spawn worktrees can be several commits behind main; fast-forward the worktree branch onto main BEFORE editing, or your diff reverts concurrent fixes at merge time
metadata:
  type: feedback
---

Fast-forward the worktree branch onto current `main` before your first edit:

```bash
git merge-base --is-ancestor HEAD main && git -C "$WT" merge --ff-only main
```

The `-C "$WT"` form is REQUIRED (#1128): the repo-root branch guard is
CWD-BLIND — a bare `git merge` is hook-blocked even when your cwd IS the
worktree, exactly like a bare `git restore .` there.

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
and HEAD is an ancestor of main, `git -C "$WT" merge --ff-only main`
(safe: only moves your own branch pointer; the repo root stays on main). Then Read
the WORKTREE copies (not the main checkout) before editing — Edit requires reads of
the exact path being edited anyway. Related: [[branch-guard-blocks-subprocess]].

**Diverged-local-main addendum (2026-07-15, #1362):** `merge --ff-only main` can
fail with "Not possible to fast-forward" when the worktree base IS the fetched
`origin/main` tip while the shared root's LOCAL `main` has diverged (unpushed
local commits; merge-base behind both). That is NOT a stale worktree — do NOT
merge local main in (it would contaminate the branch with unpushed root
commits). Diagnose with `git -C "$WT" fetch origin main` +
`git merge-base HEAD origin/main`: if HEAD == origin/main tip, stay on that
base and proceed; the server-side Step 10d rebase-merge reconciles. Check
`git log -1 main -- <target-file>` for a local-main-only edit to the same file
(a rebase-conflict risk to note, not a blocker).

**End-of-run addendum (2026-06-12):** on a LONG run, main drifts again AFTER the
startup FF — a full-suite test failure in a file you never touched is usually drift,
not your diff (verify by running the same test in the main checkout). After
committing your work, `git -C "$WT" merge main --no-edit` and re-run the
suite; resolving a conflict by hand is SAFE here (your private branch, not the
shared repo root, where conflicted merges must instead be aborted). This run: a
TrainLoraConfig test failed on drift; post-merge 3354/3354 passed, and the merge
also pre-resolved a CLAUDE.md paragraph conflict the orchestrator would otherwise
have hit at merge-to-main time.
