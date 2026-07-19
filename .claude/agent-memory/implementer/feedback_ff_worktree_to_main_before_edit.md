---
name: ff-worktree-to-main-before-edit
description: Spawn worktrees can be several commits behind main; fast-forward the worktree branch onto freshly-fetched origin/main (NEVER local main) BEFORE editing, or your diff reverts concurrent fixes at merge time
metadata:
  type: feedback
---

Fast-forward the worktree branch onto freshly-fetched `origin/main` before your
first edit — NEVER onto local `main` (see Why):

```bash
timeout 60 git -C "$WT" fetch origin "+refs/heads/main:refs/remotes/origin/main"
git -C "$WT" merge-base --is-ancestor HEAD origin/main && git -C "$WT" merge --ff-only origin/main
```

On fetch failure (offline), stay on the existing base — it carries pushed
history only (the #1214 origin/main creation base), the same ladder
`new_worktree.sh` uses.

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
**And NEVER FF onto LOCAL `main`** — it carries unpushed task-state churn from
concurrent sessions. On 2026-07-18 (#1530/#1509) this memory's old local-main
form FF'd issue-1509 from its clean origin/main base onto the local main tip
2.5 minutes after creation (branch reflog: `merge main: Fast-forward`,
14:11:10Z), importing 4 unpushed marker/status commits from 3 sessions and
forcing a Step-10d branch rebuild; `origin/main` carries every landed workflow
fix (fix merges push immediately) and zero unpushed churn.

**How to apply:** at startup, after the `git rev-parse --show-toplevel` self-check,
diff the target files against main (`diff -q "$WT/$f" "$MAIN/$f"`); if any differ,
run the bounded fetch above and, when HEAD is an ancestor of origin/main,
`git -C "$WT" merge --ff-only origin/main`
(safe: only moves your own branch pointer; the repo root stays on main). Then Read
the WORKTREE copies (not the main checkout) before editing — Edit requires reads of
the exact path being edited anyway. Related: [[branch-guard-blocks-subprocess]].

**Diverged-local-main addendum (2026-07-15, #1362; now the DEFAULT, not a
failure diagnosis):** the fetch-first `origin/main` recipe above IS the standing
default as of #1530 — this addendum's diagnosis path survives as the rationale.
A `merge --ff-only main` (the old local form) can fail with "Not possible to
fast-forward" when the worktree base IS the fetched `origin/main` tip while the
shared root's LOCAL `main` has diverged (unpushed local commits; merge-base
behind both). That is NOT a stale worktree — do NOT merge local main in (it
would contaminate the branch with unpushed root commits). If HEAD == origin/main
tip after the fetch, stay on that base and proceed; the server-side Step 10d
rebase-merge reconciles. Check `git log -1 main -- <target-file>` for a
local-main-only edit to the same file (a rebase-conflict risk to note, not a
blocker).

**End-of-run addendum (2026-06-12):** on a LONG run, main drifts again AFTER the
startup FF — a full-suite test failure in a file you never touched is usually drift,
not your diff (verify by running the same test in the main checkout). After
committing your work, re-fetch and merge the REMOTE tip —
`timeout 60 git -C "$WT" fetch origin "+refs/heads/main:refs/remotes/origin/main"`
then `git -C "$WT" merge origin/main --no-edit` — and re-run the
suite; resolving a conflict by hand is SAFE here (your private branch, not the
shared repo root, where conflicted merges must instead be aborted). This run: a
TrainLoraConfig test failed on drift; post-merge 3354/3354 passed, and the merge
also pre-resolved a CLAUDE.md paragraph conflict the orchestrator would otherwise
have hit at merge-to-main time.
