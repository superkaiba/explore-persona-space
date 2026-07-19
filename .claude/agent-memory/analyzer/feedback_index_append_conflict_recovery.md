---
name: INDEX.md append-conflict recovery without touching live task state
description: Rejected push on an eval_results/INDEX.md append — recover via scratch-worktree cherry-pick + root reset --soft convergence; NEVER materialize origin-side tasks/ renames into the shared working tree
type: feedback
---

Recovering a rejected repo-root push whose conflict is an eval_results/INDEX.md append: cherry-pick your commit in a scratch worktree detached at origin/main (`git -C` forms — a `cd <wt> && git ...` compound is hook-blocked), resolve keep-both-rows, push HEAD:main, then converge the root pointer with `git reset --soft origin/main` (ungated) + explicit-path index sync under the `~/.task-workflow/lock` flock.

**Why:** #1489 promotion pass (2026-07-18): sync_repo_root aborts on genuine INDEX.md conflicts, and the stranded local commit re-mines EVERY future sync (its replay re-conflicts on the same append anchor) until the local main pointer moves off it. Worse: a naive "materialize origin content into the working tree" step nearly clobbered LIVE task state — the origin-vs-local diff contained tasks/ status-move RENAMES in the WRONG direction (origin was BEHIND local on #1417/#1523 task state), so writing origin's layout rm'd `tasks/followups_running/1417/` while that task's session was mid-loop (restored from local tip within minutes; plan.md symlinks need mode-aware restore — `git show` of a 120000 entry prints the target string, don't write it as a file).

**How to apply:** (1) never write origin-side content over tasks/ paths — task state at the repo root is frequently AHEAD of origin; (2) check `git log origin/main..main` FIRST — the root routinely carries dozens of unpushed marker commits, so your commit lands mid-stack; (3) the full defusal: rebase the whole local stack onto origin/main in a scratch worktree branch (resolve your own commit's conflict by taking origin's side so it rebases to empty), push, then at the root under the workflow flock: `reset --soft origin/main` + pathspec-scoped index/file sync from HEAD (symlink-aware), verifying `git status` returns to the pre-incident dirty set.
