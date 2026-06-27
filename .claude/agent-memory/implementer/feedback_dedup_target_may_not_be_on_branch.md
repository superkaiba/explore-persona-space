---
name: A plan's named dedup/merge-target file may not exist on YOUR worktree branch
description: When a plan says "merge incoming X into existing Y", verify Y is actually present on the worktree branch — a recently-added repo-root file may be uncommitted (untracked in git status) and absent from the branch you forked, making the "merge" actually a clean add
type: feedback
---

When a subagent brief / plan tells you to dedup-merge or consolidate file A
into an EXISTING file B (agent-memory merges, doc consolidations), do NOT
assume B is present in your worktree. The worktree branch was forked at some
past `main`; a file B that landed in repo-root AFTER the fork — or that is
still UNCOMMITTED in repo-root (shows as `??` untracked in the repo-root
`git status`) — is NOT on your branch and `Read` will FileNotFound it.

**Why:** task #678 (2026-06-27). Plan v2 named
`feedback_worktree_path_discipline.md` as the implementer-side dedup target
for an incoming `feedback_edit_lands_in_main_not_worktree.md`. That target was
an uncommitted repo-root untracked file (newer than the `issue-678` branch
fork), so it was absent from my worktree. "Merge into B" became "B doesn't
exist here" → the correct move was to treat the incoming file as a NON-dup and
`git mv` it in (it would coexist with B only after the Step-10d merge, and they
were complementary anyway). The clarifier's "9 entries" count was also stale —
the worktree branch had only 6 implementer memos committed.

**How to apply:** before acting on a "merge into existing Y" instruction,
`ls $WT/<path-to-Y>` (or `git -C $WT ls-files <Y>`) to confirm Y is on YOUR
branch. If absent: the instruction's premise is stale — treat the incoming as a
non-duplicate add, and NOTE the discrepancy in the report's "Considered but not
done" so the reviewer/user can consolidate post-merge if the two truly overlap.
Counts of "existing N entries" in a brief are likewise branch-relative — re-`ls`
the actual worktree dir, don't trust the number.
