---
name: Claude approves branch-merge plan steps without running merge-tree
description: Plan-critic disagreement where the plan's first step merges a sibling issue branch; Claude endorses the merge step + generic conflict policy without running git merge-tree, Codex runs it and finds real conflicts in shared load-bearing files. Verify the conflicts yourself and check the policy against the plan's own "the merge brings X" list.
type: feedback
---

When a plan's §4.0 says "merge origin/issue-<M> into the worktree" with a one-line
conflict policy like "prefer issue-<M>'s versions of i<M> files; main's versions of
everything else", Claude critics APPROVE on the existence check (branch exists,
scripts absent from main) without running merge-tree. Codex runs `git merge-tree`
and finds real conflicts in SHARED files (callbacks.py, sft.py, experiments/*/
eval_trajectory.py, analyze.py) where NEITHER blanket side is correct — the branch
side carries the parent's plumbing (snapshot extension, config pass-through) and
main carries newer machinery (gauge asserts, slot-stats) the branch predates.

**Why:** The blanket policy gives zero correct guidance for exactly the files that
actually conflict (the new i<M> scripts never conflict — they don't exist on main).
Worse, the policy can contradict the plan's own §4.0 "the merge brings <feature in
shared file X>" sentence: following "prefer main for everything else" literally
discards the named feature. A plan whose written instruction, followed literally,
breaks the plan's own requirements is a plan defect (self-defeating-plan class),
not an implementer detail — the implementer follows the plan and the code-reviewer
reviews against the plan, so both downstream defenses inherit the error.

**How to apply:** Verify mechanically: `BASE=$(git merge-base main origin/issue-M);
git merge-tree "$BASE" main origin/issue-M` (git 2.34 has no --write-tree), count
`^+<<<<<<<` and map hunks to files via the `changed in both` headers. Then check
each conflicted file's per-side content (grep the branch-only feature AND the
main-only feature). If the plan's policy prescribes the wrong side for any
load-bearing conflicted file → REVISE (amend plan with an explicit per-file union
contract), even when smoke asserts would catch the breakage fail-loud — the smoke
only covers the asserts' surface, and subtler wrong-side resolutions (analyzer
gates, eval-rig divergence) slip past. Origin: task #555 round-1 methodology lens
(19 conflict hunks; main's callbacks.py had 0 snapshot_every_steps hits vs
branch's 27; plan §7 policy contradicted plan §4.0).
