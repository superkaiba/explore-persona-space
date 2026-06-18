---
name: Claude approves branch-merge plan steps without running merge-tree
description: Plans whose first step merges a sibling branch with a blanket conflict policy ("prefer branch's i<M> files; main's everything else") — run `git merge-tree` yourself; blanket policies give zero guidance for the SHARED files that actually conflict and can contradict the plan's own "the merge brings X" sentence.
type: feedback
---

**Rule:** when a plan says "merge origin/issue-<M> into the worktree" with a one-line conflict policy, verify mechanically: `BASE=$(git merge-base main origin/issue-M); git merge-tree "$BASE" main origin/issue-M`; count `^+<<<<<<<` and map hunks via the "changed in both" headers. The new i<M> scripts never conflict (absent from main); the conflicts land in SHARED load-bearing files where neither blanket side is correct (the branch carries the parent's plumbing, main carries newer machinery the branch predates). If the policy prescribes the wrong side for any load-bearing conflicted file — especially when "prefer main for everything else" literally discards a feature the plan's own §4.0 says the merge brings — REVISE with an explicit per-file union contract (self-defeating-plan class: implementer and code-reviewer both inherit the plan's error). Smoke asserts don't rescue: they cover only the asserts' surface, not wrong-side analyzer/eval divergences.

**Origin:** #555 r1 meth — 19 conflict hunks; main's callbacks.py had 0 `snapshot_every_steps` hits vs the branch's 27; plan §7 policy contradicted plan §4.0. REVISE.
