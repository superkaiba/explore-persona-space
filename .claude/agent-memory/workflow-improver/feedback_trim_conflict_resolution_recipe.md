---
name: trim-conflict-resolution-recipe
description: Resolving a stale compact-pointer (trim) branch against moved main — enumerate main-side deltas first, checkout the trim copy, fold each delta back in compact form with full text relocated into the pointer's rules file
metadata:
  type: feedback
---

When a CLAUDE.md trim/pointer branch conflicts with a moved main, do NOT
hand-resolve inside the conflict markers. Recipe (validated 2026-06-12,
trim × #616 + idle-unmapped reaper merge, branch agent-ae48729402ca7c778):

1. `git diff <merge-base> <main> -- CLAUDE.md` to ENUMERATE every main-side
   delta (there were exactly 2; the conflict hunks made it look bigger).
2. `git checkout <trim-branch> -- CLAUDE.md` to take the trim copy as the
   resolution base (resolves the conflict mechanically).
3. Fold each main-side delta back in COMPACT form: verify the pointed-to
   rules file is a strict superset of the dropped clauses FIRST (read it;
   if main's new text isn't there yet, relocate it verbatim into the rules
   file in the same edit pass — that keeps the trim's zero-info-loss union
   invariant).
4. Validate with an incident-id + READ-mandate grep over CLAUDE.md ∪
   .claude/rules/ before committing the merge.

**Why:** diff3 conflict blocks with multi-KB lines are unreadable and
hand-merging them loses clauses; the union invariant (every fact survives
in CLAUDE.md ∪ rules/) is the actual spec, not line-level merging.

**How to apply:** any future conflict between a compact-pointer
restructure branch and concurrent always-on-file edits — also applies to
[[stale scope list ≠ deflection]]-style cases where main moved after the
candidate was sketched. Pair with [[ff-worktree-to-main-before-edit]].
