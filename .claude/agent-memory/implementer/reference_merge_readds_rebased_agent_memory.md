---
name: merge-readds-rebased-agent-memory
description: Step 10d merges re-duplicate agent-memory bullets when the branch ancestry carries pre-rebase copies of commits that landed on main via gh pr merge --rebase; check exact-fragment counts vs main and restore main's copy
metadata:
  type: reference
---

In a Step 10d divergence-reconciliation merge, an `Auto-merging
.claude/agent-memory/...` line on a file the branch "never touched" is a
duplication tell, not a clean merge. Mechanism: agent-memory commits land on
main REWRITTEN by `gh pr merge --rebase` (new SHAs), while the issue branch's
ancestry keeps the pre-rebase originals; the merge-base excludes them, so the
merge re-adds bullets main already carries (#2214 round, 2026-08-20: three
#2228 bullets re-duplicated in
`codex-critic/feedback_custom_infra_lens_composition.md`).

**How to apply:** for each added block, grep an exact distinctive fragment in
BOTH `git show <MAIN_SHA>:<file>` and the merged copy — merged == main+1 per
fragment ⇒ duplicate; restore the file to main's copy (`git checkout
<MAIN_SHA> -- <file>`, own commit, lossless-verified). Main is authoritative
for agent-memory bookkeeping (the #2217 hygiene-commit precedent). Related:
[[prune-live-memory-moving-main]].
