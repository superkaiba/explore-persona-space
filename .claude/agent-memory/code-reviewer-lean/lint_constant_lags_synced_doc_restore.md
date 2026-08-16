---
name: lint-constant-lags-synced-doc-restore
description: Reviewing a "restore main's lint cap" unblock commit — verify byte-exact block vs origin/main, cap == main's (never above), exactly-one-residual-hunk, fails-pre/passes-post live pins
metadata:
  type: feedback
---

A spec-freshness sync updates `.claude/**` docs but never the branch's
`scripts/workflow_lint.py`, so a doc-size grandfather cap raised on main
(doc regrew + cap ratcheted together) leaves the worktree lint
deterministically red on bytes IDENTICAL to main (#2321 r3: SKILL.md synced
to main's 982,587 B, branch cap stale at pre-#2146 `980_400`).

**Why:** the restore commit looks like a ratchet raise; the distinction
(legitimate unblock vs evasion) is checkable, not a judgment call.

**How to apply:** four probes settle it — (1) restored block byte-exact to
`origin/main` AND cap value EQUALS main's (any value above main's = evasion);
(2) the gated doc byte-identical to `origin/main` (`git diff` = 0 bytes) with
headroom under the hygiene bar; (3) `git diff origin/main HEAD -- <lint>`
leaves ONLY the branch's previously-reviewed deliberate hunks; (4) the
committed live pins fail pre-fix / pass post-fix (here
`test_workflow_lint_skill_doc_size.py::test_live_tree_passes_no_fails`).
Merge-neutrality follows from (1)+(3): the branch never authored the block,
so the Step 10d 3-way merge yields main's bytes regardless. Related:
[[spec-freshness-sync-provenance-recipe]].
