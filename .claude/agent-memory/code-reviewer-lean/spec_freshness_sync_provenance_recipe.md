---
name: spec-freshness-sync-provenance-recipe
description: Reviewing a Step 5a "sync workflow-surface specs from origin/main" commit — per-file blob compare, and resolving a DIFF hit by matching against the sync-time main tip (post-sync main churn ≠ branch drift)
metadata:
  type: feedback
---

A Step 5a spec-freshness sync commit claims its files are checked out
verbatim from fetched origin/main. Verify per file, never by one aggregate
diff: `git rev-parse <sync-sha>:<path>` vs `git rev-parse origin/main:<path>`
after a fresh fetch. A DIFF hit is NOT automatically drift — main may have
advanced after the sync. Resolve it by matching the sync commit's blob
against main history: `git log origin/main --format='%H %ci' -- <path>`,
then blob-compare against each recent main commit; a match at the commit
that was main's tip at the sync timestamp (compare `%ci` of the sync commit
vs the main commits) proves verbatim-at-sync-time. Classify the residual as
Step 0.9 `stale-main-or-worktree` (main churn post-sync), provenance-clean.
Also check the commit subject carries the `sync workflow-surface specs from`
anchor phrase the provenance check keys on.

**Why:** #2321 R1 g8 — 4/5 files byte-identical, issue/SKILL.md differed;
the naive read ("sync not verbatim") would have been a false FAIL. The blob
matched main's 06:17 tip; the sync ran 06:41; main advanced 07:57.

**How to apply:** any split-review group owning a `spec-freshness` /
`sync ... from origin/main` commit, and any Step 0.9 adjudication where a
branch file "differs from main" on a path the branch never authored. A
one-main-commit-behind SKILL/spec file after such a sync is expected shape —
the rebase-merge drops already-applied hunks; note it, never block on it.
