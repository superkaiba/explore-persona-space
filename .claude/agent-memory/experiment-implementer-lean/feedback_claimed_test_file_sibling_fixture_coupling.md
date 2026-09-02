---
name: claimed-test-file-sibling-fixture-coupling
description: A sibling fix round may edit YOUR claimed test file's fixture to match ITS uncommitted src change — check commit-coupling before any whole-file commit (#2658 F/K round)
metadata:
  type: feedback
---

A concurrent fix round that owns a src module (e.g. `issue2658_inference.py`) may
legitimately edit the FIXTURE inside a test file claimed for YOU (their src change
forces new required kwargs the fixture must pass). Two traps follow:

1. **Whole-file staging smuggles sibling hunks.** `git add <your-test-file>` commits
   the sibling's fixture hunks too. Before committing a claimed file in a shared
   worktree, diff it and confirm the residual hunks are yours alone
   (`git diff -- <file> | grep '^@@\|^+def'`).
2. **Commit-coupling makes the committed tree red.** The sibling's fixture edit calls
   into their UNCOMMITTED src — probe the coupling with
   `git show HEAD:<src> | grep -A8 'def <entrypoint>'` (is the new kwarg at HEAD?).
   If not landed, committing the test file whole breaks the branch. SEQUENCE: commit
   your uncoupled files first, hold the coupled test file until their landing appears
   in `git log`, then re-diff (their hunks vanish from the residual) and commit yours.

**Why:** #2658 grouped F/K fix round (2026-09-02): group-J landed
`inference.py` + the unit12 fixture hunks mid-round; the residual unit12 diff
collapsed to exactly my two new tests, making the whole-file commit safe. Ten
minutes earlier the same commit would have shipped a fixture calling a
`reliability=` kwarg that did not exist at HEAD.

**How to apply:** any shared-worktree round whose claimed file set includes tests
fixturing a module another live round owns. Also expect transient pytest ERRORS
(13 fixture errors, then a hang) while the sibling's src is mid-edit — re-run after
their landing before diagnosing your own edits. Related: [[worktree-commit-and-selector-vintage]].
