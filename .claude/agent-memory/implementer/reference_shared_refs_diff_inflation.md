---
name: shared-refs-diff-inflation
description: In a worktree, origin/main can advance MID-ROUND (shared .git refs) — derive the round's diff list from its own commit (HEAD~1..HEAD), never origin/main..HEAD, or the pin-sweep/map-files scope inflates with other sessions' files
metadata:
  type: reference
---

Worktrees share `.git` refs with the main checkout, so ANY concurrent
session's `git fetch origin main` advances `origin/main` under your feet
mid-round. Two consequences observed on #2391 (2026-08-19, ~50 upstream
commits landed between my FF and my post-commit checks):

- `git diff --name-only origin/main..HEAD` (a two-dot ENDPOINT diff) then
  returns your files PLUS the inverse of every upstream commit — the #2391
  round saw 80 files for a 30-file commit. Derive the round's changed-path
  list for the `select_step9c_tests.py --map-files` pin-sweep from the
  commit itself: `git diff --name-only HEAD~1..HEAD` (or `git diff-tree
  -r --name-only --no-commit-id HEAD`).
- The selector's default `--json` run (fetched-origin/main base) inflates
  the same way: report the drift explicitly (base SHA + "moved mid-round")
  and cover the TRUE-diff-linked tests locally; Step 9c re-derives its own
  selection after the Step 10d rebase, so the upstream-linked remainder
  defers there.

Related timing fact from the same round: the no-flags `workflow_lint.py`
run exceeded a 500 s timeout under fleet load (both captures needed the
run_in_background + same-turn-poll shape with a ~29 min internal timeout);
plan §8-style lint baselines should not be given sub-10-min fences.
