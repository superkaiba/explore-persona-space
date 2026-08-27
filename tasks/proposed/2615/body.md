---
title: Pod sparse clones silently hide sibling-issue eval_results/ inputs (empty dir,
  no error)
kind: infra
tags: []
created_at: '2026-08-27T02:47:24Z'
has_clean_result: false
workflow: v1
---
Pod clones are sparse per-issue, so a driver reading a SIBLING issue's committed
`eval_results/issue_<M>/` silently sees an EMPTY directory instead of failing loud.

OBSERVED (#2569 leg 2, 2026-08-27). `pod-2569-wa`'s clone had
`core.sparseCheckout=true` with cone list:
  configs data docs eval_results/issue_2569 figures/issue_2569 scripts src tests
`scripts/issue2569_gateladder.py ladder` defaults `--config-dir
eval_results/issue_1979/config` and `--race-dir eval_results/issue_1979/race` —
55 files that ARE committed on the issue branch and DO appear in `git ls-files`
(flagged `S` = skip-worktree), but were absent from the worktree. `git pull`
reported "Already up to date" at the identical SHA, so the branch state looked
correct while the inputs were missing.

WHY IT IS A WORKFLOW GAP, not an experiment bug:
- The diagnosis is non-obvious: SHA matches, `git ls-files` finds the files, git
  reports up-to-date. Nothing points at sparse-checkout. The only tell is the
  `S` flag from `git ls-files -v`.
- The repo ALREADY has this lesson for the LOCAL side — `tests/sparse_cones.txt`
  plus the CLAUDE.md rule that a new test hard-reading
  `repo_root()/eval_results/issue_<M>/` must register its cone. The POD clone has
  no equivalent registry, so the same class recurs pod-side with no guard.
- It scales with cross-issue reuse, which the artifact-reuse rules actively
  encourage. Any plan whose driver reads a sibling's banked config hits it.

TWO CANDIDATE FIXES (not prejudging which):
(a) At provision/dispatch, add a cone for every `eval_results/issue_<M>/` path
    the plan references, the way `new_worktree.sh` pre-adds `sparse_cones.txt`.
(b) Make the read fail loud: when a default-valued input dir is empty AND
    `git ls-files` reports tracked-but-skip-worktree entries under it, raise
    naming `git sparse-checkout add <path>` as the remedy. This is the
    cheaper, more general arm — it converts every future instance from a
    silent empty read into a one-line remedy.

WORKAROUND APPLIED THIS ROUND (unblocks #2569, fixes nothing durably):
  ssh pod-2569-wa 'cd /workspace/explore-persona-space &&
    git sparse-checkout add eval_results/issue_1979'
  -> config 6 files, race 49 files materialized.

Filed from #2569; the ladder leg is proceeding on the workaround.
