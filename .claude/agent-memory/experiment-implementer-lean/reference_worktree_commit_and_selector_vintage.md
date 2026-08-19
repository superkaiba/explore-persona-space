---
name: worktree-commit-and-selector-vintage
description: guard_root_code_commit blocks plain `git commit` even when cwd is a worktree — use `git -C "$WT"` explicitly; and select_step9c_tests inflates on vintage-pinned worktrees
metadata:
  type: reference
---

Two mechanics hit on #1217 round 1 (lean implementer, worktree
`.claude/worktrees/issue-1217`):

1. **`guard_root_code_commit.sh` keys on the invocation shape, not the cwd.**
   A plain `git add ... && git commit ...` run from inside a worktree is still
   BLOCKED as an "uncertified repo-root code payload" for `tests/` paths. The
   fix is the hook's own hint: invoke as `git -C "$WT" add/commit` — worktree
   commits are gated at Step 10d, not by this hook. The blocked compound
   command means the `add` never ran either; re-stage on retry.

2. **`select_step9c_tests.py --json` on a vintage-pinned worktree is dominated
   by branch-vs-main drift, not the round's diff.** The selector diffs the
   BRANCH tree against fetched `origin/main`; a worktree deliberately pinned
   to an older spec vintage (brief-mandated) selects hundreds of tests via
   rules-pin/skills-pin entries for files the round never touched (on #1217:
   225 tests for a 2-file round). The round-scoped signal is (a) the
   `selection_reasons` entries naming YOUR changed paths and (b) the
   `--map-files <difflist>` pin-sweep. Report the full n_tests honestly,
   run the round-keyed hits in-turn, defer the vintage remainder to Step 9c
   (which runs post-merge where the drift resolves). Same class: a no-flags
   `workflow_lint.py` FAIL naming only files that differ from origin/main by
   MISSING-on-branch fix lines is vintage staleness, not round red — verify
   with `git diff --stat origin/main HEAD -- <offenders>` before treating it
   as payload-attributed.
