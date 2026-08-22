---
name: worktree-commit-git-c-form
description: guard_root_code_commit treats a `cd $WT && git commit` as a REPO-ROOT commit — worktree commits must use `git -C "$WT" commit -F <msgfile> -- <paths>`
metadata:
  type: feedback
---

Commit in a worktree with `git -C "$WT" add <paths>` then
`git -C "$WT" commit -F <msgfile> -- <paths>` — never
`cd "$WT" && git commit ...` inside the same Bash call.

**Why:** the `guard_root_code_commit.sh` PreToolUse hook keys its
worktree exemption on the literal `git -C` invocation form. A `cd`-form
commit is evaluated as a REPO-ROOT commit: it demands inline-lint-gate
certification for any scripts/src/tests payload AND its payload scan picks
up FOREIGN files from the shared root's staged index (observed on #2317,
2026-08-15: the block named a sibling session's staged test file that was
never in my pathspec). Worktree commits are gated at Step 10d, not by this
hook, so the `git -C` form passes cleanly.

**How to apply:** compose the commit message via the Write tool to a file
(guards scan Bash argv incl. heredocs), then run the `git -C` form with an
explicit pathspec. On a block naming files you never staged, do NOT chase
them — they are the root index's, not yours; rewrite in the `git -C` form.
