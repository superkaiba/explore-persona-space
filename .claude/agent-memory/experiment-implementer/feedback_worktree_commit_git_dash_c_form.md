---
name: worktree-commit-git-dash-c-form
description: Subagent worktree commits must use git -C "$WT" add/commit; a bare compound add+commit trips guard_root_code_commit as a repo-root code payload
metadata:
  type: feedback
---

In subagent shells (cwd resets between Bash calls), a bare `git add ... && git commit -F ... -- <paths>` from a worktree can be read by the `guard_root_code_commit.sh` PreToolUse hook as a REPO-ROOT code commit: it then demands the inline payload lint cert and blocks, also surfacing FOREIGN staged paths from the shared index (#2658 r18: blocked with two unrelated codex_daemon_reaper paths).

**Why:** the hook's scope detection keys on the command shape, and the worktree escape it recognizes is the explicit `git -C` form.

**How to apply:** always commit worktree payloads as `git -C "$WT" add <paths> && git -C "$WT" commit -F <msgfile> -- <paths>` with the literal worktree path. Compose the message file with the Write tool first (a blocked compound never ran its earlier clauses, so a heredoc msgfile from the same call does not exist on retry).
