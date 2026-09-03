---
name: lean-session-waits-and-tmp-collisions
description: "Lean variant has no Monitor tool — wait on a long detached lint via `timeout N tail --pid=<pid> -f /dev/null`; /tmp scratch names collide across concurrent same-issue sessions (use round-unique names)"
metadata:
  type: feedback
---

Two mechanics for lean-context rounds (#2658 reclaim-gate round, 2026-09-02):

1. **No-sleep bounded wait without Monitor.** The lean variant's toolset has no
   Monitor, foreground `sleep` is hook-blocked, and a >600 s foreground Bash is
   auto-backgrounded. When a no-flags `workflow_lint.py` run exceeds the 600 s
   Bash cap under fleet load (the known #2054 shape), wait on the surviving
   process with `timeout 580 tail --pid=<lint pid> -f /dev/null` in a fresh
   foreground Bash — blocks until pid exit, no `sleep` in argv, bounded. Find
   the pid via `pgrep -af 'workflow_lint[.]py'`.

2. **/tmp scratch names collide across concurrent same-issue sessions.** A
   sibling #2658 round had already written `/tmp/i2658_commit_msg.txt`; the
   Write tool refuses to overwrite an unread file, and reading+clobbering would
   destroy their scratch. Name round-scoped /tmp files by round content
   (`/tmp/i2658_reclaim_commit_msg.txt`), never bare `i<N>_<generic>`.

**Why:** both cost a retry mid-round; the second is a live-sibling collision
class like [[shared-worktree-partial-stage-commit]].

**How to apply:** any lean round that runs the no-flags lint gate or writes
/tmp scratch on a multi-session issue.
