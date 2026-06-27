---
name: repo_root() unfit for cache paths
description: task_workflow.repo_root() branch-guards + can reset-hard/return a managed worktree when primary is off-main; for cwd-independent cache paths use git-common-dir parent directly
type: reference
---

When making a `.claude/cache/` (or similar shared-state) path cwd-independent, do
NOT reach for `task_workflow.repo_root()` even though candidates suggest it: when
the primary checkout is parked off-`main` it (a) runs `reset --hard main` on a
managed `_task-main-pin` worktree as a side effect and (b) RETURNS that managed
worktree path, not the primary root — wrong anchor for a cache file. The right
pattern (applied in `backends/issue_dispatch.py::_main_checkout_root`, #612) is a
side-effect-free `git rev-parse --path-format=absolute --git-common-dir` run from
the MODULE's directory (never `os.getcwd()`), validate basename `.git`, take the
parent, `lru_cache` it. Test isolation: cwd-pinning fixtures (`_cd_to_tmp`) must
ALSO monkeypatch the resolver (`monkeypatch.setattr(idp, "_main_checkout_root",
lambda: tmp_path)`) — chdir alone no longer redirects the path.
