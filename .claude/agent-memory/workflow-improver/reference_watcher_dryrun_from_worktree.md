---
name: watcher-dryrun-from-worktree
description: How to demo autonomous_session_watch passes against the live registry from a worktree — override asw.PROJECT_ROOT to the main checkout (task.py branch-guards to main)
metadata:
  type: reference
---

Running `scripts/autonomous_session_watch.py` (or one of its `*_pass`
functions) from a workflow-improver worktree reads every task status as
`None`: `PROJECT_ROOT` (imported from `spawn_session`, derived from
`__file__`) points at the worktree, so the `task.py view` subprocesses run on
the non-`main` worktree branch and the branch-guard refuses.

Working demo pattern (read-only, dry-run):

```python
import sys
from pathlib import Path
sys.path.insert(0, "<worktree>/scripts")   # load YOUR edited module
import autonomous_session_watch as asw
asw.PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
asw.session_reconcile_pass(True, 2, daemon_reachable=asw._daemon_reachable())
```

Notes: run with the MAIN checkout's `.venv/bin/python` (no worktree venv
build, see [[worktree-venv-disk-full]]); `~/.eps-autonomous` +
`~/.happy` state is `Path.home()`-based so it is shared either way; the
RunPod API call needs `.env` exported (a bare `.venv/bin/python` shell lacks
it — the pod helpers fail-soft to `[]` with a warning, which is fine for a
dry-run demo). Also: this repo's ruff has C901 max-complexity 15 — adding one
branch to `_process_session_reconcile`-sized functions trips it; collapse
parallel branches (e.g. a message dict for skip actions).
