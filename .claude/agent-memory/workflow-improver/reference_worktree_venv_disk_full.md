---
name: Worktree uv venv build fails on full disk — use main venv python
description: In agent worktrees, `uv run` builds a fresh .venv (GBs); when the VM disk is full this fails with ENOSPC. Run main checkout's .venv/bin/python against the worktree's script copy instead.
type: reference
---

`uv run python scripts/workflow_lint.py ...` inside a workflow-improver
worktree triggers a full venv build at `<worktree>/.venv` — observed failing
with `No space left on device` when the VM root disk hit 100% (2026-06-09,
484G/485G used).

**Workaround that works:** invoke the main checkout's interpreter directly
against the WORKTREE's copy of the script:

```bash
/home/thomasjiralerspong/explore-persona-space/.venv/bin/python \
  <worktree>/scripts/workflow_lint.py --check-asks
/home/thomasjiralerspong/explore-persona-space/.venv/bin/python -m pytest tests/test_workflow_lint.py -x -q   # cwd = worktree
```

This is correct because `workflow_lint.py` resolves `_REPO_ROOT` from
`__file__` (line ~55), so the worktree's script copy lints worktree files.
Same pattern applies to pytest run from the worktree cwd. Remove any
partially-built `<worktree>/.venv` left behind by the failed `uv run`.
