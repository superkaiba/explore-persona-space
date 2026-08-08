---
name: Worktree pytest imports MAIN's src, not the worktree copy
description: The editable install resolves explore_persona_space to /home/.../explore-persona-space/src (main checkout); a worktree's src/ edits are invisible to pytest unless PYTHONPATH forces them
type: reference
---

When validating a worktree edit to a `src/explore_persona_space/**` module
(e.g. `backends/gcp.py`, `task_workflow.py`), pytest run with the main
checkout's venv imports the package from
`/home/thomasjiralerspong/explore-persona-space/src/...` — the MAIN
checkout's source tree, NOT the worktree's `src/` copy. The package is
installed editable against the main repo root, so `import
explore_persona_space.backends.gcp` resolves there regardless of cwd.

Consequence: a worktree edit to `src/.../gcp.py` is INVISIBLE to a plain
`pytest` run, and any test that depends on the worktree-branch's
not-yet-merged sibling commits will FAIL with a confusing "the fix didn't
fire" assertion (the test sees main's older code). On #634 this manifested
as `test_reconnect_skips_running_instance_with_terminal_phase_done`
"failing" on the issue-634 branch even though the sibling fix (03b57e34a)
was committed on that branch — pytest was reading main's gcp.py, which
lacked the fix.

Fix: force the worktree's src ahead of the install:

    PYTHONPATH="$(git rev-parse --show-toplevel)/src" \
      /home/thomasjiralerspong/explore-persona-space/.venv/bin/python \
      -m pytest tests/test_<module>.py -q

Verify resolution first:
    PYTHONPATH="$(pwd)/src" .venv/bin/python -c \
      "import explore_persona_space.backends.gcp as g; print(g.__file__)"
— must print the WORKTREE path, not the main-checkout path.

Why: confirm a "pre-existing test failure" is real before treating it as
out of scope. A bare `python` with `sys.path.insert(0,"src")` passes (uses
the worktree copy) while pytest fails (uses main) — that delta is the
tell that you're hitting this install-resolution quirk, not a genuine bug.
Pure-stdlib script edits (scripts/*.py imported by path) are unaffected;
this only bites `src/`-package imports.

UPDATE (2026-08-07, issue-2176): this trap applies when running MAIN's
venv against a worktree. When `uv run` is invoked FROM the worktree, it
builds a FRESH worktree `.venv` (216 pkgs, <1 s on warm cache) whose
editable install points at the WORKTREE root — a brand-new
`src/.../orchestrate/argcheck.py` imported fine with no PYTHONPATH.
Verify which regime you are in with the `print(g.__file__)` probe above
before assuming either direction; the disk-full ENOSPC caveat
([[worktree-uv-venv-fails-on-full-disk]]) is largely retired now that
worktrees live on /mnt/eps-data.
