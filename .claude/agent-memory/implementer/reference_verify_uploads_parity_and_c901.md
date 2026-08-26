---
name: reference-verify-uploads-parity-and-c901
description: Parity-testing an old revision of a __file__-rooted script requires staging it under <repo>/scripts/; verify_uploads.py sits under the C901<=15 full-ruleset pin - extract helpers, never noqa
metadata:
  type: reference
---

Two traps from #2578 (extending `scripts/verify_uploads.py`):

1. **Old-revision parity copies must live where `__file__` resolves the same
   repo root.** `verify_uploads.py` (and siblings) derive
   `REPO_ROOT = Path(__file__).resolve().parent.parent`; a
   `git show <BASE>:scripts/verify_uploads.py > /tmp/copy.py` parity run
   resolves REPO_ROOT=/ and silently changes behavior (git arms ERROR
   "not a git repository", eval-json/figures checks read nothing) — the
   parity diff then blames the NEW code. Stage the old revision at
   `<worktree>/scripts/<name>_parent.py` (untracked), `trap rm` it, and
   run both scripts from the same cwd.

2. **Feature branches on `LIVE_WORKFLOW_HELPERS` files trip the C901<=15
   full-ruleset pin** (`tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`)
   even though bare `ruff check` passes (per-file-ignores relax scripts/).
   Nested `def`s count toward the ENCLOSING function's mccabe complexity —
   hoisting nested helpers to module level is the cheap fix and keeps
   behavior byte-identical (re-run the suites + any live battery after).
   Run the policy pin BEFORE writing the report whenever the diff touches
   a roster file; `noqa: C901` is the wrong tool on a live gate.

**How to apply:** any round that adds branches to `verify_uploads.py`,
`task.py`, `workflow_lint.py`, or another roster file budget-checks
complexity at write time (design new logic as module-level helpers with
their own docstrings) and stages old-revision comparison copies inside the
repo tree, never /tmp.
