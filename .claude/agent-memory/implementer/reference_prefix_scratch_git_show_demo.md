---
name: prefix-scratch-git-show-demo
description: Demonstrate fail-pre-fix for tests of importlib-by-path scripts by git-show extracting the pre-fix scripts into a /tmp scratch with the NEW test file
metadata:
  type: reference
---

Tests that load `scripts/*.py` via importlib (`_SCRIPTS = Path(__file__).parent.parent / "scripts"`,
e.g. `tests/test_vm_disk_guard_slurm_src.py`) can be run against a PRE-fix
tree without any stash/checkout: `mkdir scratch/{scripts,tests}`, then
`git -C $WT show <pre-sha>:scripts/<f>.py > scratch/scripts/<f>.py` per
script, copy the NEW test file into `scratch/tests/`, and run
`$WT/.venv/bin/python -m pytest scratch/tests/<file> -k <new-tests> -p no:cacheprovider`
(PYTHONPATH=$WT/src). All new regression tests should FAIL there and PASS in
the worktree — the After-Implementation step-5 fail-pre-fix/pass-post-fix
demonstration with zero mutation of the worktree (#2147 r3: 10/10 failed
pre-fix at a6de37bf8567, 10/10 passed post-fix).

**How to apply:** name the scratch to dodge the janitor's /tmp sweep shapes
(`mktemp -d /tmp/r3pre-XXXX` — avoid `*scratch*`/`*smoke*`/`*-gate*`
prefixes), and `rm -rf` it when done. Self-contained fixture files only —
repo `tests/conftest.py` is not copied, so this works only for test files
whose fixtures are in-file (true for the importlib-loader family).
