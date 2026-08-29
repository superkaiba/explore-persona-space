---
name: prefix-scratch-git-show-demo
description: Demonstrate fail-pre-fix without mutating the worktree — git-show a /tmp scratch for importlib-by-path script tests, or monkeypatch the text seam for prose-pin tests
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

## Variant: prose-pin tests — monkeypatch the text seam, no scratch tree

Workflow-surface pin files (`tests/test_issue_skill_*.py`) read the shipped
spec through ONE module-level seam — `_text()` → `issue_skill_text()` in
`tests/test_issue_skill_lint_family_sync.py`. For those, the whole scratch
tree is unnecessary: `git show HEAD:<spec>.md > /tmp/pre.md`, then in a
throwaway script `sys.path.insert(0, $WT)`,
`import tests.<pin_module> as m`, `m._text = lambda: pre_text`, and CALL the
new test function directly, expecting `AssertionError`. Works even when the
span helpers (`_step5a_span`) slice by `text.index(<marker>)` — confirm both
markers exist in the pre-fix file first. Zero worktree mutation, so it is
safe with UNCOMMITTED edits in the tree (never `git checkout --` to undo a
mutation there — see [[mutation-restore-wipes-uncommitted]]).

Pass only the ONE file the test needs: a test that also slices a second spec
(`_automerge_span`) will raise `ValueError`, not `AssertionError`, if you
hand it a single-file text — run those separately (#2385 E5.7).

**How to apply:** name the scratch to dodge the janitor's /tmp sweep shapes
(`mktemp -d /tmp/r3pre-XXXX` — avoid `*scratch*`/`*smoke*`/`*-gate*`
prefixes), and `rm -rf` it when done. Self-contained fixture files only —
repo `tests/conftest.py` is not copied, so this works only for test files
whose fixtures are in-file (true for the importlib-loader family).
