---
name: preexisting-lint-test-failures
description: Broad `ruff check .claude scripts` (~1338 pre-existing errors) and a few tests/ modules are pre-broken; lint touched files only and prove "0 introduced" via the git-stash compare
metadata:
  type: reference
---

The success criterion "`ruff check .claude scripts` passes" is unsatisfiable
repo-wide: ~1338 pre-existing errors (B905 `zip()` strict, RUF001/RUF002
ambiguous-unicode) live in untouched experiment scripts
(`scripts/generate_sdf_variants.py`, `scripts/analyze_*.py`, ...) — none from
workflow-surface files. The agent spec §5 + workflow-fix-on-bug.md templates
already prescribe touched-files-only ruff (fixed 2026-06-09, `16a6b57a4`) — do
NOT re-emit that candidate.

**How to apply:** lint only the files you touched (`ruff check <paths>` +
`ruff format --check <paths>`). If you want a repo-wide regression signal,
stash-compare: `git stash -q && <check>; git stash pop -q`, compare counts,
report "PASS (N pre-existing, 0 introduced)". For pytest, expose only your own
breakage with `--ignore=tests/test_data_validation.py` (imports an untracked
module; pre-broken in fresh worktrees).

**A bare `uv run ruff format .` (no `--check`) REFORMATS the pre-existing
format-dirty files** — 44 out-of-scope files churned on 2026-08-25 (#2568 r3;
eps/experiments/, old issue scripts, tasks/ artifacts). Restore all but your
payload with the guard-compliant worktree form:
`git -C "$WT" status --porcelain | awk '{print $2}' | grep -vE '^(payload1|...)$' > /tmp/restore.txt`
then `xargs -a /tmp/restore.txt git -C "$WT" checkout --` (a bare
`git checkout --` without `-C <worktree>` is hook-BLOCKED as a repo-root
working-tree revert even when cwd is the worktree — the guard matches command
text, not cwd).

Known states (verify before citing — these get fixed over time):
- FIXED and fully passing — do not deselect: `tests/test_task_workflow.py`
  (79/79, `75c78e9f3`), `tests/test_verify_clean_result.py` (regex +
  date-gate fixes; 4 pre-existing E501s remain in the legacy verifier),
  `tests/test_stalled_detector_and_gc.py` (44/44, pins fc3f98719 semantics).
- Pre-broken on main as of 2026-06-11: `tests/test_workflow_yaml.py::
  test_gates_full_shape` (campaign commit `9eb2c7c57` added a second
  park_and_wait gate; test asserts len==1). Stash-compare proves it.
- **Files-mode `workflow_lint.py --files` enumeration artifacts (2026-08-24,
  #2336 b3/b4):** a restricted-enumeration run reddens on rows a full-tree run
  does not — (i) `sha-pin-domain/grandfather-stale` (#2559 class; prove with
  full-tree `--check-sha-pin-domain` → PASS rc=0) and (ii) #2235 per-issue
  import-closure rows on payload files whose sibling modules were NEVER
  tracked (e.g. `issue541_geometry_extract` → `issue541_personas`); prove by
  running the SAME `--files <file>` invocation on the unmodified MAIN checkout
  (read-only) → identical rows. Report both classes with the proof; never
  block a batch on them.
- **Cross-file FULL-SUITE flake — RESOLVED 2026-06-28 via #703.** The former
  isolation-pollution failures (HF_HOME env leak, root-logger-level leak,
  unguarded `sys.modules["worktree_audit"]` replacement, stale `_PR`
  PollResult stubs missing `stall_reason`) are fixed by an autouse env+logging
  snapshot/restore fixture in `tests/conftest.py` + a guarded `worktree_audit`
  loader in `tests/test_worktree_audit.py` + completed `_PR` stubs in
  `tests/test_issue_dispatch.py`. The full `uv run pytest tests/` suite is
  green. Treat ANY renewed full-suite-only failure as a REAL ordering
  regression to diagnose — do NOT re-add a special-case or deselect.
