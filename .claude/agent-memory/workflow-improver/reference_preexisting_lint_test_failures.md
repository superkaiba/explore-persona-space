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

Known states (verify before citing — these get fixed over time):
- FIXED and fully passing — do not deselect: `tests/test_task_workflow.py`
  (79/79, `75c78e9f3`), `tests/test_verify_clean_result.py` (regex +
  date-gate fixes; 4 pre-existing E501s remain in the legacy verifier),
  `tests/test_stalled_detector_and_gc.py` (44/44, pins fc3f98719 semantics).
- Pre-broken on main as of 2026-06-11: `tests/test_workflow_yaml.py::
  test_gates_full_shape` (campaign commit `9eb2c7c57` added a second
  park_and_wait gate; test asserts len==1). Stash-compare proves it.
