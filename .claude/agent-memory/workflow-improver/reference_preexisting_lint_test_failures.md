---
name: preexisting-lint-test-failures
description: Repo-wide `ruff check scripts` has ~1300+ pre-existing errors and tests/ has pre-broken modules; use the git-stash test to prove your diff adds none
metadata:
  type: reference
---

As of 2026-06-09, the success criterion "`ruff check .claude scripts` passes" is
unsatisfiable repo-wide: ~1338 pre-existing ruff errors live in untouched
experiment-analysis scripts (`scripts/analyze_*.py` etc.). Likewise
`tests/test_data_validation.py` fails collection in any fresh worktree (imports
the UNTRACKED `explore_persona_space.data.wrong_answers_deterministic`), and the
`tests/test_task_workflow.py::test_migrate_body_*` family (6 tests) fails on a
stale `CANONICAL_PASS_BODY` fixture.

**How to apply:** prove your diff is clean with the stash test — `git stash -q
&& <check> ; git stash pop -q` — and report "identical with edits stashed,
pre-existing" rather than chasing repo-wide failures. For pytest, deselect the
known-bad family (`-k "... and not test_migrate_body"`,
`--ignore=tests/test_data_validation.py`) to expose any failure your edits
actually introduce. Verify these are still broken before citing this memory.
