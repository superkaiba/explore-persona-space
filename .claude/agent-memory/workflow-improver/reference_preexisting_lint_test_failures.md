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
the UNTRACKED `explore_persona_space.data.wrong_answers_deterministic`).

UPDATE (2026-06-09, later run): the `test_migrate_body_*` family is FIXED
(fixture reconciliation, commit 75c78e9f3) — `tests/test_task_workflow.py`
passes fully (79/79). Do not deselect it anymore.

UPDATE (2026-06-09, later run): the `tests/test_verify_clean_result.py` 10-failure
family is FIXED — root cause was the documented `_extract_section` regex bug in
`scripts/verify_clean_result.py` (`(?:\s+.*)?$` consumed the first content line;
fix `[ \t]+`), plus 2 tests pinning a branch made unreachable by the v2 date-gate
and 1 test reading the retired `template.md` (retired with pointer comment). File
now passes fully; do not deselect it. The legacy verifier still carries 4
pre-existing E501s (lines 18/580/620/1815) — lint via stash-compare.

UPDATE (2026-06-10, later run): the `tests/test_stalled_detector_and_gc.py`
3-failure family is FIXED — the tests now pin the merged watcher's (fc3f98719,
#505) semantics: manual registrations get ALERT-ONLY (never respawn; renamed
`test_stalled_pass_manual_session_alert_only_never_respawns`), and the two
main()-wiring tests patch the real pass set (`vm_disk_pass` / `pod_safety_pass`
/ `stalled_session_pass` / `orphan_sweep_pass` / `gc_pass`; `_load_session_meta`
no longer exists). File passes fully (44/44); do not deselect it.

**How to apply:** prove your diff is clean with the stash test — `git stash -q
&& <check> ; git stash pop -q` — and report "identical with edits stashed,
pre-existing" rather than chasing repo-wide failures. For pytest, use
`--ignore=tests/test_data_validation.py` to expose any failure your edits
actually introduce. Verify these are still broken before citing this memory.
