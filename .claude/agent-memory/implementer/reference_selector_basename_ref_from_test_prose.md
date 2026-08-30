---
name: selector-basename-ref-from-test-prose
description: naming a workflow helper by filename in a test's docstring/comment creates a real Step-9c selector dependency edge — enough to break another test's curated exact-set pin
metadata:
  type: reference
---

`scripts/select_step9c_tests.py` has a **basename-ref** arm: it maps a test
to a source file when the test merely MENTIONS that file's basename, in
prose as well as in code. A docstring or comment naming a workflow helper
(`select_step9c_tests.py`, `verify_uploads.py`, ...) therefore creates a
genuine dependency edge in the selector's graph — the scan reads file TEXT,
not imports.

**Why:** #2386 unit 3 added `tests/test_cron_wrapper_log_dir_guard.py` whose
docstring named the selector while the test never reads it. The phantom edge
broke `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree`,
a deliberate exact-set pin over the live tree. The fix was removing the
filename from the prose, NOT widening the other test's curated pin — a
curated exact-set pin is an oracle; widening it to absorb an edge you
invented destroys the thing it exists to catch.

**How to apply:** when writing or editing a test, keep workflow-helper
FILENAMES out of docstrings and comments unless the test genuinely exercises
that helper. Refer to it by role ("the Step 9c selector") instead. If a
`--map-files` sweep or `tests/test_select_step9c_tests.py` goes red right
after you touched only prose, suspect this before suspecting the pin. Same
trap in the other direction: a fixture literal that looks like a real path
trips live-repo scanners ([[feedback_fixture_literal_trips_live_repo_scanner]]),
and HF-call shapes in prose trip the retry-routing lint
([[feedback_hf_call_shape_prose_trips_lint]]).
