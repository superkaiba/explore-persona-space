---
name: fixture-literal-trips-live-repo-scanner
description: A test FIXTURE literal can turn a live repo-wide invariant scanner red — tests/ is in the scan population and most test files are not allowlisted, so writing a realistic offending line into a fixture self-inflicts a fleet-wide red
metadata:
  type: feedback
---

When writing fixtures for a test that exercises one of the repo-wide invariant
scanners, the MOST REALISTIC fixture string is often the one that makes the
REAL scanner fire on your own test file. Four scanners walk the repo and flag
offending lines; their scan populations include `tests/`:

- `tests/test_no_direct_task_path_construction.py` — `_SCAN_ROOTS =
  ("src", "scripts", "tests")`, patterns `PROJECT_ROOT\s*/\s*"tasks"` and
  `\bROOT\s*/\s*"tasks"`, with a small `_FILE_ALLOWLIST` most test files are
  NOT in. **This is the dangerous one for fixtures.**
- `tests/test_no_pod_side_task_py_shellout.py` — AST-based over `scripts|src`
  only, so a string literal in a `tests/` file is structurally unreachable.
- `tests/test_no_dollar_budget_caps.py` — rglobs `scripts/` only.
- `tests/test_shared_vm_thread_caps.py` — `scripts/**` + src experiments.

So a fixture that quotes `ROOT / "tasks"` inside `tests/test_*.py` does not
merely test the scanner — it BECOMES an offender the scanner reports, turning a
green repo-wide invariant red for every session's Step 9c gate (the #1388
fleet-wedge shape). The irony to avoid: this bites hardest when the task IS
fixing a false-block in the gate.

**Why:** extraction/reporting logic is content-agnostic past the row's leading
path token, so the fixture's INTERIOR text is free — only the row GRAMMAR is
load-bearing. You can defuse the literal without weakening what the test proves.

**How to apply:** when a fixture must imitate an offending source line for one
of these scanners, pick a spelling that matches the row grammar byte-for-byte
but NOT the scanner's own pattern — e.g. `base / "tasks"` instead of
`ROOT / "tasks"` (matches neither `PROJECT_ROOT` nor `\bROOT`), and
`/max_[b]udget_usd/` instead of the bare symbol. Then RUN the four scanners in
the worktree before committing (~90 s) — do not infer safety from reading the
patterns. State the substitution in the implementation report so the reviewer
can check it did not hollow out the assertion; a reviewer will (and should)
ask both whether the fixture is still a faithful instance of the grammar and
whether the defusal is complete across every new fixture string.

Worked example: #2319 D3/D5 fixtures for the five `VIOLATION_SET_SCAN_NODES`
members. Related: [[workflow-invariant-two-place-registration]] (the other
"this test file participates in repo-wide machinery" trap) and
[[preexisting-lint-test-failures]].
