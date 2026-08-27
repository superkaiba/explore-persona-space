---
name: pytest-collection-print-capture
description: pytest's default capture swallows collection-phase plugin prints — a subprocess-pytest test asserting a collection-time sentinel in stdout must pass -s
metadata:
  type: reference
---

A pytest plugin hook that prints during COLLECTION (`pytest_collection_finish`,
`pytest_collectstart`, ...) has its output swallowed by pytest's default
global capture (`--capture=fd`) — the print only surfaces on collection
errors. A test that runs pytest in a SUBPROCESS and asserts a
collection-phase sentinel appears in `proc.stdout` (the #2369 non-vacuity
pattern: prove a drift-injection plugin actually reached the collected
module) must pass `-s` on the subprocess invocation; `sys.__stdout__` does
NOT bypass fd-level capture.

**Why:** #2369's Test A asserts `DRIFT-INJECTED ...` in subprocess stdout in
addition to rc==0 — without `-s` the sentinel assert fails even though the
injection worked.

**How to apply:** any subprocess `sys.executable -m pytest` invocation whose
assertions read plugin/conftest prints emitted BEFORE test execution: add
`-s` (safe when the inner tests don't depend on capture), or route the
sentinel through a file the outer test reads. Worked example:
`tests/test_tick_triage_drift_regression.py` (commit 3b5b2d7f2c3).
