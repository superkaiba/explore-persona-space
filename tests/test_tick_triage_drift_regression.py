"""Regression pins for #2369 — module-level wall-clock capture in test files.

``tests/test_tick_triage.py`` used to capture ``NOW = time.time()`` at module
import (pytest collection) while ``tick_triage.triage()`` measures marker /
transcript ages against the live clock at test execution. In any pytest
session where the file executes >=~20 min after collection, "fresh" fixture
events silently age past tick_triage's staleness thresholds and fail 4-12
tests against innocent branches (incidents: #2158 Step 9c gate, 22.2 min
drift -> 4 failures; #2168 gate, ~38 min drift -> 12 failures).

Two pins:

- **Test A (drift simulation, acceptance criterion 1):** run the 4 incident
  tests in a subprocess with a plugin that ages the module's clock anchor by
  2400 s at ``pytest_collection_finish`` — the same instrument the #2168
  controlled experiment used. Post-fix (autouse ``_fresh_now`` re-anchors NOW
  at test setup) the 4 tests must PASS. NON-VACUITY: the plugin resolves the
  target strictly via ``sys.modules["tests.test_tick_triage"]`` (KeyError =
  fail-loud; never a fresh import, which would age a DUPLICATE module object
  the collected tests never read) and prints a sentinel only AFTER a
  successful setattr; this test asserts the sentinel in the subprocess
  stdout in addition to ``returncode == 0``.
- **Test B (mechanical pin, acceptance criterion 3):** AST-scan the two
  fixed files for any module-scope Assign/AnnAssign whose value calls a
  wall clock (``time.time`` / ``time.monotonic`` / ``datetime.now`` /
  ``datetime.utcnow``).

The pin is deliberately scoped to the two fixed files: widening it to all of
``tests/`` would false-positive on deliberately-OLD anchors like
``AGED_TS = time.time() - 100*3600`` (drift only widens those in the SAFE
direction; #2369 plan v2 sweep disposition).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DRIFT_SECONDS = 2400
SENTINEL = "DRIFT-INJECTED tests.test_tick_triage NOW-=2400"

# The 4 tests that failed in the #2158/#2168 gate incidents (plan v2
# acceptance criterion 1).
INCIDENT_TEST_IDS = [
    "tests/test_tick_triage.py::test_api_error_after_marker_returns_stale_redrive",
    "tests/test_tick_triage.py::test_api_error_after_marker_falls_open_on_missing_transcript",
    "tests/test_tick_triage.py::test_api_error_probe_kill_switch",
    "tests/test_tick_triage.py::test_api_error_probe_incident_1689_content_string",
]

_DRIFT_PLUGIN = f'''
"""#2369 drift-injection plugin: ages tests.test_tick_triage.NOW by
{DRIFT_SECONDS} s at pytest_collection_finish — the #2168 controlled-experiment
instrument simulating a long gate session's collection->execution drift."""
import sys


def pytest_collection_finish(session):
    # NON-VACUITY CONTRACT (#2369 plan v2, critic round-1 blocking finding):
    # tests/ is a package (tests/__init__.py exists), so under pytest's
    # prepend import mode the COLLECTED module's sys.modules key is
    # "tests.test_tick_triage". Resolve STRICTLY via sys.modules — a fresh
    # import after a sys.path insertion would create a DUPLICATE module
    # object whose NOW the collected tests never read, making the pin
    # permanently, vacuously green. A miss raises KeyError (fail-loud).
    mod = sys.modules["tests.test_tick_triage"]
    mod.NOW = mod.NOW - {DRIFT_SECONDS}
    # Printed ONLY after the successful setattr above.
    print("{SENTINEL}")
'''


def test_drift_simulation_incident_tests_pass(tmp_path):
    """Test A (#2369): the 4 incident tests pass under a simulated 2400 s
    collection->execution wall-clock drift, and the drift injection is
    proven to have reached the collected module (sentinel in stdout)."""
    plugin_path = tmp_path / "issue2369_drift_plugin.py"
    plugin_path.write_text(_DRIFT_PLUGIN)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-s",  # keep the plugin's collection-time sentinel print on stdout
            "-p",
            "issue2369_drift_plugin",
            *INCIDENT_TEST_IDS,
        ],
        # cwd = repo root so pyproject.toml addopts + tests/conftest.py
        # resolve identically to the Step 9c gate environment.
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert SENTINEL in proc.stdout, (
        "#2369 non-vacuity: drift-injection sentinel missing from subprocess "
        "stdout — the plugin never aged the collected module, so a PASS "
        f"below would be vacuous.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.returncode == 0, (
        "#2369 drift regression: the 4 incident tests FAILED under a "
        f"simulated {DRIFT_SECONDS} s collection->execution drift — the "
        "per-test clock anchor (autouse _fresh_now) is broken or removed.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# Files pinned against module-scope wall-clock capture (#2369). Deliberately
# NOT all of tests/ — see module docstring (safe-direction AGED_TS anchors).
PINNED_FILES = [
    "tests/test_tick_triage.py",
    "tests/test_async_gate_rung0.py",
]

_WALL_CLOCK_SUFFIXES = ("time.time", "time.monotonic", "datetime.now", "datetime.utcnow")
# Known disclosed false negatives: alias escapes (`from time import time;
# NOW = time()`) and wrapper indirection (`NOW = _get_now()`) are not
# matched — Test A (non-vacuous per its sentinel contract) and code review
# are the catching arms for those.


def test_no_module_scope_wall_clock_capture():
    """Test B (#2369): no module-scope Assign/AnnAssign in the pinned files
    captures a wall clock — use the autouse ``_fresh_now`` per-test anchor
    (or a per-test ``now = time.time()`` threaded explicitly) instead."""
    offenders: list[str] = []
    for rel in PINNED_FILES:
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if node.value is None:
                continue
            for sub in ast.walk(node.value):
                if not isinstance(sub, ast.Call):
                    continue
                dotted = ast.unparse(sub.func)
                if any(
                    dotted == suffix or dotted.endswith("." + suffix)
                    for suffix in _WALL_CLOCK_SUFFIXES
                ):
                    offenders.append(f"{rel}:{sub.lineno} ({dotted})")
    assert not offenders, (
        "#2369: module-scope wall-clock capture reintroduced — a module-level "
        "clock anchor ages during long pytest sessions (20-40 min "
        "collection->execution drift = 4-12 spurious failures; #2158/#2168). "
        "Refresh the anchor in an autouse fixture (see _fresh_now in "
        f"tests/test_tick_triage.py) instead. Offenders: {offenders}"
    )
