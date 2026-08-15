"""Tests for scripts/cron_autonomous_session_watch.sh log naming (task #2141).

The wrapper writes its daily log under the LOCAL date (the fleet convention —
all 11 cron wrappers) — on this UTC-7 VM a UTC-dated read therefore hit a
NONEXISTENT path for 7h/day (local 17:00-23:59 = next-day UTC 00:00-06:59),
which is how a "the watcher never ran" false report gets filed (#2141
defect 2, clarifier-confirmed). The fix keeps the local write name and adds a
RELATIVE UTC-date symlink whenever the two date names differ, guarded so it
can never clobber a real file or self-link, refreshed with `ln -sfn`, and the
stale alias at a rolled-over local name is removed (symlink-only `rm`) before
the day's first append.

Shape copied from tests/test_cron_step9c_ledger_refresh.py: drive the bash
wrapper via subprocess with env overrides — EPM_WATCH_LOG_DIR (tmp log dir),
EPM_WATCH_BIN (a stub sh script recording each invocation; NO test runs the
real watcher), and EPM_WATCH_DATE_LOCAL_OVERRIDE / EPM_WATCH_DATE_UTC_OVERRIDE
(date pins, so the misaligned 7h window is reproducible at any wall time).

T1-T6 pin plan v3 §4 D4/D5:
  T1 aligned dates -> single REAL file, no symlink, pointer line on stdout.
  T2 misaligned window -> the UTC name is a RELATIVE symlink to the local
     file; reading through the UTC name returns the live log content (the
     exact read that failed in the incident).
  T3 next local day -> the stale alias at the new local name is replaced by a
     fresh REAL file; yesterday's log intact; pointer line printed again.
  T4 a pre-existing REAL file at the UTC name is never clobbered.
  T5 same-day second run appends without a duplicate pointer line.
  T6 the EPM_WATCH_BIN seam is invoked exactly once per run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parent.parent / "scripts" / "cron_autonomous_session_watch.sh"

LOCAL = "2026-08-14"  # the misaligned 7h window: local 17:00-23:59 ...
UTC = "2026-08-15"  # ... = UTC 00:00-06:59 of the NEXT date

_POINTER = "per-pass output"  # the once-per-local-day stdout pointer line


@pytest.fixture
def harness(tmp_path: Path):
    """Isolated wrapper harness: tmp log dir + a stub watcher recording each
    invocation. Every run passes all four env overrides — no test ever runs
    the real watcher or touches the real log dir."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    calls = tmp_path / "stub_calls.log"
    stub = tmp_path / "stub_watcher.sh"
    stub.write_text(f"#!/usr/bin/env bash\necho invoked >> {calls}\necho STUB_WATCHER_RAN\n")
    stub.chmod(0o755)

    def _run(local_date: str, utc_date: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(
            EPM_WATCH_LOG_DIR=str(log_dir),
            EPM_WATCH_BIN=str(stub),
            EPM_WATCH_DATE_LOCAL_OVERRIDE=local_date,
            EPM_WATCH_DATE_UTC_OVERRIDE=utc_date,
        )
        return subprocess.run(
            ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True, timeout=120
        )

    return {"log_dir": log_dir, "calls": calls, "run": _run}


def test_aligned_dates_single_real_file(harness):
    """T1: local == UTC -> one REAL file (never a symlink), stub marker +
    start/exit lines inside it, the only *.log entry, pointer line on stdout,
    exit 0."""
    res = harness["run"](UTC, UTC)

    assert res.returncode == 0
    log = harness["log_dir"] / f"{UTC}.log"
    assert log.is_file() and not log.is_symlink()
    content = log.read_text()
    assert "STUB_WATCHER_RAN" in content
    assert "autonomous_session_watch start" in content
    assert "exit=0" in content
    assert sorted(p.name for p in harness["log_dir"].glob("*.log")) == [f"{UTC}.log"]
    assert _POINTER in res.stdout  # first run of day


def test_misaligned_window_creates_utc_alias(harness):
    """T2: local != UTC -> real local file + a RELATIVE symlink at the UTC
    name; reading through the UTC name returns the log content (the exact
    read that failed in the incident)."""
    res = harness["run"](LOCAL, UTC)

    assert res.returncode == 0
    local_log = harness["log_dir"] / f"{LOCAL}.log"
    utc_log = harness["log_dir"] / f"{UTC}.log"
    assert local_log.is_file() and not local_log.is_symlink()
    assert utc_log.is_symlink()
    assert os.readlink(utc_log) == f"{LOCAL}.log"  # RELATIVE target
    assert "STUB_WATCHER_RAN" in utc_log.read_text()


def test_next_local_day_replaces_alias_with_real_file(harness):
    """T3: after a misaligned run, the local day rolls over onto the alias
    name — the wrapper removes the stale symlink (never a real file) and
    creates a fresh REAL file; the prior day's log is untouched; the pointer
    line prints again (fresh local day)."""
    harness["run"](LOCAL, UTC)  # leaves <UTC>.log -> <LOCAL>.log
    res = harness["run"](UTC, UTC)  # local day rolled over; names aligned

    assert res.returncode == 0
    new_log = harness["log_dir"] / f"{UTC}.log"
    old_log = harness["log_dir"] / f"{LOCAL}.log"
    assert new_log.is_file() and not new_log.is_symlink()
    assert new_log.read_text().count("STUB_WATCHER_RAN") == 1  # only the 2nd run's lines
    assert old_log.is_file() and "STUB_WATCHER_RAN" in old_log.read_text()  # intact
    assert _POINTER in res.stdout


def test_never_clobbers_real_file_at_utc_name(harness):
    """T4: a pre-existing REAL file at the UTC name is defended — bytes
    unchanged, no symlink replaces it, the local file is still written,
    exit 0."""
    pre = harness["log_dir"] / f"{UTC}.log"
    pre.write_text("PRECIOUS REAL FILE\n")

    res = harness["run"](LOCAL, UTC)

    assert res.returncode == 0
    assert not pre.is_symlink()
    assert pre.read_text() == "PRECIOUS REAL FILE\n"
    local_log = harness["log_dir"] / f"{LOCAL}.log"
    assert local_log.is_file() and "STUB_WATCHER_RAN" in local_log.read_text()


def test_same_day_second_run_appends_no_duplicate_pointer(harness):
    """T5: two aligned runs -> one pointer line total, both runs' markers in
    the (single, real) file."""
    res1 = harness["run"](UTC, UTC)
    res2 = harness["run"](UTC, UTC)

    assert _POINTER in res1.stdout
    assert _POINTER not in res2.stdout
    log = harness["log_dir"] / f"{UTC}.log"
    assert log.read_text().count("STUB_WATCHER_RAN") == 2


def test_bin_seam_invoked(harness):
    """T6: the EPM_WATCH_BIN stub records exactly one invocation per wrapper
    run (the seam replaces the real `uv run python` driver invocation)."""
    harness["run"](UTC, UTC)
    assert harness["calls"].read_text().count("invoked") == 1
    harness["run"](LOCAL, UTC)
    assert harness["calls"].read_text().count("invoked") == 2
