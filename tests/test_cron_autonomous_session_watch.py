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

T7-T9 pin the #2386 fail-loud log-dir guard (Pattern A — stderr-only fatal;
this wrapper defines no TELEGRAM_PUSH). Before #2386 an uncreatable or
unwritable log dir made the brace-group redirect fail, the ENTIRE watcher pass
(crash-recovery, pod-safety, every reaper) never ran, and the wrapper still
exited 0 — the highest blast radius in the fix set:
  T7 uncreatable log dir -> rc != 0, stderr FATAL naming the dir, stub NOT run.
  T8 existing-but-unwritable log dir (root-skipif) -> the probe arm fires.
  T9 the probe sits AFTER the #2141 symlink removal: a STALE DANGLING alias at
     the local log name must not false-FATAL a healthy run. The stale alias
     points into a NONEXISTENT directory, which is what makes the pin
     discriminating — `: >> "$LOG_FILE"` through such a symlink fails ENOENT,
     so a probe placed BEFORE the removal would fatal on a healthy run, while
     the shipped order removes the symlink first and the probe creates a fresh
     real file. (Unit 2 verified this ordering behaviourally; T9 pins it.)
The three log-dir modes use the same vocabulary as
tests/test_cron_wrapper_log_dir_guard.py ("ok" / "uncreatable" / "unwritable");
the setup is spelled locally rather than imported because each cron test file
weaves the log dir into its OWN env-construction idiom (this one has a fixed
four-override env, no extra_env seam).
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

# #2386 fatal-message fragments. The mkdir arm and the probe arm are
# distinguished by these, NOT by the dated log filename (which would race a
# midnight rollover). Spelled locally rather than imported: each cron test
# file weaves the guard into its own harness idiom.
_MKDIR_FATAL = "cannot create log dir"
_PROBE_FATAL = "not appendable"


@pytest.fixture
def harness(tmp_path: Path):
    """Isolated wrapper harness: tmp log dir + a stub watcher recording each
    invocation. Every run passes all four env overrides — no test ever runs
    the real watcher or touches the real log dir.

    ``run(local, utc, log_dir_setup=...)`` selects WHICH log dir is injected
    (#2386): ``"ok"`` (the pre-created default every T1-T6 call uses),
    ``"uncreatable"`` (a path under a regular FILE, so ``mkdir -p`` fails
    ENOTDIR — deterministic, no permission dependence) or ``"unwritable"``
    (an existing dir at 0o555, so ``mkdir -p`` passes and the appendability
    probe fails; root bypasses mode bits, hence the caller's skipif). Each
    mode owns a DISTINCT path under this test's own ``tmp_path``, so a fatal
    arm can never leave residue in the happy-path dir. ``log_dir_for(mode)``
    exposes the resolved path for assertions.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    calls = tmp_path / "stub_calls.log"
    stub = tmp_path / "stub_watcher.sh"
    stub.write_text(f"#!/usr/bin/env bash\necho invoked >> {calls}\necho STUB_WATCHER_RAN\n")
    stub.chmod(0o755)

    resolved: dict[str, Path] = {"ok": log_dir}

    def _log_dir_for(log_dir_setup: str) -> Path:
        """Resolve (creating on first use) the log dir for one setup mode."""
        if log_dir_setup in resolved:
            return resolved[log_dir_setup]
        if log_dir_setup == "uncreatable":
            blocker = tmp_path / "blocker"
            blocker.write_text("regular file blocking mkdir -p (ENOTDIR)\n")
            resolved[log_dir_setup] = blocker / "logs"
        elif log_dir_setup == "unwritable":
            unwritable = tmp_path / "unwritable_logs"
            unwritable.mkdir()
            unwritable.chmod(0o555)
            resolved[log_dir_setup] = unwritable
        else:  # pragma: no cover — harness misuse is a test defect
            raise ValueError(f"unknown log_dir_setup: {log_dir_setup!r}")
        return resolved[log_dir_setup]

    def _run(
        local_date: str, utc_date: str, log_dir_setup: str = "ok"
    ) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(
            EPM_WATCH_LOG_DIR=str(_log_dir_for(log_dir_setup)),
            EPM_WATCH_BIN=str(stub),
            EPM_WATCH_DATE_LOCAL_OVERRIDE=local_date,
            EPM_WATCH_DATE_UTC_OVERRIDE=utc_date,
        )
        return subprocess.run(
            ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True, timeout=120
        )

    return {"log_dir": log_dir, "calls": calls, "run": _run, "log_dir_for": _log_dir_for}


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


# ── T7-T9: the #2386 fail-loud log-dir guard (Pattern A, stderr-only) ────────


def test_uncreatable_log_dir_fails_loud(harness):
    """T7: an uncreatable $LOG_DIR (path under a regular file, ENOTDIR) exits
    non-zero with a stderr FATAL naming the dir, and the watcher stub NEVER
    runs — never the pre-#2386 silent skip-the-whole-pass-and-exit-0, which on
    this wrapper drops crash-recovery, pod-safety and every reaper."""
    log_dir = harness["log_dir_for"]("uncreatable")
    res = harness["run"](UTC, UTC, log_dir_setup="uncreatable")

    assert res.returncode != 0, f"expected non-zero exit, stderr={res.stderr!r}"
    assert "FATAL" in res.stderr
    assert _MKDIR_FATAL in res.stderr
    assert str(log_dir) in res.stderr
    assert not harness["calls"].exists(), "the watcher pass RAN despite an uncreatable log dir"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory mode bits")
def test_existing_unwritable_log_dir_fails_loud(harness):
    """T8: a $LOG_DIR that EXISTS but is unwritable (chmod 0o555) passes
    `mkdir -p` and fails the appendability probe — non-zero exit, the
    probe-specific FATAL naming the log FILE, stub never invoked."""
    log_dir = harness["log_dir_for"]("unwritable")
    res = harness["run"](UTC, UTC, log_dir_setup="unwritable")

    assert res.returncode != 0, f"expected non-zero exit, stderr={res.stderr!r}"
    assert "FATAL" in res.stderr
    # The probe arm, not the mkdir arm — mkdir -p succeeds on an existing dir.
    assert _PROBE_FATAL in res.stderr
    assert _MKDIR_FATAL not in res.stderr
    assert str(log_dir) in res.stderr
    assert not harness["calls"].exists(), "the watcher pass RAN despite an unwritable log dir"


def test_stale_dangling_alias_does_not_false_fatal(harness):
    """T9: probe-vs-#2141-symlink-removal ORDERING. A stale alias at the LOCAL
    log name pointing into a NONEXISTENT directory is removed first, so the
    probe creates a fresh real file and the run is healthy: exit 0, no FATAL,
    the stub RAN, the local log is a real file, and the UTC alias is still
    created (the probe creating $LOG_FILE does not disturb that block).

    Discriminating by construction: `: >> "$LOG_FILE"` through a symlink whose
    target directory does not exist fails ENOENT, so a probe placed before the
    removal would fatal on this healthy input."""
    log_dir = harness["log_dir"]
    stale = log_dir / f"{LOCAL}.log"
    stale.symlink_to("gone-dir/gone.log")  # dangling: target dir absent
    assert stale.is_symlink() and not stale.exists()

    res = harness["run"](LOCAL, UTC, log_dir_setup="ok")

    assert res.returncode == 0, f"stale alias false-FATALed: stderr={res.stderr!r}"
    assert "FATAL" not in res.stderr
    assert harness["calls"].read_text().count("invoked") == 1
    local_log = log_dir / f"{LOCAL}.log"
    assert local_log.is_file() and not local_log.is_symlink()
    assert "STUB_WATCHER_RAN" in local_log.read_text()
    utc_log = log_dir / f"{UTC}.log"
    assert utc_log.is_symlink()
    assert os.readlink(utc_log) == f"{LOCAL}.log"
    assert _POINTER in res.stdout  # the removed dangling alias is not a first-run log
