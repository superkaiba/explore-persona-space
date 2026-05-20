"""Tests for ``scripts/watchdog_factor_screen_365.sh`` (round-7 hardening).

These cover the five behaviors the round-6 watchdog lacked:

1. **Clean exit on dispatcher rc=0.** Watchdog log captures startup,
   cycle-done with the dispatcher's rc, and a clean exit-0 message.
2. **Stall detection + respawn.** When a fake dispatcher hangs without
   touching the dispatcher log, the watchdog kills it after STALL_GAP
   and respawns at least once. PID file is cleaned up at exit.
3. **SIGTERM handling.** Sending SIGTERM mid-cycle causes the watchdog
   to log the signal, clean up the dispatcher process, remove the PID
   file, and exit non-zero.

Tests override ``WATCHDOG_POLL_SECONDS=2`` ``WATCHDOG_STALL_GAP_SECONDS=4``
``WATCHDOG_MAX_RESPAWNS=2`` to keep each test under ~30s.

The fake "dispatcher" is just a bash one-liner passed as DISPATCH_CMD;
the watchdog spawns it via ``bash -c "$DISPATCH_CMD"`` exactly the way
it spawns the real ``uv run python ...`` dispatcher.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

# Reference contextlib at module scope so ruff doesn't strip the import.
_SUPPRESS = contextlib.suppress

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCHDOG = REPO_ROOT / "scripts" / "watchdog_factor_screen_365.sh"


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _launch_watchdog(
    tmp_path: Path,
    dispatch_cmd: str,
    *,
    poll: int = 2,
    stall_gap: int = 4,
    max_respawns: int = 2,
    cool_down: int = 1,
    pid_file: Path | None = None,
) -> tuple[subprocess.Popen, Path, Path, Path]:
    """Launch the watchdog as a subprocess.

    Returns (popen, dispatcher_log, watchdog_log, pid_file).
    """
    dispatcher_log = tmp_path / "dispatcher.log"
    watchdog_log = tmp_path / "watchdog.log"
    slab_root = tmp_path / "slab"
    slab_root.mkdir()
    if pid_file is None:
        pid_file = tmp_path / "watchdog.pid"

    env = os.environ.copy()
    env["WATCHDOG_POLL_SECONDS"] = str(poll)
    env["WATCHDOG_STALL_GAP_SECONDS"] = str(stall_gap)
    env["WATCHDOG_MAX_RESPAWNS"] = str(max_respawns)
    env["WATCHDOG_COOL_DOWN_SECONDS"] = str(cool_down)
    env["WATCHDOG_PID_FILE"] = str(pid_file)

    proc = subprocess.Popen(
        [
            "bash",
            str(WATCHDOG),
            str(dispatcher_log),
            str(slab_root),
            dispatch_cmd,
            str(watchdog_log),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # New process group so we can deliver signals only to the watchdog.
        preexec_fn=os.setsid,
    )
    return proc, dispatcher_log, watchdog_log, pid_file


def _wait_for(proc: subprocess.Popen, timeout: float) -> int | None:
    """Wait for proc to exit, return its rc or None on timeout."""
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def _kill_group(proc: subprocess.Popen) -> None:
    """Best-effort cleanup of a process group."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def test_dispatcher_clean_exit(tmp_path: Path) -> None:
    """Happy path: dispatcher runs briefly and exits 0 — watchdog exits 0
    and the log shows startup, cycle-done, and dispatcher-success.
    """
    # Sleep 3s then exit 0 — under MAX_RESPAWNS=2.
    dispatch_cmd = "sleep 3"
    proc, _, watchdog_log, pid_file = _launch_watchdog(tmp_path, dispatch_cmd)

    rc = _wait_for(proc, timeout=20)
    if rc is None:
        _kill_group(proc)
        pytest.fail("watchdog did not exit within 20s on clean dispatcher path")

    log_text = _read(watchdog_log)
    assert rc == 0, f"expected rc=0 on clean dispatcher path, got {rc}; log:\n{log_text}"
    assert "started:" in log_text, f"missing 'started:' in log:\n{log_text}"
    assert "wrote pid file" in log_text
    assert "spawning dispatcher (respawn 1/2)" in log_text
    assert "cycle done: rc=0" in log_text
    assert "dispatcher exited 0 cleanly" in log_text
    # PID file should be cleaned up on exit.
    assert not pid_file.exists(), "PID file should be removed on clean exit"


def test_stall_detected_and_dispatcher_respawned(tmp_path: Path) -> None:
    """Stall path: a fake dispatcher that ``sleep 9999`` never touches the
    dispatcher log. After STALL_GAP=4s the watchdog must kill it. With
    MAX_RESPAWNS=2 and zero forward progress (no metrics.json files
    appear), the watchdog aborts with "NO FORWARD PROGRESS" after the
    first respawn.
    """
    # `sleep 9999` never writes to the dispatcher log — perfect stall.
    dispatch_cmd = "sleep 9999"
    proc, _, watchdog_log, pid_file = _launch_watchdog(
        tmp_path,
        dispatch_cmd,
        poll=2,
        stall_gap=4,
        max_respawns=2,
    )

    # POLL=2s + STALL_GAP=4s means stall detection on poll #3 at ~6s wall time;
    # then a 60s cool-down would dominate. To keep the test under 30s we
    # cap MAX_RESPAWNS=2 and rely on the "no forward progress" exit which
    # fires before the cool-down on respawn 1.
    rc = _wait_for(proc, timeout=25)
    if rc is None:
        _kill_group(proc)
        pytest.fail(
            "watchdog did not detect stall + abort within 25s; log:\n" + _read(watchdog_log)
        )

    log_text = _read(watchdog_log)
    # Watchdog should have logged at least one heartbeat ("alive —") before
    # detecting the stall.
    assert "alive —" in log_text, "missing heartbeat 'alive —' line; log:\n" + log_text
    assert "log stall:" in log_text, "missing 'log stall:' detection; log:\n" + log_text
    assert "SIGTERM dispatcher" in log_text or "cleanup: SIGTERM" in log_text, (
        "missing dispatcher SIGTERM cleanup; log:\n" + log_text
    )
    # Either NO FORWARD PROGRESS (likely, since no metrics.json appear) or
    # MAX_RESPAWNS — both are valid abort reasons; both exit 1.
    assert rc == 1, f"expected rc=1 on stall path, got {rc}; log:\n{log_text}"
    assert "NO FORWARD PROGRESS" in log_text or "hit MAX_RESPAWNS" in log_text, (
        "expected progress-guard or max-respawns abort; log:\n" + log_text
    )
    # PID file should be cleaned up on exit.
    assert not pid_file.exists(), "PID file should be removed on stall-abort exit"


def test_sigterm_clean_shutdown(tmp_path: Path) -> None:
    """SIGTERM path: while the watchdog is mid-cycle waiting on its
    dispatcher, send SIGTERM. Watchdog must log the signal, kill the
    dispatcher, remove the PID file, and exit non-zero.
    """
    # Long-running dispatcher so we can SIGTERM the watchdog mid-cycle.
    dispatch_cmd = "sleep 9999"
    proc, _, watchdog_log, pid_file = _launch_watchdog(
        tmp_path,
        dispatch_cmd,
        poll=2,
        stall_gap=300,  # Don't let stall detection fire before we SIGTERM.
        max_respawns=5,
    )

    # Give the watchdog ~3s to start, write its PID file, spawn the dispatcher,
    # and enter the wait loop. Then SIGTERM the watchdog itself.
    time.sleep(3.0)
    assert pid_file.exists(), "PID file should exist mid-cycle"
    log_before = _read(watchdog_log)
    assert "spawning dispatcher" in log_before, (
        "dispatcher should have spawned before SIGTERM; log:\n" + log_before
    )

    proc.send_signal(signal.SIGTERM)
    rc = _wait_for(proc, timeout=20)
    if rc is None:
        _kill_group(proc)
        pytest.fail("watchdog did not exit within 20s of SIGTERM; log:\n" + _read(watchdog_log))

    log_text = _read(watchdog_log)
    assert rc != 0, f"expected non-zero rc on SIGTERM, got {rc}; log:\n{log_text}"
    assert "received SIGTERM" in log_text, f"missing SIGTERM log line; log:\n{log_text}"
    assert "cleanup: SIGTERM dispatcher" in log_text, (
        "missing dispatcher cleanup line; log:\n" + log_text
    )
    # PID file should be cleaned up on signal-driven exit.
    assert not pid_file.exists(), "PID file should be removed on SIGTERM exit"


def test_single_instance_guard(tmp_path: Path) -> None:
    """Single-instance guard: if a PID file points to a live process,
    refuse to start with rc=2.
    """
    pid_file = tmp_path / "watchdog.pid"
    # Use our own PID — guaranteed alive while pytest is running.
    pid_file.write_text(str(os.getpid()))

    proc, _, watchdog_log, _ = _launch_watchdog(
        tmp_path,
        "sleep 1",
        pid_file=pid_file,
    )
    rc = _wait_for(proc, timeout=10)
    if rc is None:
        _kill_group(proc)
        pytest.fail("watchdog did not exit on stale-pid guard")

    log_text = _read(watchdog_log)
    assert rc == 2, f"expected rc=2 on single-instance guard, got {rc}; log:\n{log_text}"
    assert "refusing to start" in log_text, "missing 'refusing to start' line; log:\n" + log_text
    # Our PID file should NOT be removed (we didn't write it; the guard
    # respects the live PID).
    assert pid_file.exists(), "PID file of another live instance must not be removed"
    assert pid_file.read_text().strip() == str(os.getpid())
