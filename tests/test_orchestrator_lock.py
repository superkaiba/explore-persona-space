"""Tests for `scripts/orchestrator_lock.py` (Task 1.2).

Exercises acquire / release / status subcommands against a temporary git repo
whose task structure is created by the shared `tmp_repo` + `registered_task`
fixtures from conftest.py.
"""

import os
import subprocess
import sys
from pathlib import Path


def _lock_cmd(tmp_repo, *args):
    return subprocess.run(
        [sys.executable, "scripts/orchestrator_lock.py", *args],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "TASK_PY_REPO_ROOT": str(tmp_repo)},
        capture_output=True,
        text=True,
    )


def test_acquire_when_unlocked_succeeds(tmp_repo, registered_task):
    r = _lock_cmd(tmp_repo, "acquire", str(registered_task))
    assert r.returncode == 0, r.stderr
    assert "acquired" in r.stdout


def test_acquire_when_locked_by_live_pid_fails(tmp_repo, registered_task):
    # Acquire as one subprocess
    _lock_cmd(tmp_repo, "acquire", str(registered_task))
    # The first subprocess has already exited (since it was a one-shot CLI invocation).
    # We need to simulate a LIVE owner. Two options:
    #   (a) Write a PID file pointing at the current pytest process (os.getpid()).
    #   (b) Spawn a long-running sleeper and write its PID.
    # Use (a) — simpler and reliable.
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    (folder / ".orchestrator.pid").write_text(f"{os.getpid()}\n2026-01-01T00:00:00\n")
    r = _lock_cmd(tmp_repo, "acquire", str(registered_task))
    assert r.returncode == 1
    assert "locked" in r.stderr.lower()


def test_acquire_when_locked_by_dead_pid_succeeds(tmp_repo, registered_task):
    # Write a stale PID file (PID 999999 — extremely unlikely to be alive).
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    (folder / ".orchestrator.pid").write_text("999999\n2026-01-01T00:00:00\n")
    r = _lock_cmd(tmp_repo, "acquire", str(registered_task))
    assert r.returncode == 0, r.stderr
    assert "reclaimed" in r.stdout


def test_status_shows_active_for_live_owner(tmp_repo, registered_task):
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    (folder / ".orchestrator.pid").write_text(f"{os.getpid()}\n2026-01-01T00:00:00\n")
    r = _lock_cmd(tmp_repo, "status", str(registered_task))
    assert r.returncode == 0
    assert "active" in r.stdout
    assert f"pid={os.getpid()}" in r.stdout


def test_status_shows_stale_for_dead_owner(tmp_repo, registered_task):
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    (folder / ".orchestrator.pid").write_text("999999\n2026-01-01T00:00:00\n")
    r = _lock_cmd(tmp_repo, "status", str(registered_task))
    assert "stale" in r.stdout


def test_status_inactive_when_no_lock(tmp_repo, registered_task):
    r = _lock_cmd(tmp_repo, "status", str(registered_task))
    assert "inactive" in r.stdout


def test_release_drops_lock(tmp_repo, registered_task):
    # Acquire (with our own pid since the subprocess exits immediately)
    _lock_cmd(tmp_repo, "acquire", str(registered_task))
    # The acquire subprocess just released-on-exit anyway. To exercise release explicitly,
    # write a lock owned by the test's own pid, then release with --force (since the
    # release subprocess won't own the lock either):
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    (folder / ".orchestrator.pid").write_text(f"{os.getpid()}\n2026-01-01T00:00:00\n")
    r = _lock_cmd(tmp_repo, "release", str(registered_task), "--force")
    assert r.returncode == 0
    assert not (folder / ".orchestrator.pid").exists()


def test_release_refuses_other_owner_without_force(tmp_repo, registered_task):
    folder = next((tmp_repo / "tasks").glob(f"*/{registered_task}"))
    (folder / ".orchestrator.pid").write_text(f"{os.getpid()}\n2026-01-01T00:00:00\n")
    r = _lock_cmd(tmp_repo, "release", str(registered_task))
    assert r.returncode == 1
    assert (folder / ".orchestrator.pid").exists()  # not deleted
