"""Bounded real-subprocess tests for durable polling and fresh-handle gates."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    """Import the worktree monitor without stubbing its subprocess/liveness bodies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("issue825_turn_k5_monitor")


def write_handle(path: Path, *, job_id: str = "fixture-new") -> None:
    """Write only the harmless identifying fields used by the monitor."""
    path.write_text(json.dumps({"backend": "fixture", "job_id": job_id, "pod_name": "fixture"}))


def test_inspect_handle_requires_new_mtime_and_valid_identity(driver, tmp_path):
    """Old, missing, and malformed handles never become live-run evidence."""
    path = tmp_path / "handle.json"
    assert driver.inspect_handle(path, time.time())["reason"] == "handle_missing"
    write_handle(path)
    os.utime(path, (1, 1))
    assert driver.inspect_handle(path, 2)["reason"] == "handle_predates_launch"
    write_handle(path)
    assert driver.inspect_handle(path, 2)["ready"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="missing backend"):
        driver.inspect_handle(path, 2)


def test_poll_keeps_heartbeat_live_and_preserves_complete_logs(driver, tmp_path):
    """A real slow probe emits multiple heartbeats before its completion JSON."""
    handle = tmp_path / "handle.json"
    write_handle(handle)
    monitor = driver.Monitor(tmp_path / "monitor", handle, 0)
    script = tmp_path / "probe.py"
    script.write_text(
        "import json, sys, time\nprint('probe starting', file=sys.stderr, flush=True)\n"
        "time.sleep(.15)\nprint(json.dumps(dict(status='running', current_phase='gen', "
        "pid_alive=True, log_tail_excerpt='chunk 3/20')))\n"
    )
    captured = []

    def observe():
        """Read the actual atomic heartbeat while the real subprocess is sleeping."""
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not captured:
            state = json.loads((monitor.out / "status.json").read_text())
            if state["monitor_status"] == "polling" and state.get("poll_elapsed_seconds", 0) > 0.07:
                captured.append(state)
                return
            time.sleep(0.01)

    observer = threading.Thread(target=observe)
    observer.start()
    result = monitor.poll([sys.executable, str(script)], timeout=2, heartbeat=0.02)
    observer.join(timeout=2)
    assert result["status"] == "running"
    assert captured and captured[0]["poll_child_pid"] > 0
    assert "probe starting" in monitor.state["poll_stderr"]["tail"]
    assert monitor.state["poll_exit_code"] == 0
    assert monitor.state["polls_completed"] == 1


def test_poll_soft_timeout_preserves_active_backend_operation(driver, tmp_path):
    """Overdue probes remain alive and finish normally, with a durable loud alert."""
    handle = tmp_path / "handle.json"
    write_handle(handle)
    monitor = driver.Monitor(tmp_path / "monitor", handle, 0)
    command = [
        sys.executable,
        "-c",
        "import time,json; print('waiting',flush=True); time.sleep(.15);"
        "print(json.dumps(dict(status='running',current_phase='gen',pid_alive=True,log_tail_excerpt='alive')))",
    ]
    result = monitor.poll(command, timeout=0.05, heartbeat=0.01)
    assert result["status"] == "running"
    assert monitor.state["poll_child_pid"] is None
    assert monitor.state["poll_exit_code"] == 0
    assert monitor.state["last_poll_overdue"] is True
    assert "waiting" in monitor.state["poll_stdout"]["tail"]
    events = [
        json.loads(line) for line in (monitor.out / "observations.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == ["poll_overdue"]


def test_signal_stop_finishes_active_backend_operation(driver, tmp_path):
    """A shutdown request waits for the active probe rather than killing provisioning."""
    handle = tmp_path / "handle.json"
    write_handle(handle)
    monitor = driver.Monitor(tmp_path / "monitor", handle, 0)
    command = [
        sys.executable,
        "-c",
        "import time,json; time.sleep(.15);"
        "print(json.dumps(dict(status='running',current_phase='gen',"
        "pid_alive=True,log_tail_excerpt='finished probe')))",
    ]

    def request_stop():
        """Use the signal handler's exact effect during a real active subprocess."""
        time.sleep(0.05)
        monitor.stopping = True

    stopper = threading.Thread(target=request_stop)
    stopper.start()
    result = monitor.run(
        command, poll_timeout=2, heartbeat=0.01, initial_interval=0.01, later_interval=0.01
    )
    stopper.join(timeout=2)
    assert result == 130
    assert monitor.state["monitor_status"] == "stopped"
    assert monitor.state["poll_exit_code"] == 0
    assert monitor.state["last_backend_result"]["log_tail_excerpt"] == "finished probe"
    events = [
        json.loads(line) for line in (monitor.out / "observations.jsonl").read_text().splitlines()
    ]
    assert "shutdown_pending" in [event["kind"] for event in events]


def test_mutable_phase_clock_does_not_invalidate_poll(driver, tmp_path):
    """The sanctioned poller's sidecar clock updates keep stable run identity."""
    handle = tmp_path / "handle.json"
    write_handle(handle)
    monitor = driver.Monitor(tmp_path / "monitor", handle, 0)
    before = driver.inspect_handle(handle, 0)
    script = tmp_path / "probe.py"
    script.write_text(
        "import json,pathlib,sys\np=pathlib.Path(sys.argv[1])\n"
        "h=json.loads(p.read_text()); h['extra']={'phase_clock':'next'}\n"
        "p.write_text(json.dumps(h))\n"
        "print(json.dumps(dict(status='done',current_phase='done',pid_alive=False,"
        "log_tail_excerpt='GPU archive verified')))\n"
    )
    assert (
        monitor.run(
            [sys.executable, str(script), str(handle)],
            poll_timeout=2,
            heartbeat=0.01,
            initial_interval=0.01,
            later_interval=0.01,
        )
        == 0
    )
    after = driver.inspect_handle(handle, 0)
    assert before["sha256"] != after["sha256"] and driver.same_run(before, after)
    assert not driver.same_run(before, after | {"attempt_id": "new-incarnation"})


def test_heartbeat_io_failure_drains_child_before_retry(driver, tmp_path, monkeypatch):
    """A failed status write cannot leave one active poll while another starts."""
    from unittest.mock import create_autospec

    handle = tmp_path / "handle.json"
    write_handle(handle)
    monitor = driver.Monitor(tmp_path / "monitor", handle, 0)
    script = tmp_path / "probe.py"
    finished = tmp_path / "finished.txt"
    script.write_text(
        "import pathlib,sys,time,json\ntime.sleep(.1)\n"
        "pathlib.Path(sys.argv[1]).write_text('complete')\n"
        "print(json.dumps(dict(status='running',current_phase='gen',pid_alive=True,"
        "log_tail_excerpt='complete')))\n"
    )
    original = driver.atomic_json
    failures = []

    def fail_one_write(path, value):
        """Inject one signature-conformant filesystem-boundary failure."""
        if not failures:
            failures.append(True)
            raise OSError("injected temporary status write failure")
        return original(path, value)

    monkeypatch.setattr(
        driver, "atomic_json", create_autospec(original, side_effect=fail_one_write)
    )
    with pytest.raises(OSError, match="injected temporary"):
        monitor.poll([sys.executable, str(script), str(finished)], timeout=2, heartbeat=0.01)
    assert finished.read_text() == "complete"
    assert monitor.state["poll_child_pid"] is None
    assert monitor.state["poll_exit_code"] == 0


def test_monitor_retries_error_then_done_retains_cpu_pending_scope(driver, tmp_path):
    """The real loop survives a failed probe and stops only on a later valid done."""
    handle = tmp_path / "handle.json"
    write_handle(handle)
    monitor = driver.Monitor(tmp_path / "monitor", handle, 0)
    script = tmp_path / "probe.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "count = pathlib.Path(__file__).with_suffix('.count')\n"
        "n = int(count.read_text()) if count.exists() else 0\n"
        "count.write_text(str(n+1))\n"
        "if n == 0: print('transient transport error',file=sys.stderr); sys.exit(3)\n"
        "print(json.dumps(dict(status='done',current_phase='done',pid_alive=False,"
        "log_tail_excerpt='GPU archive verified')))\n"
    )
    result = monitor.run(
        [sys.executable, str(script)],
        poll_timeout=2,
        heartbeat=0.01,
        initial_interval=0.01,
        later_interval=0.01,
    )
    state = json.loads((monitor.out / "status.json").read_text())
    assert result == 0 and state["monitor_status"] == "terminal"
    assert state["polls_completed"] == 2
    assert state["consecutive_poll_errors"] == 0
    assert state["experiment_complete"] is False
    assert "CPU analysis pending" in state["completion_scope"]
    events = [
        json.loads(line) for line in (monitor.out / "observations.jsonl").read_text().splitlines()
    ]
    assert [e["kind"] for e in events] == ["poll_error", "backend_observation", "terminal"]


def test_monitor_waits_for_new_handle_before_polling(driver, tmp_path):
    """Touching the old sidecar cannot impersonate a new launched job."""
    handle = tmp_path / "handle.json"
    write_handle(handle, job_id="fixture-old")
    os.utime(handle, (1, 1))
    monitor = driver.Monitor(tmp_path / "monitor", handle, 2)
    observations = []

    def replace_handle():
        """First rewrite old phase-clock metadata, then supply a different job."""
        time.sleep(0.04)
        old = json.loads(handle.read_text())
        old["extra"] = {"phase_clock": "changed, same old job"}
        handle.write_text(json.dumps(old))
        time.sleep(0.05)
        observations.append(json.loads((monitor.out / "status.json").read_text()))
        write_handle(handle, job_id="fixture-new")

    updater = threading.Thread(target=replace_handle)
    updater.start()
    command = [
        sys.executable,
        "-c",
        "import json; print(json.dumps(dict(status='dead',"
        "current_phase='failed',pid_alive=False,log_tail_excerpt='crash')))",
    ]
    result = monitor.run(
        command, poll_timeout=2, heartbeat=0.01, initial_interval=0.01, later_interval=0.01
    )
    updater.join(timeout=2)
    assert result == 1
    assert observations[0]["polls_completed"] == 0
    assert observations[0]["handle_wait"]["reason"] == "unchanged_stale_handle"
    assert monitor.state["handle"]["job_id"] == "fixture-new"


def test_parse_poll_rejects_malformed_or_unknown_status(driver, tmp_path):
    """Uninterpretable output stays a monitor/probe error, never a workload verdict."""
    path = tmp_path / "stdout.log"
    path.write_text("not json\n")
    with pytest.raises(ValueError):
        driver.parse_poll(path)
    path.write_text(
        json.dumps(
            {"status": "missing", "current_phase": "x", "pid_alive": False, "log_tail_excerpt": "x"}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="unknown backend status"):
        driver.parse_poll(path)
