"""Regression checks for stale-child detection and bounded monitor probes."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import issue2054_k5_assistant_story_monitor as monitor


def observation(latest, *, alive=True, source="a" * 40):
    return {
        "status": "running",
        "pid_alive": True,
        "outputs": {
            "runtime_exists": True,
            "source_sha": source,
            "pid_alive": alive,
            "started": 100,
            "latest_progress": latest,
        },
    }


def test_parent_heartbeat_cannot_hide_stale_child():
    result = monitor.assess_progress(observation(110), "a" * 40, 1500)
    assert "stopped advancing" in result["stall_reason"]


def test_fresh_child_is_healthy():
    result = monitor.assess_progress(observation(1450), "a" * 40, 1500)
    assert "stall_reason" not in result


def test_progressing_peer_cannot_hide_stalled_worker():
    backend = observation(1450)
    backend["outputs"]["active_workers"] = [
        {
            "started": 100,
            "latest_progress": 120,
            "model": "base",
            "stage": "generate",
            "pid_exists": True,
        }
    ]
    result = monitor.assess_progress(backend, "a" * 40, 1500)
    assert "base generate worker stopped" in result["stall_reason"]


def test_worker_exit_has_short_status_flush_grace():
    backend = observation(1490)
    backend["outputs"]["active_workers"] = [
        {
            "started": 100,
            "latest_progress": 1490,
            "model": "base",
            "stage": "generate",
            "pid_exists": False,
        }
    ]
    assert "stall_reason" not in monitor.assess_progress(backend, "a" * 40, 1500)
    assert "exited" in monitor.assess_progress(backend, "a" * 40, 1530)["stall_reason"]


def test_dead_or_wrong_source_driver_is_not_healthy():
    result = monitor.assess_progress(observation(1450, alive=False), "a" * 40, 1500)
    assert result["pid_alive"] is False
    assert "exited" in result["stall_reason"]
    with pytest.raises(RuntimeError, match="different source"):
        monitor.assess_progress(observation(1450, source="b" * 40), "a" * 40, 1500)


def test_startup_has_bounded_allowance():
    assert "stall_reason" not in monitor.assess_progress(observation(None), "a" * 40, 1000)
    assert "startup" in monitor.assess_progress(observation(None), "a" * 40, 2000)["stall_reason"]


def test_bootstrap_before_runtime_is_bounded():
    backend = {"status": "running", "outputs": {"runtime_exists": False}}
    assert "stall_reason" not in monitor.assess_progress(backend, "a" * 40, 1000, launched_at=100)
    assert (
        "bootstrap"
        in monitor.assess_progress(backend, "a" * 40, 2000, launched_at=100)["stall_reason"]
    )


def test_remote_probe_is_valid_python():
    compile(monitor.progress_script(), "remote_probe", "exec")


def test_timeout_reaps_probe(tmp_path):
    pidfile = tmp_path / "pid"
    code = f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        monitor.bounded([sys.executable, "-c", code], cwd=tmp_path, timeout=0.2)
    assert not (Path("/proc") / pidfile.read_text()).exists()


def test_json_output_is_atomic(tmp_path):
    path = tmp_path / "state" / "observation.json"
    monitor.atomic_json(path, {"checked_at": 123})
    assert json.loads(path.read_text()) == {"checked_at": 123}
    assert not path.with_suffix(".tmp").exists()


def test_terminal_probe_rechecks_completion(monkeypatch, tmp_path):
    results = iter([None, {"source_sha": "a" * 40, "verified_revision": "b" * 40}])
    monkeypatch.setattr(monitor, "completion", lambda *args: next(results))
    monkeypatch.setattr(
        monitor,
        "bounded",
        lambda *a, **kw: json.dumps(
            {
                "status": "done",
                "pid_alive": False,
                "current_phase": "workload_done",
            }
        ),
    )
    path = tmp_path / "observation.json"
    args = SimpleNamespace(
        source_sha="a" * 40,
        launched_at=100,
        backend_repo=tmp_path,
        handle_file=tmp_path / "handle.json",
        observation=path,
    )
    monitor.monitor(args)
    assert json.loads(path.read_text())["status"] == "complete"


def test_remote_inventory_detects_missing_or_changed_outputs():
    path = monitor.PREFIX + "/analysis/map.npz"
    record = {"path": path, "sha256": "c" * 64, "size": 10}
    api = SimpleNamespace(
        get_paths_info=lambda *a, **kw: [
            SimpleNamespace(path=path, size=10, lfs=SimpleNamespace(sha256="d" * 64)),
        ]
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        monitor.verify_remote_files(api, "b" * 40, [record])
    api.get_paths_info = lambda *a, **kw: []
    with pytest.raises(RuntimeError, match="absent"):
        monitor.verify_remote_files(api, "b" * 40, [record])
