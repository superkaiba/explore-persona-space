"""Bound the real provision-to-workload handoff without hiding failed experiments."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from scripts import story_persona_qwen38_monitor as monitor


def save(path, value):
    """Write isolated fixture evidence, never a real run ledger or service state."""
    path.write_text(json.dumps(value))


def digest(path):
    """Hash exactly the fixture bytes used by the production evidence reader."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def live_handoff(tmp_path, monkeypatch):
    """Exercise actual /proc identity, timeout ancestry and process-owned flocks."""
    run = tmp_path / "run"
    run.mkdir()
    attempt = run / "attempt-0001"
    attempt.mkdir()
    handle_path = tmp_path / "handle.json"
    ledger_path = tmp_path / "allocation-ledger.json"
    child_script = run / "handoff.py"
    child_script.write_text(
        "import fcntl, json, os, signal\nfrom pathlib import Path\n"
        "root = Path(__file__).parent\n"
        "locks = [(root.parent / name).open('a') for name in "
        "('handle.handoff.lock', 'allocation-ledger.lock')]\n"
        "for stream in locks: fcntl.flock(stream, fcntl.LOCK_EX)\n"
        "(root / 'owner.json').write_text(json.dumps({\n"
        "'owner_pid': os.getpid(),\n"
        "'owner_start_ticks': int(Path('/proc/self/stat').read_text()"
        ".rsplit(')', 1)[1].split()[19]),\n"
        "'owner_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))\n"
        "signal.pause()\n"
    )
    (run / "provision.py").write_text(
        "import fcntl, subprocess, sys\nfrom pathlib import Path\n"
        "root = Path(__file__).parent\n"
        "stream = (root / 'provision.lock').open('a')\n"
        "fcntl.flock(stream, fcntl.LOCK_EX)\n"
        "subprocess.run(['timeout', '120s', sys.executable, str(root / 'handoff.py'), "
        "'--attempt', str(root / 'attempt-0001')], check=True)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, str(run / "provision.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        limit = time.monotonic() + 10
        while not (run / "owner.json").exists():
            if parent.poll() is not None or time.monotonic() >= limit:
                raise RuntimeError("harmless local fixture did not start")
            time.sleep(0.01)
        config = {
            "source_sha": "a" * 40,
            "immediate_handoff": True,
            "created_at": 700,
            "provision_unit": "test-capacity.service",
            "provision_state": str(run / "provision-state.json"),
            "handle_file": str(handle_path),
            "allocation_ledger": str(ledger_path),
            "observation": str(run / "observation.json"),
            "workdir": str(tmp_path),
            "out_dir": str(tmp_path / "out"),
        }
        handle = {
            "backend": "runpod",
            "extra": {
                "pod_id": "test-pod",
                "workload_executed": False,
                "runpod_attempt_id": "rp-test",
                "expected_artifacts": {
                    "sentinel_path": "/workspace/test/completion.json",
                },
            },
        }
        save(handle_path, handle)
        save(
            ledger_path,
            {
                "allocations": [
                    {
                        "pod_id": "test-pod",
                        "source_sha": config["source_sha"],
                        "status": "allocated",
                        "paid_start_unix": 800,
                        "deadline_unix": 8000,
                    }
                ]
            },
        )
        state = {
            "source_sha": config["source_sha"],
            "status": "handoff_in_progress",
            "checked_at": 1900,
            "attempt": 1,
            "handoff_started_at": 1900,
            "handoff_deadline_unix": 2800,
        }
        save(Path(config["provision_state"]), state)
        save(
            attempt / "deployment-request.json",
            {
                "source_sha": config["source_sha"],
                "started_at": 700,
            },
        )
        save(
            attempt / "handoff-request.json",
            {
                "version": 1,
                "source_sha": config["source_sha"],
                "deployment_request_sha256": digest(attempt / "deployment-request.json"),
                "handoff_sha256": digest(child_script),
            },
        )
        claim = {
            **json.loads((run / "owner.json").read_text()),
            "checked_at": 1901,
            "pod_id": "test-pod",
            "source_sha": config["source_sha"],
            "attempt": attempt.name,
            "paid_start_unix": 800,
            "deadline_unix": 8000,
            "ledger_sha256": digest(ledger_path),
            "handle_sha256": digest(handle_path),
            "runpod_attempt_id": "rp-test",
            "sentinel_path": "/workspace/test/completion.json",
            "handoff_request_sha256": digest(attempt / "handoff-request.json"),
        }
        save(run / "handoff-claim.json", claim)
        observed = {
            "status": "stalled",
            "pid_alive": False,
            "original_backend_status": "pid-stale-workload-live",
            "stall_reason": "pid_dead_evidence: completed provision-only worker",
            "pilot": {"pids": [], "stages": [], "artifacts": {}, "chunk_count": 0},
        }

        def command(argv, *, cwd=None, stdin=None, timeout=150):
            assert argv == [
                "systemctl",
                "--user",
                "show",
                "test-capacity.service",
                "--property=ActiveState,SubState,MainPID",
            ]
            assert timeout == 15
            return f"ActiveState=active\nSubState=running\nMainPID={parent.pid}\n"

        monkeypatch.setattr(monitor, "command", command)
        yield config, handle, state, claim, observed, parent
    finally:
        if (run / "owner.json").exists():
            owner = json.loads((run / "owner.json").read_text())["owner_pid"]
            with suppress(ProcessLookupError):  # Already-exited fixture children need no signal.
                os.kill(owner, signal.SIGTERM)
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGTERM)
        parent.communicate(timeout=10)


def test_actual_process_tree_and_flocks_allow_handoff_after_old_provider_grace(live_handoff):
    config, _, _, _, observed, _ = live_handoff
    result = monitor.apply_handoff_progress(config, observed, now=2000)
    assert result["status"] == "pending"
    assert result["current_phase"] == "handoff_in_progress"
    assert result["handoff_deadline_unix"] == 2800
    assert result["allocation_deadline_unix"] == 8000
    assert "pid_alive" not in result and "stall_reason" not in result
    assert observed["pid_alive"] is False  # Original observation remains untouched.


@pytest.mark.parametrize(
    "change",
    [
        {"status": "handoff_failed"},
        {"source_sha": "b" * 40},
        {"checked_at": 2001},
        {"handoff_deadline_unix": 2801},
        {"handoff_started_at": 2001},
        {"attempt": True},
    ],
)
def test_bad_phase_evidence_cannot_excuse_missing_worker(live_handoff, change):
    config, _, state, _, observed, _ = live_handoff
    state.update(change)
    save(Path(config["provision_state"]), state)
    with pytest.raises(RuntimeError):
        monitor.apply_handoff_progress(config, observed, now=2000)


def test_fresh_heartbeat_cannot_extend_fixed_handoff_deadline(live_handoff):
    config, _, state, _, observed, _ = live_handoff
    state["checked_at"] = 2800
    save(Path(config["provision_state"]), state)
    with pytest.raises(RuntimeError, match="fixed phase deadline"):
        monitor.apply_handoff_progress(config, observed, now=2800)


def test_claim_published_after_first_clock_sample_uses_fresh_clock(live_handoff, monkeypatch):
    config, _, _, claim, observed, _ = live_handoff
    claim["checked_at"] = 2000.1
    save(Path(config["provision_state"]).parent / "handoff-claim.json", claim)
    samples = iter([2000, 2000.05, 2000.2, 2000.3])
    monkeypatch.setattr(monitor.time, "time", lambda: next(samples))
    assert monitor.apply_handoff_progress(config, observed)["status"] == "pending"


def test_deadline_expiring_during_process_checks_is_not_extended(live_handoff, monkeypatch):
    config, _, _, _, observed, _ = live_handoff
    samples = iter([2799.7, 2799.8, 2799.9, 2800])
    monkeypatch.setattr(monitor.time, "time", lambda: next(samples))
    with pytest.raises(RuntimeError, match="fixed deadline during verification"):
        monitor.apply_handoff_progress(config, observed)


def test_partial_claim_is_an_error_not_startup_grace(live_handoff):
    config, _, _, _, observed, _ = live_handoff
    (Path(config["provision_state"]).parent / "handoff-claim.json").write_text('{"owner_pid":')
    with pytest.raises(json.JSONDecodeError):
        monitor.apply_handoff_progress(config, observed, now=1920)


def test_no_claim_has_only_thirty_second_process_startup_window(live_handoff):
    config, _, _, _, observed, _ = live_handoff
    (Path(config["provision_state"]).parent / "handoff-claim.json").unlink()
    assert monitor.apply_handoff_progress(config, observed, now=1930)["status"] == "pending"
    with pytest.raises(RuntimeError, match="claim missing"):
        monitor.apply_handoff_progress(config, observed, now=1930.01)


@pytest.mark.parametrize(
    "change",
    [
        {"owner_start_ticks": 1},
        {"owner_boot_id": "old-boot"},
        {"owner_pid": 0},
        {"source_sha": "b" * 40},
        {"pod_id": "other-pod"},
        {"deadline_unix": 9000},
        {"paid_start_unix": 900},
        {"ledger_sha256": "0" * 64},
        {"runpod_attempt_id": "old-attempt"},
        {"sentinel_path": "/wrong"},
        {"handoff_request_sha256": "0" * 64},
        {"checked_at": 1899},
    ],
)
def test_stale_or_mismatched_claim_is_not_live_handoff(live_handoff, change):
    config, _, _, claim, observed, _ = live_handoff
    claim.update(change)
    save(Path(config["provision_state"]).parent / "handoff-claim.json", claim)
    with pytest.raises(RuntimeError):
        monitor.apply_handoff_progress(config, observed, now=2000)


def test_missing_flock_owner_fails_even_with_valid_heartbeat(live_handoff):
    config, _, _, _, observed, _ = live_handoff
    lock = Path(config["handle_file"]).with_suffix(".handoff.lock")
    lock.unlink()
    lock.touch()  # Existing path is insufficient; kernel lock remains on old inode.
    with pytest.raises(RuntimeError, match="lock is not owned"):
        monitor.apply_handoff_progress(config, observed, now=2000)


def test_dead_service_does_not_gain_grace(live_handoff, monkeypatch):
    config, _, _, _, observed, _ = live_handoff

    def command(argv, *, cwd=None, stdin=None, timeout=150):
        return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"

    monkeypatch.setattr(monitor, "command", command)
    with pytest.raises(RuntimeError, match="not actively running"):
        monitor.apply_handoff_progress(config, observed, now=2000)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"status": "done"},
        {"reachability_alarm": True},
        {"startup_probe_error": "TimeoutExpired"},
        {"stall_reason": monitor.PROGRESS_STALL},
        {"pilot": {"pids": [123]}},
        {"pilot": {"stages": ["capture"]}},
        {"pilot": {"artifacts": {"smoke.json": {}}}},
        {"pilot": {"chunk_count": 1}},
        {"pilot": {"error_lines": ["Traceback"]}},
    ],
)
def test_real_workload_and_error_evidence_never_masked(live_handoff, change):
    config, _, state, _, observed, _ = live_handoff
    state["status"] = "handoff_failed"
    save(Path(config["provision_state"]), state)
    observed.update(change)
    assert monitor.apply_handoff_progress(config, observed, now=2000) is observed


def test_executed_or_default_run_uses_original_backend(live_handoff):
    config, handle, _, _, observed, _ = live_handoff
    handle["extra"]["workload_executed"] = True
    save(Path(config["handle_file"]), handle)
    assert monitor.apply_handoff_progress(config, observed, now=2000) is observed
    config.pop("immediate_handoff")
    Path(config["handle_file"]).unlink()
    assert monitor.apply_handoff_progress(config, observed, now=2000) is observed


def test_handle_changed_without_claim_link_fails_but_canonical_link_is_accepted(live_handoff):
    config, handle, _, _, observed, _ = live_handoff
    handle["extra"]["workload_cmd"] = "LD_LIBRARY_PATH=/compat original"
    save(Path(config["handle_file"]), handle)
    with pytest.raises(RuntimeError, match="claim does not match"):
        monitor.apply_handoff_progress(config, observed, now=2000)
    claim_path = Path(config["provision_state"]).parent / "handoff-claim.json"
    handle["extra"]["handoff_claim_sha256"] = digest(claim_path)
    save(Path(config["handle_file"]), handle)
    assert monitor.apply_handoff_progress(config, observed, now=2000)["status"] == "pending"


def test_request_phase_race_uses_immutable_request_clock(live_handoff):
    config, _, state, _, observed, _ = live_handoff
    state.update(status="requesting_capacity", checked_at=1899)
    save(Path(config["provision_state"]), state)
    assert monitor.apply_handoff_progress(config, observed, now=1899)["current_phase"] == (
        "canonical_provision_handoff"
    )
    state["checked_at"] = 1900
    save(Path(config["provision_state"]), state)
    with pytest.raises(RuntimeError, match="fixed deadline"):
        monitor.apply_handoff_progress(config, observed, now=1900)


def test_monitor_publishes_awaiting_handoff_then_probes_real_failure(live_handoff, monkeypatch):
    config, _, _, _, observed, _ = live_handoff
    system_command = monitor.command
    probes = []

    def command(argv, *, cwd=None, stdin=None, timeout=150):
        if argv[0] == "systemctl":
            return system_command(argv, cwd=cwd, stdin=stdin, timeout=timeout)
        assert "--probe-handle" in argv and timeout == 240
        probes.append(argv)
        return json.dumps(observed if len(probes) == 1 else {"status": "failed"})

    def sleep(seconds):
        saved = json.loads(Path(config["observation"]).read_text())
        assert saved["status"] == "awaiting_handoff"
        assert saved["backend_observation"]["current_phase"] == "handoff_in_progress"

    monkeypatch.setattr(monitor, "command", command)
    monkeypatch.setattr(monitor.time, "time", lambda: 2000)
    monkeypatch.setattr(monitor.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="backend requires recovery: failed"):
        monitor.monitor(config)
    assert len(probes) == 2


def test_completed_during_process_validation_reprobes_instead_of_stale_grace(
    live_handoff, monkeypatch
):
    config, handle, _, _, observed, _ = live_handoff

    def command(argv, *, cwd=None, stdin=None, timeout=150):
        handle["extra"].update(workload_executed=True, synced_sha=config["source_sha"])
        save(Path(config["handle_file"]), handle)
        return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"

    monkeypatch.setattr(monitor, "command", command)
    assert monitor.apply_handoff_progress(config, observed, now=2000) is None
