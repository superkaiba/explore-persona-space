"""Fault-injection tests for the deployed tick, including real failed subprocesses."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/experiment_watchdog.py"
spec = importlib.util.spec_from_file_location("experiment_watchdog", SOURCE)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class FakeRuntime:
    def __init__(self):
        self.units = {"monitor.service": dict(LoadState="loaded", ActiveState="active")}
        self.starts, self.messages = [], []
        self.fail_delivery = False
        self.ambiguous_start = False

    def unit(self, name):
        return self.units.get(name, dict(LoadState="not-found", ActiveState="inactive"))

    def notify(self, config, message):
        if self.fail_delivery:
            raise RuntimeError("transport unavailable")
        self.messages.append(message)

    def start_recovery(self, config_path, config, unit, directory):
        self.starts.append(unit)
        self.units[unit] = dict(LoadState="loaded", ActiveState="active")
        if self.ambiguous_start:
            raise subprocess.TimeoutExpired("systemd-run", 20)


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "runbook.md").write_text("Only recover the fake test experiment.")
    codex = tmp_path / "codex"
    codex.write_text("test executable")
    config = dict(
        version=1,
        name="test-run",
        monitor_unit="monitor.service",
        observation=str(tmp_path / "observation.json"),
        state_dir=str(tmp_path / "state"),
        workdir=str(tmp_path),
        runbook=str(tmp_path / "runbook.md"),
        codex=str(codex),
        source_sha="a" * 40,
        stale_seconds=300,
        recovery_seconds=180,
        max_recoveries_per_day=2,
        notify_argv=["/bin/echo"],
        uv_environment=str(tmp_path),
    )
    path = tmp_path / "config.json"
    W.write_json(path, config)
    observation = dict(
        source_sha="a" * 40,
        status="waiting_for_gpu_aggregates",
        checked_at=1000,
        backend_observation=dict(status="running", pid_alive=True, last_log_mtime_sec_ago=10),
    )
    W.write_json(Path(config["observation"]), observation)
    return path, config, observation, FakeRuntime()


def state(config):
    return json.loads((Path(config["state_dir"]) / "watchdog.json").read_text())


def fail_monitor(config, observation, runtime):
    # The original incident: monitor notices a worker error, records it, exits.
    result = subprocess.run(
        [sys.executable, "-c", "raise RuntimeError('upload failed')"], capture_output=True
    )
    assert result.returncode != 0
    observation.update(status="backend_failed")
    W.write_json(Path(config["observation"]), observation)
    runtime.units["monitor.service"]["ActiveState"] = "failed"


def test_real_crash_detect_recover_verify_and_quiet_repeated_checks(setup):
    path, config, observation, runtime = setup
    W.tick(path, runtime, 1000)
    assert not runtime.starts and not runtime.messages
    fail_monitor(config, observation, runtime)
    W.tick(path, runtime, 1060)
    assert len(runtime.starts) == 1
    assert any("failure detected" in text for text in runtime.messages)
    W.tick(path, runtime, 1120)
    assert len(runtime.starts) == 1
    # Real recovery subprocess restores the durable monitor observation.
    observation.update(status="waiting_for_gpu_aggregates", checked_at=1180)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            config["observation"],
            json.dumps(observation),
        ],
        check=True,
    )
    runtime.units["monitor.service"]["ActiveState"] = "active"
    runtime.units[runtime.starts[0]]["ActiveState"] = "inactive"
    W.tick(path, runtime, 1180)
    assert state(config)["incident"] is None
    assert any("fresh checks confirm recovery" in text for text in runtime.messages)
    delivered = len(runtime.messages)
    W.tick(path, runtime, 1240)
    assert len(runtime.messages) == delivered and len(runtime.starts) == 1


@pytest.mark.parametrize(
    "change", ["stale", "future", "wrong_source", "dead_pid", "stale_log", "missing_backend"]
)
def test_stale_or_invalid_evidence_never_counts_as_healthy(setup, change):
    path, config, observation, runtime = setup
    if change == "stale":
        observation["checked_at"] = 0
    elif change == "future":
        observation["checked_at"] = 9999
    elif change == "wrong_source":
        observation["source_sha"] = "b" * 40
    elif change == "dead_pid":
        observation["backend_observation"]["pid_alive"] = False
    elif change == "stale_log":
        observation["backend_observation"]["last_log_mtime_sec_ago"] = 1000
    else:
        observation.pop("backend_observation")
    W.write_json(Path(config["observation"]), observation)
    W.tick(path, runtime, 1000)
    assert len(runtime.starts) == 1
    assert state(config)["observed_status"] == "failed"


def test_notification_failure_is_persisted_and_retried(setup):
    path, config, observation, runtime = setup
    fail_monitor(config, observation, runtime)
    runtime.fail_delivery = True
    W.tick(path, runtime, 1000)
    assert all(n["delivered_at"] is None for n in state(config)["notifications"])
    runtime.fail_delivery = False
    W.tick(path, runtime, 1061)
    assert all(n["delivered_at"] is not None for n in state(config)["notifications"])
    assert len(runtime.starts) == 1
    delivered = len(runtime.messages)
    W.tick(path, runtime, 1122)
    assert len(runtime.messages) == delivered


def test_watchdog_dies_after_dispatch_without_duplicate_recovery(setup):
    path, config, observation, runtime = setup
    fail_monitor(config, observation, runtime)
    runtime.ambiguous_start = True
    with pytest.raises(subprocess.TimeoutExpired):
        W.tick(path, runtime, 1000)
    assert state(config)["worker"]["unit"] == runtime.starts[0]
    # A new tick reads the persisted identity after the preceding tick crashed.
    runtime.ambiguous_start = False
    W.tick(path, runtime, 1060)
    assert len(runtime.starts) == 1
    assert any("failure detected" in t for t in runtime.messages)


def test_failed_recovery_is_bounded_and_false_success_is_rejected(setup):
    path, config, observation, runtime = setup
    fail_monitor(config, observation, runtime)
    W.tick(path, runtime, 1000)
    for now in [1060, 1120, 1180, 1240]:
        runtime.units[runtime.starts[-1]].update(ActiveState="inactive", Result="success")
        W.tick(path, runtime, now)
    assert len(runtime.starts) == 2
    assert not any("confirm recovery" in text for text in runtime.messages)
    assert sum("limit reached" in text for text in runtime.messages) == 1


def test_crash_creating_attempt_directory_does_not_wedge_next_tick(setup, monkeypatch):
    path, config, observation, runtime = setup
    fail_monitor(config, observation, runtime)
    mkdir = Path.mkdir

    def crash_after_mkdir(self, *args, **kwargs):
        mkdir(self, *args, **kwargs)
        if self.name == "recovery-1":
            raise RuntimeError("injected process death after mkdir")

    monkeypatch.setattr(Path, "mkdir", crash_after_mkdir)
    with pytest.raises(RuntimeError, match="injected"):
        W.tick(path, runtime, 1000)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    W.tick(path, runtime, 1060)
    assert len(runtime.starts) == 1
    assert state(config)["worker"]["directory"].endswith("recovery-2")


def test_recovery_cannot_use_pre_attempt_observation_after_budget_exhausted(setup):
    path, config, observation, runtime = setup
    fail_monitor(config, observation, runtime)
    W.tick(path, runtime, 1060)
    observation["status"] = "waiting_for_gpu_aggregates"
    W.write_json(Path(config["observation"]), observation)
    runtime.units["monitor.service"]["ActiveState"] = "active"
    for now in (1120, 1180, 1240):
        runtime.units[runtime.starts[-1]]["ActiveState"] = "inactive"
        W.tick(path, runtime, now)
    assert not any("confirm recovery" in t for t in runtime.messages)
    assert len(runtime.starts) == 2
    assert state(config)["incident"] is not None


def test_completion_requires_verified_matching_results(setup):
    path, config, observation, runtime = setup
    observation.update(
        status="complete", results={"source_sha": "b" * 40, "verified_revision": "r"}
    )
    W.write_json(Path(config["observation"]), observation)
    # No completion claim for wrong-source results, even with a live service.
    W.tick(path, runtime, 1000)
    assert not any("experiment completed" in text for text in runtime.messages)
    observation["results"]["source_sha"] = config["source_sha"]
    W.write_json(Path(config["observation"]), observation)
    runtime.units["monitor.service"]["ActiveState"] = "inactive"
    W.tick(path, runtime, 2000)
    W.tick(path, runtime, 2060)
    assert sum("experiment completed" in text for text in runtime.messages) == 1


def test_canary_required_before_install(setup):
    path, config, _, _ = setup
    with pytest.raises(FileNotFoundError):
        W.install(path)
    W.write_json(Path(config["state_dir"]) / "canary.json", dict(digest="wrong", passed_at=1000))
    with pytest.raises(AssertionError):
        W.install(path)


def test_systemd_arguments_do_not_expand_dollars_or_specifiers():
    assert W.systemd_quote('/tmp/a "$b" %n') == '"/tmp/a \\"$$b\\" %%n"'
