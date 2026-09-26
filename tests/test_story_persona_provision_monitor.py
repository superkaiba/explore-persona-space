"""Capacity waiting requires a real service and fresh provider-result evidence."""

import json
from pathlib import Path

import pytest

from scripts import story_persona_qwen38_monitor as monitor


@pytest.fixture
def config(tmp_path):
    """Use isolated observation files; no systemd or provider mutations occur."""
    return {
        "source_sha": "a" * 40,
        "created_at": 1000,
        "provision_unit": "test-capacity.service",
        "provision_state": str(tmp_path / "provision.json"),
        "handle_file": str(tmp_path / "handle.json"),
        "observation": str(tmp_path / "observation.json"),
        "out_dir": str(tmp_path / "out"),
        "workdir": str(tmp_path),
    }


def save_state(config, **overrides):
    """Record a controlled provider observation for the actual monitor reader."""
    state = {
        "source_sha": config["source_sha"],
        "checked_at": 1000,
        "status": "waiting_for_capacity",
        "attempt": 1,
        "provider_refusal_verified": True,
    }
    state.update(overrides)
    Path(config["provision_state"]).write_text(json.dumps(state))


@pytest.fixture
def active_service(monkeypatch):
    """Assert that liveness comes from systemd rather than the saved heartbeat."""
    def command(argv, *, timeout):
        assert argv == [
            "systemctl", "--user", "show", "test-capacity.service",
            "--property=ActiveState,SubState,MainPID",
        ]
        assert timeout == 15
        return "ActiveState=active\nSubState=running\nMainPID=123\n"

    monkeypatch.setattr(monitor, "command", command)


@pytest.mark.parametrize(
    ("status", "age"), [("requesting_capacity", 1260), ("waiting_for_capacity", 150)]
)
def test_live_capacity_evidence_has_bounded_phase_specific_freshness(
    config, active_service, status, age
):
    save_state(config, status=status)
    result = monitor.provision_observation(config, now=1000 + age)
    assert result["status"] == "pending" and result["current_phase"] == status
    assert result["attempt"] == 1 and result["provider_checked_at"] == 1000
    with pytest.raises(RuntimeError, match="stale or malformed"):
        monitor.provision_observation(config, now=1000 + age + 0.01)


def test_missing_state_is_only_allowed_during_active_startup(config, active_service):
    assert monitor.provision_observation(config, now=1030)["current_phase"] == (
        "capacity_service_startup"
    )
    with pytest.raises(RuntimeError, match="missing beyond startup"):
        monitor.provision_observation(config, now=1030.01)


@pytest.mark.parametrize(
    "service",
    [
        "ActiveState=failed\nSubState=failed\nMainPID=0\n",
        "ActiveState=active\nSubState=exited\nMainPID=0\n",
        "ActiveState=active\nSubState=running\nMainPID=0\n",
    ],
)
def test_fresh_state_does_not_hide_dead_service(config, monkeypatch, service):
    save_state(config)
    monkeypatch.setattr(monitor, "command", lambda *a, **kw: service)
    with pytest.raises(RuntimeError, match="not actively running"):
        monitor.provision_observation(config, now=1001)


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_sha": "b" * 40},
        {"status": "failed"},
        {"status": "complete"},
        {"checked_at": 1002},
        {"checked_at": "1000"},
        {"checked_at": float("nan")},
        {"attempt": 0},
        {"attempt": True},
        {"provider_refusal_verified": False},
        {"provider_refusal_verified": "true"},
    ],
)
def test_invalid_capacity_state_fails_loudly(config, active_service, overrides):
    save_state(config, **overrides)
    with pytest.raises(RuntimeError):
        monitor.provision_observation(config, now=1001)


def test_malformed_json_is_not_a_capacity_wait(config, active_service):
    Path(config["provision_state"]).write_text("not-json")
    with pytest.raises(json.JSONDecodeError):
        monitor.provision_observation(config, now=1001)


def test_source_repin_requires_new_matching_provider_evidence(config, active_service):
    save_state(config)
    config["source_sha"] = "b" * 40
    with pytest.raises(RuntimeError, match="source pin"):
        monitor.provision_observation(config, now=1001)
    save_state(config)
    assert monitor.provision_observation(config, now=1001)["status"] == "pending"


def test_long_capacity_wait_transitions_to_actual_backend(config, monkeypatch):
    """Exercise the loop beyond both old limits, then demand the real handle probe."""
    clock = [1001]
    monkeypatch.setattr(monitor.time, "time", lambda: clock[0])
    save_state(config)
    probes = []

    def command(argv, **kwargs):
        if argv[0] == "systemctl":
            return "ActiveState=active\nSubState=running\nMainPID=123\n"
        assert "--probe-handle" in argv and config["handle_file"] in argv
        probes.append(argv)
        return json.dumps({"status": "failed", "pilot": {"pids": []}})

    sleeps = []

    def sleep(seconds):
        observation = json.loads(Path(config["observation"]).read_text())
        assert observation["status"] == "awaiting_launch"
        assert observation["backend_observation"]["current_phase"] == "waiting_for_capacity"
        sleeps.append(seconds)
        if len(sleeps) == 1:
            clock[0] += 7 * 3600
            save_state(config, checked_at=clock[0], attempt=400)
        else:
            Path(config["handle_file"]).write_text("{}")

    monkeypatch.setattr(monitor, "command", command)
    monkeypatch.setattr(monitor.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="backend requires recovery: failed"):
        monitor.monitor(config)
    assert len(sleeps) == 2 and len(probes) == 1
    observation = json.loads(Path(config["observation"]).read_text())
    assert observation["status"] == "backend_failed"
    assert config["created_at"] == 1000


def test_default_monitor_keeps_original_launch_timeout(config, monkeypatch):
    del config["provision_unit"], config["provision_state"]
    monkeypatch.setattr(monitor.time, "time", lambda: 2801)
    with pytest.raises(RuntimeError, match="within 30 minutes"):
        monitor.monitor(config)


def test_handle_published_during_capacity_probe_takes_over(config, monkeypatch):
    """A retiring provision unit must yield to the newly published real handle."""
    monkeypatch.setattr(monitor.time, "time", lambda: 1001)
    probes = []

    def command(argv, **kwargs):
        if argv[0] == "systemctl":
            Path(config["handle_file"]).write_text("{}")
            return "ActiveState=inactive\nSubState=dead\nMainPID=0\n"
        assert "--probe-handle" in argv
        probes.append(argv)
        return json.dumps({"status": "failed", "pilot": {"pids": []}})

    monkeypatch.setattr(monitor, "command", command)
    with pytest.raises(RuntimeError, match="backend requires recovery: failed"):
        monitor.monitor(config)
    assert len(probes) == 1
    assert json.loads(Path(config["observation"]).read_text())["status"] == "backend_failed"


def test_partial_capacity_configuration_fails(config):
    del config["provision_state"]
    with pytest.raises(ValueError, match="both provision_unit and provision_state"):
        monitor.monitor(config)
