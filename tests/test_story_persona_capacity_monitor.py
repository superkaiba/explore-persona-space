"""Exercise capacity, handoff and failure boundaries without allocating hardware."""

import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec
from urllib.error import HTTPError

import pytest

from scripts import experiment_watchdog as W
from scripts import runpod_api as API
from scripts import story_persona_capacity_monitor as M


def quote(status=None, price=None):
    """Construct the provider's exact quoted-shape response."""
    q = {"stockStatus": status, "uninterruptablePrice": price}
    return {
        "myself": {"id": "personal-user", "teams": []},
        "gpuTypes": [{"id": "NVIDIA H200", "memoryInGb": 141, "secure": q, "community": dict(q)}],
    }


class FakeRuntime:
    """Substitute only API, systemd and notification boundaries."""

    def __init__(self):
        self.data = quote()
        self.state = "inactive"
        self.starts = 0
        self.messages = []
        self.pods = []
        self.fail_notification = False
        self.ambiguous_start = False

    def query(self):
        return self.data

    def owned_pods(self, config):
        return self.pods

    def unit(self, name):
        return {"ActiveState": self.state}

    def start(self, config):
        self.starts += 1
        self.state = "active"
        if self.ambiguous_start:
            raise subprocess.TimeoutExpired("systemctl", 20)

    def notify(self, config, state, path, now):
        for item in state["notifications"]:
            if item["delivered_at"] is None and not self.fail_notification:
                self.messages.append(item["message"])
                item["delivered_at"] = now
        M.write(path, state)


@pytest.fixture
def setup(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"max_gpu_hours":17}')
    c = dict(
        source_sha="a" * 40,
        capacity_state=str(tmp_path / "state.json"),
        observation=str(tmp_path / "observation.json"),
        interval_seconds=300,
        max_attempts_per_day=2,
        allocation_ledger=str(ledger),
        worker_unit="resume.service",
        handle_file=str(tmp_path / "handle.json"),
        terminal_file=str(tmp_path / "terminal.json"),
        pod_name="pod-2673-crossmodel-deepseek",
    )
    return c, FakeRuntime()


def state(c):
    """Read persisted state, not an in-memory shadow."""
    return json.loads(Path(c["capacity_state"]).read_text())


def finish(c, status):
    """Publish worker outcome and exit evidence in the current attempt directory."""
    a = state(c)["worker"]
    d = Path(a["directory"])
    M.write(
        d / "outcome.json",
        {
            "attempt": a["number"],
            "source_sha": c["source_sha"],
            "status": status,
            "comparison_git_revision": "b" * 40,
            "report_url": "https://github.com/superkaiba/explore-persona-space/blob/"
            + "b" * 40
            + "/report.md",
        },
    )
    M.write(d / "exit.json", {"returncode": 0, "ended_at": 1100})


def test_quiet_wait_and_malformed_capacity(setup):
    c, r = setup
    assert M.tick(c, r, 1000)["status"] == "waiting_for_capacity"
    assert M.tick(c, r, 1300)["status"] == "waiting_for_capacity"
    assert r.starts == 0 and not r.messages
    for status, price in [("Low", None), ("OutOfStock", 10), ("Low", float("nan"))]:
        assert M.eligible(quote(status, price)) == []
    with pytest.raises(KeyError):
        M.eligible({})


def test_single_dispatch_survives_ambiguous_start_and_notification_failure(setup):
    c, r = setup
    r.data = quote("Low", 36.72)
    r.ambiguous_start = True
    with pytest.raises(subprocess.TimeoutExpired):
        M.tick(c, r, 1000)
    assert state(c)["worker"]["number"] == 1
    r.fail_notification = True
    assert M.tick(c, r, 1030)["status"] == "continuation_running"
    assert r.starts == 1


def test_delivery_retry_and_capacity_loss(setup):
    c, r = setup
    r.data = quote("Low", 36.72)
    r.fail_notification = True
    M.tick(c, r, 1000)
    assert state(c)["notifications"][0]["delivered_at"] is None
    r.fail_notification = False
    M.tick(c, r, 1030)
    assert len(r.messages) == 1
    r.state = "inactive"
    finish(c, "capacity_lost")
    assert M.tick(c, r, 1100)["status"] == "capacity_lost_waiting_again"
    M.tick(c, r, 1200)
    assert r.starts == 1
    M.tick(c, r, 1400)
    assert r.starts == 2


def test_repeated_verified_capacity_losses_do_not_exhaust_daily_attempts(setup):
    c, r = setup
    c["interval_seconds"] = 60
    r.data = quote("Low", 36.72)
    for number in range(1, 5):
        now = 1000 + (number - 1) * 120
        assert M.tick(c, r, now)["status"] == "continuation_requested"
        assert state(c)["worker"]["number"] == number
        r.state = "inactive"
        finish(c, "capacity_lost")
        M.write(
            Path(state(c)["worker"]["directory"]) / "exit.json",
            {"returncode": 0, "ended_at": now + 10},
        )
        assert M.tick(c, r, now + 10)["status"] == "capacity_lost_waiting_again"
        assert state(c)["retry_after"] == now + 70
    assert state(c)["verified_capacity_losses"] == [1, 2, 3, 4]
    assert r.starts == 4


def test_unverified_attempts_still_count_towards_daily_limit(setup):
    c, r = setup
    M.write(c["capacity_state"], {"attempts": [800, 900], "worker": None, "notifications": []})
    r.data = quote("Low", 36.72)
    assert M.tick(c, r, 1000)["status"] == "capacity_available_attempt_limit"
    assert r.starts == 0


@pytest.mark.parametrize("bad", ["pod", "handle", "ledger"])
def test_capacity_loss_cannot_discard_an_allocation(setup, bad):
    c, r = setup
    r.data = quote("Low", 36.72)
    M.tick(c, r, 1000)
    r.state = "inactive"
    finish(c, "capacity_lost")
    if bad == "pod":
        r.pods = [{"id": "actual-pod"}]
    elif bad == "handle":
        Path(c["handle_file"]).write_text("{}")
    else:
        Path(c["allocation_ledger"]).write_text('{"changed":true}')
    with pytest.raises(ValueError):
        M.tick(c, r, 1100)
    assert state(c)["worker"]
    assert not state(c).get("verified_capacity_losses")


def test_missing_outcome_and_stale_logs_surface_to_watchdog(setup):
    c, r = setup
    r.data = quote("Low", 36.72)
    M.tick(c, r, 1000)
    o = M.tick(c, r, 2200)
    assert o["backend_observation"]["last_log_mtime_sec_ago"] == 1200
    r.state = "failed"
    assert M.tick(c, r, 2201)["backend_observation"]["status"] == "gate"


def terminal_record(c):
    """Full verification evidence expected from the independent publication check."""
    return dict(
        source_sha=c["source_sha"],
        model_key="deepseek",
        verified_revision="c" * 40,
        continuation_attempt=1,
        checked_at=1050,
        handle_file=c["handle_file"],
        all_remote_names_sizes_hashes_pass=True,
        git_json_equals_immutable_hf=True,
        smoke_passed=True,
        row_count=1920,
        chunks=240,
        selected_layers=[15, 30, 45, 60],
        file_count=261,
        total_bytes=1000,
        capture_fingerprint="d" * 64,
        analysis_fingerprint="e" * 64,
    )


def publish_terminal(c, terminal):
    """Bind the outcome to the exact terminal bytes as the real worker must."""
    M.write(c["terminal_file"], terminal)
    outcome_path = Path(state(c)["worker"]["directory"]) / "outcome.json"
    outcome = json.loads(outcome_path.read_text())
    outcome["terminal_sha256"] = hashlib.sha256(Path(c["terminal_file"]).read_bytes()).hexdigest()
    M.write(outcome_path, outcome)


@pytest.mark.parametrize(
    "bad",
    [
        {"model_key": "qwen"},
        {"checked_at": 10},
        {"smoke_passed": False},
        {"continuation_attempt": 2},
        {"verified_revision": "main"},
        {"row_count": 1919},
    ],
)
def test_verified_completion_requires_fresh_full_evidence(setup, bad):
    c, r = setup
    r.data = quote("Low", 36.72)
    M.tick(c, r, 1000)
    r.state = "inactive"
    finish(c, "complete")
    terminal = {**terminal_record(c), **bad}
    publish_terminal(c, terminal)
    with pytest.raises(ValueError):
        M.tick(c, r, 1100)
    publish_terminal(c, terminal_record(c))
    assert M.tick(c, r, 1101)["status"] == "complete"


def test_completion_is_not_sticky_until_notifications_acknowledged(setup):
    c, r = setup
    r.data = quote("Low", 36.72)
    M.tick(c, r, 1000)
    r.state = "inactive"
    finish(c, "complete")
    publish_terminal(c, terminal_record(c))
    r.fail_notification = True
    assert M.tick(c, r, 1101)["status"] == "completion_notification_pending"
    dead = SimpleNamespace(unit=lambda name: {"ActiveState": "inactive"})
    assert (
        W.assessment({**c, "monitor_unit": "monitor.service", "stale_seconds": 900}, dead, 1102)[0]
        == "failed"
    )
    r.fail_notification = False
    assert M.tick(c, r, 1161)["status"] == "complete"


def test_runtime_start_body_and_query_shape(setup, monkeypatch):
    c, _ = setup
    r = M.Runtime.__new__(M.Runtime)
    r.expected_account_id = "personal-user"
    run = create_autospec(subprocess.run, return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(M.subprocess, "run", run)
    r.start(c)
    assert run.call_args.args[0] == ["systemctl", "--user", "start", "--no-block", "resume.service"]

    def graphql(query, variables=None, timeout=60):
        return quote("Low", 36.72)

    r.transport = create_autospec(graphql, side_effect=graphql)
    assert M.eligible(r.query())
    assert "gpuCount:8" in r.transport.call_args.args[0]
    assert "myself { id teams { id } }" in r.transport.call_args.args[0]
    r.system = SimpleNamespace(unit=create_autospec(W.Runtime.unit, instance=True))


@pytest.mark.parametrize(
    "account", [{"id": "wrong", "teams": []}, {"id": "personal-user", "teams": [{"id": "team"}]}]
)
def test_actual_query_rejects_account_change(account):
    r = M.Runtime.__new__(M.Runtime)
    r.expected_account_id = "personal-user"

    def graphql(query, variables=None, timeout=60):
        return {**quote("Low", 36.72), "myself": account}

    r.transport = create_autospec(graphql, side_effect=graphql)
    with pytest.raises(ValueError, match="identity/scope changed"):
        r.query()


def test_owned_pods_verifies_personal_identity(setup):
    c, _ = setup
    r = M.Runtime.__new__(M.Runtime)
    r.expected_account_id = "personal-user"

    def graphql(query, variables=None, timeout=60):
        return {
            "myself": {
                "id": "personal-user",
                "teams": [],
                "pods": [
                    {"id": "owned", "name": c["pod_name"]},
                    {"id": "unrelated", "name": "another"},
                ],
            }
        }

    r.transport = create_autospec(graphql, side_effect=graphql)
    assert r.owned_pods(c) == [{"id": "owned", "name": c["pod_name"]}]


@pytest.mark.parametrize("team", ["legacy-team-that-must-not-be-sent", ""])
def test_personal_runtime_uses_headerless_transport_and_retries(setup, monkeypatch, tmp_path, team):
    c, _ = setup
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    c.update(
        account_scope="personal",
        expected_account_id="personal-user",
        workdir=str(Path(M.__file__).resolve().parent.parent),
        dotenv=str(env_file),
        watchdog_helper=W.__file__,
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "test-api-key-not-real")
    monkeypatch.setenv("RUNPOD_TEAM_ID", team)
    monkeypatch.setattr(API.time, "sleep", create_autospec(API.time.sleep))
    boundary = create_autospec(
        API.urlrequest.urlopen,
        side_effect=[
            HTTPError(API.GRAPHQL_URL, 503, "temporary", {}, io.BytesIO(b"temporary")),
            io.BytesIO(json.dumps({"data": quote("Low", 36.72)}).encode()),
        ],
    )
    monkeypatch.setattr(API.urlrequest, "urlopen", boundary)
    runtime = M.Runtime(c)
    assert M.eligible(runtime.query())
    assert boundary.call_count == 2
    for call in boundary.call_args_list:
        request = call.args[0]
        assert request.get_header("X-team-id") is None
        assert request.get_header("Authorization") == "Bearer test-api-key-not-real"
        assert request.get_full_url() == API.GRAPHQL_URL


def test_worker_body_executes_real_subprocess(setup, tmp_path):
    c, r = setup
    r.data = quote("Low", 36.72)
    M.tick(c, r, 1000)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Test handoff only; no experiment.")
    exe = tmp_path / "codex"
    exe.write_text('#!/bin/sh\ncat >/dev/null\nprintf "executed harmless boundary\\n"\n')
    exe.chmod(0o700)
    c.update(
        continuation_prompt=str(prompt),
        continuation_seconds=10,
        codex=str(exe),
        workdir=str(tmp_path),
    )
    r.helper = W
    assert M.worker(c, r) == 0
    directory = Path(state(c)["worker"]["directory"])
    assert json.loads((directory / "exit.json").read_text())["returncode"] == 0
    assert "executed harmless boundary" in (directory / "worker.log").read_text()
