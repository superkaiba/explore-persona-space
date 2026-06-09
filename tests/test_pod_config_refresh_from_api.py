"""Tests for ``pod_config.cmd_refresh_from_api`` — pulling live host/port
from the RunPod API into ``pods.conf`` outside an explicit provision/resume.

Regression target: task #488 incident on 2026-06-09. A
SUPPLY_CONSTRAINT-blocked ``pod.py resume`` hard-exited, the pod later came
back at a NEW SSH port via a retry path that bypassed
``_upsert_pods_conf``, and the autonomous session's SSH polling loop spun
for 13+ hours at $32/hr on the pre-stop port because ``pods.conf`` stayed
stale. ``cmd_sync`` only propagates ``pods.conf`` OUTWARD; nothing pulled
the live API INWARD until now.

The fix adds ``pod_config.cmd_refresh_from_api(pods, pod_name | None)``,
wired as ``pod.py config --refresh-from-api [<name>]``. These tests pin
its contract:

* Fresh host/port from the live API land in ``pods.conf`` (the rows
  handed to ``write_pods_conf``).
* Pods with ``manual_override=True`` are NOT overwritten — instead a WARN
  surfaces (same discipline as ``_upsert_pods_conf`` / ``cmd_update``).
* Non-RUNNING pods are SKIPPED (single-pod mode fails loud; bulk mode
  warns and continues), because SSH endpoints don't exist for them.
* Pods missing from the live API are SKIPPED (single-pod fails loud;
  bulk warns and continues), because we can't infer their endpoint.
* Pods not in ``pods.conf`` fail loud in single-pod mode (typo guard).
* Tests stub ``parse_pods_conf`` / ``write_pods_conf`` / ``cmd_sync``
  directly (mirroring the pattern in
  ``test_pod_config_sync_preserves_manual.py``) — the function under test
  re-parses pods.conf inside its lock, so a default-arg ``path=PODS_CONF``
  captured at function-def time would defeat ``monkeypatch.setattr`` on
  the module constant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
from pod_config import Pod  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _info(
    name: str,
    *,
    pod_id: str = "live_id",
    desired_status: str = "RUNNING",
    gpu_count: int = 1,
    ssh_host: str | None = "10.0.0.1",
    ssh_port: int | None = 22000,
) -> PodInfo:
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status=desired_status,
        gpu_count=gpu_count,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        created_at="2026-06-09T00:00:00Z",
    )


def _row(name: str, host: str = "1.2.3.4", port: int = 11111) -> Pod:
    return Pod(name=name, host=host, port=port, gpus=1, gpu_type="H100", label=f"thomas-{name}")


def _copy_rows(rows: list[Pod]) -> list[Pod]:
    return [
        Pod(name=p.name, host=p.host, port=p.port, gpus=p.gpus, gpu_type=p.gpu_type, label=p.label)
        for p in rows
    ]


def _write_sidecar_with_overrides(path: Path, overrides: dict[str, bool]) -> None:
    """Write a minimal pods_ephemeral.json carrying only the manual_override
    flag for each named pod — that's all ``_read_manual_overrides`` reads.
    """
    payload = {
        "version": 2,
        "updated_at": "2026-06-09T00:00:00Z",
        "pods": {
            name: {
                "name": name,
                "pod_id": f"{name}_id",
                "issue": int(name.split("-")[-1]) if "-" in name else 0,
                "gpu_intent": "custom",
                "ttl_days": 7,
                "stopped_at": None,
                "notes": "",
                "manual_override": flag,
                "extra": {},
            }
            for name, flag in overrides.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture
def stubbed_pods_conf(monkeypatch):
    """Stub ``parse_pods_conf`` / ``write_pods_conf`` / ``cmd_sync`` and
    ``locked_pods_conf`` so the function under test never touches the real
    repo state. Mirrors the pattern in
    ``test_pod_config_sync_preserves_manual.py`` — necessary because
    ``parse_pods_conf`` captures ``PODS_CONF`` as a default arg at
    function-def time, so a plain ``monkeypatch.setattr(pod_config,
    "PODS_CONF", tmp)`` is ignored by callers inside the locked region.

    Returns a state dict the test can mutate (``state["rows"]`` for the
    fake on-disk pods.conf) and read (``state["written"]`` for what
    ``write_pods_conf`` was handed, ``state["sync_called"]`` for whether
    ``cmd_sync`` was invoked, ``state["sync_rows"]`` for its argument).
    """
    import contextlib

    state: dict[str, object] = {
        "rows": [],
        "written": None,
        "sync_called": False,
        "sync_rows": None,
    }

    def fake_parse() -> list[Pod]:
        # Return a fresh copy each call so the function under test mutates
        # the snapshot it gets back, not our fixture's underlying list.
        return _copy_rows(state["rows"])  # type: ignore[arg-type]

    def fake_write(rows: list[Pod]) -> None:
        state["written"] = _copy_rows(rows)

    def fake_sync(rows: list[Pod]) -> None:
        state["sync_called"] = True
        state["sync_rows"] = _copy_rows(rows)

    @contextlib.contextmanager
    def noop_lock():
        # The real lock holds an flock on ``PODS_CONF_LOCK`` — out of scope
        # for these unit tests. Concurrency is covered by
        # ``test_pod_config_locking.py``.
        yield

    monkeypatch.setattr(pod_config, "parse_pods_conf", fake_parse)
    monkeypatch.setattr(pod_config, "write_pods_conf", fake_write)
    monkeypatch.setattr(pod_config, "cmd_sync", fake_sync)
    monkeypatch.setattr(pod_config, "locked_pods_conf", noop_lock)

    return state


@pytest.fixture
def isolated_sidecar(tmp_path, monkeypatch):
    """Point ``pod_config.PODS_EPHEMERAL_JSON`` at a tmpdir copy so
    ``_read_manual_overrides`` reads the test sidecar, not the real one."""
    sidecar = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", sidecar)
    return sidecar


@pytest.fixture
def stub_list_team_pods(monkeypatch):
    """Stub ``runpod_api.list_team_pods``. ``cmd_refresh_from_api`` does a
    lazy ``from runpod_api import list_team_pods`` at call time, so we
    patch the module-level name and the lazy import picks up the stub.
    """
    import runpod_api

    class _Stub:
        def __init__(self) -> None:
            self.return_value: list[PodInfo] = []

        def __call__(self) -> list[PodInfo]:
            return list(self.return_value)

    stub = _Stub()
    monkeypatch.setattr(runpod_api, "list_team_pods", stub)
    return stub


# ---------------------------------------------------------------------------
# Happy path: refresh ALL pods → live host/port land in pods.conf
# ---------------------------------------------------------------------------


def test_refresh_all_updates_stale_pods_conf(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """Two pods in pods.conf are stale; the live API has fresh host/port.
    ``cmd_refresh_from_api(pods, None)`` writes the fresh values to
    ``pods.conf`` and triggers a downstream ``cmd_sync``.
    """
    stale = [
        _row("pod-488", host="1.1.1.1", port=11111),
        _row("pod-500", host="2.2.2.2", port=22222),
    ]
    stubbed_pods_conf["rows"] = _copy_rows(stale)

    stub_list_team_pods.return_value = [
        _info("pod-488", ssh_host="103.207.149.130", ssh_port=18166),
        _info("pod-500", ssh_host="9.9.9.9", ssh_port=33333),
    ]

    pod_config.cmd_refresh_from_api(_copy_rows(stale), None)

    written = stubbed_pods_conf["written"]
    assert written is not None
    by_name = {p.name: (p.host, p.port) for p in written}
    assert by_name["pod-488"] == ("103.207.149.130", 18166)
    assert by_name["pod-500"] == ("9.9.9.9", 33333)
    assert stubbed_pods_conf["sync_called"] is True

    out = capsys.readouterr().out
    assert "1.1.1.1:11111 -> 103.207.149.130:18166" in out
    assert "Updating pods.conf with live API host/port" in out


# ---------------------------------------------------------------------------
# Single-pod mode
# ---------------------------------------------------------------------------


def test_refresh_single_pod_updates_only_target(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods
):
    """When ``pod_name`` is given, only that pod is refreshed even if other
    pods are also stale (lets the orchestrator target a known stale row).
    """
    rows = [
        _row("pod-488", host="1.1.1.1", port=11111),
        _row("pod-500", host="2.2.2.2", port=22222),
    ]
    stubbed_pods_conf["rows"] = _copy_rows(rows)

    stub_list_team_pods.return_value = [
        _info("pod-488", ssh_host="103.207.149.130", ssh_port=18166),
        _info("pod-500", ssh_host="9.9.9.9", ssh_port=33333),
    ]

    pod_config.cmd_refresh_from_api(_copy_rows(rows), "pod-488")

    written = stubbed_pods_conf["written"]
    by_name = {p.name: (p.host, p.port) for p in written}
    assert by_name["pod-488"] == ("103.207.149.130", 18166)
    # The untargeted pod is left alone.
    assert by_name["pod-500"] == ("2.2.2.2", 22222)


# ---------------------------------------------------------------------------
# manual_override is honored
# ---------------------------------------------------------------------------


def test_refresh_respects_manual_override(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """A pod flagged ``manual_override=True`` in pods_ephemeral.json must
    NOT be overwritten, even if the live API has different values. This is
    the same discipline ``_upsert_pods_conf`` enforces in pod_lifecycle.
    """
    rows = [_row("pod-488", host="1.1.1.1", port=11111)]
    stubbed_pods_conf["rows"] = _copy_rows(rows)
    _write_sidecar_with_overrides(isolated_sidecar, {"pod-488": True})

    stub_list_team_pods.return_value = [
        _info("pod-488", ssh_host="103.207.149.130", ssh_port=18166),
    ]

    pod_config.cmd_refresh_from_api(_copy_rows(rows), None)

    # No write because the manual_override blocked the only candidate change.
    assert stubbed_pods_conf["written"] is None
    assert stubbed_pods_conf["sync_called"] is False

    err = capsys.readouterr().err
    assert "manual_override=True" in err
    assert "pod-488" in err


# ---------------------------------------------------------------------------
# Skip / fail on non-RUNNING pods
# ---------------------------------------------------------------------------


def test_refresh_skips_non_running_pod_in_bulk_mode(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """An EXITED pod can't have its endpoint refreshed — its SSH mapping is
    gone. In bulk mode we WARN and skip (don't blank out the existing row).
    """
    rows = [
        _row("pod-488", host="1.1.1.1", port=11111),
        _row("pod-500", host="2.2.2.2", port=22222),
    ]
    stubbed_pods_conf["rows"] = _copy_rows(rows)
    stub_list_team_pods.return_value = [
        _info("pod-488", desired_status="EXITED", ssh_host=None, ssh_port=None),
        _info("pod-500", ssh_host="9.9.9.9", ssh_port=33333),
    ]

    pod_config.cmd_refresh_from_api(_copy_rows(rows), None)

    written = stubbed_pods_conf["written"]
    assert written is not None
    by_name = {p.name: (p.host, p.port) for p in written}
    # EXITED pod untouched.
    assert by_name["pod-488"] == ("1.1.1.1", 11111)
    # RUNNING sibling refreshed.
    assert by_name["pod-500"] == ("9.9.9.9", 33333)

    err = capsys.readouterr().err
    assert "desiredStatus=EXITED" in err
    assert "pod-488" in err


def test_refresh_single_non_running_pod_fails_loud(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """Single-pod mode is a directed action; if the named pod isn't RUNNING
    we fail loud with an actionable message (the user almost certainly
    meant to refresh a running pod).
    """
    rows = [_row("pod-488", host="1.1.1.1", port=11111)]
    stubbed_pods_conf["rows"] = _copy_rows(rows)
    stub_list_team_pods.return_value = [
        _info("pod-488", desired_status="EXITED", ssh_host=None, ssh_port=None),
    ]

    with pytest.raises(SystemExit) as excinfo:
        pod_config.cmd_refresh_from_api(_copy_rows(rows), "pod-488")
    assert excinfo.value.code == 1
    assert stubbed_pods_conf["written"] is None

    err = capsys.readouterr().err
    assert "desiredStatus=EXITED" in err
    assert "pod.py resume" in err


# ---------------------------------------------------------------------------
# Pod missing from live API
# ---------------------------------------------------------------------------


def test_refresh_skips_pod_missing_from_api_in_bulk_mode(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """A row in pods.conf with no matching live API entry (terminated
    externally) is skipped with a WARN in bulk mode."""
    rows = [
        _row("pod-488", host="1.1.1.1", port=11111),
        _row("pod-500", host="2.2.2.2", port=22222),
    ]
    stubbed_pods_conf["rows"] = _copy_rows(rows)
    # API has only pod-500.
    stub_list_team_pods.return_value = [
        _info("pod-500", ssh_host="9.9.9.9", ssh_port=33333),
    ]

    pod_config.cmd_refresh_from_api(_copy_rows(rows), None)

    written = stubbed_pods_conf["written"]
    by_name = {p.name: (p.host, p.port) for p in written}
    assert by_name["pod-488"] == ("1.1.1.1", 11111)
    assert by_name["pod-500"] == ("9.9.9.9", 33333)

    err = capsys.readouterr().err
    assert "not in the live RunPod API" in err
    assert "pod-488" in err


def test_refresh_single_pod_missing_from_api_fails_loud(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """Single-pod mode fails loud when the named pod is gone from the API
    (user almost certainly wants to know rather than silently no-op)."""
    rows = [_row("pod-488", host="1.1.1.1", port=11111)]
    stubbed_pods_conf["rows"] = _copy_rows(rows)
    stub_list_team_pods.return_value = []

    with pytest.raises(SystemExit) as excinfo:
        pod_config.cmd_refresh_from_api(_copy_rows(rows), "pod-488")
    assert excinfo.value.code == 1
    assert stubbed_pods_conf["written"] is None
    err = capsys.readouterr().err
    assert "not in the live RunPod API" in err


# ---------------------------------------------------------------------------
# Unknown pod name in single-pod mode
# ---------------------------------------------------------------------------


def test_refresh_unknown_pod_name_in_single_mode_fails_loud(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """If the user passes a pod name not in pods.conf, fail loud — this is
    almost always a typo. The check fires BEFORE the lock + API call, so we
    don't even need the API stub to ring."""
    rows = [_row("pod-488", host="1.1.1.1", port=11111)]
    stubbed_pods_conf["rows"] = _copy_rows(rows)

    with pytest.raises(SystemExit) as excinfo:
        pod_config.cmd_refresh_from_api(_copy_rows(rows), "pod-999")
    assert excinfo.value.code == 1
    assert stubbed_pods_conf["written"] is None
    err = capsys.readouterr().err
    assert "pod-999" in err
    assert "not found in pods.conf" in err


# ---------------------------------------------------------------------------
# No-op safety
# ---------------------------------------------------------------------------


def test_refresh_noop_when_already_in_sync(
    stubbed_pods_conf, isolated_sidecar, stub_list_team_pods, capsys
):
    """pods.conf already matches the live API — no write, no sync, clean
    "all in sync" message."""
    rows = [_row("pod-488", host="103.207.149.130", port=18166)]
    stubbed_pods_conf["rows"] = _copy_rows(rows)
    stub_list_team_pods.return_value = [
        _info("pod-488", ssh_host="103.207.149.130", ssh_port=18166),
    ]

    pod_config.cmd_refresh_from_api(_copy_rows(rows), None)

    assert stubbed_pods_conf["written"] is None
    assert stubbed_pods_conf["sync_called"] is False

    out = capsys.readouterr().out
    assert "already at 103.207.149.130:18166" in out
    assert "already match the live RunPod API" in out
