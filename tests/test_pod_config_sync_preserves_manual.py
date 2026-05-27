"""Regression tests for the manual-override flag (post-mortem from task #391).

Bug: ``pod.py config --update pod-391 --host A --port B`` would correctly
write the manual values to ``pods.conf``, but a later auto-refresh path in
``pod_lifecycle.py`` (drift repair in ``_load_state`` or host/port write in
``_upsert_pods_conf``) would silently clobber them if the live RunPod API
returned a DIFFERENT pod that happened to share the ``pod-391`` name. The
SSH alias would be repointed without warning and downstream subagents would
run on the wrong pod.

Fix: ``cmd_update`` flips ``manual_override=True`` in pods_ephemeral.json
for the matching ephemeral pod entry. The auto-refresh paths check the flag
and skip the overwrite, surfacing a stderr WARN instead. Cleared via
``--clear-override``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
import pod_lifecycle  # noqa: E402
from pod_config import Pod  # noqa: E402
from pod_lifecycle import (  # noqa: E402
    EphemeralMetadata,
    EphemeralPod,
    _load_state,
    _read_metadata_file,
    _upsert_pods_conf,
    _write_metadata_file,
)
from runpod_api import PodInfo  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (mirror test_pod_lifecycle.py)
# ---------------------------------------------------------------------------


def _info(
    name: str,
    *,
    pod_id: str,
    desired_status: str = "RUNNING",
    gpu_count: int = 1,
    ssh_host: str | None = "1.2.3.4",
    ssh_port: int | None = 12345,
) -> PodInfo:
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status=desired_status,
        gpu_count=gpu_count,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        created_at="2026-05-26T00:00:00Z",
    )


def _meta(
    name: str,
    *,
    issue: int,
    pod_id: str,
    manual_override: bool = False,
) -> EphemeralMetadata:
    return EphemeralMetadata(
        name=name,
        pod_id=pod_id,
        issue=issue,
        gpu_intent="custom",
        ttl_days=7,
        stopped_at=None,
        notes="",
        manual_override=manual_override,
    )


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point both pod_lifecycle and pod_config at a tmpdir sidecar JSON.

    Both modules see the same file; pod_lifecycle writes through
    ``_write_metadata_file`` and pod_config reads/writes the same path
    through ``_set_manual_override``.
    """
    state_file = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state_file)
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", state_file)
    return state_file


@pytest.fixture
def stub_list_team_pods(monkeypatch):
    class _Stub:
        def __init__(self):
            self.return_value: list[PodInfo] = []

        def __call__(self):
            return list(self.return_value)

    stub = _Stub()
    monkeypatch.setattr(pod_lifecycle, "list_team_pods", stub)
    return stub


# ---------------------------------------------------------------------------
# Core scenario: name-collision in live API + manual override on sidecar
# ---------------------------------------------------------------------------


def test_load_state_preserves_pod_id_when_manual_override_true(
    isolated_state, stub_list_team_pods, capsys
):
    """Sidecar has pod-391 with pod_id=NEW and manual_override=True.
    Live API returns a DIFFERENT pod also named pod-391 with pod_id=OLD.
    _load_state must KEEP the sidecar pod_id and WARN — not silently repair.

    Reproduces the #391 silent-clobber: pre-fix, _load_state would have
    rewritten pods_ephemeral.json with pod_id=OLD on the next read.
    """
    _write_metadata_file(
        {
            "pod-391": _meta(
                "pod-391",
                issue=391,
                pod_id="new_pod_id",
                manual_override=True,
            ),
        }
    )
    stub_list_team_pods.return_value = [
        _info("pod-391", pod_id="stale_old_pod_id"),
    ]

    state = _load_state()

    # In-memory view keeps the sidecar's pod_id, not the API's.
    assert state["pod-391"].pod_id == "new_pod_id"

    # Sidecar file UNCHANGED — no silent repair-through-write.
    on_disk = _read_metadata_file()
    assert on_disk["pod-391"].pod_id == "new_pod_id"
    assert on_disk["pod-391"].manual_override is True

    # WARN surfaced so the user sees the divergence.
    err = capsys.readouterr().err
    assert "manual_override=True" in err
    assert "pod-391" in err
    assert "stale_old_pod_id" in err


def test_load_state_drift_repair_still_runs_without_override(
    isolated_state, stub_list_team_pods, capsys
):
    """Counterfactual: same setup but manual_override=False. Drift-repair
    behaves as before #391 (silently overwrites sidecar pod_id with live).
    Guards against the fix accidentally suppressing the existing repair path.
    """
    _write_metadata_file(
        {
            "pod-7": _meta(
                "pod-7",
                issue=7,
                pod_id="stale_id",
                manual_override=False,
            ),
        }
    )
    stub_list_team_pods.return_value = [_info("pod-7", pod_id="live_id")]

    state = _load_state()
    assert state["pod-7"].pod_id == "live_id"
    assert _read_metadata_file()["pod-7"].pod_id == "live_id"
    err = capsys.readouterr().err
    assert "drifted" in err  # existing drift-repair WARN


def test_upsert_pods_conf_keeps_manual_host_port(monkeypatch, capsys):
    """The pods.conf row has manually-set host/port. _upsert_pods_conf is
    called with an EphemeralPod whose live info disagrees AND
    manual_override=True. Host/port must NOT be overwritten; gpus/gpu_type/
    label (not user-overrideable via --update) ARE refreshed.
    """
    rows = [
        Pod(
            name="pod-391",
            host="31.24.80.42",
            port=10439,
            gpus=1,
            gpu_type="H100",
            label="thomas-pod-391",
        ),
    ]

    captured: dict[str, list[Pod]] = {}
    monkeypatch.setattr(pod_lifecycle, "parse_pods_conf", lambda: rows)
    monkeypatch.setattr(pod_lifecycle, "write_pods_conf", lambda r: captured.setdefault("rows", r))
    monkeypatch.setattr(pod_lifecycle, "cmd_sync", lambda r: None)

    pod = EphemeralPod(
        metadata=_meta(
            "pod-391",
            issue=391,
            pod_id="new_pod_id",
            manual_override=True,
        ),
        info=_info(
            "pod-391",
            pod_id="stale_old_pod_id",
            ssh_host="9.9.9.9",  # would clobber the manual value
            ssh_port=22999,
            gpu_count=4,  # legitimate refresh — not via --update
        ),
    )
    _upsert_pods_conf(pod)

    out_rows = captured["rows"]
    assert len(out_rows) == 1
    # Manual values survive.
    assert out_rows[0].host == "31.24.80.42"
    assert out_rows[0].port == 10439
    # Non-overrideable fields still refresh.
    assert out_rows[0].gpus == 4
    assert out_rows[0].gpu_type == "H100"
    assert out_rows[0].label == "thomas-pod-391"

    err = capsys.readouterr().err
    assert "refusing to overwrite" in err
    assert "pod-391" in err


def test_upsert_pods_conf_no_warn_when_host_port_already_match(monkeypatch, capsys):
    """If manual_override=True but the live host/port happen to MATCH what
    pods.conf already has, no WARN should fire (nothing was clobbered).
    Avoids alarm noise during the steady-state case.
    """
    rows = [
        Pod(
            name="pod-391",
            host="31.24.80.42",
            port=10439,
            gpus=1,
            gpu_type="H100",
            label="thomas-pod-391",
        ),
    ]
    captured: dict[str, list[Pod]] = {}
    monkeypatch.setattr(pod_lifecycle, "parse_pods_conf", lambda: rows)
    monkeypatch.setattr(pod_lifecycle, "write_pods_conf", lambda r: captured.setdefault("rows", r))
    monkeypatch.setattr(pod_lifecycle, "cmd_sync", lambda r: None)

    pod = EphemeralPod(
        metadata=_meta("pod-391", issue=391, pod_id="x", manual_override=True),
        info=_info("pod-391", pod_id="x", ssh_host="31.24.80.42", ssh_port=10439),
    )
    _upsert_pods_conf(pod)

    assert capsys.readouterr().err == ""
    assert captured["rows"][0].host == "31.24.80.42"


def test_upsert_pods_conf_overwrites_without_override(monkeypatch):
    """Counterfactual: manual_override=False → existing behavior preserved
    (host/port overwritten from live info). Guards the fix against
    over-restricting the common case.
    """
    rows = [
        Pod(
            name="pod-5",
            host="0.0.0.0",
            port=1,
            gpus=1,
            gpu_type="H100",
            label="stale",
        ),
    ]
    captured: dict[str, list[Pod]] = {}
    monkeypatch.setattr(pod_lifecycle, "parse_pods_conf", lambda: rows)
    monkeypatch.setattr(pod_lifecycle, "write_pods_conf", lambda r: captured.setdefault("rows", r))
    monkeypatch.setattr(pod_lifecycle, "cmd_sync", lambda r: None)

    pod = EphemeralPod(
        metadata=_meta("pod-5", issue=5, pod_id="x", manual_override=False),
        info=_info("pod-5", pod_id="x", ssh_host="5.5.5.5", ssh_port=22001),
    )
    _upsert_pods_conf(pod)

    assert captured["rows"][0].host == "5.5.5.5"
    assert captured["rows"][0].port == 22001


# ---------------------------------------------------------------------------
# cmd_update + cmd_clear_override flip the flag end-to-end
# ---------------------------------------------------------------------------


def test_cmd_update_sets_manual_override_on_sidecar(isolated_state, monkeypatch, capsys):
    """End-to-end: cmd_update writes pods.conf AND flips manual_override=True
    in pods_ephemeral.json. The downstream cmd_sync is stubbed because
    ~/.ssh/config and ~/.claude/mcp.json are out of test scope.
    """
    # Seed sidecar with a registered ephemeral pod (manual_override=False).
    _write_metadata_file({"pod-391": _meta("pod-391", issue=391, pod_id="old_id")})
    # Stub write_pods_conf and cmd_sync to avoid touching real files.
    monkeypatch.setattr(pod_config, "write_pods_conf", lambda pods: None)
    monkeypatch.setattr(pod_config, "cmd_sync", lambda pods: None)

    rows = [
        Pod(
            name="pod-391",
            host="0.0.0.0",
            port=1,
            gpus=1,
            gpu_type="H100",
            label="thomas-pod-391",
        )
    ]
    pod_config.cmd_update(rows, "pod-391", host="31.24.80.42", port=10439)

    # In-memory row updated.
    assert rows[0].host == "31.24.80.42"
    assert rows[0].port == 10439
    # Sidecar flag flipped.
    on_disk = _read_metadata_file()
    assert on_disk["pod-391"].manual_override is True
    # User-facing breadcrumb.
    assert "manual_override for pod-391" in capsys.readouterr().out


def test_cmd_clear_override_round_trips(isolated_state, capsys):
    """cmd_clear_override flips manual_override back to False."""
    _write_metadata_file({"pod-391": _meta("pod-391", issue=391, pod_id="x", manual_override=True)})
    pod_config.cmd_clear_override("pod-391")
    assert _read_metadata_file()["pod-391"].manual_override is False
    assert "True -> False" in capsys.readouterr().out


def test_cmd_update_no_op_for_pod_not_in_sidecar(isolated_state, monkeypatch, capsys):
    """A permanent pod (e.g. ``pod1``) isn't in pods_ephemeral.json. cmd_update
    must succeed and NOT crash; the flag is silently skipped because there's
    no auto-refresh path to protect against for non-managed names.
    """
    # Sidecar exists but has no entry for the target name.
    _write_metadata_file({})
    monkeypatch.setattr(pod_config, "write_pods_conf", lambda pods: None)
    monkeypatch.setattr(pod_config, "cmd_sync", lambda pods: None)

    rows = [
        Pod(name="pod1", host="0.0.0.0", port=1, gpus=1, gpu_type="H100", label="perm"),
    ]
    pod_config.cmd_update(rows, "pod1", host="9.9.9.9", port=22)
    assert rows[0].host == "9.9.9.9"
    # No exception, and no "manual_override for pod1" status line printed.
    out = capsys.readouterr().out
    assert "manual_override for pod1" not in out


# ---------------------------------------------------------------------------
# Schema forward-compat: legacy sidecars without manual_override default False
# ---------------------------------------------------------------------------


def test_legacy_sidecar_without_manual_override_defaults_false(isolated_state, stub_list_team_pods):
    """A pods_ephemeral.json written by the pre-fix version of pod_lifecycle.py
    has no ``manual_override`` field. The new reader must default it to False
    so the existing drift-repair path keeps running. Without this, every old
    sidecar would be silently treated as overridden, and drift-repair would
    silently stop working everywhere.
    """
    legacy_blob = {
        "version": 2,
        "updated_at": "2026-05-20T00:00:00Z",
        "pods": {
            "pod-200": {
                "name": "pod-200",
                "pod_id": "stale",
                "issue": 200,
                "gpu_intent": "custom",
                "ttl_days": 7,
                "stopped_at": None,
                "notes": "",
                # NB: no manual_override key (legacy schema)
            }
        },
    }
    isolated_state.write_text(json.dumps(legacy_blob))
    stub_list_team_pods.return_value = [_info("pod-200", pod_id="live")]

    state = _load_state()
    # Drift-repair ran because the default is False.
    assert state["pod-200"].pod_id == "live"
    assert _read_metadata_file()["pod-200"].pod_id == "live"
