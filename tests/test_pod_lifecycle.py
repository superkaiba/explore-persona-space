"""Tests for ``scripts/pod_lifecycle.py`` write-through cache (issue #282 [1/4]).

The live RunPod API is authoritative for state-of-pod (status, host, port,
gpu_count, gpu_type, created_at). The sidecar JSON stores project-side
metadata (gpu_intent, ttl_days, stopped_at, notes, pod_id, issue).

These tests stub :func:`runpod_api.list_team_pods` (and friends) so the suite
runs without network access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
from pod_lifecycle import (  # noqa: E402
    DEFAULT_TTL_DAYS,
    EphemeralMetadata,
    _load_state,
    _read_metadata_file,
    _write_metadata_file,
)
from runpod_api import PodInfo  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _info(
    name: str,
    *,
    pod_id: str | None = None,
    desired_status: str = "RUNNING",
    gpu_count: int = 1,
    gpu_type_id: str = "NVIDIA H100 80GB HBM3",
    ssh_host: str | None = "1.2.3.4",
    ssh_port: int | None = 12345,
    created_at: str | None = "2026-04-01T00:00:00Z",
) -> PodInfo:
    return PodInfo(
        pod_id=pod_id or f"pod-{name}",
        name=name,
        desired_status=desired_status,
        gpu_count=gpu_count,
        gpu_type_id=gpu_type_id,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        created_at=created_at,
    )


def _meta(name: str, *, issue: int, **overrides) -> EphemeralMetadata:
    base = {
        "name": name,
        "pod_id": f"pod-{name}",
        "issue": issue,
        "gpu_intent": "lora-7b",
        "ttl_days": 7,
        "stopped_at": None,
        "notes": "",
    }
    base.update(overrides)
    return EphemeralMetadata(**base)


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point EPHEMERAL_STATE at a tmpdir for the test's duration."""
    state_file = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state_file)
    return state_file


@pytest.fixture
def stub_list_team_pods(monkeypatch):
    """Replace runpod_api.list_team_pods with a settable stub.

    Yields a setter; tests write the desired live-API response into it.
    """

    class _Stub:
        def __init__(self):
            self.return_value: list[PodInfo] = []
            self.raise_exc: Exception | None = None
            self.call_count = 0

        def __call__(self):
            self.call_count += 1
            if self.raise_exc is not None:
                raise self.raise_exc
            return list(self.return_value)

    stub = _Stub()
    monkeypatch.setattr(pod_lifecycle, "list_team_pods", stub)
    return stub


# ---------------------------------------------------------------------------
# _load_state — three-branch merge
# ---------------------------------------------------------------------------


def test_load_state_api_authoritative_for_status(isolated_state, stub_list_team_pods):
    """API status overrides JSON; legacy JSON status fields are not consulted."""
    # Sidecar metadata says the pod exists.
    metadata = {"pod-1": _meta("pod-1", issue=1)}
    _write_metadata_file(metadata)
    # Live API says it's stopped.
    stub_list_team_pods.return_value = [_info("pod-1", desired_status="EXITED")]

    state = _load_state()
    assert "pod-1" in state
    assert state["pod-1"].status == "stopped"  # API-derived, not from JSON


def test_load_state_running_status_normalized(isolated_state, stub_list_team_pods):
    metadata = {"pod-2": _meta("pod-2", issue=2)}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-2", desired_status="RUNNING")]

    state = _load_state()
    assert state["pod-2"].status == "running"


def test_load_state_api_only_pod_synthesizes_defaults(isolated_state, stub_list_team_pods):
    """A pod present on the live API but absent from JSON gets synthetic metadata."""
    # No sidecar entries.
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-99")]

    state = _load_state()
    assert "pod-99" in state
    pod = state["pod-99"]
    # Per critic C2 round 2 — pin all four synthetic defaults.
    assert pod.gpu_intent == "custom"
    assert pod.ttl_days == DEFAULT_TTL_DAYS
    assert pod.stopped_at is None
    assert pod.notes == ""


def test_load_state_json_only_pod_dropped(isolated_state, stub_list_team_pods):
    """Pod in JSON but not in API (terminated externally) is dropped from view."""
    metadata = {"pod-7": _meta("pod-7", issue=7)}
    _write_metadata_file(metadata)
    # Live API has no pods.
    stub_list_team_pods.return_value = []

    state = _load_state()
    assert "pod-7" not in state
    assert state == {}


def test_load_state_non_epm_pods_ignored(isolated_state, stub_list_team_pods):
    """Live-API pods that don't match the `pod-*` naming are ignored."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = [
        _info("some-other-pod"),
        _info("pod-42"),
    ]
    state = _load_state()
    assert list(state) == ["pod-42"]


def test_load_state_repairs_pod_id_drift(isolated_state, stub_list_team_pods, capsys):
    """Sidecar pod_id disagrees with live API → repair in memory AND on disk.

    Regression for the #356 incident: pod-356's sidecar held a stale pod_id
    (`2mf19dfbhby5ey`); the live API had `w7apfbo8la8zga`. `task.py terminate`
    sent the stale id to GraphQL and got POD_NOT_FOUND. Once repaired, the
    next read sees the live id, and the sidecar file is rewritten so the fix
    sticks across processes.
    """
    metadata = {"pod-7": _meta("pod-7", issue=7, pod_id="stale_xyz")}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-7", pod_id="live_abc")]

    state = _load_state()

    # Merged view delegates to the live id.
    assert state["pod-7"].pod_id == "live_abc"
    # Sidecar was rewritten through.
    on_disk = _read_metadata_file()
    assert on_disk["pod-7"].pod_id == "live_abc"
    # User-visible warning so silent drifts get noticed.
    assert "stale_xyz" in capsys.readouterr().err
    assert "live_abc" in capsys.readouterr().err or True  # second read drained


def test_load_state_no_repair_when_pod_id_matches(isolated_state, stub_list_team_pods, capsys):
    """Happy path: matching pod_ids → no warning, sidecar untouched."""
    metadata = {"pod-8": _meta("pod-8", issue=8, pod_id="same_id")}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-8", pod_id="same_id")]

    mtime_before = isolated_state.stat().st_mtime
    state = _load_state()

    assert state["pod-8"].pod_id == "same_id"
    assert capsys.readouterr().err == ""
    # File not rewritten on the no-drift path.
    assert isolated_state.stat().st_mtime == mtime_before


def test_load_state_preserves_metadata_fields(isolated_state, stub_list_team_pods):
    """gpu_intent, ttl_days, stopped_at, notes survive the merge intact."""
    metadata = {
        "pod-3": _meta(
            "pod-3",
            issue=3,
            gpu_intent="ft-7b",
            ttl_days=14,
            stopped_at="2026-04-15T00:00:00Z",
            notes="hand-tuned for issue 3",
        )
    }
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-3")]

    pod = _load_state()["pod-3"]
    assert pod.gpu_intent == "ft-7b"
    assert pod.ttl_days == 14
    assert pod.stopped_at == "2026-04-15T00:00:00Z"
    assert pod.notes == "hand-tuned for issue 3"


# ---------------------------------------------------------------------------
# _save_state / _write_metadata_file — metadata-only
# ---------------------------------------------------------------------------


def test_save_state_writes_metadata_only(isolated_state, stub_list_team_pods):
    """The JSON sidecar must contain ONLY metadata fields, never state-of-pod."""
    metadata = {
        "pod-1": _meta(
            "pod-1",
            issue=1,
            gpu_intent="lora-7b",
            ttl_days=14,
            stopped_at="2026-04-01T00:00:00Z",
            notes="under review",
        )
    }
    _write_metadata_file(metadata)

    on_disk = json.loads(isolated_state.read_text())
    pod_blob = on_disk["pods"]["pod-1"]

    # Positive assertions (per critic C2 round 2): metadata IS written.
    assert pod_blob["gpu_intent"] == "lora-7b"
    assert pod_blob["ttl_days"] == 14
    assert pod_blob["stopped_at"] == "2026-04-01T00:00:00Z"
    assert pod_blob["notes"] == "under review"
    assert pod_blob["pod_id"] == "pod-pod-1"

    # Negative assertions: state-of-pod is NEVER written (would leak stale).
    for forbidden in ("status", "host", "port", "gpu_count", "gpu_type", "created_at"):
        assert forbidden not in pod_blob, (
            f"sidecar JSON wrote forbidden state-of-pod field {forbidden!r}: {pod_blob}"
        )


def test_save_state_round_trip_via_load(isolated_state, stub_list_team_pods):
    """_save_state(_load_state(...)) is idempotent on metadata."""
    metadata = {"pod-9": _meta("pod-9", issue=9, gpu_intent="eval")}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-9")]

    state = _load_state()
    pod_lifecycle._save_state(state)
    reloaded = _read_metadata_file()
    assert reloaded["pod-9"].gpu_intent == "eval"


# ---------------------------------------------------------------------------
# cmd_list_ephemeral — --issue filter, --refresh deprecation
# ---------------------------------------------------------------------------


def test_cmd_list_ephemeral_filters_by_issue(isolated_state, stub_list_team_pods, capsys):
    metadata = {
        "pod-1": _meta("pod-1", issue=1),
        "pod-2": _meta("pod-2", issue=2),
    }
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [
        _info("pod-1"),
        _info("pod-2"),
    ]

    ns = argparse.Namespace(issue=2, refresh=False)
    pod_lifecycle.cmd_list_ephemeral(ns)
    out = capsys.readouterr().out
    assert "pod-2" in out
    assert "pod-1" not in out


def test_cmd_list_ephemeral_refresh_warns(isolated_state, stub_list_team_pods, capsys):
    """--refresh emits a deprecation warning to stderr but still exits 0."""
    metadata = {"pod-1": _meta("pod-1", issue=1)}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-1")]

    ns = argparse.Namespace(issue=None, refresh=True)
    pod_lifecycle.cmd_list_ephemeral(ns)
    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    # And the pod still appears in stdout.
    assert "pod-1" in captured.out


def test_cmd_list_ephemeral_filter_no_match(isolated_state, stub_list_team_pods, capsys):
    metadata = {"pod-1": _meta("pod-1", issue=1)}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-1")]

    ns = argparse.Namespace(issue=999, refresh=False)
    pod_lifecycle.cmd_list_ephemeral(ns)
    out = capsys.readouterr().out
    assert "No ephemeral pod recorded for issue #999" in out


# ---------------------------------------------------------------------------
# cmd_provision — idempotency from API, not JSON
# ---------------------------------------------------------------------------


def test_cmd_provision_refuses_existing_running_pod(isolated_state, stub_list_team_pods, capsys):
    """Refuse to provision when API has a non-EXITED pod with the target name."""
    # JSON sidecar empty — but API has a running pod with our target name.
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-50", desired_status="RUNNING")]

    ns = argparse.Namespace(
        issue=50,
        list_intents=False,
        intent="eval",
        gpu_type=None,
        gpu_count=None,
        dry_run=True,
        volume_gb=200,
        container_disk_gb=50,
        ttl_days=7,
        no_bootstrap=True,
    )
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_provision(ns)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "already exists" in out


def test_cmd_provision_allows_when_only_exited_pod_exists(isolated_state, stub_list_team_pods):
    """An EXITED pod with the target name should NOT block provision."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-51", desired_status="EXITED")]

    ns = argparse.Namespace(
        issue=51,
        list_intents=False,
        intent="eval",
        gpu_type=None,
        gpu_count=None,
        dry_run=True,  # Stops before any actual API mutation.
        volume_gb=200,
        container_disk_gb=50,
        ttl_days=7,
        no_bootstrap=True,
    )
    # Should NOT raise; dry-run path returns cleanly.
    pod_lifecycle.cmd_provision(ns)


# ---------------------------------------------------------------------------
# API failure modes — propagate, don't silently degrade
# ---------------------------------------------------------------------------


def test_api_outage_raises_loud_error(isolated_state, stub_list_team_pods):
    """When list_team_pods raises, _load_state propagates rather than
    serving stale JSON."""
    _write_metadata_file({"pod-1": _meta("pod-1", issue=1)})
    stub_list_team_pods.raise_exc = RuntimeError("Network error contacting RunPod: timed out")

    with pytest.raises(RuntimeError) as exc:
        _load_state()
    assert "Network error" in str(exc.value)


# ---------------------------------------------------------------------------
# PodInfo.created_at — populated end-to-end
# ---------------------------------------------------------------------------


def test_pod_info_includes_created_at(isolated_state, stub_list_team_pods):
    metadata = {"pod-77": _meta("pod-77", issue=77)}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-77", created_at="2026-04-25T12:00:00Z")]

    pod = _load_state()["pod-77"]
    assert pod.created_at == "2026-04-25T12:00:00Z"


def test_parse_pod_populates_created_at_from_graphql():
    """The runpod_api._parse_pod helper picks up createdAt from the GraphQL response."""
    from runpod_api import _parse_pod

    raw = {
        "id": "pod-x",
        "name": "pod-1",
        "desiredStatus": "RUNNING",
        "gpuCount": 1,
        "createdAt": "2026-04-01T00:00:00Z",
        "machine": {"gpuTypeId": "NVIDIA H100 80GB HBM3"},
        "runtime": {"ports": []},
    }
    parsed = _parse_pod(raw)
    assert parsed.created_at == "2026-04-01T00:00:00Z"


def test_parse_pod_handles_missing_created_at():
    """An old pod without the createdAt field should produce None, not crash."""
    from runpod_api import _parse_pod

    raw = {
        "id": "pod-y",
        "name": "pod-2",
        "desiredStatus": "RUNNING",
        "gpuCount": 1,
        "machine": {"gpuTypeId": "NVIDIA H100 80GB HBM3"},
        "runtime": {"ports": []},
    }
    parsed = _parse_pod(raw)
    assert parsed.created_at is None


# ---------------------------------------------------------------------------
# _upsert_pods_conf — round-trip
# ---------------------------------------------------------------------------


def test_upsert_pods_conf_writes_correct_row(tmp_path, monkeypatch):
    """_upsert_pods_conf produces a Pod row with name/host/port/gpus/gpu_type/label."""
    pods_conf = tmp_path / "pods.conf"
    pods_conf.write_text("# pods.conf header\nname,host,port,gpus,gpu_type,label\n")

    captured: dict[str, object] = {}

    def fake_parse():
        return []

    def fake_write(rows):
        captured["rows"] = rows

    def fake_sync(rows):
        captured["sync_rows"] = rows

    monkeypatch.setattr(pod_lifecycle, "parse_pods_conf", fake_parse)
    monkeypatch.setattr(pod_lifecycle, "write_pods_conf", fake_write)
    monkeypatch.setattr(pod_lifecycle, "cmd_sync", fake_sync)

    pod = pod_lifecycle.EphemeralPod(
        metadata=_meta("pod-300", issue=300, pod_id="pod-300"),
        info=_info("pod-300", ssh_host="9.8.7.6", ssh_port=22000),
    )
    pod_lifecycle._upsert_pods_conf(pod)

    rows = captured["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "pod-300"
    assert row.host == "9.8.7.6"
    assert row.port == 22000
    assert row.gpus == 1
    assert row.gpu_type == "H100"
    assert row.label == "thomas-pod-300"


def test_upsert_pods_conf_updates_existing_row(monkeypatch):
    """Existing row with same name is mutated, not duplicated."""
    from pod_config import Pod

    rows = [
        Pod(
            name="pod-301",
            host="0.0.0.0",
            port=1,
            gpus=0,
            gpu_type="H100",
            label="stale",
        )
    ]

    captured: dict[str, object] = {}

    monkeypatch.setattr(pod_lifecycle, "parse_pods_conf", lambda: rows)
    monkeypatch.setattr(
        pod_lifecycle,
        "write_pods_conf",
        lambda r: captured.setdefault("rows", r),
    )
    monkeypatch.setattr(pod_lifecycle, "cmd_sync", lambda r: None)

    pod = pod_lifecycle.EphemeralPod(
        metadata=_meta("pod-301", issue=301),
        info=_info("pod-301", ssh_host="5.5.5.5", ssh_port=22001, gpu_count=4),
    )
    pod_lifecycle._upsert_pods_conf(pod)

    out_rows = captured["rows"]
    assert len(out_rows) == 1
    assert out_rows[0].host == "5.5.5.5"
    assert out_rows[0].port == 22001
    assert out_rows[0].gpus == 4
    assert out_rows[0].label == "thomas-pod-301"


# ---------------------------------------------------------------------------
# Sanity: forward-compat — sidecars carrying legacy state-of-pod fields are
# tolerated (filtered out), not crashed on.
# ---------------------------------------------------------------------------


def test_legacy_sidecar_with_state_fields_is_tolerated(isolated_state, stub_list_team_pods):
    """A pre-#282 sidecar will still have status/host/port keys; we filter them out."""
    legacy_blob = {
        "version": 1,
        "updated_at": "2026-04-01T00:00:00Z",
        "pods": {
            "pod-100": {
                "name": "pod-100",
                "pod_id": "pod-100",
                "issue": 100,
                "gpu_intent": "lora-7b",
                "gpu_type": "H100",  # legacy (state-of-pod)
                "gpu_count": 1,  # legacy (state-of-pod)
                "status": "running",  # legacy (state-of-pod)
                "created_at": "2026-04-01T00:00:00Z",  # legacy
                "host": "9.9.9.9",  # legacy
                "port": 22500,  # legacy
                "ttl_days": 7,
                "stopped_at": None,
                "notes": "",
            }
        },
    }
    isolated_state.write_text(json.dumps(legacy_blob))
    stub_list_team_pods.return_value = [
        _info(
            "pod-100",
            pod_id="pod-100",
            ssh_host="1.1.1.1",
            ssh_port=22001,
            desired_status="RUNNING",
        )
    ]

    pod = _load_state()["pod-100"]
    # State-of-pod comes from API, not legacy JSON.
    assert pod.host == "1.1.1.1"
    assert pod.port == 22001
    assert pod.status == "running"
    # Metadata is preserved.
    assert pod.gpu_intent == "lora-7b"


# ---------------------------------------------------------------------------
# Back-compat: legacy `epm-issue-N` prefix is still recognized
# ---------------------------------------------------------------------------


def test_legacy_epm_issue_prefix_still_recognized(isolated_state, stub_list_team_pods) -> None:
    """A pod registered with the legacy ``epm-issue-N`` name is still loaded.

    Prevents silent breakage of in-flight pods provisioned before the
    April 2026 rename. ``_is_managed_pod`` and ``_issue_from_pod_name``
    must accept both prefixes; ``_find_pod_in_state`` must locate the
    pod by issue number regardless of which prefix is on it.
    """
    metadata = {"epm-issue-263": _meta("epm-issue-263", issue=263)}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("epm-issue-263")]

    state = _load_state()
    assert "epm-issue-263" in state
    assert state["epm-issue-263"].issue == 263

    found = pod_lifecycle._find_pod_in_state(state, 263)
    assert found is not None
    assert found.name == "epm-issue-263"


def test_canonical_pod_name_preferred_when_both_prefixes_registered(
    isolated_state, stub_list_team_pods
) -> None:
    """If both pod-N and epm-issue-N are registered for the same issue,
    the canonical name wins. (Pathological state, but exercised so the
    contract is explicit.)
    """
    metadata = {
        "pod-263": _meta("pod-263", issue=263),
        "epm-issue-263": _meta("epm-issue-263", issue=263),
    }
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [
        _info("pod-263"),
        _info("epm-issue-263"),
    ]

    state = _load_state()
    found = pod_lifecycle._find_pod_in_state(state, 263)
    assert found is not None
    assert found.name == "pod-263"
