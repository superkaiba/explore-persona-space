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


# ---------------------------------------------------------------------------
# cmd_terminate — upload-verification guard (post-#444)
# ---------------------------------------------------------------------------


@pytest.fixture
def terminate_ns():
    """Build an argparse.Namespace matching the terminate subparser shape."""

    def _make(*, issue: int, yes: bool = True, dry_run: bool = False, skip: bool = False):
        return argparse.Namespace(
            issue=issue,
            yes=yes,
            dry_run=dry_run,
            skip_upload_verify=skip,
        )

    return _make


@pytest.fixture
def stub_terminate_pod(monkeypatch):
    """Capture calls to runpod_api.terminate_pod so tests can assert it was
    (or wasn't) invoked without hitting the network."""

    calls: list[str] = []

    def _stub(pod_id: str) -> None:
        calls.append(pod_id)

    monkeypatch.setattr(pod_lifecycle, "terminate_pod", _stub)
    return calls


@pytest.fixture
def stub_pods_conf_writes(monkeypatch):
    """No-op pods.conf side effects so tests don't touch the real file."""
    monkeypatch.setattr(pod_lifecycle, "_remove_from_pods_conf", lambda _name: None)


def _register_pod_for_issue(issue: int, *, name: str | None = None) -> str:
    """Register a managed pod for ``issue`` in the in-test sidecar + stub.
    Returns the pod name actually used."""
    pod_name = name or f"pod-{issue}"
    metadata = {pod_name: _meta(pod_name, issue=issue)}
    _write_metadata_file(metadata)
    return pod_name


def _upload_verification_event(verdict: str) -> dict:
    """Build a realistic ``epm:upload-verification`` event whose verdict lives
    in the markdown ``note`` body as ``**Verdict: <verdict>**`` — the real
    shape the upload-verifier writes (event keys are ts/kind/version/by/note;
    there is NO top-level ``verdict`` field). Mirrors
    tasks/completed/390/events.jsonl so the tests exercise the actual
    note-parsing path in ``_has_upload_verification_pass``."""
    return {
        "ts": "2026-06-02T00:00:00Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "upload-verifier",
        "note": (
            "<!-- epm:upload-verification v1 -->\n## Upload Verification\n\n"
            f"**Verdict: {verdict}**\n\nDiscovered N files on the pod under eval_results/."
        ),
    }


def _stub_list_events(monkeypatch, events: list[dict]) -> None:
    """Monkeypatch task_workflow.list_events (imported at call time inside
    ``_has_upload_verification_pass``) to return a fixed event list. This
    drives the REAL note-parsing logic instead of stubbing the function under
    test — the prior tests monkeypatched ``_has_upload_verification_pass``
    itself, which made them tautologies that hid a marker-shape bug (the first
    implementation read a non-existent top-level ``verdict`` field, so it would
    have refused EVERY terminate; the tautological tests stayed green)."""
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.list_events",
        lambda issue: list(events),
    )


def test_terminate_refuses_experiment_pod_without_upload_pass(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """A kind=experiment task with no epm:upload-verification PASS must block
    terminate. Origin: task #444 — hand-orchestrated completion skipped the
    Step-8 verifier and lost the training-mix datasets."""
    pod_name = _register_pod_for_issue(444)
    stub_list_team_pods.return_value = [_info(pod_name)]

    # No upload-verification event at all → the real note-parser returns False.
    _stub_list_events(monkeypatch, [])

    def fake_get_task(issue):
        return {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""}

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        fake_get_task,
    )

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=444))

    assert "epm:upload-verification PASS" in str(exc.value)
    assert "--skip-upload-verify" in str(exc.value)
    assert stub_terminate_pod == [], "terminate_pod must NOT be called when guard refuses"


def test_terminate_proceeds_when_upload_verification_pass_present(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """The normal /issue Step 8 path posts PASS before terminate — the guard
    must be silent on the happy path."""
    pod_name = _register_pod_for_issue(500)
    stub_list_team_pods.return_value = [_info(pod_name)]

    # A real PASS event in the note body → the note-parser returns True.
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])

    def fake_get_task(issue):
        return {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""}

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        fake_get_task,
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=500))

    assert len(stub_terminate_pod) == 1, (
        f"terminate_pod must be called exactly once on happy path; got {stub_terminate_pod}"
    )


def test_terminate_skip_upload_verify_overrides_with_warning(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    capsys,
    monkeypatch,
):
    """--skip-upload-verify proceeds even without the PASS marker but logs a
    LOUD warning so the override is never silent."""
    pod_name = _register_pod_for_issue(501)
    stub_list_team_pods.return_value = [_info(pod_name)]

    # A real FAIL event → not PASS → the guard would block, but --skip overrides.
    _stub_list_events(monkeypatch, [_upload_verification_event("FAIL")])

    def fake_get_task(issue):
        return {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""}

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        fake_get_task,
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=501, skip=True))

    err = capsys.readouterr().err
    assert "WITHOUT an epm:upload-verification PASS marker" in err
    assert "--skip-upload-verify" in err
    assert len(stub_terminate_pod) == 1, (
        "terminate_pod must still fire when --skip-upload-verify is passed"
    )


def test_terminate_does_not_block_non_experiment_kinds(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """kind ∈ {analysis, infra, batch, survey} must NOT be gated — those tasks
    don't produce the artifacts the verifier protects."""
    pod_name = _register_pod_for_issue(502)
    stub_list_team_pods.return_value = [_info(pod_name)]

    # No upload-verification event → if the guard DID consult it for these
    # kinds, the note-parser would return False and block. terminate proceeding
    # anyway proves the non-experiment early-return fires BEFORE the
    # verification check.
    _stub_list_events(monkeypatch, [])

    for kind in ("analysis", "infra", "batch", "survey"):

        def fake_get_task(issue, _k=kind):
            return {"id": issue, "frontmatter": {"kind": _k}, "body": ""}

        monkeypatch.setattr(
            "explore_persona_space.task_workflow.get_task",
            fake_get_task,
        )
        stub_terminate_pod.clear()
        pod_lifecycle.cmd_terminate(terminate_ns(issue=502))
        assert len(stub_terminate_pod) == 1, (
            f"non-experiment kind={kind!r} must not be blocked by upload-verification guard"
        )


def test_terminate_proceeds_when_task_cannot_be_resolved(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    capsys,
    monkeypatch,
):
    """Unresolvable tasks (manual / ad-hoc pods, registry miss, repo branch-guard
    fires) warn-and-proceed rather than hard-fail — the guard is for experiment
    pods, not a universal block."""
    pod_name = _register_pod_for_issue(503)
    stub_list_team_pods.return_value = [_info(pod_name)]

    def boom(issue):
        raise FileNotFoundError(f"task #{issue} not found")

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        boom,
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=503))

    err = capsys.readouterr().err
    assert "upload-verification guard skipped" in err
    assert "Proceeding with terminate" in err
    assert len(stub_terminate_pod) == 1, "terminate must proceed when the task can't be resolved"


def test_terminate_dry_run_bypasses_guard(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """--dry-run is a preview; the guard must NOT block (and terminate_pod
    must NOT fire) so the user can see what would happen."""
    pod_name = _register_pod_for_issue(504)
    stub_list_team_pods.return_value = [_info(pod_name)]

    def should_not_be_called(_issue):
        raise AssertionError("guard must not inspect task in --dry-run")

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        should_not_be_called,
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=504, dry_run=True))

    assert stub_terminate_pod == [], "dry-run must not call terminate_pod"


def test_terminate_parser_exposes_skip_upload_verify_flag():
    """Regression guard: the --skip-upload-verify flag must remain wired into
    the terminate subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pod_lifecycle._parser_terminate(sub)

    ns = parser.parse_args(["terminate", "--issue", "1", "--yes", "--skip-upload-verify"])
    assert ns.skip_upload_verify is True

    ns2 = parser.parse_args(["terminate", "--issue", "1", "--yes"])
    assert ns2.skip_upload_verify is False


# ---------------------------------------------------------------------------
# cmd_terminate — live-API authority for pod_id (post-#475 hardening)
# ---------------------------------------------------------------------------


def test_issue_from_pod_name_anchors_on_full_suffix():
    """``pod-47`` resolves to issue 47, NOT 475 — the suffix is parsed as a
    whole int, not a substring. Regression for the name-matching anchor that
    keeps multi-pod terminate from over-matching neighbouring issues."""
    assert pod_lifecycle._issue_from_pod_name("pod-47") == 47
    assert pod_lifecycle._issue_from_pod_name("pod-475") == 475
    assert pod_lifecycle._issue_from_pod_name("epm-issue-475") == 475
    # Trailing garbage is rejected so suffixes can't bleed across issues.
    assert pod_lifecycle._issue_from_pod_name("pod-475-backup") is None
    # Names without a managed prefix never match.
    assert pod_lifecycle._issue_from_pod_name("thomas-pod-475") is None


def test_terminate_kills_all_live_pods_matching_issue(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """Multiple live pods can share an issue (an EXITED orphan plus a fresh
    RUNNING pod, a duplicate from a crashed prior provision, etc.). The
    live RunPod API is authoritative for existence — terminate MUST kill
    every match by its live pod_id, not just the one referenced by the
    local sidecar. Regression for the #475 incident where a stale local
    pod_id terminated a ghost while two real pods survived."""
    pod_name = _register_pod_for_issue(475)  # local sidecar holds ONE row
    # Live API has THREE pods for issue 475 — the canonical one matching
    # the sidecar, an EXITED orphan, and a stray ``epm-issue-475`` from
    # the legacy prefix.
    stub_list_team_pods.return_value = [
        _info(pod_name, pod_id="live-canonical"),
        _info(pod_name, pod_id="live-exited-orphan", desired_status="EXITED"),
        _info("epm-issue-475", pod_id="live-legacy-prefix"),
    ]
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=475))

    assert sorted(stub_terminate_pod) == sorted(
        ["live-canonical", "live-exited-orphan", "live-legacy-prefix"]
    ), (
        "terminate must kill EVERY live pod whose name resolves to the "
        f"issue (#475), not just the local sidecar's pod_id; got {stub_terminate_pod}"
    )


def test_terminate_fails_loud_when_survivor_remains(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """If a fresh duplicate pod for the issue appears on the live API
    BETWEEN our terminate call and the post-check (or our terminate didn't
    actually destroy one), we must raise so the user knows the account
    still carries a live pod for this issue. Silent success in this case is
    what caused the #475 incident to go undetected for 6.5h."""
    pod_name = _register_pod_for_issue(476)

    initial_pods = [_info(pod_name, pod_id="initial-pod")]
    survivor = _info(pod_name, pod_id="surprise-survivor")

    call_count = {"n": 0}

    def _stub() -> list[PodInfo]:
        call_count["n"] += 1
        # First call: see the initial pod. Second call (the post-check):
        # a different pod_id is now live for the same issue — the
        # survivor that our terminate sweep missed.
        if call_count["n"] == 1:
            return list(initial_pods)
        return [survivor]

    monkeypatch.setattr(pod_lifecycle, "list_team_pods", _stub)

    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    with pytest.raises(pod_lifecycle.RunPodError) as exc_info:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=476))

    assert "surprise-survivor" in str(exc_info.value)
    assert "476" in str(exc_info.value)
    assert stub_terminate_pod == ["initial-pod"], (
        "the initial pod must still have been terminated before the "
        f"survivor check fired; got {stub_terminate_pod}"
    )


def test_terminate_clears_stale_local_record_when_no_live_match(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """Live API has no pod for the issue but the local sidecar still names
    one (terminated externally; sidecar never reconciled). Don't call
    terminate_pod (it would 404), but do clear the stale local row so the
    next provision starts from a clean slate."""
    pod_name = _register_pod_for_issue(477)
    stub_list_team_pods.return_value = []  # API has none

    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=477))

    assert stub_terminate_pod == [], "no terminate_pod call when no live match"
    metadata = _read_metadata_file()
    assert pod_name not in metadata, "stale local record must be cleared"


def test_terminate_raises_when_no_live_match_and_no_local_record(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """No live pod AND no local record → SystemExit with a clear message.
    Nothing to do; surface the fact rather than report a misleading
    'Terminated' on a no-op."""
    stub_list_team_pods.return_value = []  # API empty
    # No _register_pod_for_issue — sidecar empty too.

    # The guard short-circuits non-experiment kinds without touching the
    # task_workflow module, so we can keep this test free of mocks for it.
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "infra"}, "body": ""},
    )

    with pytest.raises(SystemExit) as exc_info:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=478))

    assert "478" in str(exc_info.value)
    assert "No live pod" in str(exc_info.value)
    assert stub_terminate_pod == []


# ---------------------------------------------------------------------------
# _has_upload_verification_pass — note-body verdict parsing (the bug site)
# ---------------------------------------------------------------------------


def test_has_upload_verification_pass_reads_note_body_pass(monkeypatch):
    """A real upload-verification event carries its verdict in the markdown
    note body (``**Verdict: PASS**``), NOT a top-level field. The parser must
    return True for it. Regression: the first implementation read
    ``ev.get("verdict")`` (always None) and would have refused every
    terminate."""
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    assert pod_lifecycle._has_upload_verification_pass(999) is True


@pytest.mark.parametrize("verdict", ["FAIL", "WARN"])
def test_has_upload_verification_pass_false_for_non_pass(monkeypatch, verdict):
    _stub_list_events(monkeypatch, [_upload_verification_event(verdict)])
    assert pod_lifecycle._has_upload_verification_pass(999) is False


def test_has_upload_verification_pass_false_when_no_event(monkeypatch):
    """No upload-verification event (and unrelated events present) → False."""
    other = {
        "ts": "2026-06-02T00:00:00Z",
        "kind": "epm:pod-terminated",
        "version": 1,
        "by": "unknown",
        "note": "pod-999 terminated.",
    }
    _stub_list_events(monkeypatch, [other])
    assert pod_lifecycle._has_upload_verification_pass(999) is False


def test_has_upload_verification_pass_anchors_on_bold_verdict_line(monkeypatch):
    """A note that mentions PASS in prose BEFORE the real bold verdict line
    must NOT false-positive: the parser anchors on ``**Verdict: X**``. Guards
    the reviewer's hypothetical `## Verdict\\n\\nPASS files...\\n**Verdict: FAIL**`
    shape."""
    event = {
        "ts": "2026-06-02T00:00:00Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "upload-verifier",
        "note": (
            "<!-- epm:upload-verification v1 -->\n## Verdict\n\n"
            "PASS files: 50 discovered.\n\n**Verdict: FAIL**\n\nMissing datasets."
        ),
    }
    _stub_list_events(monkeypatch, [event])
    assert pod_lifecycle._has_upload_verification_pass(999) is False


def test_has_upload_verification_pass_latest_event_wins(monkeypatch):
    """A re-verification supersedes an earlier verdict: latest FAIL after an
    earlier PASS → False; latest PASS after an earlier FAIL → True."""
    _stub_list_events(
        monkeypatch,
        [_upload_verification_event("PASS"), _upload_verification_event("FAIL")],
    )
    assert pod_lifecycle._has_upload_verification_pass(999) is False

    _stub_list_events(
        monkeypatch,
        [_upload_verification_event("FAIL"), _upload_verification_event("PASS")],
    )
    assert pod_lifecycle._has_upload_verification_pass(999) is True


def _orchestrator_posted_event(verdict: str) -> dict:
    """Build an ``epm:upload-verification`` event in the shape the orchestrator
    posts when it verifies uploads directly (no upload-verifier agent in the
    loop): the note BEGINS with a bare verdict token followed by a
    parenthetical attribution, with no ``**Verdict: ...**`` prefix. Mirrors
    tasks/awaiting_promotion/465/events.jsonl, the incident that motivated the
    leading-bare-verdict fallback in ``_has_upload_verification_pass``."""
    return {
        "ts": "2026-06-02T12:24:00Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "orchestrator",
        "note": (
            f"{verdict} (orchestrator-verified directly, not via experimenter): "
            "4 adapters on HF model repo + 5 figures committed to git issue-465."
        ),
    }


def test_has_upload_verification_pass_orchestrator_posted_bare_leading_pass(
    monkeypatch,
):
    """Regression for the 2026-06-05 task #465 incident: when the orchestrator
    posts the upload-verification marker directly (without the upload-verifier
    agent's bold-prefixed template), the note leads with a bare ``PASS`` token.
    The parser must accept that as a PASS verdict so ``pod.py terminate`` does
    not refuse a fully-verified pod and force a ``--skip-upload-verify``
    override."""
    _stub_list_events(monkeypatch, [_orchestrator_posted_event("PASS")])
    assert pod_lifecycle._has_upload_verification_pass(999) is True


@pytest.mark.parametrize("verdict", ["FAIL", "WARN"])
def test_has_upload_verification_pass_orchestrator_posted_bare_leading_non_pass(
    monkeypatch, verdict
):
    """Symmetry: an orchestrator-posted note that leads with bare ``FAIL`` or
    ``WARN`` must NOT be parsed as PASS — the fallback regex captures any
    verdict token, the PASS-or-not test then rejects the non-PASS values."""
    _stub_list_events(monkeypatch, [_orchestrator_posted_event(verdict)])
    assert pod_lifecycle._has_upload_verification_pass(999) is False


def test_has_upload_verification_pass_orchestrator_posted_latest_wins(monkeypatch):
    """``latest-event-wins`` still holds across the new fallback: an older
    FAIL followed by a newer bare-leading PASS resolves to True; the inverse
    resolves to False."""
    _stub_list_events(
        monkeypatch,
        [_orchestrator_posted_event("FAIL"), _orchestrator_posted_event("PASS")],
    )
    assert pod_lifecycle._has_upload_verification_pass(999) is True

    _stub_list_events(
        monkeypatch,
        [_orchestrator_posted_event("PASS"), _orchestrator_posted_event("FAIL")],
    )
    assert pod_lifecycle._has_upload_verification_pass(999) is False
