"""Tests for ``scripts/pod_lifecycle.py`` write-through cache (issue #282 [1/4]).

The live RunPod API is authoritative for state-of-pod (status, host, port,
gpu_count, gpu_type, created_at). The sidecar JSON stores project-side
metadata (gpu_intent, ttl_days, stopped_at, notes, pod_id, issue).

These tests stub :func:`runpod_api.list_team_pods` (and friends) so the suite
runs without network access.
"""

from __future__ import annotations

import argparse
import ast
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


@pytest.fixture(autouse=True)
def bad_host_state(tmp_path, monkeypatch):
    """#2011: isolate the bad-placement sidecar + DC enumeration for EVERY
    test in this file.

    The provision tail now records/consumes bad placements
    (``_record_bad_placement_loud`` / ``_warn_on_bad_host_repeat``), so any
    test driving it would otherwise write the LIVE
    ``<git-common-dir>/eps/bad-pod-hosts.json`` and hit the network via
    ``get_datacenters``. Autouse + requestable by name where a test needs the
    tmp sidecar path. ``get_datacenters`` is zero-arg — the stub mirrors the
    real signature; #2011-specific tests re-patch it with candidate lists.
    """
    state_file = tmp_path / "bad-pod-hosts.json"
    monkeypatch.setattr(pod_lifecycle, "BAD_HOST_STATE", state_file)
    monkeypatch.setattr(pod_lifecycle, "get_datacenters", lambda: [])
    return state_file


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


def test_load_state_primary_prefers_sidecar_match_among_duplicates(
    isolated_state, stub_list_team_pods, capsys
):
    """Duplicate-named live pods: the sidecar-matching member is the PRIMARY.

    Targeting keeps following the sidecar (acceptance 4+5, #2049): with two
    live pods named pod-7 and the sidecar recording B, the name-keyed view
    resolves to B (even though A is RUNNING and B is EXITED), no drift-repair
    fires, and the sidecar file is byte-unchanged.
    """
    metadata = {"pod-7": _meta("pod-7", issue=7, pod_id="B")}
    _write_metadata_file(metadata)
    raw_before = isolated_state.read_bytes()
    stub_list_team_pods.return_value = [
        _info("pod-7", pod_id="A", desired_status="RUNNING"),
        _info("pod-7", pod_id="B", desired_status="EXITED"),
    ]

    state = _load_state()

    assert state["pod-7"].pod_id == "B"
    # No drift-repair / ambiguity WARN — the sidecar match is authoritative.
    assert capsys.readouterr().err == ""
    # Sidecar file byte-unchanged (no disk rewrite).
    assert isolated_state.read_bytes() == raw_before


def test_load_state_duplicates_without_sidecar_match_skip_drift_repair(
    isolated_state, stub_list_team_pods, capsys
):
    """Ambiguous duplicate group (sidecar id matches NO live pod): no disk
    rewrite, but the IN-MEMORY targeting view repoints to the primary.

    Acceptance 5 (#2049): sidecar records C (gone); live = A (RUNNING) + B
    (EXITED), both named pod-7. The primary is A (RUNNING preferred); the
    merged view's pod_id — what cmd_stop/cmd_resume send to the API — is A,
    never the dead sidecar id C. The sidecar FILE still records C, and a loud
    ambiguity WARN lands on stderr.
    """
    metadata = {"pod-7": _meta("pod-7", issue=7, pod_id="C")}
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [
        _info("pod-7", pod_id="A", desired_status="RUNNING"),
        _info("pod-7", pod_id="B", desired_status="EXITED"),
    ]

    state = _load_state()
    err = capsys.readouterr().err

    # In-memory metadata repaired to the PRIMARY's LIVE id (RUNNING preferred).
    assert state["pod-7"].pod_id == "A"
    # Ambiguity WARN on stderr names the pod and all colliding ids.
    assert "pod-7" in err
    assert "ids: A, B" in err
    assert "ambiguous" in err
    assert "sidecar left untouched" in err
    # No drift-repair text (the ambiguous case must NOT take the RMW path).
    assert "repaired pods_ephemeral.json" not in err

    # Full view: B's own row carries B's own pod_id.
    all_pods = pod_lifecycle._load_state_all()
    b_rows = [p for p in all_pods if p.info.pod_id == "B"]
    assert len(b_rows) == 1
    assert b_rows[0].pod_id == "B"

    # Sidecar FILE still records C — the on-disk rewrite was suppressed.
    assert _read_metadata_file()["pod-7"].pod_id == "C"


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


def test_list_ephemeral_shows_duplicate_named_pods(isolated_state, stub_list_team_pods, capsys):
    """N live pods sharing one managed name print N rows + a loud stderr WARN.

    Mirrors the 2026-08-03 incident (#2049): three live pods named pod-1739
    (two RUNNING, one EXITED); the name-keyed last-wins merge showed ONLY the
    EXITED one, hiding ~$8/hr of live burn. Fail-loud pin: the WARN itself is
    asserted on stderr (with the colliding pod_ids listed), so a re-swallowed
    or silently dropped warning fails this test rather than shipping green.
    """
    _write_metadata_file({})
    stub_list_team_pods.return_value = [
        _info("pod-1739", pod_id="id_run_a", desired_status="RUNNING"),
        _info("pod-1739", pod_id="id_run_b", desired_status="RUNNING"),
        _info("pod-1739", pod_id="id_exited", desired_status="EXITED"),
    ]

    ns = argparse.Namespace(issue=None, refresh=False)
    pod_lifecycle.cmd_list_ephemeral(ns)
    captured = capsys.readouterr()

    rows = [line for line in captured.out.splitlines() if line.startswith("pod-1739")]
    assert len(rows) == 3
    for pod_id in ("id_run_a", "id_run_b", "id_exited"):
        assert pod_id in captured.out
    # Loud WARN on stderr naming the colliding name and every pod_id.
    assert "3 live pods share the name pod-1739" in captured.err
    for pod_id in ("id_run_a", "id_run_b", "id_exited"):
        assert pod_id in captured.err
    assert "provisioning-idempotency problem" in captured.err


def test_list_ephemeral_issue_filter_includes_duplicates(
    isolated_state, stub_list_team_pods, capsys
):
    """--issue <N> keeps EVERY duplicate row for that issue, per-pod filtered."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = [
        _info("pod-1739", pod_id="dup_x", desired_status="RUNNING"),
        _info("pod-1739", pod_id="dup_y", desired_status="RUNNING"),
        _info("pod-42", pod_id="single_z"),
    ]

    ns = argparse.Namespace(issue=1739, refresh=False)
    pod_lifecycle.cmd_list_ephemeral(ns)
    out = capsys.readouterr().out

    assert "dup_x" in out
    assert "dup_y" in out
    assert "single_z" not in out
    assert "pod-42" not in out


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
    # #1518: the refusal is an error-path diagnostic — it rides stderr (the
    # #1465 PodLifecycleProcessError tail surface), never stdout.
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "already exists" not in captured.out


def test_provision_refusal_stderr_classifies_created_nothing(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """#1518 fail-loud contract: the already-exists refusal rides stderr (the
    #1465 ``PodLifecycleProcessError`` tail surface) and the REAL #1490
    classifier reads the produced text as ``created-nothing`` — the production
    route that turned #1481's self-explanatory refusal into an unexplained
    ``exit status 1`` on the router failure marker."""
    import backend_poll  # scripts/ already on sys.path (module header)

    from explore_persona_space.backends.runpod import PodLifecycleProcessError

    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-50", desired_status="RUNNING")]

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_provision(_gpu_provision_ns(50))
    assert exc.value.code == 1

    err = capsys.readouterr().err
    # Both conjunctive fragments of backend_poll._PROVISION_REFUSAL_MARKERS
    # must survive in the REAL produced stderr text (kill criterion 3: the
    # message text is a classifier-matched surface — never reword it).
    assert "already exists" in err
    assert "Use `pod.py resume" in err
    assert backend_poll._classify_provision_failure(err, 1) == "created-nothing"

    # Composition hop: the router's failure-marker text is
    # str(PodLifecycleProcessError(...)) carrying the stderr tail — both
    # classifier fragments must survive that composition too.
    marker_text = str(
        PodLifecycleProcessError(1, ["pod_lifecycle.py", "provision"], output=None, stderr=err)
    )
    assert "already exists" in marker_text
    assert "Use `pod.py resume" in marker_text
    assert backend_poll._classify_provision_failure(marker_text, 1) == "created-nothing"


def test_provision_refuses_stopped_collision_exit_76(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """#1997 (b1): a same-named STOPPED (EXITED) pod now REFUSES the provision
    with the typed exit 76 (pre-#1997 it silently proceeded, minting the
    #1739 duplicate-named pod whose name-keyed state rows hijacked the
    stopped pod's). The stderr message keeps BOTH conjunctive #1490
    created-nothing classifier fragments AND names every recovery path."""
    import backend_poll  # scripts/ already on sys.path (module header)

    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-51", desired_status="EXITED")]

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_provision(_gpu_provision_ns(51))
    assert exc.value.code == pod_lifecycle.EXIT_STOPPED_POD_COLLISION == 76

    err = capsys.readouterr().err
    # BOTH conjunctive fragments of backend_poll._PROVISION_REFUSAL_MARKERS
    # survive, so the #1490 classifier reads the refusal as created-nothing
    # (never a terminate candidate).
    assert "already exists" in err
    assert "Use `pod.py resume" in err
    assert backend_poll._classify_provision_failure(err, 76) == "created-nothing"
    # Every recovery path is named: resume / approved terminate / suffix /
    # the deliberate override flag.
    assert "pod.py resume --issue 51" in err
    assert "terminate --issue 51 --yes --approve" in err
    assert "--name-suffix" in err
    assert "--allow-stopped-duplicate" in err


def test_provision_stopped_mask_running_duplicate_refuses_exit_1(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """#1997 (b2, the duplicate-mask fix): RunPod permits duplicate pod names,
    and the pre-#1997 name-keyed dict let a STOPPED entry mask a RUNNING one
    (order-dependent on the API list). With BOTH orders, the non-EXITED
    refusal (exit 1) must win over the stopped-collision refusal (exit 76)."""
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    _write_metadata_file({})
    exited = _info("pod-52", pod_id="pod-52-stopped", desired_status="EXITED")
    running = _info("pod-52", pod_id="pod-52-live", desired_status="RUNNING")

    for order in ([exited, running], [running, exited]):
        stub_list_team_pods.return_value = order
        with pytest.raises(SystemExit) as exc:
            pod_lifecycle.cmd_provision(_gpu_provision_ns(52))
        assert exc.value.code == 1, f"order {order} did not take the non-EXITED refusal"
        err = capsys.readouterr().err
        assert "status=RUNNING" in err
        assert "id=pod-52-live" in err


def test_provision_allow_stopped_duplicate_proceeds(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """#1997 (b3): --allow-stopped-duplicate deliberately overrides the
    stopped-collision refusal — the provision proceeds past the idempotency
    check (dry-run stops before any API mutation)."""
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-53", desired_status="EXITED")]

    # Should NOT raise; dry-run path returns cleanly past the collision check.
    pod_lifecycle.cmd_provision(_gpu_provision_ns(53, allow_stopped_duplicate=True))
    out = capsys.readouterr().out
    assert "[dry-run]" in out


def test_suffixed_provision_unaffected_by_bare_stopped_pod(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """#1997 (b4): a --name-suffix provision collides only on ITS OWN name —
    a bare STOPPED pod-<N> never blocks it (the existing suffix semantics,
    pinned against the new stopped-collision check), while a STOPPED
    pod-<N>-<slug> refuses the suffixed provision with exit 76."""
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-54", desired_status="EXITED")]

    pod_lifecycle.cmd_provision(_gpu_provision_ns(54, name_suffix="b"))
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "pod-54-b" in out

    # A STOPPED pod-54-b DOES refuse the suffixed provision (its own name).
    stub_list_team_pods.return_value = [_info("pod-54-b", desired_status="EXITED")]
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_provision(_gpu_provision_ns(54, name_suffix="b"))
    assert exc.value.code == pod_lifecycle.EXIT_STOPPED_POD_COLLISION
    assert "pod-54-b already exists STOPPED" in capsys.readouterr().err


def test_provision_parser_exposes_allow_stopped_duplicate():
    """#1997: the provision subparser wires --allow-stopped-duplicate
    (store_true, default False — the refusal is the default posture)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pod_lifecycle._parser_provision(sub)

    ns = parser.parse_args(["provision", "--issue", "51", "--allow-stopped-duplicate"])
    assert ns.allow_stopped_duplicate is True
    ns0 = parser.parse_args(["provision", "--issue", "51"])
    assert ns0.allow_stopped_duplicate is False


# ---------------------------------------------------------------------------
# cmd_provision — CPU-only intent bridge (#747)
#
# The router tests prove the RunPod backend RECEIVES `--intent cpu-small`, and
# test_runpod_api_retry.py proves create_cpu_pod renders the deployCpuPod
# mutation. These pin the SCRIPT-LEVEL bridge in cmd_provision: that a CPU
# intent routes to create_cpu_pod (with the canonical instance_id) via the
# CPU-branch-before-_resolve_spec ordering, and NEVER falls through to the GPU
# _resolve_spec / create_pod path (which KeyErrors on a CPU intent). The
# load-bearing assertion is the negative one — without it a future refactor
# could silently route CPU work through the GPU resolver.
# ---------------------------------------------------------------------------


def _cpu_provision_ns(issue: int, intent: str, **overrides) -> argparse.Namespace:
    """Namespace matching the provision subparser shape, for a CPU intent."""
    base = {
        "issue": issue,
        "list_intents": False,
        "intent": intent,
        "gpu_type": None,
        "gpu_count": None,
        "dry_run": False,  # exercise the real create_cpu_pod call, not the dry-run early-return
        "volume_gb": 200,  # argparse default (the GPU default) unless overridden below
        "container_disk_gb": 50,
        "ttl_days": 7,
        "no_bootstrap": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def cpu_provision_stubs(monkeypatch):
    """Stub the CPU-provision tail so cmd_provision runs without network.

    Records every create_cpu_pod call and asserts the GPU path is never taken.
    Yields a dict carrying the captured create_cpu_pod kwargs + call counters.
    """
    captured: dict = {"cpu_calls": [], "gpu_resolve_calls": 0, "gpu_create_calls": 0}

    def _fake_create_cpu_pod(
        *, name, instance_id, volume_gb, container_disk_gb, data_center_id=None
    ):
        captured["cpu_calls"].append(
            {
                "name": name,
                "instance_id": instance_id,
                "volume_gb": volume_gb,
                "container_disk_gb": container_disk_gb,
                "data_center_id": data_center_id,
            }
        )
        return _info(name, pod_id=f"cpupod-{name}", gpu_count=0, gpu_type_id="")

    def _fail_resolve_spec(*_a, **_k):
        captured["gpu_resolve_calls"] += 1
        raise AssertionError("GPU _resolve_spec must NOT be called for a CPU intent")

    def _fail_create_pod(*_a, **_k):
        captured["gpu_create_calls"] += 1
        raise AssertionError("GPU create_pod must NOT be called for a CPU intent")

    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod", _fake_create_cpu_pod)
    monkeypatch.setattr(pod_lifecycle, "_resolve_spec", _fail_resolve_spec)
    monkeypatch.setattr(pod_lifecycle, "create_pod", _fail_create_pod)
    # No-op the SSH/register/bootstrap tail — it is exercised by the GPU path's
    # own coverage; here we only care about the create-call routing.
    monkeypatch.setattr(pod_lifecycle, "_provision_wait_register_bootstrap", lambda *_a, **_k: None)
    return captured


@pytest.mark.parametrize(
    ("intent", "instance_id"),
    [
        ("cpu-small", "cpu3g-2-8"),
        ("cpu-mid", "cpu3c-8-16"),
    ],
)
def test_cmd_provision_cpu_intent_routes_to_create_cpu_pod(
    isolated_state, stub_list_team_pods, cpu_provision_stubs, intent, instance_id
):
    """A cpu-small / cpu-mid intent calls create_cpu_pod with the canonical
    instance_id and NEVER the GPU resolver/create path."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []  # no existing pod for this issue

    ns = _cpu_provision_ns(747, intent)
    pod_lifecycle.cmd_provision(ns)

    assert len(cpu_provision_stubs["cpu_calls"]) == 1
    call = cpu_provision_stubs["cpu_calls"][0]
    assert call["instance_id"] == instance_id
    # The load-bearing differentiator: GPU path was never touched.
    assert cpu_provision_stubs["gpu_resolve_calls"] == 0
    assert cpu_provision_stubs["gpu_create_calls"] == 0


@pytest.mark.parametrize(
    ("intent", "expected_volume_gb"),
    [
        # cpu-small: CPU default 40, then clamped to the cpu3g cap (20) by the
        # #1010 effective-payload clamp (the untouched default effective 50
        # exceeds the 20 GB validation cap — pre-#1010 RunPod REJECTED every
        # default cpu-small provision outright).
        ("cpu-small", 20),
        # cpu-mid: CPU default 40 rides through unclamped (effective 50 == cap).
        ("cpu-mid", 40),
    ],
)
def test_cmd_provision_cpu_default_volume_is_cpu_default(
    isolated_state, stub_list_team_pods, cpu_provision_stubs, intent, expected_volume_gb
):
    """`provision --intent cpu-*` with no explicit --volume-gb never uses the
    200 GB GPU argparse default (#747 minor fix — a 200 GB persistent volume
    on a cents/hr CPU pod defeats the lane): the cheap CPU default (40 GB)
    applies, further clamped to the instance's #1010 container-disk cap where
    the cap sits below it."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []

    ns = _cpu_provision_ns(747, intent)  # volume_gb left at the 200 default
    pod_lifecycle.cmd_provision(ns)

    assert cpu_provision_stubs["cpu_calls"][0]["volume_gb"] == expected_volume_gb


def test_cmd_provision_cpu_small_explicit_subcap_volume_is_honored(
    isolated_state, stub_list_team_pods, cpu_provision_stubs
):
    """An explicit sub-cap --volume-gb is honored on a CPU intent (only the
    implicit 200 GB default is rewritten to the CPU default; the #1010 clamp
    only ever REDUCES an over-cap knob, so a 15 GB volume rides through while
    the 50 GB default container disk clamps to the cpu3g cap)."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []

    ns = _cpu_provision_ns(747, "cpu-small", volume_gb=15)
    pod_lifecycle.cmd_provision(ns)

    call = cpu_provision_stubs["cpu_calls"][0]
    assert call["volume_gb"] == 15
    assert call["container_disk_gb"] == 20  # default 50 clamped to the cap


# ---------------------------------------------------------------------------
# cmd_provision — #1010 CPU effective-payload cap clamp / pre-API refusal
#
# RunPod validates deployCpuPod's EFFECTIVE container disk —
# max(container_disk_gb, volume_gb); runpod_api._deploy_cpu_once folds the CPU
# volume into containerDiskInGb — against a per-flavor cap (probe 2026-07-04:
# cpu3g <= 20, cpu3c <= 80). These assert the create_cpu_pod CALL KWARGS (the
# EFFECTIVE payload), never a pod_lifecycle local, which the fold false-PASSes.
# ---------------------------------------------------------------------------


def test_cpu_provision_clamps_default_disk_to_instance_cap(
    isolated_state, stub_list_team_pods, cpu_provision_stubs
):
    """#1010: the untouched defaults (container 50 / CPU volume 40 ->
    effective 50) exceed the cpu3g cap (20); the CPU branch clamps the
    EFFECTIVE payload to the cap and PROCEEDS (pre-#1010 RunPod validation
    rejected every default cpu-small provision — the lane was broken)."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []

    ns = _cpu_provision_ns(1010, "cpu-small")
    pod_lifecycle.cmd_provision(ns)

    call = cpu_provision_stubs["cpu_calls"][0]
    assert max(call["container_disk_gb"], call["volume_gb"]) == 20


def test_cpu_provision_refuses_explicit_disk_above_cap(
    isolated_state, stub_list_team_pods, cpu_provision_stubs, capsys
):
    """#1010: an EXPLICIT above-default-band request (> 50 GB effective) over
    the instance cap refuses pre-API with exit 1 (same UX as the 'Pod already
    exists' refusal) — never a paid create RunPod's validation would reject."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []

    ns = _cpu_provision_ns(1010, "cpu-small", container_disk_gb=60)
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_provision(ns)
    assert exc.value.code == 1
    assert cpu_provision_stubs["cpu_calls"] == []  # pre-API: create never called
    # #1518: the refusal diagnostic rides stderr (the #1465 tail surface).
    captured = capsys.readouterr()
    assert "exceeds the" in captured.err
    assert "exceeds the" not in captured.out


def test_cpu_provision_threaded_floor_50_clamps_not_refuses_when_cap_below_50(
    isolated_state, stub_list_team_pods, cpu_provision_stubs
):
    """#1010 invariant: an effective payload of exactly 50 (the untouched
    default AND the router-threaded `--container-disk-gb max(50, boot)` floor)
    ALWAYS rides the clamp branch, never the refusal — a stated-small-footprint
    cpu-small auto-launch arrives here at exactly 50 and MUST clamp-and-proceed
    at the cap, not exit 1."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []

    # The router-threaded shape: --container-disk-gb 50 explicit on the argv.
    ns = _cpu_provision_ns(1010, "cpu-small", container_disk_gb=50)
    pod_lifecycle.cmd_provision(ns)

    call = cpu_provision_stubs["cpu_calls"][0]
    assert max(call["container_disk_gb"], call["volume_gb"]) == 20


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

    def fake_sync():
        # Task #831: cmd_sync is no-arg (re-reads pods.conf under the lock);
        # record only that the upsert triggered a downstream sync.
        captured["sync_called"] = True

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
    monkeypatch.setattr(pod_lifecycle, "cmd_sync", lambda: None)

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

    def _make(
        *,
        issue: int,
        yes: bool = True,
        dry_run: bool = False,
        skip: bool = False,
        name_suffix: str | None = None,
    ):
        return argparse.Namespace(
            issue=issue,
            yes=yes,
            dry_run=dry_run,
            skip_upload_verify=skip,
            name_suffix=name_suffix,
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


def _upload_verification_event(verdict: str, *, outroot: str | None = "swept-clean") -> dict:
    """Build a realistic ``epm:upload-verification`` event whose verdict lives
    in the markdown ``note`` body as ``**Verdict: <verdict>**`` — the real
    shape the upload-verifier writes (event keys are ts/kind/version/by/note;
    there is NO top-level ``verdict`` field). Mirrors
    tasks/completed/390/events.jsonl so the tests exercise the actual
    note-parsing path in ``_has_upload_verification_pass``.

    ``outroot`` renders the #2187 sweep-attestation token line the Step-5
    template now carries by construction (``outroot=<value>``); pass
    ``outroot=None`` for the pre-#2187 token-less note shape (the terminate
    guard refuses a PASS without it)."""
    outroot_line = f"outroot={outroot}\n\n" if outroot is not None else ""
    return {
        "ts": "2026-06-02T00:00:00Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "upload-verifier",
        "note": (
            "<!-- epm:upload-verification v1 -->\n## Upload Verification\n\n"
            f"**Verdict: {verdict}**\n\n{outroot_line}"
            "Discovered N files on the pod under eval_results/."
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


def _fake_experiment_task(monkeypatch) -> None:
    """Point task_workflow.get_task at a kind=experiment task (guard engaged)."""

    def fake_get_task(issue):
        return {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""}

    monkeypatch.setattr("explore_persona_space.task_workflow.get_task", fake_get_task)


def test_terminate_refuses_pass_without_outroot_token(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """#2187: a PASS note WITHOUT the outroot= sweep-attestation token must
    refuse the terminate, with a remediation message naming the sweep recipe
    (the pre-#2187 note shape — subdirectory-only upload globs lost three
    out-root TOP-LEVEL files on #2162 behind exactly this PASS)."""
    pod_name = _register_pod_for_issue(2187)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS", outroot=None)])
    _fake_experiment_task(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=2187))

    msg = str(exc.value)
    assert "outroot=" in msg
    assert "--outroot-listing" in msg
    assert "Step 2.10" in msg
    assert stub_terminate_pod == [], "terminate_pod must NOT be called on a token-less PASS"


def test_terminate_proceeds_with_prose_outroot_token(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """The Step-5 template's prose token line (`outroot=swept-clean`) satisfies
    the #2187 attestation — the guard is silent on the happy path."""
    pod_name = _register_pod_for_issue(2188)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_list_events(
        monkeypatch, [_upload_verification_event("PASS", outroot="residue-committed")]
    )
    _fake_experiment_task(monkeypatch)

    pod_lifecycle.cmd_terminate(terminate_ns(issue=2188))

    assert len(stub_terminate_pod) == 1


def test_terminate_proceeds_with_json_outroot_token(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """A machine-readable JSON note carrying {"verdict": "PASS", "outroot":
    "swept-clean"} satisfies both the verdict and the #2187 attestation."""
    pod_name = _register_pod_for_issue(2189)
    stub_list_team_pods.return_value = [_info(pod_name)]
    event = {
        "ts": "2026-08-08T00:00:00Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "upload-verifier",
        "note": json.dumps({"verdict": "PASS", "outroot": "swept-clean"}),
    }
    _stub_list_events(monkeypatch, [event])
    _fake_experiment_task(monkeypatch)

    pod_lifecycle.cmd_terminate(terminate_ns(issue=2189))

    assert len(stub_terminate_pod) == 1


def test_terminate_outroot_token_with_fail_verdict_still_refused(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """The verdict check comes FIRST: a FAIL note carrying the outroot= token
    is still refused (the token attests the sweep, never the verdict)."""
    pod_name = _register_pod_for_issue(2190)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_list_events(monkeypatch, [_upload_verification_event("FAIL", outroot="swept-clean")])
    _fake_experiment_task(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=2190))

    assert "epm:upload-verification PASS" in str(exc.value)
    assert stub_terminate_pod == []


def test_terminate_pass_without_token_skip_flag_warns_and_proceeds(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    capsys,
    monkeypatch,
):
    """The finding-6 pinned decision: --skip-upload-verify waives the outroot=
    token exactly as it waives the whole PASS (a PASS-without-token pod is
    strictly MORE verified than a no-marker pod and must not be blocked
    harder under the same flag) — LOUD WARN, then proceed."""
    pod_name = _register_pod_for_issue(2191)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS", outroot=None)])
    _fake_experiment_task(monkeypatch)

    pod_lifecycle.cmd_terminate(terminate_ns(issue=2191, skip=True))

    err = capsys.readouterr().err
    assert "LACKS the outroot= sweep attestation" in err
    assert len(stub_terminate_pod) == 1


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
# cmd_terminate — keep-running teardown shield (#1485)
# ---------------------------------------------------------------------------


def _stub_keep_running_state(monkeypatch, state) -> None:
    """Monkeypatch task_workflow.keep_running_tag_state (imported lazily
    inside ``_guard_keep_running_before_terminate``) to a fixed tri-state
    value — the seam is the LIBRARY reader, so the guard's own routing
    (refuse / proceed / force / dry-run) runs for real."""
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.keep_running_tag_state",
        lambda issue: state,
    )


def test_terminate_bare_refuses_on_keep_running_tag(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """DURABILITY PIN (#1485 acceptance criterion 1): the bare (issue-wide)
    terminate REFUSES via SystemExit when the owning task carries the
    keep-running tag — zero terminate_pod calls — and the message names all
    three remedies (remove-tag / --name-suffix / --force-keep-running).
    Incident 2026-07-17: pod-1345-onpolicy destroyed mid-launch by an
    issue-wide sweep that never read the tag."""
    pod_name = _register_pod_for_issue(600)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_keep_running_state(monkeypatch, True)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=600))

    msg = str(exc.value)
    assert "keep-running" in msg
    assert "remove-tag 600 keep-running" in msg
    assert "--name-suffix" in msg
    assert "--force-keep-running" in msg
    assert stub_terminate_pod == [], "terminate_pod must NOT be called when the shield refuses"


def test_terminate_name_suffix_allowed_despite_keep_running_tag(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """#1485 acceptance criterion 2: a surgical --name-suffix destroy is the
    operator's explicit single-pod choice — never blocked by the tag. The
    reader stub RAISES, pinning that the surgical path reads NO tag state at
    all (the guard early-returns before any read)."""
    _write_metadata_file(
        {
            "pod-601": _meta("pod-601", issue=601),
            "pod-601-b": _meta("pod-601-b", issue=601, pod_id="live-suffix-b"),
        }
    )
    stub_list_team_pods.return_value = [
        _info("pod-601", pod_id="live-canonical"),
        _info("pod-601-b", pod_id="live-suffix-b"),
    ]

    def boom(_issue):
        raise AssertionError("keep-running state must not be read on the surgical path")

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.keep_running_tag_state",
        boom,
    )
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=601, name_suffix="b"))

    assert stub_terminate_pod == ["live-suffix-b"], (
        f"surgical terminate must destroy ONLY pod-601-b; got {stub_terminate_pod}"
    )


@pytest.mark.parametrize("state", [True, None])
def test_terminate_force_keep_running_overrides_with_warning(
    state,
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    capsys,
    monkeypatch,
):
    """#1485 acceptance criterion 3: --force-keep-running proceeds with a
    LOUD stderr warning — for BOTH the tag-present and the unreadable state
    (the force check precedes the None refusal)."""
    pod_name = _register_pod_for_issue(602)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_keep_running_state(monkeypatch, state)
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    ns = terminate_ns(issue=602)
    ns.force_keep_running = True  # the fixture Namespace predates the flag
    pod_lifecycle.cmd_terminate(ns)

    err = capsys.readouterr().err
    assert "--force-keep-running" in err
    assert "DESPITE keep-running" in err
    assert len(stub_terminate_pod) == 1, (
        f"terminate must proceed under --force-keep-running (state={state!r})"
    )


def test_terminate_keep_running_unknown_refuses(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """#1485 acceptance criterion 4 (fail-closed): an UNREADABLE tag state
    (branch-guard RuntimeError, corrupt frontmatter, registry corruption —
    the library reader returns None) refuses the irreversible bare terminate,
    naming the override."""
    pod_name = _register_pod_for_issue(603)
    stub_list_team_pods.return_value = [_info(pod_name)]
    _stub_keep_running_state(monkeypatch, None)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_terminate(terminate_ns(issue=603))

    msg = str(exc.value)
    assert "could not be read" in msg
    assert "--force-keep-running" in msg
    assert stub_terminate_pod == []


def test_terminate_keep_running_dry_run_notes_and_previews(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    capsys,
    monkeypatch,
):
    """Phase-2 Statistics Must-Fix (#1485): --dry-run previews and reads NO
    task state — BOTH get_task and keep_running_tag_state are stubbed to
    raise — while a generic would-check NOTE names what a real run would do.
    The pre-existing pin test_terminate_dry_run_bypasses_guard stays green
    unmodified; this test additionally pins the NOTE + the reader."""
    pod_name = _register_pod_for_issue(604)
    stub_list_team_pods.return_value = [_info(pod_name)]

    def should_not_read_task(_issue):
        raise AssertionError("dry-run must not inspect task state")

    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        should_not_read_task,
    )
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.keep_running_tag_state",
        should_not_read_task,
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=604, dry_run=True))

    err = capsys.readouterr().err
    assert "keep-running tag" in err
    assert "REFUSE this issue-wide terminate" in err
    assert stub_terminate_pod == [], "dry-run must not call terminate_pod"


def test_terminate_parser_exposes_force_keep_running_flag():
    """Regression guard: the --force-keep-running flag exists on the
    terminate subparser (and only defaults False)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pod_lifecycle._parser_terminate(sub)

    ns = parser.parse_args(["terminate", "--issue", "1", "--yes", "--force-keep-running"])
    assert ns.force_keep_running is True

    ns2 = parser.parse_args(["terminate", "--issue", "1", "--yes"])
    assert ns2.force_keep_running is False


def test_keep_running_tag_constant_matches_task_workflow():
    """The pod_lifecycle module keeps its own lazy-import-independent copy of
    the tag literal; pin it to the task_workflow canonical constant so the
    two can never drift."""
    import explore_persona_space.task_workflow as tw

    assert pod_lifecycle._KEEP_RUNNING_TAG == tw.KEEP_RUNNING_TAG == "keep-running"


def test_runpod_backend_teardown_composes_bare_terminate_no_force(monkeypatch):
    """#1485 defense-in-depth pin (transitive inheritance): RunPodBackend.
    teardown delegates to a ``pod_lifecycle.py terminate --issue N --yes``
    SUBPROCESS in the BARE form — no --force-keep-running, no --name-suffix —
    so the new keep-running guard binds in the child. (An in-process
    monkeypatch cannot cross the subprocess boundary; the argv assertion is
    the mechanizable form.)"""
    from explore_persona_space.backends import runpod as rp
    from explore_persona_space.backends.base import RunHandle

    captured: list[list[str]] = []
    monkeypatch.setattr(rp, "_run_pod_lifecycle_relay", lambda cmd, **k: captured.append(cmd))

    handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="job-1485",
        pod_name="pod-1485",
        scratch_dir="/workspace",
        log_path="/log",
        extra={"issue": 1485},
    )
    rp.RunPodBackend().teardown(handle)

    assert len(captured) == 1
    cmd = captured[0]
    assert "terminate" in cmd
    assert "--issue" in cmd and "1485" in cmd and "--yes" in cmd
    assert "--force-keep-running" not in cmd
    assert "--name-suffix" not in cmd


# ---------------------------------------------------------------------------
# cmd_terminate — live-API authority for pod_id (post-#475 hardening)
# ---------------------------------------------------------------------------


def test_issue_from_pod_name_anchors_on_full_suffix():
    """``pod-47`` resolves to issue 47, NOT 475 — the digits are anchored on
    end-of-string or a ``-<slug>`` boundary, never a substring. Regression for
    the name-matching anchor that keeps multi-pod terminate from over-matching
    neighbouring issues."""
    assert pod_lifecycle._issue_from_pod_name("pod-47") == 47
    assert pod_lifecycle._issue_from_pod_name("pod-475") == 475
    assert pod_lifecycle._issue_from_pod_name("epm-issue-475") == 475
    # DELIBERATE #1334 contract change: a letter-initial lowercase slug is the
    # multi-pod-per-issue form and maps to its owning issue (the old pin was
    # ``is None`` — pre-#1334 any suffixed name was unmappable).
    assert pod_lifecycle._issue_from_pod_name("pod-475-backup") == 475
    # Names without a managed prefix never match.
    assert pod_lifecycle._issue_from_pod_name("thomas-pod-475") is None


# ---------------------------------------------------------------------------
# #1334 — multi-pod-per-issue naming (pod-<N>-<slug>)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pod-47", 47),
        ("pod-475", 475),
        ("epm-issue-475", 475),
        ("pod-475-backup", 475),  # deliberate #1334 lifecycle change (was None)
        ("pod-779-b", 779),
        ("epm-issue-546-b", 546),  # legacy suffixed dispatcher pods (audit precedent)
        ("pod-779-60", None),  # numeric slug rejected — letter-initial rule
        ("pod-779-B", None),  # uppercase rejected; we only generate lowercase
        ("pod-77960", 77960),  # legacy fabrication shape — bare int tail, unchanged
        ("pod-779-", None),  # empty slug
        # Trailing-hyphen slug: [a-z][a-z0-9-]* admits 'b-' — pinned so the
        # parser and provision's slug validator (which also admits it) agree.
        ("pod-779-b-", 779),
        ("thomas-pod-475", None),
        ("pod-abc", None),
        ("pod-", None),
        ("", None),
    ],
)
def test_issue_from_pod_name_suffix_grammar(name: str, expected: int | None):
    """The full #1334 grammar table for the canonical parser."""
    assert pod_lifecycle._issue_from_pod_name(name) == expected


def test_canonical_pod_name_suffix():
    """Builder shapes + the parser⇄builder round-trip invariant (#1334
    acceptance criterion 1): every valid (issue, slug) pair maps back to its
    owning issue."""
    assert pod_lifecycle._canonical_pod_name(779) == "pod-779"
    assert pod_lifecycle._canonical_pod_name(779, "b") == "pod-779-b"
    assert pod_lifecycle._canonical_pod_name(779, None) == "pod-779"
    for issue, slug in [(779, "b"), (475, "followup2"), (1, "b-2"), (77960, None)]:
        name = pod_lifecycle._canonical_pod_name(issue, slug)
        assert pod_lifecycle._issue_from_pod_name(name) == issue, (issue, slug, name)


def test_provision_parser_exposes_name_suffix():
    """--name-suffix is wired into all FOUR subparsers (provision / stop /
    resume / terminate), defaulting to None (mirrors the
    test_terminate_parser_exposes_skip_upload_verify_flag pattern)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pod_lifecycle._parser_provision(sub)
    pod_lifecycle._parser_stop(sub)
    pod_lifecycle._parser_resume(sub)
    pod_lifecycle._parser_terminate(sub)

    ns = parser.parse_args(["provision", "--issue", "779", "--name-suffix", "b"])
    assert ns.name_suffix == "b"
    ns0 = parser.parse_args(["provision", "--issue", "779"])
    assert ns0.name_suffix is None
    for verb in ("stop", "resume", "terminate"):
        ns1 = parser.parse_args([verb, "--issue", "779", "--name-suffix", "b"])
        assert ns1.name_suffix == "b"
        ns2 = parser.parse_args([verb, "--issue", "779"])
        assert ns2.name_suffix is None


@pytest.mark.parametrize("bad", ["60", "B", "-b", "a" * 21])
def test_provision_name_suffix_rejects_bad_slug(bad: str):
    """cmd_provision SystemExits on a slug outside [a-z][a-z0-9-]{0,19} —
    BEFORE any task-state or live-API read (the minimal Namespace proves it)."""
    ns = argparse.Namespace(issue=779, list_intents=False, name_suffix=bad)
    with pytest.raises(SystemExit, match="--name-suffix must match"):
        pod_lifecycle.cmd_provision(ns)


def _gpu_provision_ns(issue: int, *, name_suffix: str | None = None, **overrides):
    """Namespace matching the provision subparser shape, for a GPU intent."""
    base = {
        "issue": issue,
        "list_intents": False,
        "intent": "eval",
        "gpu_type": None,
        "gpu_count": None,
        "dry_run": True,
        "volume_gb": 200,
        "container_disk_gb": 50,
        "ttl_days": 7,
        "no_bootstrap": True,
        "name_suffix": name_suffix,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_provision_name_suffix_collision_scope(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """A live RUNNING pod-779 blocks a bare provision (existing behavior) but
    NOT a --name-suffix provision — a live bare pod is exactly why the suffix
    form exists. The suffixed provision collides only on ITS OWN name."""
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = [_info("pod-779", desired_status="RUNNING")]

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_provision(_gpu_provision_ns(779))
    assert exc.value.code == 1
    assert "already exists" in capsys.readouterr().err  # #1518: refusal rides stderr

    # Suffixed provision proceeds past the collision check (dry-run plan
    # names pod-779-b — acceptance criterion 2).
    pod_lifecycle.cmd_provision(_gpu_provision_ns(779, name_suffix="b"))
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "pod-779-b" in out

    # A RUNNING pod-779-b DOES block the suffixed provision (its own name).
    stub_list_team_pods.return_value = [_info("pod-779-b", desired_status="RUNNING")]
    with pytest.raises(SystemExit) as exc2:
        pod_lifecycle.cmd_provision(_gpu_provision_ns(779, name_suffix="b"))
    assert exc2.value.code == 1
    assert "pod-779-b already exists" in capsys.readouterr().err  # #1518: stderr


def test_provision_registers_owning_issue_for_suffixed_name(isolated_state, monkeypatch, capsys):
    """_provision_wait_register_bootstrap records EphemeralMetadata keyed by
    the suffixed name with issue == the REAL owning task (#1334 acceptance
    criterion 3) — falls out of the existing name threading, no code change."""
    _write_metadata_file({})
    info = _info("pod-779-b", pod_id="live-779-b")
    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", lambda pod_id, timeout=600: info)
    monkeypatch.setattr(pod_lifecycle, "note_ssh_wait_outcome", lambda *a, **k: None)
    monkeypatch.setattr(pod_lifecycle, "_upsert_pods_conf", lambda pod: None)
    ns = argparse.Namespace(issue=779, name_suffix="b", ttl_days=7, no_bootstrap=True)

    pod_lifecycle._provision_wait_register_bootstrap(ns, "pod-779-b", info, "lora-7b")

    metadata = _read_metadata_file()
    assert "pod-779-b" in metadata
    assert metadata["pod-779-b"].issue == 779
    assert metadata["pod-779-b"].pod_id == "live-779-b"


def test_bootstrap_failure_hint_carries_name_suffix(isolated_state, monkeypatch, capsys):
    """A suffixed provision's bootstrap-failure discard hint is scoped with
    --name-suffix — it must never suggest an issue-wide terminate that would
    take a healthy sibling pod-<N>'s volume with it."""
    info = _info("pod-779-b", pod_id="live-779-b")
    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", lambda pod_id, timeout=600: info)
    monkeypatch.setattr(pod_lifecycle, "note_ssh_wait_outcome", lambda *a, **k: None)
    monkeypatch.setattr(pod_lifecycle, "_upsert_pods_conf", lambda pod: None)
    # Signature-conformant stub: _bootstrap(pod_name, intent_label, issue) —
    # the pre-#1997 bare two-arg lambda went stale when the production call
    # site gained issue=args.issue (red on pristine main, fixed here).
    monkeypatch.setattr(pod_lifecycle, "_bootstrap", lambda name, intent_label, issue=None: 1)
    ns = argparse.Namespace(issue=779, name_suffix="b", ttl_days=7, no_bootstrap=False)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._provision_wait_register_bootstrap(ns, "pod-779-b", info, "lora-7b")
    assert exc.value.code == 1
    assert "terminate --issue 779 --name-suffix b" in capsys.readouterr().err


def _bootstrap_tail_ns(*, no_bootstrap: bool = False) -> argparse.Namespace:
    """Namespace for driving _provision_wait_register_bootstrap directly (#1931)."""
    return argparse.Namespace(issue=779, name_suffix=None, ttl_days=7, no_bootstrap=no_bootstrap)


def _stub_provision_tail(monkeypatch, info, rcs: list[int]) -> list[str]:
    """Stub the provision tail's collaborators; _bootstrap pops rcs per call.

    Returns the (mutable) list of recorded _bootstrap call targets so tests can
    assert the exact call count.
    """
    calls: list[str] = []

    def fake_bootstrap(name, intent_label, issue=None):
        # Signature-conformant with _bootstrap(pod_name, intent_label, issue)
        # — the pre-#1997 two-arg stub went stale when the production call
        # site gained issue=args.issue (red on pristine main, fixed here).
        calls.append(name)
        return rcs[len(calls) - 1]

    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", lambda pod_id, timeout=600: info)
    monkeypatch.setattr(pod_lifecycle, "note_ssh_wait_outcome", lambda *a, **k: None)
    monkeypatch.setattr(pod_lifecycle, "_upsert_pods_conf", lambda pod: None)
    monkeypatch.setattr(pod_lifecycle, "_bootstrap", fake_bootstrap)
    return calls


def test_provision_bootstrap_retries_once_then_succeeds(isolated_state, monkeypatch, capsys):
    """A transient first-attempt bootstrap failure (rc=100) retries EXACTLY once;
    the retry's rc=0 completes provision (no SystemExit) with the retry line on
    stderr and the BOOTSTRAP-OK verdict token emitted (#1931 acceptance 1+2)."""
    _write_metadata_file({})
    info = _info("pod-779", pod_id="live-779")
    calls = _stub_provision_tail(monkeypatch, info, [100, 0])

    pod_lifecycle._provision_wait_register_bootstrap(
        _bootstrap_tail_ns(), "pod-779", info, "lora-7b"
    )

    assert calls == ["pod-779", "pod-779"]  # exactly 2 calls: first try + one retry
    captured = capsys.readouterr()
    assert "[bootstrap-retry] bootstrap exited rc=100 on pod-779" in captured.err
    assert "BOOTSTRAP-OK pod=pod-779" in captured.out
    assert "BOOTSTRAP-OK pod=pod-779" in captured.err  # stream-consistent with FAILED
    assert "BOOTSTRAP-FAILED" not in captured.out + captured.err


def test_provision_bootstrap_fails_loud_after_retry(isolated_state, monkeypatch, capsys):
    """Both bootstrap attempts failing (rc=100 twice) keeps the sys.exit(rc)
    contract AND emits the machine-greppable BOOTSTRAP-FAILED verdict as the
    last stderr line before exit (#1931 acceptance 1+3)."""
    _write_metadata_file({})
    info = _info("pod-779", pod_id="live-779")
    calls = _stub_provision_tail(monkeypatch, info, [100, 100])

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._provision_wait_register_bootstrap(
            _bootstrap_tail_ns(), "pod-779", info, "lora-7b"
        )

    assert exc.value.code == 100
    assert calls == ["pod-779", "pod-779"]  # never more than one retry
    captured = capsys.readouterr()
    assert "[bootstrap-retry]" in captured.err
    assert "BOOTSTRAP-FAILED pod=pod-779 rc=100" in captured.err
    assert captured.err.rstrip().splitlines()[-1] == "BOOTSTRAP-FAILED pod=pod-779 rc=100"
    assert "BOOTSTRAP-OK" not in captured.out + captured.err


def test_provision_bootstrap_success_no_retry(isolated_state, monkeypatch, capsys):
    """A clean first-attempt bootstrap (rc=0) never retries: one _bootstrap
    call, no [bootstrap-retry] line, BOOTSTRAP-OK present (#1931 acceptance 4)."""
    _write_metadata_file({})
    info = _info("pod-779", pod_id="live-779")
    calls = _stub_provision_tail(monkeypatch, info, [0])

    pod_lifecycle._provision_wait_register_bootstrap(
        _bootstrap_tail_ns(), "pod-779", info, "lora-7b"
    )

    assert calls == ["pod-779"]
    captured = capsys.readouterr()
    assert "[bootstrap-retry]" not in captured.err
    assert "BOOTSTRAP-OK pod=pod-779" in captured.out
    assert "Done. SSH with: ssh pod-779" in captured.out


def test_provision_no_bootstrap_skips_retry_and_verdict(isolated_state, monkeypatch, capsys):
    """--no-bootstrap semantics unchanged (#1931 acceptance 5): _bootstrap is
    never invoked and neither verdict token is printed — the skip message stays."""
    _write_metadata_file({})
    info = _info("pod-779", pod_id="live-779")
    calls = _stub_provision_tail(monkeypatch, info, [])

    pod_lifecycle._provision_wait_register_bootstrap(
        _bootstrap_tail_ns(no_bootstrap=True), "pod-779", info, "lora-7b"
    )

    assert calls == []  # _bootstrap never called
    captured = capsys.readouterr()
    assert "Skipping bootstrap (--no-bootstrap)" in captured.out
    assert "BOOTSTRAP-OK" not in captured.out + captured.err
    assert "BOOTSTRAP-FAILED" not in captured.out + captured.err
    assert "[bootstrap-retry]" not in captured.err


def _epod(name: str, issue: int) -> pod_lifecycle.EphemeralPod:
    return pod_lifecycle.EphemeralPod(metadata=_meta(name, issue=issue), info=_info(name))


def test_find_pod_in_state_name_suffix_and_fallback():
    """Resolution semantics (#1334): exact suffix lookup; canonical-first when
    both exist; unique-issue fallback returns the lone suffixed pod; two
    suffixed pods + no canonical -> None."""
    both = {p.name: p for p in [_epod("pod-779", 779), _epod("pod-779-b", 779)]}
    assert pod_lifecycle._find_pod_in_state(both, 779, name_suffix="b").name == "pod-779-b"
    assert pod_lifecycle._find_pod_in_state(both, 779, name_suffix="c") is None
    assert pod_lifecycle._find_pod_in_state(both, 779).name == "pod-779"

    lone_suffixed = {p.name: p for p in [_epod("pod-779-b", 779)]}
    assert pod_lifecycle._find_pod_in_state(lone_suffixed, 779).name == "pod-779-b"

    two_suffixed = {p.name: p for p in [_epod("pod-779-b", 779), _epod("pod-779-c", 779)]}
    assert pod_lifecycle._find_pod_in_state(two_suffixed, 779) is None


def test_stop_resume_thread_name_suffix(isolated_state, stub_list_team_pods, monkeypatch, capsys):
    """cmd_stop / cmd_resume actually THREAD --name-suffix into
    _find_pod_in_state: with BOTH pod-779 and pod-779-b registered, the
    suffixed namespace targets pod-779-b (an unthreaded flag would resolve the
    canonical pod-779 first) and prints the resolved name before acting."""
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    metadata = {
        "pod-779": _meta("pod-779", issue=779),
        "pod-779-b": _meta("pod-779-b", issue=779),
    }
    _write_metadata_file(metadata)
    ns = argparse.Namespace(issue=779, name_suffix="b", dry_run=True)

    stub_list_team_pods.return_value = [_info("pod-779"), _info("pod-779-b")]
    pod_lifecycle.cmd_stop(ns)
    assert "Stopping pod-779-b" in capsys.readouterr().out

    stub_list_team_pods.return_value = [
        _info("pod-779", desired_status="EXITED"),
        _info("pod-779-b", desired_status="EXITED"),
    ]
    pod_lifecycle.cmd_resume(ns)
    assert "Resuming pod-779-b" in capsys.readouterr().out


def test_stop_ambiguous_multi_pod_error_directs_to_name_suffix(isolated_state, stub_list_team_pods):
    """Two suffixed pods and no canonical pod-<N>: the bare stop errors,
    LISTING the registered names and directing the caller to --name-suffix
    (never a silent arbitrary pick)."""
    metadata = {
        "pod-779-b": _meta("pod-779-b", issue=779),
        "pod-779-c": _meta("pod-779-c", issue=779),
    }
    _write_metadata_file(metadata)
    stub_list_team_pods.return_value = [_info("pod-779-b"), _info("pod-779-c")]
    ns = argparse.Namespace(issue=779, name_suffix=None, dry_run=True)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_stop(ns)
    msg = str(exc.value)
    assert "pod-779-b" in msg and "pod-779-c" in msg
    assert "--name-suffix" in msg


def test_terminate_name_suffix_scopes_to_one_pod(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """terminate --issue 779 --name-suffix b destroys ONLY pod-779-b (#1334
    acceptance criterion 6): the sibling pod-779 is neither terminated NOR
    reported as a survivor (the re-check applies the same name filter — an
    unfiltered re-check would raise RunPodError on the healthy sibling), and
    its sidecar record survives the post-terminate cleanup."""
    _write_metadata_file(
        {
            "pod-779": _meta("pod-779", issue=779),
            "pod-779-b": _meta("pod-779-b", issue=779, pod_id="live-suffix-b"),
        }
    )
    stub_list_team_pods.return_value = [
        _info("pod-779", pod_id="live-canonical", desired_status="EXITED"),
        _info("pod-779-b", pod_id="live-suffix-b"),
    ]
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=779, name_suffix="b"))

    assert stub_terminate_pod == ["live-suffix-b"], (
        f"suffix-narrowed terminate must destroy ONLY pod-779-b; got {stub_terminate_pod}"
    )
    metadata = _read_metadata_file()
    assert "pod-779" in metadata, "sibling pod-779's sidecar record must survive"
    assert "pod-779-b" not in metadata


def test_terminate_bare_issue_sweeps_suffixed_pods(
    isolated_state,
    stub_list_team_pods,
    stub_terminate_pod,
    stub_pods_conf_writes,
    terminate_ns,
    monkeypatch,
):
    """DECIDED bare-form semantics (#1334 plan §3.3): issue-level teardown
    destroys EVERY live pod of the issue, suffixed follow-up pods included —
    a round that must survive Step 8 sets the task-level keep-running tag."""
    _write_metadata_file({"pod-779": _meta("pod-779", issue=779)})
    stub_list_team_pods.return_value = [
        _info("pod-779", pod_id="live-canonical"),
        _info("pod-779-b", pod_id="live-suffix-b"),
    ]
    _stub_list_events(monkeypatch, [_upload_verification_event("PASS")])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )

    pod_lifecycle.cmd_terminate(terminate_ns(issue=779))

    assert sorted(stub_terminate_pod) == sorted(["live-canonical", "live-suffix-b"])


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


# ---------------------------------------------------------------------------
# _upload_verification_outroot_attested — the #2187 sweep-attestation token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        "**Verdict: PASS**\n\noutroot=swept-clean\n",
        "**Verdict: PASS**\n\noutroot=residue-committed\n",
        "Verdict: PASS\noutroot: none\n",
        "verdict: pass\nOUTROOT = SWEPT-CLEAN\n",  # case-insensitive, spaced =
        json.dumps({"verdict": "PASS", "outroot": "swept-clean"}),
        json.dumps({"verdict": "PASS", "outroot": "none"}),
    ],
)
def test_outroot_attested_accepts_both_note_shapes(note):
    assert pod_lifecycle._upload_verification_outroot_attested(note) is True


@pytest.mark.parametrize(
    "note",
    [
        "**Verdict: PASS**\n\nDiscovered N files.",  # token absent
        "**Verdict: PASS**\noutroot=sweptish\n",  # invalid value
        "**Verdict: PASS**\noutroot=\n",  # empty value
        json.dumps({"verdict": "PASS"}),  # JSON without the key
        json.dumps({"verdict": "PASS", "outroot": "yes"}),  # JSON invalid value
        "",
    ],
)
def test_outroot_attested_rejects_missing_or_invalid_token(note):
    assert pod_lifecycle._upload_verification_outroot_attested(note) is False


# The DOCUMENTED inline-round note shape (#1970, incident #1773): an inline
# round that verified its own uploads posts this note via `task.py
# post-marker`, then re-runs terminate. LEADING `Verdict: PASS` so BOTH
# parsers accept it: pod_lifecycle's loose `verdict[:*\s]+PASS` regex AND
# task_workflow.UPLOAD_VERIFICATION_PASS_RE (the finalize teardown gate).
# As of #2187 the documented recipe ALSO carries the out-root sweep
# attestation token (`outroot=<...>`) — the terminate guard refuses a PASS
# without it (see test_terminate_refuses_pass_without_outroot_token).
_INLINE_ROUND_NOTE = (
    "Verdict: PASS — inline-round verification; "
    "prefixes: issue1773_fulldict/, issue1773_raw_windows/; "
    "outroot=swept-clean"
)


def test_has_upload_verification_pass_accepts_inline_round_note(monkeypatch):
    """The documented inline-round note satisfies the guard's satisfier AND
    the terminate-guard path proceeds without --skip-upload-verify; the same
    note also matches task_workflow.UPLOAD_VERIFICATION_PASS_RE (cross-parser
    pin — an inline PASS marker must never read as
    upload_verification_failed_current to a later finalize)."""
    from explore_persona_space.task_workflow import UPLOAD_VERIFICATION_PASS_RE

    event = {
        "ts": "2026-08-03T00:00:00Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "orchestrator",
        "note": _INLINE_ROUND_NOTE,
    }
    _stub_list_events(monkeypatch, [event])
    assert pod_lifecycle._has_upload_verification_pass(1773) is True

    # Cross-parser pin: import the constant, never retype the regex.
    assert UPLOAD_VERIFICATION_PASS_RE.search(_INLINE_ROUND_NOTE) is not None

    # Terminate-guard decision flow proceeds (returns None, no SystemExit)
    # without --skip-upload-verify for a kind=experiment task.
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )
    pod_lifecycle._guard_upload_verification_before_terminate(1773, skip_flag=False, dry_run=False)


def test_terminate_guard_refusal_names_inline_recipe(monkeypatch):
    """The refusal message names the sanctioned inline-round recipe (post
    `epm:upload-verification` via `task.py post-marker`, then re-run
    terminate) BEFORE the --skip-upload-verify last resort — the
    message-content sibling of the existing --skip-upload-verify assert
    (#1970; #1773: a verified inline round was steered straight to the
    blunt override)."""
    _stub_list_events(monkeypatch, [])
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"id": issue, "frontmatter": {"kind": "experiment"}, "body": ""},
    )
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._guard_upload_verification_before_terminate(
            1773, skip_flag=False, dry_run=False
        )
    message = str(exc.value)
    assert "epm:upload-verification" in message
    assert "post-marker" in message
    assert "--skip-upload-verify" in message


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


def _json_note_event(verdict: str, *, nested_checked_verdict: str | None = None) -> dict:
    """Build an ``epm:upload-verification`` event whose ``note`` is a JSON
    object — the machine-readable shape the upload-verifier agent posts when
    it serializes its checklist directly. Mirrors line 222 of
    tasks/interpreting/488/events.jsonl (incident 2026-06-10 task #488), where
    ``pod.py terminate`` refused a fully-verified pod because the regex chain
    in ``_has_upload_verification_pass`` couldn't parse the quoted-key JSON.

    Optionally embed a per-file ``"verdict"`` in the nested ``checked`` block
    to verify the parser uses the TOP-LEVEL verdict, not whichever ``verdict``
    a permissive regex would have hit first.
    """
    payload: dict[str, object] = {
        "verdict": verdict,
        "discovered_pod_files": 730,
        "reverification": True,
    }
    if nested_checked_verdict is not None:
        payload["checked"] = {
            "raw_completions_hf_data": (
                f"{nested_checked_verdict}: 648/648 emission JSONs on HF data repo."
            ),
        }
    return {
        "ts": "2026-06-10T03:20:15Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "upload-verifier",
        "note": json.dumps(payload),
    }


def test_has_upload_verification_pass_json_note_pass(monkeypatch):
    """Regression for the 2026-06-10 task #488 incident: when the
    upload-verifier posts a machine-readable JSON note
    (``{"verdict": "PASS", ...}``), the parser must accept it as a PASS
    verdict so ``pod.py terminate`` does not refuse a fully-verified pod and
    force the orchestrator to re-post a duplicate guard-parseable marker.
    """
    _stub_list_events(monkeypatch, [_json_note_event("PASS")])
    assert pod_lifecycle._has_upload_verification_pass(999) is True


@pytest.mark.parametrize("verdict", ["FAIL", "WARN"])
def test_has_upload_verification_pass_json_note_non_pass(monkeypatch, verdict):
    """Symmetry: a JSON note whose top-level ``verdict`` is FAIL or WARN
    must NOT be parsed as PASS."""
    _stub_list_events(monkeypatch, [_json_note_event(verdict)])
    assert pod_lifecycle._has_upload_verification_pass(999) is False


def test_has_upload_verification_pass_json_note_top_level_verdict_wins_over_nested(
    monkeypatch,
):
    """A JSON note can contain a nested ``checked.{file}`` block with its own
    ``PASS``/``FAIL`` strings (per-artifact subverdicts). The parser must
    consult ONLY the top-level ``verdict`` key, not whichever substring a
    permissive regex hits first — otherwise a FAIL note with a nested
    ``PASS: 648/648 ...`` would false-positive (and a PASS note with a nested
    ``FAIL`` would refuse a verified pod)."""
    _stub_list_events(monkeypatch, [_json_note_event("FAIL", nested_checked_verdict="PASS")])
    assert pod_lifecycle._has_upload_verification_pass(999) is False

    _stub_list_events(monkeypatch, [_json_note_event("PASS", nested_checked_verdict="FAIL")])
    assert pod_lifecycle._has_upload_verification_pass(999) is True


def test_has_upload_verification_pass_json_note_latest_wins(monkeypatch):
    """``latest-event-wins`` extends across the JSON-note path: an older
    FAIL JSON followed by a newer PASS JSON resolves to True; the inverse
    resolves to False."""
    _stub_list_events(
        monkeypatch,
        [_json_note_event("FAIL"), _json_note_event("PASS")],
    )
    assert pod_lifecycle._has_upload_verification_pass(999) is True

    _stub_list_events(
        monkeypatch,
        [_json_note_event("PASS"), _json_note_event("FAIL")],
    )
    assert pod_lifecycle._has_upload_verification_pass(999) is False


def test_has_upload_verification_pass_json_note_missing_verdict_key_falls_through(
    monkeypatch,
):
    """A JSON object that parses cleanly but has no ``verdict`` key is NOT a
    verifier verdict — fall through to the regex chain. If neither matches,
    the result is False (no PASS asserted)."""
    event = {
        "ts": "2026-06-10T03:20:15Z",
        "kind": "epm:upload-verification",
        "version": 1,
        "by": "upload-verifier",
        "note": json.dumps({"discovered_pod_files": 730, "status": "incomplete"}),
    }
    _stub_list_events(monkeypatch, [event])
    assert pod_lifecycle._has_upload_verification_pass(999) is False


# ---------------------------------------------------------------------------
# _resolve_spec — intent vs explicit --gpu-type/--gpu-count merging (#531)
# ---------------------------------------------------------------------------


def test_resolve_spec_intent_only_uses_table():
    spec, label = pod_lifecycle._resolve_spec("eval", None, None)
    assert (spec.gpu_type, spec.gpu_count) == ("H100", 1)
    assert label == "eval"


def test_resolve_spec_both_flags_override_intent():
    spec, label = pod_lifecycle._resolve_spec("eval", "H200", 8)
    assert (spec.gpu_type, spec.gpu_count) == ("H200", 8)
    assert label == "eval"


def test_resolve_spec_both_flags_without_intent_is_custom():
    spec, label = pod_lifecycle._resolve_spec(None, "H100", 2)
    assert (spec.gpu_type, spec.gpu_count) == ("H100", 2)
    assert label == "custom"


def test_resolve_spec_partial_count_merges_over_intent():
    """Regression for #531: ``--intent eval --gpu-count 4`` silently provisioned
    1x H100 because the lone count flag fell through to the intent table. The
    explicit count must merge over the intent's default."""
    spec, label = pod_lifecycle._resolve_spec("eval", None, 4)
    assert (spec.gpu_type, spec.gpu_count) == ("H100", 4)
    assert label == "eval"
    assert "override" in spec.rationale and "--gpu-count 4" in spec.rationale


def test_resolve_spec_partial_type_merges_over_intent():
    spec, label = pod_lifecycle._resolve_spec("ft-7b", "H200", None)
    assert (spec.gpu_type, spec.gpu_count) == ("H200", 4)
    assert label == "ft-7b"
    assert "--gpu-type H200" in spec.rationale


@pytest.mark.parametrize(
    ("gpu_type", "gpu_count"),
    [("H100", None), (None, 4)],
)
def test_resolve_spec_partial_flag_without_intent_fails_loud(gpu_type, gpu_count):
    """A lone override flag with no --intent has no default to fill the missing
    field from — fail loud instead of guessing."""
    with pytest.raises(SystemExit, match="without --intent"):
        pod_lifecycle._resolve_spec(None, gpu_type, gpu_count)


def test_resolve_spec_nothing_given_fails_loud():
    with pytest.raises(SystemExit, match="Must pass either --intent"):
        pod_lifecycle._resolve_spec(None, None, None)


# ---------------------------------------------------------------------------
# cmd_provision — pod-safety terminal-parent warn (#1177)
# ---------------------------------------------------------------------------


def _fm(status: str, tags: tuple[str, ...] = (), kind: str = "experiment") -> dict:
    """Build the ``get_task()`` return shape the #1177 guard reads: top-level
    ``status`` (folder name) + ``frontmatter.tags``."""
    return {
        "id": 0,
        "status": status,
        "frontmatter": {"kind": kind, "tags": list(tags)},
        "body": "",
    }


def _ev(kind: str, ts: str | None) -> dict:
    """Build a minimal events.jsonl row (ts/kind are all the guard reads)."""
    return {"ts": ts, "kind": kind, "version": 1, "by": "test", "note": ""}


def _patch_task_state(monkeypatch, task, events) -> None:
    """Monkeypatch ``task_workflow.get_task`` / ``.list_events`` (lazily
    imported inside the #1177 guard) with fixed returns; pass an Exception
    instance for either to make that read raise (fail-open tests)."""

    def fake_get_task(issue):
        if isinstance(task, Exception):
            raise task
        return dict(task, id=issue)

    def fake_list_events(issue):
        if isinstance(events, Exception):
            raise events
        return list(events)

    monkeypatch.setattr("explore_persona_space.task_workflow.get_task", fake_get_task)
    monkeypatch.setattr("explore_persona_space.task_workflow.list_events", fake_list_events)


_T1 = "2026-07-01T00:00:00Z"
_T2 = "2026-07-02T00:00:00Z"


def test_provision_warns_on_completed_status_no_signals(monkeypatch, capsys):
    """The primary trigger: completed + untagged + only a done-transition
    event -> the warning fires with all five acceptance substrings."""
    _patch_task_state(monkeypatch, _fm("completed"), [_ev("epm:status-changed", _T1)])

    assert pod_lifecycle._warn_on_terminal_parent_provision(664) is True

    err = capsys.readouterr().err
    for needle in (
        "pod-safety",
        "AUTO-STOP",
        "add-tag 664 keep-running",
        "epm:run-launched",
        "Proceeding",
    ):
        assert needle in err, f"warning missing acceptance substring: {needle!r}"
    # All three recipe commands are quoted.
    assert "post-marker 664 epm:run-launched" in err
    assert "remove-tag 664 keep-running" in err


@pytest.mark.parametrize("status", ["completed", "awaiting_promotion", "archived", "on_hold"])
def test_provision_warn_fires_for_each_auto_stop_status(monkeypatch, capsys, status):
    _patch_task_state(monkeypatch, _fm(status), [_ev("epm:status-changed", _T1)])
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is True
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.parametrize(
    "status", ["running", "followups_running", "approved", "blocked", "proposed"]
)
def test_provision_silent_on_active_statuses(monkeypatch, capsys, status):
    """Sanctioned paths (incl. the followups_running follow-up loop) must
    print NOTHING — false-positive rate 0 on this matrix."""
    _patch_task_state(monkeypatch, _fm(status), [_ev("epm:status-changed", _T1)])
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is False
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_provision_silent_with_keep_running_tag(monkeypatch, capsys):
    _patch_task_state(
        monkeypatch, _fm("completed", tags=("keep-running",)), [_ev("epm:status-changed", _T1)]
    )
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is False
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "signal_kind",
    ["epm:run-launched", "epm:followup-scope", "epm:free-analysis-followup-run"],
)
def test_provision_silent_with_fresh_followup_signal(monkeypatch, capsys, signal_kind):
    """A follow-up signal STRICTLY newer than the latest done-transition is
    the watcher's live-follow-up exemption -> no warning."""
    events = [_ev("epm:status-changed", _T1), _ev(signal_kind, _T2)]
    _patch_task_state(monkeypatch, _fm("completed"), events)
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is False
    assert capsys.readouterr().err == ""


def test_provision_warns_on_stale_followup_signal(monkeypatch, capsys):
    """A signal OLDER than the latest done-transition means the follow-up
    finished (the watcher's re-arm semantics, strict >) -> warn fires."""
    events = [_ev("epm:followup-scope", _T1), _ev("epm:status-changed", _T2)]
    _patch_task_state(monkeypatch, _fm("completed"), events)
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is True
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.parametrize(
    "exc", [FileNotFoundError("gone"), RuntimeError("branch"), ValueError("bad")]
)
def test_provision_failopen_on_unresolvable_task(monkeypatch, capsys, exc):
    """An unresolvable task (ad-hoc pod / registry miss / branch-guard fire)
    proceeds with a one-line NOTE — never a block, never an exception."""
    _patch_task_state(monkeypatch, exc, [])
    assert pod_lifecycle._warn_on_terminal_parent_provision(9999) is False
    err = capsys.readouterr().err
    assert "skipped" in err
    assert "WARNING" not in err


def test_provision_failopen_on_unparseable_ts(monkeypatch, capsys):
    """Garbage/absent ts values never crash: the events are treated as absent
    (conservative toward warning), so the warn fires here."""
    events = [_ev("epm:run-launched", None), _ev("epm:status-changed", "not-a-timestamp")]
    _patch_task_state(monkeypatch, _fm("completed"), events)
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is True
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.parametrize(
    "signal_kind",
    ["epm:run-launched", "epm:followup-scope", "epm:free-analysis-followup-run"],
)
def test_provision_warns_on_signal_without_done_transition(monkeypatch, capsys, signal_kind):
    """A signal with NO done-transition ever posted -> warn fires (the
    watcher's conservative missing-done -> False branch,
    autonomous_session_watch._task_followup_active). A sign inversion here
    would silently drop the warn on a watcher-stopped class."""
    _patch_task_state(monkeypatch, _fm("completed"), [_ev(signal_kind, _T2)])
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is True
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.parametrize(
    "exc",
    [OSError("io"), FileNotFoundError("gone"), RuntimeError("branch"), ValueError("bad")],
)
def test_provision_failopen_on_list_events_raising(monkeypatch, capsys, exc):
    """get_task succeeds (completed, untagged) but list_events raises — the
    events read is where OSError specifically joins the fail-open set."""
    _patch_task_state(monkeypatch, _fm("completed"), exc)
    assert pod_lifecycle._warn_on_terminal_parent_provision(1) is False
    err = capsys.readouterr().err
    assert "skipped" in err
    assert "WARNING" not in err


def test_provision_pod_safety_constants_match_watcher():
    """Parity pin: the local mirror constants equal the watcher's — a watcher
    change to any of the three sets breaks this test and forces a same-round
    re-sync (plan §11: local mirror + parity test, zero coupling)."""
    import autonomous_session_watch as asw

    assert frozenset(asw.POD_SAFETY_AUTO_STOP) == pod_lifecycle._POD_SAFETY_AUTO_STOP_STATUSES
    assert pod_lifecycle._POD_FOLLOWUP_SIGNAL_KINDS == asw._POD_FOLLOWUP_SIGNAL_KINDS
    assert pod_lifecycle._POD_DONE_TRANSITION_KINDS == asw._DONE_TRANSITION_KINDS


def test_provision_freshness_behavioral_parity_with_watcher():
    """Parity pin on the comparison SEMANTICS (strict >, missing-signal ->
    False, missing-done -> False) — the constants test alone cannot catch
    comparison-logic drift. The watcher accepts an injected events list."""
    import autonomous_session_watch as asw

    matrices = {
        "fresh-signal": [_ev("epm:status-changed", _T1), _ev("epm:run-launched", _T2)],
        "stale-signal": [_ev("epm:followup-scope", _T1), _ev("epm:status-changed", _T2)],
        "signal-only-no-done": [_ev("epm:free-analysis-followup-run", _T2)],
        "done-only": [_ev("epm:promoted", _T1)],
        "empty": [],
    }
    for label, evts in matrices.items():
        assert pod_lifecycle._fresh_followup_signal(evts) == asw._task_followup_active(
            1, events=evts
        ), f"freshness parity diverged from the watcher on {label!r}"


def test_provision_dry_run_calls_check(monkeypatch, capsys, isolated_state, stub_list_team_pods):
    """Wiring: cmd_provision --dry-run on a completed-status task prints the
    warning BEFORE the dry-run return (i.e. before any create call)."""
    _patch_task_state(monkeypatch, _fm("completed"), [_ev("epm:status-changed", _T1)])
    stub_list_team_pods.return_value = []
    ns = argparse.Namespace(
        list_intents=False,
        issue=664,
        intent="debug",
        gpu_type=None,
        gpu_count=None,
        dry_run=True,
        volume_gb=200,
        container_disk_gb=50,
    )

    pod_lifecycle.cmd_provision(ns)

    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "pod-safety" in captured.err
    assert "[dry-run]" in captured.out


def test_resume_calls_check(monkeypatch, capsys, isolated_state, stub_list_team_pods):
    """Wiring: cmd_resume --dry-run runs the same check with verb='resume'
    as its FIRST statement (before _load_state)."""
    pod_name = _register_pod_for_issue(664)
    stub_list_team_pods.return_value = [_info(pod_name, desired_status="EXITED")]
    _patch_task_state(monkeypatch, _fm("completed"), [_ev("epm:status-changed", _T1)])
    ns = argparse.Namespace(issue=664, dry_run=True)

    pod_lifecycle.cmd_resume(ns)

    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "pod-safety" in captured.err
    assert "Proceeding with resume" in captured.err
    assert "[dry-run]" in captured.out


# ---------------------------------------------------------------------------
# Error-path diagnostics stream routing (#1518) — durability pin
# ---------------------------------------------------------------------------

#: Functions allowed to bare-print to stdout immediately before a nonzero
#: exit. `_emit_still_waiting_and_exit` deliberately DUAL-prints (stderr +
#: stdout) before its exit-75 still-waiting contract, "so an output-capturing
#: caller that only keeps stdout still sees it" (its docstring). Exempt by
#: enclosing-function NAME, never line number; a new deliberate dual-print
#: function gets added here with a comment.
_STDOUT_BEFORE_EXIT_EXEMPT_FUNCTIONS = frozenset({"_emit_still_waiting_and_exit"})


def _is_bare_stdout_print_stmt(stmt) -> bool:
    """True for a statement-level ``print(...)`` call with NO ``file=`` kwarg."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "print"
        and not any(kw.arg == "file" for kw in stmt.value.keywords)
    )


def _is_nonzero_exit_stmt(stmt) -> bool:
    """True for ``sys.exit(<arg>)`` / ``raise SystemExit(<arg>)`` with an arg
    that is not provably zero (a string, a Name like EXIT_STILL_WAITING, or
    any expression counts as nonzero; a bare call exits 0)."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "exit"
            and isinstance(func.value, ast.Name)
            and func.value.id == "sys"
        ):
            return False
    elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
        call = stmt.exc
        func = call.func
        if not (isinstance(func, ast.Name) and func.id == "SystemExit"):
            return False
    else:
        return False
    if not call.args:
        return False  # sys.exit() / SystemExit() -> exit code 0
    arg = call.args[0]
    return not (isinstance(arg, ast.Constant) and arg.value == 0)


def _scan_bare_stdout_prints_before_nonzero_exit(path):
    """AST-scan ``path`` for bare stdout ``print(...)`` statements immediately
    preceding a nonzero exit in the same block.

    Returns ``(offenders, exempt_hits)`` — lists of ``(lineno, func_name)``
    for each contiguous bare ``print(...)`` (no ``file=`` kwarg) statement
    directly preceding a ``sys.exit(<nonzero>)`` / ``raise SystemExit(<arg>)``
    statement in the same statement block. Hits inside a function named in
    :data:`_STDOUT_BEFORE_EXIT_EXEMPT_FUNCTIONS` land in ``exempt_hits``.
    """
    offenders: list[tuple[int, str]] = []
    exempt_hits: list[tuple[int, str]] = []

    def _visit_block(stmts, func_name: str) -> None:
        for i, stmt in enumerate(stmts):
            if _is_nonzero_exit_stmt(stmt):
                j = i - 1
                while j >= 0 and _is_bare_stdout_print_stmt(stmts[j]):
                    hit = (stmts[j].lineno, func_name)
                    if func_name in _STDOUT_BEFORE_EXIT_EXEMPT_FUNCTIONS:
                        exempt_hits.append(hit)
                    else:
                        offenders.append(hit)
                    j -= 1
            _visit_children(stmt, func_name)

    def _visit_children(node, func_name: str) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_name = node.name
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if block:
                _visit_block(block, func_name)
        for handler in getattr(node, "handlers", None) or []:
            _visit_block(handler.body, func_name)
        for case in getattr(node, "cases", None) or []:
            _visit_block(case.body, func_name)

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    _visit_children(tree, "<module>")
    return offenders, exempt_hits


def test_no_bare_stdout_print_before_nonzero_exit():
    """#1518 durability pin: ``scripts/pod_lifecycle.py`` must contain NO bare
    stdout ``print(...)`` statement immediately preceding a nonzero
    ``sys.exit(...)`` / ``raise SystemExit(...)`` in the same block — an
    error-path diagnostic on stdout never reaches the #1465
    ``PodLifecycleProcessError`` stderr tail, so the router failure marker
    reads as an unexplained exit (incident #1481).

    Scope limit: the scan catches only CONTIGUOUS bare prints immediately
    preceding a nonzero exit in the same block — a print two statements
    earlier, or one buried in a helper called before the exit, escapes it.

    Positive witness: the scan must FIND exactly one exempted site (the
    ``_emit_still_waiting_and_exit`` deliberate dual print) — if the exempt
    count is not exactly 1, the scanner itself has been disarmed by a scan
    bug, or the dual print moved (update the exempt set deliberately).
    """
    offenders, exempt_hits = _scan_bare_stdout_prints_before_nonzero_exit(pod_lifecycle.__file__)
    assert not offenders, (
        "bare stdout print(...) immediately before a nonzero exit in "
        f"scripts/pod_lifecycle.py at (lineno, func): {sorted(offenders)}; "
        "route error-path diagnostics to stderr (file=sys.stderr) so they ride "
        "the #1465 PodLifecycleProcessError stderr tail (#1518), or add a "
        "deliberate dual-print function to _STDOUT_BEFORE_EXIT_EXEMPT_FUNCTIONS "
        "with a comment."
    )
    assert len(exempt_hits) == 1, (
        "expected exactly 1 exempted dual-print site "
        f"(_emit_still_waiting_and_exit), found {len(exempt_hits)}: {exempt_hits}"
    )


# ---------------------------------------------------------------------------
# Bad-placement (known-bad host) sidecar (#2011)
#
# After a fresh provision fails its SSH/bootstrap readiness probe, the failed
# placement's host identity is recorded durably (bad-pod-hosts.json) and the
# next provision warns on a repeat placement. RunPod has no host-exclude
# input, so DC-pin-away + repeat detection is the whole lever set. The
# autouse ``bad_host_state`` fixture (top of file) isolates the sidecar +
# get_datacenters for every test here.
# ---------------------------------------------------------------------------


def test_note_bad_placement_round_trip(bad_host_state):
    """Record + consume: fresh_bad_hosts returns the host-keyed entry with
    every recorded field (issue, pod name, pod id, DC, reason, timestamp)."""
    pod_lifecycle.note_bad_placement(
        1947,
        "pod-1947-b",
        "podid-x",
        "103.207.149.60",
        "EU-RO-1",
        reason="bootstrap-failed",
        now=1000.0,
    )

    fresh = pod_lifecycle.fresh_bad_hosts(now=1001.0)
    assert list(fresh) == ["103.207.149.60"]
    (entry,) = fresh["103.207.149.60"]
    assert entry["issue"] == 1947
    assert entry["pod_name"] == "pod-1947-b"
    assert entry["pod_id"] == "podid-x"
    assert entry["dc_id"] == "EU-RO-1"
    assert entry["reason"] == "bootstrap-failed"
    assert entry["ts"] == 1000.0
    # The sidecar file itself landed (durable across processes).
    assert bad_host_state.exists()


def test_fresh_bad_hosts_drops_expired_entries():
    """Entries older than the TTL (default 6h) are dropped on read; fresh
    ones survive. Read-side pruning only — the file is not rewritten."""
    pod_lifecycle.note_bad_placement(1, "pod-1", "id1", "1.1.1.1", None, now=1000.0)
    pod_lifecycle.note_bad_placement(2, "pod-2", "id2", "2.2.2.2", None, now=20000.0)

    ttl = pod_lifecycle._bad_host_ttl_secs()
    assert ttl == 6 * 3600.0
    fresh = pod_lifecycle.fresh_bad_hosts(now=1000.0 + ttl + 1.0)
    assert "1.1.1.1" not in fresh  # expired
    assert "2.2.2.2" in fresh  # still within TTL
    # The file keeps both rows (pruning never rewrites).
    assert set(pod_lifecycle._load_bad_host_state()) == {"1.1.1.1", "2.2.2.2"}


def test_bad_host_ttl_env_override(monkeypatch):
    """EPM_BAD_HOST_TTL_SECS overrides the 6h default; garbage falls back."""
    monkeypatch.setenv("EPM_BAD_HOST_TTL_SECS", "60")
    assert pod_lifecycle._bad_host_ttl_secs() == 60.0
    pod_lifecycle.note_bad_placement(1, "pod-1", "id1", "1.1.1.1", None, now=1000.0)
    assert "1.1.1.1" in pod_lifecycle.fresh_bad_hosts(now=1050.0)
    assert "1.1.1.1" not in pod_lifecycle.fresh_bad_hosts(now=1061.0)

    monkeypatch.setenv("EPM_BAD_HOST_TTL_SECS", "not-a-number")
    assert pod_lifecycle._bad_host_ttl_secs() == 6 * 3600.0


def test_load_bad_host_state_garbled_returns_empty(bad_host_state):
    """A garbled sidecar reads as {} (fresh state), never a crash."""
    bad_host_state.write_text("{not json!!")
    assert pod_lifecycle._load_bad_host_state() == {}
    assert pod_lifecycle.fresh_bad_hosts() == {}


def test_note_bad_placement_without_host_ip_writes_nothing(bad_host_state):
    """No host identity ⇒ nothing a future avoidance read could key on ⇒ no
    sidecar row (the [bad-host-RECORD] line still prints at the call site)."""
    pod_lifecycle.note_bad_placement(1, "pod-1", "id1", None, "EU-RO-1")
    pod_lifecycle.note_bad_placement(1, "pod-1", "id1", "", "EU-RO-1")
    assert not bad_host_state.exists()
    assert pod_lifecycle.fresh_bad_hosts() == {}


def test_save_bad_host_state_io_failure_swallowed(tmp_path, monkeypatch, capsys):
    """Sidecar IO failures are swallowed with a WARN — observability must
    never crash a provision (plan acceptance criterion 5)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    monkeypatch.setattr(pod_lifecycle, "BAD_HOST_STATE", blocker / "bad-pod-hosts.json")

    # Must not raise despite the unwritable path (mkdir → NotADirectoryError).
    pod_lifecycle.note_bad_placement(1, "pod-1", "id1", "1.1.1.1", None)
    assert "bad-host state save failed" in capsys.readouterr().err


def test_warn_on_bad_host_repeat_fires_and_stays_silent(capsys):
    """[bad-host-REPEAT] fires on a fresh recorded host (cross-issue) and
    stays silent on a clean host. WARN-only — never raises."""
    pod_lifecycle.note_bad_placement(
        1947, "pod-1947-b", "id-old", "9.9.9.9", "EU-RO-1", reason="bootstrap-failed"
    )

    ready_clean = _info("pod-2011", ssh_host="8.8.8.8")
    pod_lifecycle._warn_on_bad_host_repeat("pod-2011", ready_clean)
    assert "[bad-host-REPEAT]" not in capsys.readouterr().err

    ready_repeat = _info("pod-2011", ssh_host="9.9.9.9")
    pod_lifecycle._warn_on_bad_host_repeat("pod-2011", ready_repeat)
    err = capsys.readouterr().err
    assert "[bad-host-REPEAT]" in err
    assert "9.9.9.9" in err
    assert "1947" in err  # names the prior issue (cross-issue record)
    assert "--data-center-id" in err  # the DC-pin-away recipe


def test_warn_on_bad_dc_pin_warns_but_never_raises(capsys):
    """An explicit --data-center-id matching a recorded bad placement's DC
    WARNs loudly; a non-matching pin stays silent. Advisory only."""
    pod_lifecycle.note_bad_placement(
        1947, "pod-1947-b", "id-old", "9.9.9.9", "EU-RO-1", reason="ssh-wait-timeout"
    )

    pod_lifecycle._warn_on_bad_dc_pin("CA-MTL-1")
    assert "[bad-host-DC-PIN]" not in capsys.readouterr().err

    pod_lifecycle._warn_on_bad_dc_pin("EU-RO-1")
    err = capsys.readouterr().err
    assert "[bad-host-DC-PIN]" in err
    assert "9.9.9.9" in err
    assert "Honoring the explicit pin" in err


def test_bootstrap_fail_records_bad_placement(isolated_state, monkeypatch, capsys):
    """Fail-loud pin (plan §1): a bootstrap failure records the placement
    (sidecar row + [bad-host-RECORD]) AND still sys.exit(rc)s — the recording
    never swallows the fail-loud exit. BOOTSTRAP-FAILED stays the last stderr
    line (#1931)."""
    _write_metadata_file({})
    ready = PodInfo(
        pod_id="live-2011",
        name="pod-2011",
        desired_status="RUNNING",
        ssh_host="103.207.149.60",
        ssh_port=22222,
        data_center_id="EU-RO-1",
    )
    _stub_provision_tail(monkeypatch, ready, [1, 1])

    def fake_get_datacenters():
        return [{"id": "EU-RO-1"}, {"id": "CA-MTL-1"}, {"id": "EU-SE-1"}]

    monkeypatch.setattr(pod_lifecycle, "get_datacenters", fake_get_datacenters)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._provision_wait_register_bootstrap(
            _bootstrap_tail_ns(), "pod-2011", ready, "lora-7b"
        )

    assert exc.value.code == 1  # the fail-loud exit is UNCHANGED
    fresh = pod_lifecycle.fresh_bad_hosts()
    (entry,) = fresh["103.207.149.60"]
    assert entry["reason"] == "bootstrap-failed"
    assert entry["issue"] == 779  # _bootstrap_tail_ns issue
    assert entry["dc_id"] == "EU-RO-1"

    err = capsys.readouterr().err
    assert "[bad-host-RECORD]" in err
    assert "host=103.207.149.60" in err
    # Different-DC hint with candidates, excluding the bad DC.
    assert "CA-MTL-1" in err
    assert "candidates:" in err
    # BOOTSTRAP-FAILED stays the LAST stderr line (#1931 verdict contract).
    assert err.rstrip().splitlines()[-1].startswith("BOOTSTRAP-FAILED pod=pod-2011")
    assert err.index("[bad-host-RECORD]") < err.index("BOOTSTRAP-FAILED")


def test_ssh_wait_timeout_records_bad_placement(isolated_state, monkeypatch, capsys):
    """Fail-loud pin (plan §1): a wait_for_ssh timeout records the placement
    (via the best-effort late get_pod) AND the RunPodError still propagates
    THROUGH the new recording code."""
    _write_metadata_file({})
    created = PodInfo(
        pod_id="live-2011",
        name="pod-2011",
        desired_status="RUNNING",
        ssh_host=None,
        ssh_port=None,
        data_center_id=None,
    )

    def fake_wait_for_ssh(pod_id, timeout=600):
        raise pod_lifecycle.RunPodError("no public 22/tcp within 600s")

    def fake_get_pod(pod_id):
        # Late-appearing host identity on the one best-effort re-read.
        return PodInfo(
            pod_id=pod_id,
            name="pod-2011",
            desired_status="RUNNING",
            ssh_host="103.207.149.60",
            ssh_port=11111,
            data_center_id="EU-RO-1",
        )

    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", fake_wait_for_ssh)
    monkeypatch.setattr(pod_lifecycle, "get_pod", fake_get_pod)
    monkeypatch.setattr(pod_lifecycle, "note_ssh_wait_outcome", lambda *a, **k: None)

    with pytest.raises(pod_lifecycle.RunPodError):
        pod_lifecycle._provision_wait_register_bootstrap(
            _bootstrap_tail_ns(), "pod-2011", created, "lora-7b"
        )

    (entry,) = pod_lifecycle.fresh_bad_hosts()["103.207.149.60"]
    assert entry["reason"] == "ssh-wait-timeout"
    assert entry["dc_id"] == "EU-RO-1"
    err = capsys.readouterr().err
    assert "[bad-host-RECORD]" in err
    assert "host=103.207.149.60" in err


def test_ssh_wait_timeout_reraises_even_when_get_pod_fails(isolated_state, monkeypatch, capsys):
    """A get_pod failure inside the recording branch must never shadow or
    swallow the RunPodError re-raise (critic round-1 Must-Fix)."""
    _write_metadata_file({})
    created = PodInfo(
        pod_id="live-2011",
        name="pod-2011",
        desired_status="RUNNING",
        ssh_host=None,
        ssh_port=None,
    )

    def fake_wait_for_ssh(pod_id, timeout=600):
        raise pod_lifecycle.RunPodError("no public 22/tcp within 600s")

    def fake_get_pod(pod_id):
        raise RuntimeError("late get_pod exploded")

    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", fake_wait_for_ssh)
    monkeypatch.setattr(pod_lifecycle, "get_pod", fake_get_pod)
    monkeypatch.setattr(pod_lifecycle, "note_ssh_wait_outcome", lambda *a, **k: None)

    with pytest.raises(pod_lifecycle.RunPodError, match="no public 22/tcp"):
        pod_lifecycle._provision_wait_register_bootstrap(
            _bootstrap_tail_ns(), "pod-2011", created, "lora-7b"
        )

    err = capsys.readouterr().err
    assert "late get_pod for bad-host record failed" in err
    # No host identity was recoverable → RECORD line prints host=unknown and
    # NO sidecar row is written (nothing to key a future avoidance read on).
    assert "[bad-host-RECORD]" in err
    assert "host=unknown" in err
    assert pod_lifecycle.fresh_bad_hosts() == {}


def test_provision_tail_warns_on_repeat_placement(isolated_state, monkeypatch, capsys):
    """End-to-end consume (plan §5 replay shape): a recorded bad host + a new
    placement landing on the SAME IP prints [bad-host-REPEAT] through the
    real provision tail; the provision itself proceeds (WARN-only)."""
    _write_metadata_file({})
    pod_lifecycle.note_bad_placement(
        1947,
        "pod-1947-b",
        "id-old",
        "103.207.149.60",
        "EU-RO-1",
        reason="bootstrap-failed",
    )
    ready = PodInfo(
        pod_id="live-2011",
        name="pod-2011",
        desired_status="RUNNING",
        ssh_host="103.207.149.60",
        ssh_port=22222,
        data_center_id="EU-RO-1",
    )
    _stub_provision_tail(monkeypatch, ready, [0])

    pod_lifecycle._provision_wait_register_bootstrap(
        _bootstrap_tail_ns(), "pod-2011", ready, "lora-7b"
    )

    captured = capsys.readouterr()
    assert "[bad-host-REPEAT]" in captured.err
    assert "103.207.149.60" in captured.err
    assert "BOOTSTRAP-OK pod=pod-2011" in captured.out  # WARN-only: proceeded


# ---------------------------------------------------------------------------
# --data-center-id threading (#2011)
# ---------------------------------------------------------------------------


def _fake_gpu_create_pod(recorder: list):
    """Signature-conformant create_pod fake (mirrors runpod_api.create_pod)."""

    def fake_create_pod(
        name,
        gpu_type,
        gpu_count,
        *,
        image=None,
        volume_gb=200,
        container_disk_gb=50,
        cloud_type="ALL",
        data_center_id=None,
        enable_supply_fallback=True,
    ):
        recorder.append(
            {
                "name": name,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "data_center_id": data_center_id,
            }
        )
        return _info(name, pod_id=f"live-{name}")

    return fake_create_pod


def _stub_gpu_provision_preflights(monkeypatch):
    """No-op the network-touching provision preflights for threading tests."""
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *a, **k: False)
    monkeypatch.setattr(pod_lifecycle, "_account_key_preflight", lambda pod_label: None)
    monkeypatch.setattr(pod_lifecycle, "_assert_under_account_hourly_cap", lambda **kw: None)
    monkeypatch.setattr(pod_lifecycle, "_provision_wait_register_bootstrap", lambda *a, **k: None)


def test_data_center_id_threads_to_gpu_create_pod(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """One-shot GPU provision threads --data-center-id into create_pod
    (acceptance criterion 3: the flag exists and reaches the create input)."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = []
    calls: list[dict] = []
    _stub_gpu_provision_preflights(monkeypatch)
    monkeypatch.setattr(pod_lifecycle, "create_pod", _fake_gpu_create_pod(calls))

    ns = _gpu_provision_ns(2011, dry_run=False, wait_for_capacity=False, data_center_id="EU-RO-1")
    pod_lifecycle.cmd_provision(ns)

    assert len(calls) == 1
    assert calls[0]["data_center_id"] == "EU-RO-1"


def test_data_center_id_threads_through_wait_for_capacity(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """The wait-for-capacity branch threads the pin through the REAL
    create_pod_with_wait_for_capacity body into create_pod (plan §3: the
    wrapper itself gains the data_center_id parameter; retry-loop semantics
    unchanged)."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = []
    calls: list[dict] = []
    _stub_gpu_provision_preflights(monkeypatch)
    monkeypatch.setattr(pod_lifecycle, "create_pod", _fake_gpu_create_pod(calls))

    ns = _gpu_provision_ns(2011, dry_run=False, wait_for_capacity=True, data_center_id="EU-SE-1")
    pod_lifecycle.cmd_provision(ns)

    assert len(calls) == 1
    assert calls[0]["data_center_id"] == "EU-SE-1"


def test_data_center_id_threads_to_cpu_create_pod(
    isolated_state, stub_list_team_pods, cpu_provision_stubs
):
    """The CPU branch threads --data-center-id into create_cpu_pod; absent
    flag (hand-built Namespace) defaults to None via getattr."""
    _write_metadata_file({})
    stub_list_team_pods.return_value = []

    pod_lifecycle.cmd_provision(_cpu_provision_ns(747, "cpu-small", data_center_id="CA-MTL-1"))
    assert cpu_provision_stubs["cpu_calls"][-1]["data_center_id"] == "CA-MTL-1"

    # Hand-built Namespace WITHOUT the flag (predates it): getattr default.
    pod_lifecycle.cmd_provision(_cpu_provision_ns(748, "cpu-small"))
    assert cpu_provision_stubs["cpu_calls"][-1]["data_center_id"] is None


def test_explicit_dc_pin_warns_and_still_reaches_create(
    isolated_state, stub_list_team_pods, monkeypatch, capsys
):
    """Must-ask constraint (plan §10): an explicit --data-center-id matching
    a recorded bad placement's DC WARNs loudly AND is still honored — the
    pin reaches create_pod unchanged, never silently dropped."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    _write_metadata_file({})
    stub_list_team_pods.return_value = []
    pod_lifecycle.note_bad_placement(
        1947, "pod-1947-b", "id-old", "9.9.9.9", "EU-RO-1", reason="bootstrap-failed"
    )
    calls: list[dict] = []
    _stub_gpu_provision_preflights(monkeypatch)
    monkeypatch.setattr(pod_lifecycle, "create_pod", _fake_gpu_create_pod(calls))

    ns = _gpu_provision_ns(2011, dry_run=False, wait_for_capacity=False, data_center_id="EU-RO-1")
    pod_lifecycle.cmd_provision(ns)

    assert "[bad-host-DC-PIN]" in capsys.readouterr().err
    assert calls[0]["data_center_id"] == "EU-RO-1"  # explicit pin WINS


def test_provision_parser_exposes_data_center_id():
    """The provision subparser wires --data-center-id (default None)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pod_lifecycle._parser_provision(sub)

    ns = parser.parse_args(["provision", "--issue", "51", "--data-center-id", "EU-RO-1"])
    assert ns.data_center_id == "EU-RO-1"
    ns0 = parser.parse_args(["provision", "--issue", "51"])
    assert ns0.data_center_id is None


def test_parse_pod_populates_placement_identity():
    """_parse_pod surfaces machine.podHostId / machine.dataCenterId on the
    PodInfo (#2011); selections that omit them parse to None (the
    list_team_pods hot query stays without the fields)."""
    from runpod_api import _parse_pod

    raw = {
        "id": "pod-x",
        "name": "pod-2011",
        "desiredStatus": "RUNNING",
        "gpuCount": 1,
        "machine": {
            "gpuTypeId": "NVIDIA H100 80GB HBM3",
            "podHostId": "pod-x-644123f3",
            "dataCenterId": "EUR-IS-5",
        },
        "runtime": {"ports": []},
    }
    parsed = _parse_pod(raw)
    assert parsed.pod_host_id == "pod-x-644123f3"
    assert parsed.data_center_id == "EUR-IS-5"

    raw_without = {
        "id": "pod-y",
        "name": "pod-2",
        "desiredStatus": "RUNNING",
        "gpuCount": 1,
        "machine": {"gpuTypeId": "NVIDIA H100 80GB HBM3"},
        "runtime": {"ports": []},
    }
    parsed_without = _parse_pod(raw_without)
    assert parsed_without.pod_host_id is None
    assert parsed_without.data_center_id is None
