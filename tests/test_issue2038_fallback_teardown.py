"""#2038 — terminate pod on failed RunPod fallback launch.

Two components under test (plan #2038 §3):

* **Component 1** (``backends/runpod.py``): the exact-pod-id round-trip from
  ``pods_ephemeral.json`` into ``handle.extra["pod_id"]`` (omit-when-absent),
  and the post-provision protective wrapper in ``RunPodBackend.launch`` — a
  non-``RunPodWorkloadStartError`` exception after the provision subprocess
  best-effort terminates the just-provisioned pod BY EXACT ID and re-raises
  the ORIGINAL exception; the #954 ``RunPodWorkloadStartError`` diagnosis-lane
  path stays byte-unchanged in behavior (regression-pinned here).

* **Component 2** (``backends/issue_dispatch.py``): the superseded-fallback
  reap at ``dispatch_for_issue``'s prior-sidecar snapshot seam — the pure
  decision table (:func:`decide_superseded_runpod_reap`), the effectful
  best-effort wrapper (injectable seams, network-free here), and the
  end-to-end wiring incl. the ``superseded_runpod_reaped`` audit record on
  the new handle's extra.

Every test is hermetic: the provision subprocess, SSH exec leg, RunPod
GraphQL API, and keep-running tag reads are all injected/monkeypatched at
their boundaries; the production bodies of every #2038-added function are
executed for real (code-style § one production-body test per seam-stubbed
function).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pytest

from explore_persona_space.backends import issue_dispatch as idp
from explore_persona_space.backends import runpod as RP
from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
from explore_persona_space.backends.base import RunHandle, RunSpec
from explore_persona_space.backends.issue_dispatch import (
    SupersededReapDecision,
    decide_superseded_runpod_reap,
)
from explore_persona_space.backends.kill_approval import verified_teardown_active

# Mirror of tests/test_runpod_workload_exec.py::_PRE_954_SUCCESS_EXTRA_KEYS
# (declared locally — test modules are not importable as a package). The
# #2038 ``pod_id`` key is CONDITIONAL (omit-when-absent), exactly like the
# #1118 footprint keys; the exact-set assertions below pin BOTH shapes.
_PRE_954_SUCCESS_EXTRA_KEYS = frozenset(
    {
        "intent",
        "issue",
        "pid_file",
        "runpod_attempt_id",
        "workload_cmd",
        "hydra_args",
        "gpus",
        "time_budget_hours",
        "repo_branch",
        "workload_executed",
        EXPECTED_ARTIFACTS_HANDLE_KEY,
    }
)


def _explode(*_a, **_k):
    raise AssertionError("seam must NOT be reached on this path")


def _spec(*, extra: dict | None = None, **overrides) -> RunSpec:
    return RunSpec(
        issue=2038,
        intent="lora-7b",
        backend="runpod",
        workload_cmd="bash scripts/issue2038_dispatch.sh",
        extra=extra or {},
        **overrides,
    )


@pytest.fixture(autouse=True)
def _tmp_pods_ephemeral(tmp_path, monkeypatch):
    """Point the live pods_ephemeral.json resolver at a per-test tmp file.

    ``pod_config.PODS_EPHEMERAL_JSON`` is the documented monkeypatch seam
    (returned verbatim by ``resolve_live_pods_ephemeral`` when it differs
    from the tracked seed) — keeps every ``launch()`` here off the shared-VM
    live sidecar while still executing ``_provisioned_pod_id``'s REAL body.
    """
    import scripts.pod_config as pod_config

    p = tmp_path / "pods_ephemeral.json"
    p.write_text('{"version": 2, "pods": {}}', encoding="utf-8")
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", p)
    return p


def _seed_pod_row(sidecar, pod_name: str, pod_id: str) -> None:
    sidecar.write_text(
        json.dumps({"version": 2, "pods": {pod_name: {"pod_id": pod_id, "issue": 2038}}}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Component 1 — pod-id round-trip (§5 row C1-roundtrip)
# ---------------------------------------------------------------------------


def test_provisioned_pod_id_reads_live_sidecar(_tmp_pods_ephemeral):
    """Real body: the id comes from the pods_ephemeral.json row, or None."""
    _seed_pod_row(_tmp_pods_ephemeral, "pod-2038", "rpabc123")
    assert RP._provisioned_pod_id("pod-2038") == "rpabc123"
    # Row absent -> None (id-less fallback).
    assert RP._provisioned_pod_id("pod-9999") is None


def test_provisioned_pod_id_degrades_on_missing_or_malformed(tmp_path, monkeypatch, caplog):
    import scripts.pod_config as pod_config

    # Missing file -> None + WARN, never a raise.
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", tmp_path / "absent.json")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.backends.runpod"):
        assert RP._provisioned_pod_id("pod-2038") is None
    assert "pods_ephemeral.json" in caplog.text

    # Malformed JSON -> None, never a raise.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", bad)
    assert RP._provisioned_pod_id("pod-2038") is None


def test_launch_extra_carries_pod_id_when_captured(monkeypatch, _tmp_pods_ephemeral):
    """Success-path handle extra == the pre-#954 exact set PLUS pod_id —
    the deliberate additive update to the exact-set pin (#2038)."""
    _seed_pod_row(_tmp_pods_ephemeral, "pod-2038", "rpabc123")
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    handle = RP.RunPodBackend().launch(_spec())
    assert handle.extra["pod_id"] == "rpabc123"
    assert set(handle.extra.keys()) == _PRE_954_SUCCESS_EXTRA_KEYS | {"pod_id"}
    # job_id semantics unchanged (#1122 empty-job_id = no-match contract).
    assert handle.job_id == ""


def test_launch_extra_omits_pod_id_when_absent(monkeypatch):
    """Omit-when-absent: an empty live sidecar keeps the legacy exact set."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    handle = RP.RunPodBackend().launch(_spec())
    assert "pod_id" not in handle.extra
    assert set(handle.extra.keys()) == _PRE_954_SUCCESS_EXTRA_KEYS


# ---------------------------------------------------------------------------
# Component 1 — post-provision emergency teardown (§5 rows C1-terminate,
# C1-no-id, C1-exec-succeeded, C1-regression, mask guard)
# ---------------------------------------------------------------------------


def test_post_provision_failure_terminates_by_exact_id_and_reraises(monkeypatch):
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: "rpDEAD1")

    def _mint_boom():
        raise RuntimeError("mint exploded")

    monkeypatch.setattr(RP, "mint_runpod_attempt_id", _mint_boom)
    calls: list[dict] = []
    monkeypatch.setattr(
        RP,
        "_terminate_just_provisioned",
        lambda **kw: calls.append(kw) or True,
    )
    with pytest.raises(RuntimeError, match="mint exploded"):
        RP.RunPodBackend().launch(_spec())
    assert len(calls) == 1
    assert calls[0]["pod_id"] == "rpDEAD1"
    assert calls[0]["pod_name"] == "pod-2038"
    assert calls[0]["issue"] == 2038
    assert "mint exploded" in calls[0]["cause"]


def test_post_provision_failure_without_pod_id_is_loud_noop(monkeypatch, caplog):
    """No captured id -> NEVER a name-keyed / issue-wide terminate; loud log."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: None)
    monkeypatch.setattr(RP, "mint_runpod_attempt_id", _explode)
    import scripts.runpod_api as runpod_api

    monkeypatch.setattr(runpod_api, "terminate_pod", _explode)
    with (
        caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.runpod"),
        pytest.raises(AssertionError, match="seam must NOT"),
    ):
        RP.RunPodBackend().launch(_spec())
    assert "cannot terminate by exact id" in caplog.text
    assert "pod.py terminate --issue 2038" in caplog.text


def test_post_start_failure_never_terminates_running_workload(monkeypatch, caplog):
    """Exec leg SUCCEEDED, later failure -> pod left RUNNING, loud log."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: "rpLIVE")
    monkeypatch.setattr(RP, "_execute_workload_on_pod", lambda spec, **kw: {"workload_pid": 777})
    monkeypatch.setattr(RP, "_terminate_just_provisioned", _explode)
    # Make the terminal handle build raise AFTER the workload started: the
    # launch-local `from ... import build_expected_artifacts_declaration`
    # resolves at call time, so patching the artifacts module attr lands.
    import explore_persona_space.backends.artifacts as artifacts

    def _declaration_boom(**_kw):
        raise RuntimeError("handle build exploded")

    monkeypatch.setattr(artifacts, "build_expected_artifacts_declaration", _declaration_boom)
    with (
        caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.runpod"),
        pytest.raises(RuntimeError, match="handle build exploded"),
    ):
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    assert "left RUNNING because its workload already started" in caplog.text


def test_workload_start_error_diagnosis_lane_unchanged(monkeypatch):
    """C1-regression pin (plan §5): the #954 path is behavior-unchanged —
    partial handle attached, NO terminate, pod stays RUNNING for diagnosis."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: "rpDIAG")
    monkeypatch.setattr(RP, "_terminate_just_provisioned", _explode)

    def _exec_boom(spec, **kw):
        raise RP.RunPodWorkloadStartError("branch sync of pod-2038 timed out")

    monkeypatch.setattr(RP, "_execute_workload_on_pod", _exec_boom)
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    partial = ei.value.handle
    assert partial is not None
    assert partial.extra["workload_executed"] is False
    assert "timed out" in partial.extra["workload_start_error"]
    # The partial handle ALSO carries the captured id (diagnosis aid).
    assert partial.extra["pod_id"] == "rpDIAG"


def test_mask_guard_original_exception_never_swallowed(monkeypatch, caplog):
    """Fail-loud negative control (plan §5): even a RAISING teardown never
    masks the ORIGINAL launch exception."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: "rpDEAD2")

    def _mint_boom():
        raise ValueError("original-cause")

    monkeypatch.setattr(RP, "mint_runpod_attempt_id", _mint_boom)

    def _teardown_boom(**_kw):
        raise RuntimeError("teardown exploded")

    monkeypatch.setattr(RP, "_terminate_just_provisioned", _teardown_boom)
    with (
        caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.runpod"),
        pytest.raises(ValueError, match="original-cause"),
    ):
        RP.RunPodBackend().launch(_spec())
    assert "emergency teardown wrapper itself raised" in caplog.text


def test_terminate_just_provisioned_exact_id_under_grant():
    """Real body: terminate by EXACT id, inside the verified_teardown grant."""
    calls: list[str] = []

    def fake_terminate(pod_id: str) -> bool:
        assert verified_teardown_active(), "terminate must run under the owner-driven grant"
        calls.append(pod_id)
        return True

    ok = RP._terminate_just_provisioned(
        pod_id="rpX1", pod_name="pod-2038", issue=2038, cause="c", terminate_fn=fake_terminate
    )
    assert ok is True
    assert calls == ["rpX1"]
    assert not verified_teardown_active(), "grant must not leak past the call"


def test_terminate_just_provisioned_failure_never_raises(caplog):
    def fake_terminate(pod_id: str) -> bool:
        raise RuntimeError("api down")

    with caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.runpod"):
        ok = RP._terminate_just_provisioned(
            pod_id="rpX2", pod_name="pod-2038", issue=2038, cause="c", terminate_fn=fake_terminate
        )
    assert ok is False
    assert "FAILED" in caplog.text


def test_terminate_just_provisioned_no_id_is_noop():
    ok = RP._terminate_just_provisioned(
        pod_id=None, pod_name="pod-2038", issue=2038, cause="c", terminate_fn=_explode
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Component 2 — pure decision table (network-free; §5 rows C2-*)
# ---------------------------------------------------------------------------


def _decide(**over) -> SupersededReapDecision:
    kw: dict = dict(
        issue=2038,
        prior_backend="runpod",
        prior_pod_name="pod-2038",
        prior_pod_id="rpOLD",
        prior_workload_executed=False,
        prior_workload_start_error="RunPodWorkloadStartError: branch sync timed out",
        new_backend="fellows",
        new_pod_name=None,
        new_pod_id=None,
        live_matches_fn=lambda: [("rpOLD", "RUNNING")],
        keep_running_fn=lambda: False,
    )
    kw.update(over)
    return decide_superseded_runpod_reap(**kw)


def test_decide_terminate_on_workload_start_error():
    d = _decide()
    assert d.action == "terminate"
    assert d.target_pod_id == "rpOLD"
    assert d.surface_marker is False


def test_decide_stop_on_workload_executed_true():
    """Critic note 3: the workload_executed-true arm stays a reversible STOP."""
    d = _decide(prior_workload_start_error=None, prior_workload_executed=True)
    assert d.action == "stop"
    assert d.target_pod_id == "rpOLD"


def test_decide_provision_only_skips_with_marker_when_running():
    d = _decide(prior_workload_start_error=None, prior_workload_executed=False)
    assert d.action == "skip-provision-only"
    assert d.surface_marker is True  # RUNNING match -> durable marker owed


def test_decide_provision_only_no_marker_when_not_running():
    d = _decide(
        prior_workload_start_error=None,
        prior_workload_executed=False,
        live_matches_fn=lambda: [("rpOLD", "EXITED")],
    )
    assert d.action == "skip-provision-only"
    assert d.surface_marker is False


def test_decide_keep_running_tag_blocks_destruction():
    d = _decide(keep_running_fn=lambda: True)
    assert d.action == "skip-keep-running"
    assert d.surface_marker is True


def test_decide_keep_running_unreadable_fails_closed():
    d = _decide(keep_running_fn=lambda: None)
    assert d.action == "skip-keep-running-unreadable"
    assert d.surface_marker is True


def test_decide_prior_not_runpod_is_cheap_skip():
    """No live read, no tag read — the common non-RunPod prior."""
    d = _decide(prior_backend="nibi", live_matches_fn=_explode, keep_running_fn=_explode)
    assert d.action == "skip-prior-not-runpod"


@pytest.mark.parametrize("name", ["pod-99", "rogue-pod", "", "pod20380", "pod-2038x"])
def test_decide_unmanaged_or_foreign_name_never_touched(name):
    d = _decide(prior_pod_name=name, live_matches_fn=_explode, keep_running_fn=_explode)
    assert d.action == "skip-unmanaged-name"


def test_decide_name_grammar_pod17_vs_pod1739():
    """#1334 one-parser grammar: pod-1739 belongs to 1739, never to 17."""
    d = _decide(issue=17, prior_pod_name="pod-1739", live_matches_fn=_explode)
    assert d.action == "skip-unmanaged-name"
    d2 = _decide(issue=1739, prior_pod_name="pod-1739")
    assert d2.action == "terminate"
    # Suffixed multi-pod form parses to the same owner.
    d3 = _decide(issue=1739, prior_pod_name="pod-1739-b")
    assert d3.action == "terminate"


def test_decide_same_pod_identity_never_destroys_new_launch():
    d = _decide(
        new_backend="runpod",
        new_pod_name="pod-2038",
        new_pod_id="rpOLD",
        live_matches_fn=_explode,
        keep_running_fn=_explode,
    )
    assert d.action == "skip-same-pod"


def test_decide_live_read_failed_skips():
    d = _decide(live_matches_fn=lambda: None, keep_running_fn=_explode)
    assert d.action == "skip-live-read-failed"


def test_decide_pod_gone_by_id():
    d = _decide(live_matches_fn=lambda: [("rpOTHER", "RUNNING")], keep_running_fn=_explode)
    assert d.action == "skip-pod-gone"


def test_decide_idless_zero_matches_pod_gone():
    d = _decide(prior_pod_id=None, live_matches_fn=lambda: [], keep_running_fn=_explode)
    assert d.action == "skip-pod-gone"


def test_decide_idless_single_match_acts():
    """Legacy id-less prior x exactly 1 live match (new launch elsewhere) -> act."""
    d = _decide(prior_pod_id=None, live_matches_fn=lambda: [("rpX", "RUNNING")])
    assert d.action == "terminate"
    assert d.target_pod_id == "rpX"


def test_decide_idless_excludes_new_pod_id_then_acts():
    d = _decide(
        prior_pod_id=None,
        new_backend="runpod",
        new_pod_name="pod-2038",
        new_pod_id="rpNEW",
        live_matches_fn=lambda: [("rpNEW", "RUNNING"), ("rpOLD", "RUNNING")],
    )
    assert d.action == "terminate"
    assert d.target_pod_id == "rpOLD"


def test_decide_idless_same_name_unknown_new_id_skips_with_marker():
    d = _decide(
        prior_pod_id=None,
        new_backend="runpod",
        new_pod_name="pod-2038",
        new_pod_id=None,
        live_matches_fn=lambda: [("rpX", "RUNNING")],
        keep_running_fn=_explode,
    )
    assert d.action == "skip-new-pod-id-unknown"
    assert d.surface_marker is True


def test_decide_idless_ambiguous_matches_skip_with_marker():
    """Critic note 1 row: the #1739 duplicate-name replay — id-less prior,
    >1 live exact-name matches, ≥1 RUNNING -> skip + durable marker."""
    d = _decide(
        prior_pod_id=None,
        live_matches_fn=lambda: [("rpA", "RUNNING"), ("rpB", "EXITED")],
        keep_running_fn=_explode,
    )
    assert d.action == "skip-name-ambiguous"
    assert d.surface_marker is True


# ---------------------------------------------------------------------------
# Component 2 — effectful wrapper (injected seams; network-free)
# ---------------------------------------------------------------------------


def _prior_runpod_handle(*, issue: int = 2038, extra: dict | None = None) -> RunHandle:
    base_extra = {
        "issue": issue,
        "workload_cmd": "bash scripts/x.sh",
        "hydra_args": [],
        "workload_executed": False,
        "workload_start_error": "RunPodWorkloadStartError: start leg died",
        "pod_id": "rpOLD",
    }
    if extra is not None:
        base_extra.update(extra)
    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name=f"pod-{issue}",
        scratch_dir="/workspace",
        log_path=f"/workspace/logs/issue-{issue}.log",
        extra=base_extra,
    )


def _prior_dict(tmp_path, handle: RunHandle) -> dict:
    """Round-trip through the REAL sidecar snapshot helper (#2038 handle key)."""
    sidecar = tmp_path / f"issue-{handle.extra['issue']}-handle.json"
    idp.write_handle_sidecar(handle, sidecar)
    prior = idp._prior_sidecar_failover_extras(sidecar)
    assert prior is not None
    return prior


def _new_handle(*, backend: str = "fellows", issue: int = 2038, extra: dict | None = None):
    return RunHandle(
        backend=backend,
        cluster=backend if backend != "runpod" else None,
        job_id="job-9",
        pod_name=f"pod-{issue}" if backend == "runpod" else None,
        scratch_dir="/s",
        log_path="/l",
        extra={"issue": issue, **(extra or {})},
    )


def test_prior_sidecar_snapshot_carries_full_handle(tmp_path):
    """#2038: the snapshot's additive ``handle`` key exposes the failure-path
    extras the #1122 ``extra`` filter deliberately drops."""
    prior = _prior_dict(tmp_path, _prior_runpod_handle())
    h = prior["handle"]
    assert isinstance(h, RunHandle)
    assert h.extra["pod_id"] == "rpOLD"
    assert h.extra["workload_start_error"].startswith("RunPodWorkloadStartError")
    # The #1122 filtered extra does NOT carry the failure-path keys.
    assert "workload_start_error" not in prior["extra"]


def test_wrapper_terminate_records_audit_and_posts_marker(tmp_path):
    prior = _prior_dict(tmp_path, _prior_runpod_handle())
    terminated: list[str] = []
    notes: list[tuple[int, str]] = []

    def fake_terminate(pod_id: str) -> bool:
        assert verified_teardown_active()
        terminated.append(pod_id)
        return True

    record = idp._reap_superseded_runpod_fallback(
        issue=2038,
        prior=prior,
        new_handle=_new_handle(),
        live_matches_fn=lambda: [("rpOLD", "RUNNING")],
        keep_running_fn=lambda: False,
        terminate_fn=fake_terminate,
        stop_fn=_explode,
        marker_poster=lambda issue, note: notes.append((issue, note)),
    )
    assert terminated == ["rpOLD"]
    assert record is not None
    assert record["pod_name"] == "pod-2038"
    assert record["pod_id"] == "rpOLD"
    assert record["action"] == "terminate"
    datetime.fromisoformat(record["ts"])  # ISO-8601 parseable
    assert len(notes) == 1
    assert notes[0][0] == 2038
    assert "terminate" in notes[0][1]


def test_wrapper_terminate_failure_records_failed_never_raises(tmp_path, caplog):
    prior = _prior_dict(tmp_path, _prior_runpod_handle())

    def fake_terminate(pod_id: str) -> bool:
        raise RuntimeError("api down")

    with caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.issue_dispatch"):
        record = idp._reap_superseded_runpod_fallback(
            issue=2038,
            prior=prior,
            new_handle=_new_handle(),
            live_matches_fn=lambda: [("rpOLD", "RUNNING")],
            keep_running_fn=lambda: False,
            terminate_fn=fake_terminate,
            stop_fn=_explode,
            marker_poster=lambda issue, note: None,
        )
    assert record is not None
    assert record["action"] == "terminate-failed"
    assert "may still be billing" in caplog.text


def test_wrapper_stop_arm_reversible(tmp_path):
    prior = _prior_dict(
        tmp_path,
        _prior_runpod_handle(extra={"workload_executed": True, "workload_start_error": None}),
    )
    stopped: list[str] = []
    record = idp._reap_superseded_runpod_fallback(
        issue=2038,
        prior=prior,
        new_handle=_new_handle(),
        live_matches_fn=lambda: [("rpOLD", "RUNNING")],
        keep_running_fn=lambda: False,
        terminate_fn=_explode,
        stop_fn=lambda pod_id: stopped.append(pod_id),
        marker_poster=lambda issue, note: None,
    )
    assert stopped == ["rpOLD"]
    assert record is not None
    assert record["action"] == "stop"


def test_wrapper_running_skip_posts_durable_marker(tmp_path):
    """Critic note 1: ANY skip whose live read shows the prior pod RUNNING
    posts a durable epm:progress note (here: keep-running tag set)."""
    prior = _prior_dict(tmp_path, _prior_runpod_handle())
    notes: list[str] = []
    record = idp._reap_superseded_runpod_fallback(
        issue=2038,
        prior=prior,
        new_handle=_new_handle(),
        live_matches_fn=lambda: [("rpOLD", "RUNNING")],
        keep_running_fn=lambda: True,
        terminate_fn=_explode,
        stop_fn=_explode,
        marker_poster=lambda issue, note: notes.append(note),
    )
    assert record is None  # skips never accrete audit keys on the sidecar
    assert len(notes) == 1
    assert "SKIPPED (skip-keep-running)" in notes[0]
    assert "still RUNNING" in notes[0]


def test_wrapper_ambiguous_name_skip_posts_durable_marker(tmp_path):
    """Critic note 1's named row: the legacy id-less 0/>1 name-match skip."""
    prior = _prior_dict(tmp_path, _prior_runpod_handle(extra={"pod_id": None}))
    notes: list[str] = []
    record = idp._reap_superseded_runpod_fallback(
        issue=2038,
        prior=prior,
        new_handle=_new_handle(),
        live_matches_fn=lambda: [("rpA", "RUNNING"), ("rpB", "RUNNING")],
        keep_running_fn=_explode,
        terminate_fn=_explode,
        stop_fn=_explode,
        marker_poster=lambda issue, note: notes.append(note),
    )
    assert record is None
    assert len(notes) == 1
    assert "skip-name-ambiguous" in notes[0]


def test_wrapper_marker_post_failure_degrades_to_warn(tmp_path, caplog):
    prior = _prior_dict(tmp_path, _prior_runpod_handle())
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.backends.issue_dispatch"):
        record = idp._reap_superseded_runpod_fallback(
            issue=2038,
            prior=prior,
            new_handle=_new_handle(),
            live_matches_fn=lambda: [("rpOLD", "RUNNING")],
            keep_running_fn=lambda: False,
            terminate_fn=lambda pod_id: True,
            stop_fn=_explode,
            marker_poster=_explode,
        )
    assert record is not None  # the reap itself still happened + is recorded
    assert "durable marker post failed" in caplog.text


def test_wrapper_non_runpod_prior_is_silent_noop(tmp_path):
    nibi_prior = _prior_dict(
        tmp_path,
        RunHandle(
            backend="nibi",
            cluster="nibi",
            job_id="42",
            pod_name="pod-2038",
            scratch_dir="/s",
            log_path="/l",
            extra={"issue": 2038, "workload_cmd": "bash x.sh", "hydra_args": []},
        ),
    )
    record = idp._reap_superseded_runpod_fallback(
        issue=2038,
        prior=nibi_prior,
        new_handle=_new_handle(),
        live_matches_fn=_explode,
        keep_running_fn=_explode,
        terminate_fn=_explode,
        stop_fn=_explode,
        marker_poster=_explode,
    )
    assert record is None


def test_wrapper_defensive_on_prior_without_handle_key():
    """A pre-#2038 snapshot shape (no ``handle`` key) is a silent no-op."""
    record = idp._reap_superseded_runpod_fallback(
        issue=2038,
        prior={"backend": "runpod", "pod_name": "pod-2038", "job_id": "", "extra": {}},
        new_handle=_new_handle(),
        live_matches_fn=_explode,
        keep_running_fn=_explode,
        terminate_fn=_explode,
        stop_fn=_explode,
        marker_poster=_explode,
    )
    assert record is None


def test_wrapper_prior_none_returns_none():
    assert (
        idp._reap_superseded_runpod_fallback(issue=2038, prior=None, new_handle=_new_handle())
        is None
    )


def test_default_live_name_matches_exact_name_only(monkeypatch):
    """pod-17 / pod-173 must NOT match pod-1739 (never a prefix match)."""
    import scripts.runpod_api as runpod_api
    from scripts.runpod_api import PodInfo

    pods = [
        PodInfo(pod_id="a", name="pod-17", desired_status="RUNNING"),
        PodInfo(pod_id="b", name="pod-1739", desired_status="RUNNING"),
        PodInfo(pod_id="c", name="pod-173", desired_status="EXITED"),
        PodInfo(pod_id="d", name="pod-1739", desired_status="EXITED"),
        PodInfo(pod_id="e", name="pod-1739-b", desired_status="RUNNING"),
    ]
    monkeypatch.setattr(runpod_api, "list_team_pods", lambda: pods)
    assert idp._default_live_name_matches("pod-1739") == [("b", "RUNNING"), ("d", "EXITED")]


def test_default_live_name_matches_failure_returns_none(monkeypatch, caplog):
    import scripts.runpod_api as runpod_api

    def _api_boom():
        raise RuntimeError("graphql down")

    monkeypatch.setattr(runpod_api, "list_team_pods", _api_boom)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.backends.issue_dispatch"):
        assert idp._default_live_name_matches("pod-2038") is None
    assert "live pod listing failed" in caplog.text


# ---------------------------------------------------------------------------
# Component 2 — dispatch_for_issue integration (§5 row C2-integration)
# ---------------------------------------------------------------------------


class _MockBackend:
    """Minimal launch-only backend for the dispatch integration tests."""

    def __init__(self, kind: str = "nibi") -> None:
        self._kind = kind
        self.launches: list[RunSpec] = []

    @property
    def name(self):
        return self._kind

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        self.launches.append(spec)
        return RunHandle(
            backend=self._kind,
            cluster=self._kind if self._kind != "runpod" else None,
            job_id="job-1",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/scratch",
            log_path="/log",
            extra={"issue": spec.issue},
        )

    def estimate_start(self, spec: RunSpec):
        from datetime import UTC, datetime

        return datetime.now(tz=UTC)

    def estimate_start_seconds(self, spec: RunSpec):
        return 0.0

    def poll(self, handle: RunHandle):
        raise NotImplementedError

    def fetch_logs(self, handle: RunHandle) -> str:
        return ""

    def fetch_results(self, handle: RunHandle) -> None:
        return None

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        return True

    def teardown(self, handle: RunHandle) -> None:
        return None


@pytest.fixture
def tmp_lease_store(tmp_path):
    from explore_persona_space.backends.router import LeaseStore

    return LeaseStore(lease_dir=tmp_path / ".eps-routing")


def test_dispatch_for_issue_reaps_superseded_runpod_fallback(
    tmp_path, tmp_lease_store, monkeypatch
):
    """End-to-end: prior failed-RunPod sidecar + fresh nibi dispatch -> the
    superseded pod is terminated by exact id and the audit record lands on
    BOTH the returned handle and the authoritative sidecar JSON."""
    issue = 300
    sidecar = tmp_path / f"issue-{issue}-handle.json"
    idp.write_handle_sidecar(_prior_runpod_handle(issue=issue), sidecar)

    monkeypatch.setattr(idp, "_default_live_name_matches", lambda name: [("rpOLD", "RUNNING")])
    monkeypatch.setattr(idp, "_default_keep_running_state", lambda i: False)
    notes: list[tuple[int, str]] = []
    monkeypatch.setattr(idp, "_default_reap_marker_poster", lambda i, note: notes.append((i, note)))
    terminated: list[str] = []
    import scripts.runpod_api as runpod_api

    def fake_terminate(pod_id: str) -> bool:
        assert verified_teardown_active()
        terminated.append(pod_id)
        return True

    monkeypatch.setattr(runpod_api, "terminate_pod", fake_terminate)

    nibi = _MockBackend(kind="nibi")
    outcome = idp.dispatch_for_issue(
        RunSpec(issue=issue, intent="lora-7b", backend="nibi"),
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
        marker_poster=lambda **kw: None,
    )
    assert len(nibi.launches) == 1
    assert terminated == ["rpOLD"]
    record = outcome.result.handle.extra["superseded_runpod_reaped"]
    assert record["action"] == "terminate"
    assert record["pod_id"] == "rpOLD"
    # The authoritative sidecar carries the audit record too.
    on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
    assert on_disk["extra"]["superseded_runpod_reaped"]["action"] == "terminate"
    assert notes, "the terminate arm owes a durable epm:progress note"
    assert notes[0][0] == issue
    assert "terminate" in notes[0][1]


def test_dispatch_for_issue_fresh_sidecar_no_reap(tmp_path, tmp_lease_store, monkeypatch):
    """No prior sidecar -> the reap never runs (no live read, no audit key)."""
    monkeypatch.setattr(idp, "_default_live_name_matches", _explode)
    nibi = _MockBackend(kind="nibi")
    sidecar = tmp_path / "issue-301-handle.json"
    outcome = idp.dispatch_for_issue(
        RunSpec(issue=301, intent="lora-7b", backend="nibi"),
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
        marker_poster=lambda **kw: None,
    )
    assert "superseded_runpod_reaped" not in outcome.result.handle.extra


def test_dispatch_for_issue_reap_crash_never_blocks_launch(
    tmp_path, tmp_lease_store, monkeypatch, caplog
):
    """Best-effort contract: an unexpected reap crash degrades to a loud log
    and the launch outcome is returned untouched."""
    issue = 302
    sidecar = tmp_path / f"issue-{issue}-handle.json"
    idp.write_handle_sidecar(_prior_runpod_handle(issue=issue), sidecar)
    monkeypatch.setattr(idp, "_reap_superseded_runpod_fallback", _explode)
    nibi = _MockBackend(kind="nibi")
    with caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.issue_dispatch"):
        outcome = idp.dispatch_for_issue(
            RunSpec(issue=issue, intent="lora-7b", backend="nibi"),
            runpod_backend=_MockBackend(kind="runpod"),
            free_backends={"nibi": nibi},
            is_started=lambda _b, _h: True,
            lease_store=tmp_lease_store,
            handle_sidecar_path=sidecar,
            marker_poster=lambda **kw: None,
        )
    assert len(nibi.launches) == 1
    assert "superseded_runpod_reaped" not in outcome.result.handle.extra
    assert "superseded-RunPod reap failed" in caplog.text
