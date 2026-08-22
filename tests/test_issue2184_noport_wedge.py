"""#2184: CPU no-port-wedge typed detection + teardown-interlocked DC rotation.

Covers the plan §5 rows: the typed ``RunPodNoPortWedgeError`` raise contract in
``wait_for_ssh`` (RUNNING-only; other timeouts stay bare ``RunPodError``), the
teardown-disposition interlock (rotation re-creates ONLY on a CONFIRMED
``"terminated"`` teardown; ``--keep-on-bootstrap-failure`` and explicit user
DC pins disable rotation; unknown/failed teardown and unidentifiable wedged DC
refuse), the bias-away candidate exclusion (wedged + dry + fresh #2011
bad-placement DCs), both rotation budgets (wedge budget ``>=`` post-append →
exactly 1+budget wedge-bearing creates; dry-DC cap), the ``CPU-LANE-DRY``
typed residual refusal (stderr verdict + ``EXIT_CPU_LANE_DRY`` = 77), the
armed-wait split (attempt 1 via the #2238 wait-for-capacity loop, rotation via
plain ``create_cpu_pod``), the unchanged first-attempt no-capacity fail-loud,
and the rule-docs anchor pins.

All RunPod API seams are monkeypatched at the ``pod_lifecycle`` namespace
(module-global late binding — the ``tests/test_pod_lifecycle.py`` convention);
NO test touches a pod or the network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time as real_time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
import runpod_api  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

ISSUE = 930
POD_NAME = f"pod-{ISSUE}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _info(
    pod_id: str,
    *,
    desired_status: str = "RUNNING",
    ssh_host: str | None = None,
    ssh_port: int | None = None,
    data_center_id: str | None = None,
) -> PodInfo:
    """A CPU-pod PodInfo (gpu fields None — the deployCpuPod parse shape)."""
    return PodInfo(
        pod_id=pod_id,
        name=POD_NAME,
        desired_status=desired_status,
        gpu_count=None,
        gpu_type_id=None,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        created_at="2026-08-18T00:00:00Z",
        data_center_id=data_center_id,
    )


def _ns(**overrides) -> argparse.Namespace:
    """Namespace matching the provision subparser shape, for a CPU intent."""
    base = {
        "issue": ISSUE,
        "list_intents": False,
        "intent": "cpu-small",
        "gpu_type": None,
        "gpu_count": None,
        "dry_run": False,
        "volume_gb": 200,  # argparse default (GPU default) → CPU default kicks in
        "container_disk_gb": 50,
        "ttl_days": 7,
        "no_bootstrap": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolate every sidecar + env knob so no test touches live state.

    EPHEMERAL_STATE / BAD_HOST_STATE / _SSH_WAIT_STATE_PATH → tmp;
    EPM_AUTONOMOUS_SESSION + both rotation env knobs deleted (defaults);
    get_datacenters → [] (tests re-patch with candidate lists);
    list_team_pods → [] (idempotency scan); network-adjacent preflights no-op'd;
    pods.conf upsert no-op'd (the success leg would write the LIVE file).
    """
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", tmp_path / "pods_ephemeral.json")
    monkeypatch.setattr(pod_lifecycle, "BAD_HOST_STATE", tmp_path / "bad-pod-hosts.json")
    monkeypatch.setattr(pod_lifecycle, "_SSH_WAIT_STATE_PATH", tmp_path / "ssh-wait-alarm.json")
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    monkeypatch.delenv("EPM_CPU_DC_ROTATION_MAX_WEDGES", raising=False)
    monkeypatch.delenv("EPM_CPU_DC_ROTATION_MAX_DCS", raising=False)
    monkeypatch.setattr(pod_lifecycle, "get_datacenters", lambda: [])
    monkeypatch.setattr(pod_lifecycle, "list_team_pods", lambda: [])
    monkeypatch.setattr(pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *_a: None)
    monkeypatch.setattr(pod_lifecycle, "_account_key_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(pod_lifecycle, "_upsert_pods_conf", lambda *_a, **_k: None)
    return tmp_path


def _wire(monkeypatch, plan: list[tuple[str, str | None]], *, terminate_raises: bool = False):
    """Wire create/wait/get/terminate fakes; the provision TAIL stays REAL.

    ``plan[i]`` governs the i-th ``create_cpu_pod`` call:
      ("wedge", dc) → pod created; the REAL tail's faked ``wait_for_ssh``
                       raises the typed wedge (PodInfo carries ``dc``);
      ("ok", dc)    → pod created; ``wait_for_ssh`` returns a ready PodInfo;
      ("dry", None) → ``create_cpu_pod`` raises ``RunPodNoCapacityError``.

    Only the external API boundary is faked (``wait_for_ssh`` /
    ``terminate_pod`` / ``get_pod`` / ``create_cpu_pod``);
    ``_provision_wait_register_bootstrap`` and ``_teardown_failed_provision``
    run their REAL bodies (MF-A coverage), incl. the real kill-gate grant
    (thread-local only). Returns the ordered events list
    (``("create", dc_pin)`` / ``("terminate", pod_id)``).
    """
    events: list[tuple[str, str | None]] = []
    state = {"n": 0}
    pods: dict[str, tuple[str, PodInfo]] = {}

    def fake_create_cpu_pod(
        *, name, instance_id, volume_gb, container_disk_gb, data_center_id=None
    ):
        i = state["n"]
        state["n"] += 1
        events.append(("create", data_center_id))
        kind, dc = plan[i]
        if kind == "dry":
            raise runpod_api.RunPodNoCapacityError("no CPU capacity (test)")
        info = _info(f"pod{i}", data_center_id=dc)
        pods[info.pod_id] = (kind, info)
        return info

    def fake_wait_for_ssh(pod_id, timeout=600, poll_interval=10):
        kind, info = pods[pod_id]
        if kind == "wedge":
            raise runpod_api.RunPodNoPortWedgeError(
                f"Pod {pod_id} did not expose public 22/tcp within {timeout}s. "
                f"Last desiredStatus: RUNNING (RUNNING-but-no-port wedge, #2184)",
                info=info,
            )
        return _info(pod_id, ssh_host="1.2.3.4", ssh_port=22, data_center_id=info.data_center_id)

    def fake_get_pod(pod_id):
        return pods[pod_id][1]

    def fake_terminate_pod(pod_id):
        events.append(("terminate", pod_id))
        if terminate_raises:
            raise RuntimeError("terminate API down (test)")

    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod", fake_create_cpu_pod)
    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", fake_wait_for_ssh)
    monkeypatch.setattr(pod_lifecycle, "get_pod", fake_get_pod)
    monkeypatch.setattr(pod_lifecycle, "terminate_pod", fake_terminate_pod)
    return events


def _dcs(monkeypatch, *ids: str) -> None:
    monkeypatch.setattr(pod_lifecycle, "get_datacenters", lambda: [{"id": i} for i in ids])


# ---------------------------------------------------------------------------
# wait_for_ssh raise contract (runpod_api)
# ---------------------------------------------------------------------------


class _FakeTime:
    """Deterministic time for wait_for_ssh: sleep() advances the clock."""

    def __init__(self):
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.now += secs


def test_wait_for_ssh_raises_typed_wedge_on_running_no_port(monkeypatch):
    """RUNNING + no 22/tcp at timeout → the TYPED RunPodNoPortWedgeError,
    carrying the last PodInfo and a None teardown_disposition slot."""
    monkeypatch.setattr(runpod_api, "time", _FakeTime())
    wedged = _info("w1", desired_status="RUNNING", data_center_id="EU-RO-1")
    monkeypatch.setattr(runpod_api, "get_pod", lambda pod_id: wedged)
    with pytest.raises(runpod_api.RunPodNoPortWedgeError) as ei:
        runpod_api.wait_for_ssh("w1", timeout=30, poll_interval=10)
    assert ei.value.info is wedged
    assert ei.value.teardown_disposition is None
    assert "did not expose public 22/tcp" in str(ei.value)
    assert "RUNNING-but-no-port wedge" in str(ei.value)


def test_wait_for_ssh_bare_error_on_non_running_timeout(monkeypatch):
    """A non-RUNNING timeout keeps the EXACT bare RunPodError type — the
    typed wedge is scoped to the RUNNING-but-no-port shape only."""
    monkeypatch.setattr(runpod_api, "time", _FakeTime())
    monkeypatch.setattr(runpod_api, "get_pod", lambda pod_id: _info("x1", desired_status="EXITED"))
    with pytest.raises(runpod_api.RunPodError) as ei:
        runpod_api.wait_for_ssh("x1", timeout=30, poll_interval=10)
    assert type(ei.value) is runpod_api.RunPodError
    assert "did not expose public 22/tcp" in str(ei.value)


# ---------------------------------------------------------------------------
# Rotation happy path + interlock ordering (real tail, real teardown)
# ---------------------------------------------------------------------------


def test_wedge_rotates_to_different_dc_after_confirmed_teardown(monkeypatch, capsys, tmp_path):
    """First create wedges (EU-RO-1) → REAL #2060 teardown terminates it →
    rotation re-creates pinned to a DIFFERENT DC, excluding the wedged DC AND
    a fresh #2011 bad-placement DC; terminate strictly precedes create #2."""
    (tmp_path / "bad-pod-hosts.json").write_text(
        json.dumps({"5.6.7.8": [{"ts": real_time.time(), "dc_id": "SEED-BAD", "issue": 1}]})
    )
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2", "SEED-BAD")
    events = _wire(monkeypatch, [("wedge", "EU-RO-1"), ("ok", "US-KS-2")])
    pod_lifecycle.cmd_provision(_ns())
    assert events == [("create", None), ("terminate", "pod0"), ("create", "US-KS-2")]
    captured = capsys.readouterr()
    assert "BOOTSTRAP-FAILED-TERMINATED pod=pod-930" in captured.err
    assert "[cpu-dc-rotation]" in captured.out


def test_rotation_notice_names_confirmed_teardown_and_budget(monkeypatch, capsys):
    """The rotation stdout line names the wedged DC, the CONFIRMED teardown,
    and the wedge-budget position (k of 1+budget)."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")
    _wire(monkeypatch, [("wedge", "EU-RO-1"), ("ok", "US-KS-2")])
    pod_lifecycle.cmd_provision(_ns())
    out = capsys.readouterr().out
    assert "[cpu-dc-rotation] no-port wedge in EU-RO-1 (teardown CONFIRMED)" in out
    assert "retrying pinned to US-KS-2" in out
    assert "(1 of 3 wedge-bearing attempts used)" in out


# ---------------------------------------------------------------------------
# CPU-LANE-DRY residual refusals (exit 77, reason-tagged)
# ---------------------------------------------------------------------------


def _assert_dry(excinfo, capsys, reason: str) -> str:
    """Shared refusal asserts: exit 77, typed cause, verdict first line."""
    assert excinfo.value.code == pod_lifecycle.EXIT_CPU_LANE_DRY == 77
    assert isinstance(excinfo.value.__cause__, runpod_api.RunPodCpuLaneDryError)
    err = capsys.readouterr().err
    assert f"CPU-LANE-DRY reason={reason}" in err
    # Residual route: the sanctioned GPU-intent fallback command is named.
    assert f"provision --issue {ISSUE} --intent eval" in err
    # RAM/disk-fit caveat rides every refusal.
    assert "RAM" in err and "disk" in err.lower()
    return err


def test_candidates_exhausted_refuses_exit_77(monkeypatch, capsys):
    """One candidate DC total (== the wedged one) → candidates-exhausted."""
    _dcs(monkeypatch, "EU-RO-1")
    events = _wire(monkeypatch, [("wedge", "EU-RO-1")])
    with pytest.raises(SystemExit) as ei:
        pod_lifecycle.cmd_provision(_ns())
    err = _assert_dry(ei, capsys, "candidates-exhausted")
    assert [e[0] for e in events] == ["create", "terminate"]
    # Verdict FIRST-LINE shape (machine-greppable single token grammar).
    m = re.search(
        r"^CPU-LANE-DRY reason=candidates-exhausted intent=cpu-small "
        r"instance=cpu3g-2-8 wedged_dcs=EU-RO-1 dry_dcs=none \(#2184\)$",
        err,
        re.MULTILINE,
    )
    assert m is not None, err


def test_dry_dc_cap_exhausted_refuses(monkeypatch, capsys):
    """Rotation advances over dry (no-capacity) DCs; the dry-DC cap
    (EPM_CPU_DC_ROTATION_MAX_DCS, >= post-append) bounds the sweep."""
    monkeypatch.setenv("EPM_CPU_DC_ROTATION_MAX_DCS", "2")
    _dcs(monkeypatch, "DC-A", "DC-B", "DC-C", "DC-D", "DC-E")
    events = _wire(monkeypatch, [("wedge", "DC-A"), ("dry", None), ("dry", None)])
    with pytest.raises(SystemExit) as ei:
        pod_lifecycle.cmd_provision(_ns())
    err = _assert_dry(ei, capsys, "dc-cap-exhausted")
    assert [e[0] for e in events] == ["create", "terminate", "create", "create"]
    assert "dry_dcs=DC-B,DC-C" in err


def test_wedge_budget_exactly_three_wedge_bearing_creates(monkeypatch, capsys):
    """Default budget 2 → exactly 1 + budget = 3 wedge-bearing creates; the
    third wedge refuses (>= post-append), each preceded by its teardown."""
    _dcs(monkeypatch, "DC-A", "DC-B", "DC-C", "DC-D", "DC-E")
    events = _wire(monkeypatch, [("wedge", "DC-A"), ("wedge", "DC-B"), ("wedge", "DC-C")])
    with pytest.raises(SystemExit) as ei:
        pod_lifecycle.cmd_provision(_ns())
    err = _assert_dry(ei, capsys, "wedge-budget-exhausted")
    assert [e[0] for e in events] == [
        "create",
        "terminate",
        "create",
        "terminate",
        "create",
        "terminate",
    ]
    assert len([e for e in events if e[0] == "create"]) == 3
    assert "wedged_dcs=DC-A,DC-B,DC-C" in err


def test_env_budget_one_allows_two_wedge_bearing_creates(monkeypatch, capsys):
    """EPM_CPU_DC_ROTATION_MAX_WEDGES=1 → 1 + 1 = 2 wedge-bearing creates."""
    monkeypatch.setenv("EPM_CPU_DC_ROTATION_MAX_WEDGES", "1")
    _dcs(monkeypatch, "DC-A", "DC-B", "DC-C")
    events = _wire(monkeypatch, [("wedge", "DC-A"), ("wedge", "DC-B")])
    with pytest.raises(SystemExit) as ei:
        pod_lifecycle.cmd_provision(_ns())
    _assert_dry(ei, capsys, "wedge-budget-exhausted")
    assert len([e for e in events if e[0] == "create"]) == 2


def test_teardown_unconfirmed_refuses_without_recreate(monkeypatch, capsys):
    """MF-A: terminate_pod RAISES → REAL _teardown_failed_provision returns
    "failed" → rotation REFUSES (teardown-unconfirmed) with NO second create,
    naming the possibly-billing pod + the manual terminate command."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")
    events = _wire(monkeypatch, [("wedge", "EU-RO-1")], terminate_raises=True)
    with pytest.raises(SystemExit) as ei:
        pod_lifecycle.cmd_provision(_ns())
    err = _assert_dry(ei, capsys, "teardown-unconfirmed")
    assert len([e for e in events if e[0] == "create"]) == 1
    assert "STILL BE BILLING" in err
    assert "pod0" in err
    assert f"pod.py terminate --issue {ISSUE}" in err


def test_keep_flag_disables_rotation_and_propagates_typed_wedge(monkeypatch, capsys):
    """MF-A: --keep-on-bootstrap-failure → NO terminate, NO rotation; the
    typed wedge propagates (today's single-attempt shape) with the kept
    disposition threaded on."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")
    events = _wire(monkeypatch, [("wedge", "EU-RO-1")])
    with pytest.raises(runpod_api.RunPodNoPortWedgeError) as ei:
        pod_lifecycle.cmd_provision(_ns(keep_on_bootstrap_failure=True))
    assert ei.value.teardown_disposition == "kept"
    assert events == [("create", None)]  # no terminate, no re-create
    err = capsys.readouterr().err
    assert "rotation DISABLED" in err


def test_first_attempt_no_capacity_propagates_exact_type(monkeypatch):
    """A plain (unarmed) first-attempt no-capacity miss keeps today's
    fail-loud propagation — the EXACT RunPodNoCapacityError type, never a
    rotation and never a CPU-LANE-DRY refusal."""
    events = _wire(monkeypatch, [("dry", None)])
    with pytest.raises(runpod_api.RunPodNoCapacityError) as ei:
        pod_lifecycle.cmd_provision(_ns())
    assert type(ei.value) is runpod_api.RunPodNoCapacityError
    assert events == [("create", None)]


def test_dc_enumeration_failure_refuses(monkeypatch, capsys):
    """get_datacenters failure after a confirmed-teardown wedge → the
    dc-enumeration-failed refusal, never a blind re-create."""

    def _boom():
        raise runpod_api.RunPodError("datacenters enumeration down (test)")

    events = _wire(monkeypatch, [("wedge", "EU-RO-1")])
    monkeypatch.setattr(pod_lifecycle, "get_datacenters", _boom)
    with pytest.raises(SystemExit) as ei:
        pod_lifecycle.cmd_provision(_ns())
    _assert_dry(ei, capsys, "dc-enumeration-failed")
    assert len([e for e in events if e[0] == "create"]) == 1


# ---------------------------------------------------------------------------
# Helper-level interlock rows (user pin / unknown DC / armed wait)
# ---------------------------------------------------------------------------


def _helper_ns(**overrides) -> argparse.Namespace:
    base = {"issue": ISSUE, "name_suffix": None, "ttl_days": 7, "no_bootstrap": True}
    base.update(overrides)
    return argparse.Namespace(**base)


def _run_helper(
    monkeypatch,
    *,
    wait_for_capacity: bool = False,
    explicit_wait_flag: bool = False,
    data_center_id: str | None = None,
    tail_plan: dict[str, str | None] | None = None,
):
    """Drive _provision_cpu_with_rotation with a FAKE tail (helper-level rows).

    ``tail_plan`` maps pod_id → teardown_disposition; a listed pod's tail
    raises the typed wedge with that disposition threaded on; unlisted pods
    succeed. Returns (created kwargs list, wait-loop call count).
    """
    tail_plan = tail_plan or {}
    created: list[dict] = []
    wait_loop_calls = {"n": 0}
    counter = {"n": 0}

    def _mk_info(dc):
        i = counter["n"]
        counter["n"] += 1
        return _info(f"hpod{i}", data_center_id=dc)

    def fake_create(*, name, instance_id, volume_gb, container_disk_gb, data_center_id=None):
        created.append({"data_center_id": data_center_id})
        return _mk_info(data_center_id)

    def fake_wait_loop(*, name, instance_id, volume_gb, container_disk_gb, data_center_id=None):
        wait_loop_calls["n"] += 1
        return _mk_info(data_center_id)

    def fake_tail(args, name, info, intent_label):
        if info.pod_id in tail_plan:
            exc = runpod_api.RunPodNoPortWedgeError("wedge (test)", info=info)
            exc.teardown_disposition = tail_plan[info.pod_id]
            raise exc

    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod", fake_create)
    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod_with_wait_for_capacity", fake_wait_loop)
    monkeypatch.setattr(pod_lifecycle, "_provision_wait_register_bootstrap", fake_tail)
    pod_lifecycle._provision_cpu_with_rotation(
        _helper_ns(),
        POD_NAME,
        "cpu3g-2-8",
        "cpu-small",
        20,
        20,
        wait_for_capacity=wait_for_capacity,
        explicit_wait_flag=explicit_wait_flag,
        data_center_id=data_center_id,
    )
    return created, wait_loop_calls["n"]


def test_user_pin_wedged_refuses_never_rotates(monkeypatch, capsys):
    """An explicitly pinned DC that wedges refuses (user-pin-wedged) even on
    a CONFIRMED teardown — rotation never overrides an operator pin."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")
    with pytest.raises(SystemExit) as ei:
        _run_helper(
            monkeypatch,
            data_center_id="EU-RO-1",
            tail_plan={"hpod0": "terminated"},
        )
    err = _assert_dry(ei, capsys, "user-pin-wedged")
    assert "re-run WITHOUT the pin" in err


def test_disposition_none_refuses_teardown_unconfirmed(monkeypatch, capsys):
    """A wedge whose teardown disposition never threaded (None — e.g. a path
    that raised before _teardown_failed_provision ran) refuses as
    teardown-unconfirmed: only an explicit "terminated" licenses a re-create."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")
    with pytest.raises(SystemExit) as ei:
        _run_helper(monkeypatch, tail_plan={"hpod0": None})
    _assert_dry(ei, capsys, "teardown-unconfirmed")


def test_wedged_dc_unknown_with_confirmed_teardown(monkeypatch, capsys):
    """Disposition "terminated" but the wedged pod's DC is unresolvable →
    wedged-dc-unknown (bias-away rotation is impossible without a DC)."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")
    with pytest.raises(SystemExit) as ei:
        _run_helper(monkeypatch, tail_plan={"hpod0": "terminated"})
    _assert_dry(ei, capsys, "wedged-dc-unknown")


def test_armed_wait_first_attempt_only_rotation_uses_plain_create(monkeypatch, capsys):
    """#2238 armed wait: attempt 1 goes through the wait-for-capacity loop
    EXACTLY once; rotation retries use PLAIN create_cpu_pod (fast per-DC
    capacity probes — an unbounded wait pinned to one DC would wedge it)."""
    _dcs(monkeypatch, "EU-RO-1", "US-KS-2")

    # The wait-loop pod (hpod0) wedges with DC EU-RO-1 threaded via its info.
    created: list[dict] = []
    wait_loop_calls = {"n": 0}

    def fake_wait_loop(*, name, instance_id, volume_gb, container_disk_gb, data_center_id=None):
        wait_loop_calls["n"] += 1
        return _info("wl0", data_center_id="EU-RO-1")

    def fake_create(*, name, instance_id, volume_gb, container_disk_gb, data_center_id=None):
        created.append({"data_center_id": data_center_id})
        return _info(f"rot{len(created)}", data_center_id=data_center_id)

    def fake_tail(args, name, info, intent_label):
        if info.pod_id == "wl0":
            exc = runpod_api.RunPodNoPortWedgeError("wedge (test)", info=info)
            exc.teardown_disposition = "terminated"
            raise exc

    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod", fake_create)
    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod_with_wait_for_capacity", fake_wait_loop)
    monkeypatch.setattr(pod_lifecycle, "_provision_wait_register_bootstrap", fake_tail)
    pod_lifecycle._provision_cpu_with_rotation(
        _helper_ns(),
        POD_NAME,
        "cpu3g-2-8",
        "cpu-small",
        20,
        20,
        wait_for_capacity=True,
        explicit_wait_flag=False,
        data_center_id=None,
    )
    assert wait_loop_calls["n"] == 1
    assert created == [{"data_center_id": "US-KS-2"}]
    out = capsys.readouterr().out
    # The #2238 auto-enable promise line still prints once (before attempt 1).
    assert "auto-enabling --wait-for-capacity" in out


# ---------------------------------------------------------------------------
# Error taxonomy + docs anchors
# ---------------------------------------------------------------------------


def test_error_taxonomy_and_exit_code_pins():
    """Both new classes subclass RunPodError but NOT RunPodNoCapacityError
    (the wait-for-capacity loop must never blind-retry a wedge); exit 77 is
    distinct within the structured-exit family (75/76/77)."""
    assert issubclass(runpod_api.RunPodNoPortWedgeError, runpod_api.RunPodError)
    assert issubclass(runpod_api.RunPodCpuLaneDryError, runpod_api.RunPodError)
    assert not issubclass(runpod_api.RunPodNoPortWedgeError, runpod_api.RunPodNoCapacityError)
    assert not issubclass(runpod_api.RunPodCpuLaneDryError, runpod_api.RunPodNoCapacityError)
    assert pod_lifecycle.EXIT_CPU_LANE_DRY == 77
    assert (
        len(
            {
                pod_lifecycle.EXIT_STILL_WAITING,
                pod_lifecycle.EXIT_STOPPED_POD_COLLISION,
                pod_lifecycle.EXIT_CPU_LANE_DRY,
            }
        )
        == 3
    )


def test_rule_docs_name_provision_time_wedge_coverage():
    """The three rule files carry BOTH #2184 anchors, and the superseded
    "covered by the watcher's wedge arm" claim is gone from
    compute-backends.md (the watcher is now the last of three layers)."""
    docs = {
        "compute-backends": REPO_ROOT / ".claude" / "rules" / "compute-backends.md",
        "pods": REPO_ROOT / ".claude" / "rules" / "pods.md",
        "gotchas": REPO_ROOT / ".claude" / "rules" / "gotchas.md",
    }
    for label, path in docs.items():
        text = path.read_text(encoding="utf-8")
        assert "RunPodNoPortWedgeError" in text, label
        assert "CPU-LANE-DRY" in text, label
    cb = docs["compute-backends"].read_text(encoding="utf-8")
    assert "covered by the watcher's wedge arm" not in cb
