"""Unit tests for the #692 RunPod no-port wedge watcher backstop.

This is the watcher-side companion to ``tests/test_runpod_wedge_detection.py``
(which pins the POLLER-side ``backend_poll._maybe_escalate_runpod_wedge``). It
covers the new pieces in ``scripts/autonomous_session_watch.py``:

* the extracted raw predicate ``backend_poll._pod_is_runpod_runtime_wedged``
  (one direct unit test; the existing wedge-detection suite proves the poller
  refactor is byte-equivalent);
* the pure decision fn ``decide_pod_wedge`` (the full decision table + the
  pinned off-by-one boundaries, MF5);
* the tri-state keep-running wrapper ``_wedge_keep_running`` (MF2) + the
  fail-closed ``_wedge_inputs_safe`` gate;
* the wedge-state forward-carry / round-trip through ``_save_pod_safety_state``
  (MF3) and the ``_clear_wedge_state`` onset-clock clear (MF1);
* the end-to-end wedge arm in ``_process_pod`` / ``_process_wedged_pod`` —
  the dedicated wedge clock (MF1), the pod_id-change reset (MF4), the
  once-per-episode alert dedup, and the DONE-task fall-through (MF6).

All RunPod live-API I/O is mocked (``_running_managed_issue_pods`` returns a
synthetic 4-tuple), ``task.py`` reads are monkeypatched, the inputs-on-HF gate
is monkeypatched, and per-pod state files use the ``isolated_registry`` tmp dir.
No GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import backend_poll as bp  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

K = bp.RUNPOD_WEDGE_K_SEC  # 900s — the SAME floor the poller uses (no duplicate literal)


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the per-pod state dir at a tmp dir (mirrors the main suite)."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _wedged_info(pod_id: str = "p692", name: str = "pod-692") -> PodInfo:
    """A live ``PodInfo`` in the RAW #664 wedge: RUNNING + no public port."""
    return PodInfo(pod_id=pod_id, name=name, desired_status="RUNNING", ssh_host=None, ssh_port=None)


def _healthy_info(pod_id: str = "p692", name: str = "pod-692") -> PodInfo:
    """A healthy live ``PodInfo``: RUNNING + a public SSH port present."""
    return PodInfo(
        pod_id=pod_id, name=name, desired_status="RUNNING", ssh_host="1.2.3.4", ssh_port=22000
    )


# ===========================================================================
# 1. The extracted raw predicate (composition surface (b))
# ===========================================================================


def test_pred_wedged_running_no_port():
    # RUNNING + no public port -> the raw wedge condition is True.
    assert bp._pod_is_runpod_runtime_wedged(_wedged_info()) is True


def test_pred_healthy_port_present():
    # RUNNING + a public port -> healthy, never wedge-classified.
    assert bp._pod_is_runpod_runtime_wedged(_healthy_info()) is False


def test_pred_gone_or_exited():
    # None (pod gone) and a non-RUNNING pod both take the ordinary dead path.
    assert bp._pod_is_runpod_runtime_wedged(None) is False
    exited = PodInfo(
        pod_id="p", name="pod-1", desired_status="EXITED", ssh_host=None, ssh_port=None
    )
    assert bp._pod_is_runpod_runtime_wedged(exited) is False


# ===========================================================================
# 2. The pure decision fn `decide_pod_wedge` (decision table + boundaries)
# ===========================================================================


def _decide(**over):
    kw = dict(
        wedged_for=K + 1,
        k_floor=K,
        wedge_missed=1,  # so new_wedge_missed=2 == threshold -> confirmed
        threshold=2,
        alerted=False,
        keep_running=False,
        inputs_ok=True,
    )
    kw.update(over)
    return asw.decide_pod_wedge(**kw)


def test_wedge_boundary_zero():
    # MF5: a freshly-onset wedge (wedged_for=0) never stops on tick 1 -> KEEP.
    assert _decide(wedged_for=0.0, wedge_missed=0) == ("keep", 0)


def test_wedge_boundary_eq_k():
    # MF5: exactly at the K floor -> KEEP (matches the poller's strict `> K`).
    assert _decide(wedged_for=float(K), wedge_missed=0) == ("keep", 0)


def test_wedge_below_k():
    # Below K, even with an accumulated miss count, the wedge has not matured
    # -> KEEP and RESET the miss counter (a brief no-port blip never accrues).
    assert _decide(wedged_for=K - 100.0, wedge_missed=5) == ("keep", 0)


def test_wedge_unconfirmed_increments_keep():
    # Past K but not yet confirmed for >=threshold consecutive checks: accumulate.
    # threshold=2, wedge_missed=0 -> new=1 < 2 -> KEEP with the incremented count.
    assert _decide(wedged_for=K + 1, wedge_missed=0, threshold=2) == ("keep", 1)


def test_wedge_boundary_confirm_transition_failover():
    # MF5: the action transitions EXACTLY at wedge_missed + 1 == threshold.
    # threshold=3: wedge_missed=1 -> new=2 < 3 -> KEEP; wedge_missed=2 -> new=3
    # -> act (#770: the confirmed provably-safe action is "terminate-failover",
    # formerly "stop"). The KEEP (unconfirmed) half is unchanged.
    assert _decide(wedged_for=K + 1, wedge_missed=1, threshold=3, inputs_ok=True) == ("keep", 2)
    assert _decide(wedged_for=K + 1, wedge_missed=2, threshold=3, inputs_ok=True) == (
        "terminate-failover",
        0,
    )


def test_wedge_decide_returns_terminate_failover_on_confirmed_safe():
    # #770: confirmed past K + inputs verified on HF + keep_running is the
    # literal False -> the IRREVERSIBLE terminate+failover fires (was "stop").
    assert _decide(keep_running=False, inputs_ok=True) == ("terminate-failover", 0)


def test_wedge_alert_inputs_unverified():
    # Confirmed + inputs NOT verified on HF -> ALERT-only (never strand work).
    assert _decide(keep_running=False, inputs_ok=False) == ("alert", 0)


def test_wedge_alert_keep_running_true():
    # Confirmed + the keep-running tag is present -> ALERT-only (tag exemption).
    # Even with inputs_ok=True the terminate is suppressed.
    assert _decide(keep_running=True, inputs_ok=True) == ("alert", 0)


def test_wedge_alert_keep_running_unknown():
    # MF2 closure: confirmed + inputs_ok=True but the keep-running read FAILED
    # ("unknown") -> ALERT, NOT terminate. A persistent tag-read failure must
    # never silently override a (possibly present) keep-running tag on a
    # live-work pod.
    assert _decide(keep_running="unknown", inputs_ok=True) == ("alert", 0)


def test_wedge_decision_invariant_terminate_only_on_literal_false():
    # The decision invariant (MF2, #770): ("terminate-failover", _) is the ONLY
    # irreversible action and is returned ONLY when keep_running is the literal
    # False AND inputs_ok is True. Sweep the keep_running x inputs_ok cross
    # product and assert no other combination terminates AND that "stop" is never
    # returned (the action set is now {terminate-failover, alert, keep}).
    for keep_running in (True, "unknown"):
        for inputs_ok in (True, False):
            action, _ = _decide(keep_running=keep_running, inputs_ok=inputs_ok)
            assert action != "terminate-failover", (keep_running, inputs_ok)
            assert action != "stop", (keep_running, inputs_ok)
    # Only literal False + True terminates; False + False alerts.
    assert _decide(keep_running=False, inputs_ok=True)[0] == "terminate-failover"
    assert _decide(keep_running=False, inputs_ok=False)[0] == "alert"
    # "stop" is NEVER produced by the pure fn after #770 (across the full grid).
    for keep_running in (True, False, "unknown"):
        for inputs_ok in (True, False):
            assert _decide(keep_running=keep_running, inputs_ok=inputs_ok)[0] != "stop"


def test_wedge_decide_maturity_axis_never_terminates_below_confirmation():
    # SR3: across the maturity axis (below-K, past-K-but-unconfirmed), the new
    # action "terminate-failover" is NEVER returned for ANY (keep_running,
    # inputs_ok) combination — only a CONFIRMED matured wedge can terminate.
    for keep_running in (True, False, "unknown"):
        for inputs_ok in (True, False):
            # below-K: wedged_for <= k_floor -> always KEEP regardless of gates.
            below = _decide(
                wedged_for=K - 100.0,
                wedge_missed=5,
                keep_running=keep_running,
                inputs_ok=inputs_ok,
            )
            assert below[0] != "terminate-failover", ("below-K", keep_running, inputs_ok)
            # past-K-unconfirmed: new_wedge_missed (= wedge_missed + 1) < threshold.
            unconfirmed = _decide(
                wedged_for=K + 1,
                wedge_missed=0,
                threshold=2,  # new=1 < 2 -> not yet confirmed
                keep_running=keep_running,
                inputs_ok=inputs_ok,
            )
            assert unconfirmed[0] != "terminate-failover", (
                "past-K-unconfirmed",
                keep_running,
                inputs_ok,
            )


# ===========================================================================
# 3. The tri-state keep-running wrapper (MF2) + the fail-closed inputs gate
# ===========================================================================


def _fake_task_view(monkeypatch, *, returncode=0, stdout=None, raise_exc=None):
    class _Out:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout if stdout is not None else ""

    def _run(*_a, **_k):
        if raise_exc is not None:
            raise raise_exc
        return _Out()

    monkeypatch.setattr(asw.subprocess, "run", _run)


def test_wedge_keep_running_present(monkeypatch):
    _fake_task_view(monkeypatch, stdout=json.dumps({"frontmatter": {"tags": ["keep-running"]}}))
    assert asw._wedge_keep_running(692) is True


def test_wedge_keep_running_absent(monkeypatch):
    _fake_task_view(monkeypatch, stdout=json.dumps({"frontmatter": {"tags": ["foo"]}}))
    assert asw._wedge_keep_running(692) is False


def test_wedge_keep_running_unknown_nonzero_rc(monkeypatch):
    _fake_task_view(monkeypatch, returncode=1, stdout="boom")
    assert asw._wedge_keep_running(692) == "unknown"


def test_wedge_keep_running_unknown_parse_error(monkeypatch):
    _fake_task_view(monkeypatch, returncode=0, stdout="not json{")
    assert asw._wedge_keep_running(692) == "unknown"


def test_wedge_keep_running_unknown_subprocess_error(monkeypatch):
    _fake_task_view(monkeypatch, raise_exc=OSError("no such file"))
    assert asw._wedge_keep_running(692) == "unknown"


def test_wedge_inputs_safe_no_handle_fails_closed(monkeypatch, tmp_path):
    # No persisted handle sidecar -> cannot gate -> fail-closed False (ALERT-only).
    import explore_persona_space.backends.issue_dispatch as idp

    missing = tmp_path / "issue-692-handle.json"
    monkeypatch.setattr(idp, "resolve_handle_sidecar_path", lambda issue: (missing, [missing]))
    assert asw._wedge_inputs_safe(692) is False


def test_wedge_inputs_safe_exception_fails_closed(monkeypatch):
    # Any exception inside the gate (transport / parse / import) -> False.
    import explore_persona_space.backends.issue_dispatch as idp

    def _boom(issue):
        raise RuntimeError("transport down")

    monkeypatch.setattr(idp, "resolve_handle_sidecar_path", _boom)
    assert asw._wedge_inputs_safe(692) is False


# ===========================================================================
# 4. State persistence (MF3 round-trip) + the onset-clock clear (MF1)
# ===========================================================================


def test_wedge_state_roundtrip(isolated_registry):
    # MF3: the three wedge fields survive a save + load round-trip.
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=1000.0,
        wedge_missed=1,
        wedge_alerted=True,
        prev={"first_seen": 500.0},
    )
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == 1000.0
    assert loaded["wedge_missed"] == 1
    assert loaded["wedge_alerted"] is True
    assert loaded["pod_id"] == "p692"
    # The pod-incarnation first_seen GC anchor is preserved, not clobbered.
    assert loaded["first_seen"] == 500.0


def test_wedge_state_carry_forward_on_status_class_save(isolated_registry):
    # A status-class save (no wedge kwargs) must NOT drop an accumulated wedge
    # episode's state -> the wedge fields carry forward untouched (MF3).
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=1000.0,
        wedge_missed=1,
        wedge_alerted=True,
    )
    prev = asw._load_pod_safety_state(692)
    # A later status-class save passes NO wedge kwargs (the default _CARRY).
    asw._save_pod_safety_state(
        692, "p692", missed=1, alerted=False, last_progress_ts=7.0, prev=prev
    )
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == 1000.0
    assert loaded["wedge_missed"] == 1
    assert loaded["wedge_alerted"] is True
    assert loaded["missed"] == 1  # the status-class field advanced


def test_clear_wedge_state_clears_clock_keeps_anchor(isolated_registry):
    # MF1: _clear_wedge_state resets the wedge fields to the onset-cleared state
    # while keeping the pod-incarnation first_seen GC anchor intact (NOT a
    # whole-file clear).
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=2,
        alerted=True,
        last_progress_ts=9.0,
        wedge_first_seen=1000.0,
        wedge_missed=3,
        wedge_alerted=True,
        prev={"first_seen": 500.0},
    )
    asw._clear_wedge_state(692, "p692")
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    assert loaded["wedge_missed"] == 0
    assert loaded["wedge_alerted"] is False
    # GC anchor + status-class counters preserved.
    assert loaded["first_seen"] == 500.0
    assert loaded["missed"] == 2
    assert loaded["alerted"] is True


# ===========================================================================
# 5. End-to-end wedge arm via `_process_pod` / `_process_wedged_pod`
# ===========================================================================


def _patch_wedge_io(
    monkeypatch,
    *,
    status="running",
    keep_running="unknown",
    inputs_ok=False,
    failover_outcome="failover",
):
    """Monkeypatch the wedge arm's I/O: task status, the tri-state keep-running
    read, the inputs-on-HF gate, and the action helpers. Returns the recorders
    ``(stops, posts, failovers)``.

    ``_stop_pod`` is recorded so a test can assert the wedge arm NEVER calls it
    after #770 (the confirmed provably-safe path is terminate+failover, not
    stop). ``_wedge_failover`` is stubbed to record its issue and return
    ``failover_outcome`` — the end-to-end tests exercise the DISPATCH branch
    (marker/state) on each outcome; the helper's own outcome logic is unit-tested
    separately against a stubbed ``backend_poll._failover_wedged_runpod``."""
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    failovers: list[int] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: keep_running)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: inputs_ok)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: failovers.append(issue) or failover_outcome,
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    # Guard: a non-wedge fall-through path must not touch the status-class I/O
    # in these wedge tests (they all pass a wedged info + non-DONE status).
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    return stops, posts, failovers


def test_wedge_clock_onset_heal_then_wedge(isolated_registry, monkeypatch):
    # MF1 closure: a pod RUNNING for >> K (its pod-incarnation first_seen is the
    # BOOT time) then loses its port. The FIRST wedged tick must stamp a fresh
    # wedge_first_seen=now -> wedged_for=0 <= K -> KEEP, NOT stop on a vacuous K.
    now = 2_000_000.0
    # Prior state: a long-running healthy pod (first_seen way in the past, no
    # wedge fields yet).
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        prev={"first_seen": now - 10 * K},
    )
    stops, _posts, _fo = _patch_wedge_io(monkeypatch, status="running")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []  # the K floor measures the no-port episode, not pod uptime
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == now  # stamped at ONSET this tick


def test_wedge_first_tick_zero_keeps(isolated_registry, monkeypatch):
    # The first wedged tick of a fresh incarnation -> wedged_for=0 -> KEEP.
    now = 1_000_000.0
    stops, posts, _fo = _patch_wedge_io(monkeypatch, status="approved")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert posts == []
    assert asw._load_pod_safety_state(692)["wedge_first_seen"] == now


def test_wedge_below_k_keeps(isolated_registry, monkeypatch):
    # A second tick still within K -> KEEP (no maturation).
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K - 100.0),  # 100s short of K
        wedge_missed=0,
        wedge_alerted=False,
        prev={"first_seen": now - (K - 100.0), "pod_id": "p692"},
    )
    stops, _posts, _fo = _patch_wedge_io(monkeypatch, status="running")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []


def test_wedge_unconfirmed_keeps_increments(isolated_registry, monkeypatch):
    # Past K, first confirmed-past-K tick (wedge_missed 0 -> new 1 < threshold 2)
    # -> KEEP, increment, NO alert yet.
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=0,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, _fo = _patch_wedge_io(monkeypatch, status="running")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert posts == []
    assert asw._load_pod_safety_state(692)["wedge_missed"] == 1


def test_wedge_confirming_transition_terminate_failover(isolated_registry, monkeypatch):
    # #770: the confirming tick (wedge_missed 1 -> new 2 == threshold),
    # inputs_ok=True, keep_running=False dispatches the IRREVERSIBLE
    # terminate+failover (NOT the reversible _stop_pod) exactly here, posts a
    # wedge-failover marker, and clears the wedge state. _stop_pod is NEVER
    # called on this path (the reversible stop cannot heal a dead host, #763).
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, failovers = _patch_wedge_io(
        monkeypatch,
        status="running",
        keep_running=False,
        inputs_ok=True,
        failover_outcome="failover",
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert failovers == [692]  # the terminate+failover recovery fired
    assert stops == []  # the reversible _stop_pod is NEVER called by the wedge arm
    labels = [label for _i, label in posts]
    assert "wedge-failover" in labels
    assert "wedge-stop" not in labels
    # The wedge fields are cleared on the failover, the first_seen GC anchor survives.
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    assert loaded["wedge_missed"] == 0


def test_wedge_inputs_unsafe_alerts(isolated_registry, monkeypatch):
    # Confirmed past K but inputs NOT verified on HF -> ALERT-only, NEVER stop.
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=False
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert failovers == []  # inputs-unverified -> ALERT-only, never terminate
    assert "wedge-alert" in [label for _i, label in posts]
    assert asw._load_pod_safety_state(692)["wedge_alerted"] is True


def test_wedge_keep_running_true_alerts(isolated_registry, monkeypatch):
    # Confirmed past K + inputs_ok=True but the keep-running tag is present
    # -> ALERT-only (the tag exemption matches the status-class arm).
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=True, inputs_ok=True
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert failovers == []  # the keep-running tag exempts the terminate
    assert "wedge-alert" in [label for _i, label in posts]


def test_wedge_keep_running_unknown_alerts_never_terminates(isolated_registry, monkeypatch):
    # MF2 closure end-to-end: confirmed + inputs_ok=True + keep_running="unknown"
    # -> ALERT, and neither _stop_pod nor the terminate+failover is ever called.
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running="unknown", inputs_ok=True
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert failovers == []  # a persistent tag-read failure never terminates
    assert "wedge-alert" in [label for _i, label in posts]


def test_wedge_alert_dedup_once_per_episode(isolated_registry, monkeypatch):
    # Two confirmed gated-off ticks in a row post the alert marker ONCE
    # (deduped via the persisted wedge_alerted flag).
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=False
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    asw._process_pod(692, "p692", _wedged_info(), now + 600, dry_run=False, threshold=2)
    wedge_alerts = [label for _i, label in posts if label == "wedge-alert"]
    assert len(wedge_alerts) == 1  # once per episode, not per tick
    assert stops == []  # the gated-off (inputs-unverified) path never stops
    assert failovers == []  # nor does it terminate
    assert asw._load_pod_safety_state(692)["wedge_alerted"] is True


def test_wedge_pod_id_change_reset(isolated_registry, monkeypatch):
    # MF4 closure: prev state carries a confirmed wedge from an OLD pod_id; the
    # live tick observes a FRESH pod_id (the issue was re-provisioned). All wedge
    # fields reset -> the fresh pod is KEPT, not stopped on stale wedge state.
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p_OLD",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - 10 * K,  # an old, matured wedge
        wedge_missed=5,  # well past threshold
        wedge_alerted=True,
        prev={"first_seen": now - 10 * K, "pod_id": "p_OLD"},
    )
    stops, _posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True
    )
    # The live pod has a NEW pod_id with no port (freshly wedged this tick).
    asw._process_pod(692, "p_NEW", _wedged_info(pod_id="p_NEW"), now, dry_run=False, threshold=2)
    assert stops == []  # stale wedge state from the old pod must not stop the new one
    assert failovers == []  # nor terminate it — the reset re-stamps the onset clock
    loaded = asw._load_pod_safety_state(692)
    assert loaded["pod_id"] == "p_NEW"
    assert loaded["wedge_first_seen"] == now  # re-stamped fresh for the new incarnation
    assert loaded["wedge_missed"] == 0


def test_wedge_done_task_falls_through_to_status_class_stop(isolated_registry, monkeypatch):
    # MF6 closure: a wedged pod whose task status is DONE (completed) does NOT go
    # to the wedge arm's ALERT-default — it falls through to the status-class
    # DONE auto-stop arm, which stops it after the 2-miss guard. Here the
    # status-class arm fires (no keep-running tag, no live follow-up).
    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: None)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None: False)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    # A wedge-arm helper that MUST NOT be reached for a DONE-status wedged pod.
    monkeypatch.setattr(
        asw,
        "_process_wedged_pod",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wedge arm reached on DONE task")),
    )
    info = _wedged_info()
    # Tick 1: missed 0 -> 1, no stop. Tick 2: hits the 2-miss guard -> STOP via
    # the status-class DONE arm (the canonical escaped-pod handler).
    asw._process_pod(692, "p692", info, now, dry_run=False, threshold=2)
    assert stops == []
    asw._process_pod(692, "p692", info, now, dry_run=False, threshold=2)
    assert stops == [692]
    assert "auto-stop" in [label for _i, label in posts]


def test_non_wedged_pod_clears_stale_wedge_state(isolated_registry, monkeypatch):
    # A pod that is NOT wedged (port present) clears any stale wedge clock so a
    # one-tick blip never matures, then proceeds to the status-class arm.
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - 10 * K,
        wedge_missed=5,
        wedge_alerted=True,
        prev={"first_seen": now - 10 * K, "pod_id": "p692"},
    )
    stops: list[int] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: now - 3600)  # fresh
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda issue, note, dry_run, label: None)
    asw._process_pod(692, "p692", _healthy_info(), now, dry_run=False, threshold=2)
    assert stops == []
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    assert loaded["wedge_missed"] == 0
    assert loaded["wedge_alerted"] is False


# ===========================================================================
# 6. The _wedge_failover helper (#770) — outcome mapping against a stubbed
#    backend_poll._failover_wedged_runpod, the handle-reconstruction degrade,
#    and the dry-run no-side-effects contract.
# ===========================================================================


def _stub_handle_sidecar(monkeypatch, tmp_path, *, exists=True, read_raises=None):
    """Stub the sidecar resolvers `_wedge_failover` imports lazily from
    `explore_persona_space.backends.issue_dispatch`. Returns the sidecar Path."""
    import explore_persona_space.backends.issue_dispatch as idp

    sidecar = tmp_path / "issue-692-handle.json"
    if exists:
        sidecar.write_text("{}")  # presence-only; read_handle_sidecar is stubbed

    monkeypatch.setattr(idp, "resolve_handle_sidecar_path", lambda issue: (sidecar, [sidecar]))

    class _Handle:
        pod_name = "pod-692"
        job_id = "j692"

    def _read(path):
        if read_raises is not None:
            raise read_raises
        return _Handle()

    monkeypatch.setattr(idp, "read_handle_sidecar", _read)
    return sidecar


def _stub_failover_fn(monkeypatch, returns=None, raises=None):
    """Stub `backend_poll._failover_wedged_runpod` to record its call kwargs and
    return `returns` (or raise `raises`). Returns the recorder list."""
    calls: list[dict] = []

    def _fake(*, issue, handle, result, sidecar):
        calls.append({"issue": issue, "handle": handle, "result": result, "sidecar": sidecar})
        if raises is not None:
            raise raises
        return returns

    monkeypatch.setattr(bp, "_failover_wedged_runpod", _fake)
    return calls


def test_wedge_failover_dispatches_failover_fn(monkeypatch, tmp_path):
    # SR / item 6: given a resolvable handle + a running-shaped return, the helper
    # returns "failover" AND calls _failover_wedged_runpod with `issue`, the
    # reconstructed `handle`, a `result` shim exposing the two attributes it reads
    # (current_phase + log_tail_excerpt), and the resolved `sidecar` path.
    sidecar = _stub_handle_sidecar(monkeypatch, tmp_path)
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "failover"
    assert len(calls) == 1
    c = calls[0]
    assert c["issue"] == 692
    assert c["sidecar"] == sidecar
    assert c["handle"].pod_name == "pod-692"
    # The result shim exposes ONLY the two attrs _failover_wedged_runpod reads.
    assert isinstance(c["result"].current_phase, str)
    assert isinstance(c["result"].log_tail_excerpt, str)


def test_wedge_failover_no_handle_degrades_to_alert(monkeypatch, tmp_path):
    # item 7: no persisted sidecar -> the helper returns "alert" and NEVER calls
    # _failover_wedged_runpod (never terminates blind).
    _stub_handle_sidecar(monkeypatch, tmp_path, exists=False)
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "alert"
    assert calls == []  # never reached the failover call


def test_wedge_failover_handle_read_raises_degrades_to_alert(monkeypatch, tmp_path):
    # item 8: read_handle_sidecar raises -> "alert", no failover call.
    _stub_handle_sidecar(monkeypatch, tmp_path, read_raises=RuntimeError("parse boom"))
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "alert"
    assert calls == []


def test_wedge_failover_raises_degrades_to_alert(monkeypatch, tmp_path):
    # item 9: _failover_wedged_runpod itself raises -> "alert" (fail-loud log, no
    # crash of the whole watcher tick), no double-action.
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, raises=RuntimeError("router exploded"))
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "alert"


def test_wedge_failover_already_handled_is_noop(monkeypatch, tmp_path):
    # item 10 (cross-actor idempotency): _failover_wedged_runpod returns the
    # bounded-once terminal JSON -> the helper returns "already-handled". The
    # no-double-fire is enforced INSIDE _failover_wedged_runpod via its own
    # _runpod_wedge_already_handled check (pinned by test_backend_poll.py /
    # test_runpod_wedge_detection.py); here we exercise the watcher's branch on
    # the returned reason.
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(
        monkeypatch, returns={"status": "dead", "reason": "runpod_wedge_already_handled"}
    )
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "already-handled"


def test_wedge_failover_no_capacity_terminal(monkeypatch, tmp_path):
    # item 11: terminated but RunPod unavailable for the re-provision ->
    # "no-capacity" (the capacity-retry pass re-drives; the watcher does NOT
    # re-arm here).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, returns={"status": "dead", "reason": "no_compute_available"})
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "no-capacity"


def test_wedge_failover_inputs_unverified_blocks(monkeypatch, tmp_path):
    # item 12: a PARTIAL cell -> _failover_wedged_runpod refuses to terminate and
    # returns reason=runpod_wedge_inputs_unverified -> the helper returns
    # "blocked" (the terminate did NOT happen; a human resolves, halt-crit #2).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(
        monkeypatch, returns={"status": "dead", "reason": "runpod_wedge_inputs_unverified"}
    )
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert out == "blocked"


def test_wedge_failover_dry_run_no_side_effects(monkeypatch, tmp_path):
    # item 13: dry_run=True -> the helper returns "failover" WITHOUT calling
    # _failover_wedged_runpod (no real terminate / re-provision).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    out = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=True)
    assert out == "failover"
    assert calls == []  # dry-run never calls the irreversible recovery


def test_wedge_failover_marker_sentinel_ignored_by_staleness():
    # item 14: the new failover sentinel is in the watcher-internal sentinel set
    # the staleness clock ignores (so a wedge-failover marker never resets the
    # orphan/stalled clocks) — alongside the two existing wedge sentinels.
    assert asw._WEDGE_FAILOVER_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._WEDGE_ALERT_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._WEDGE_STOP_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


# ===========================================================================
# 7. Standing-recommendation tests (adversarial-planner SR1 / SR2)
# ===========================================================================


def test_wedge_failover_reason_strings_match_backend_poll_source():
    # SR1: cross-module reason-string parity. The literal reasons _wedge_failover
    # branches on MUST match the reasons backend_poll._failover_wedged_runpod /
    # _relaunch_fresh_runpod actually emit — a drift in either module would make
    # the watcher mis-classify a real terminal JSON (e.g. read a no-capacity
    # terminal as "blocked"). Discover the reasons from backend_poll's SOURCE and
    # assert each is covered by the helper's branch logic.
    import inspect

    # The reasons _wedge_failover explicitly branches on (status=="running" is the
    # success shape; "already-handled" / "no-capacity" are explicit reason
    # branches; the rest fall through to "blocked").
    branched = {"runpod_wedge_already_handled", "no_compute_available"}
    # The reasons backend_poll EMITS from the wedge-failover call chain.
    src = inspect.getsource(bp._failover_wedged_runpod) + inspect.getsource(
        bp._relaunch_fresh_runpod
    )
    emitted = {
        "runpod_wedge_already_handled",
        "no_compute_available",
        "runpod_wedge_inputs_unverified",
        "sidecar_persistence_failed",
        "runpod_wedge_relaunch_spec_missing",
    }
    for reason in emitted:
        assert f'"{reason}"' in src, f"backend_poll no longer emits reason={reason!r}"
    # Every explicitly-branched reason is one backend_poll actually emits (no
    # dead branch on a renamed reason). The non-branched emitted reasons all map
    # to "blocked" via the helper's else-fall-through (pinned by
    # test_wedge_failover_all_blocked_reasons_map_to_blocked).
    assert branched <= emitted


@pytest.mark.parametrize(
    "reason",
    [
        "runpod_wedge_inputs_unverified",
        "sidecar_persistence_failed",
        "runpod_wedge_relaunch_spec_missing",
    ],
)
def test_wedge_failover_all_blocked_reasons_map_to_blocked(monkeypatch, tmp_path, reason):
    # SR1 (companion): every non-success, non-already-handled, non-no-capacity
    # terminal reason backend_poll can emit maps to the "blocked" outcome.
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, returns={"status": "dead", "reason": reason})
    assert asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False) == "blocked"


def test_wedge_blocked_outcome_not_terminated_by_watcher_clock_cleared(
    isolated_registry, monkeypatch
):
    # SR2: on outcome == "blocked" the WATCHER itself does NOT terminate the pod
    # (the terminate decision lives entirely inside _failover_wedged_runpod, which
    # for a "blocked" reason did NOT terminate) — i.e. the watcher never calls
    # _stop_pod, and it clears the wedge clock (the human is the resolver, so a
    # re-stamp that re-fires next tick is wrong). The end-to-end dispatch is
    # exercised via the stubbed _wedge_failover returning "blocked".
    now = 1_000_000.0
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=now - (K + 50.0),
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": now - (K + 50.0), "pod_id": "p692"},
    )
    stops, posts, failovers = _patch_wedge_io(
        monkeypatch,
        status="running",
        keep_running=False,
        inputs_ok=True,
        failover_outcome="blocked",
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert failovers == [692]  # the failover recovery WAS consulted
    assert stops == []  # but the watcher itself NEVER reversibly stops the pod
    assert "wedge-failover" in [label for _i, label in posts]
    # The wedge clock IS cleared (the human resolves the terminal block; a
    # re-stamp would re-fire the failover every tick).
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    assert loaded["wedge_missed"] == 0
    # The pod-incarnation first_seen GC anchor survives.
    assert loaded["first_seen"] == now - (K + 50.0)
