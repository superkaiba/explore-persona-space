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
import time
from datetime import datetime
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import backend_poll as bp  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

K = bp.RUNPOD_WEDGE_K_SEC  # 900s — the SAME floor the poller uses (no duplicate literal)

# Genuine helper bodies captured at import time (BEFORE any fixture stubs
# them) — the #1667 tests restore these to exercise the REAL code paths
# (the keep_running_owner file's _REAL_APPEND convention).
_REAL_OWNER_LIVE = asw._wedge_owner_live
_REAL_POST_PROGRESS = asw._post_progress_marker


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
    owner_live=False,
    failover_outcome="failover",
    failover_terminal=None,
):
    """Monkeypatch the wedge arm's I/O: task status, the tri-state keep-running
    read, the inputs-on-HF gate, and the action helpers. Returns the recorders
    ``(stops, posts, failovers)``.

    ``_stop_pod`` is recorded so a test can assert the wedge arm NEVER calls it
    after #770 (the confirmed provably-safe path is terminate+failover, not
    stop). ``_wedge_failover`` is stubbed to record its issue and return the
    ``(failover_outcome, failover_terminal)`` tuple — the end-to-end tests
    exercise the DISPATCH branch (marker/state) on each outcome; the helper's
    own outcome logic is unit-tested separately against a stubbed
    ``backend_poll._failover_wedged_runpod``.

    ``posts`` records progress-marker calls as ``(issue, label)``, epm:failure
    calls as ``(issue, "FAILURE", note)``, and set-status-blocked calls as
    ``(issue, "BLOCKED")`` — one ordered stream so a test can assert both the
    failure marker AND the status change fired (the no-capacity/blocked
    redrive contract), plus that the failure marker preceded the clock-clear."""
    stops: list[int] = []
    posts: list[tuple] = []
    failovers: list[int] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: keep_running)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: inputs_ok)
    # #1667 owner-liveness probe: default False (no live owner) so the
    # pre-#1667 tests keep their terminate behavior byte-equivalent.
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: owner_live)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: (
            failovers.append(issue) or (failover_outcome, failover_terminal)
        ),
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    monkeypatch.setattr(
        asw,
        "_post_failure_marker",
        lambda issue, note, dry_run: posts.append((issue, "FAILURE", note)) or True,
    )
    monkeypatch.setattr(
        asw,
        "_set_status_blocked",
        lambda issue, dry_run: posts.append((issue, "BLOCKED")) or True,
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
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
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


def test_wedge_on_hold_task_falls_through_to_status_class_stop(isolated_registry, monkeypatch):
    # #980 mirror of the MF6 closure for the user-paused status: a wedged pod
    # whose task is `on_hold` does NOT go to the wedge arm (whose confirmed-safe
    # path would terminate + RELAUNCH a workload the user deliberately paused) —
    # it falls through to the status-class auto-stop arm, which stops it after
    # the 2-miss guard (no keep-running tag, no live follow-up).
    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "on_hold")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: None)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    # A wedge-arm helper that MUST NOT be reached for a paused-status wedged pod.
    monkeypatch.setattr(
        asw,
        "_process_wedged_pod",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wedge arm reached on on_hold task")),
    )
    info = _wedged_info()
    # Tick 1: missed 0 -> 1, no stop. Tick 2: hits the 2-miss guard -> STOP via
    # the status-class auto-stop arm (the canonical escaped-pod handler).
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
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "failover"
    # The running-shaped success carries the recovery's terminal JSON dict.
    assert terminal == {"status": "running"}
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
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "alert"
    assert terminal is None  # no recovery dict in scope on the alert degrade
    assert calls == []  # never reached the failover call


def test_wedge_failover_handle_read_raises_degrades_to_alert(monkeypatch, tmp_path):
    # item 8: read_handle_sidecar raises -> "alert", no failover call.
    _stub_handle_sidecar(monkeypatch, tmp_path, read_raises=RuntimeError("parse boom"))
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "alert"
    assert terminal is None
    assert calls == []


def test_wedge_failover_raise_after_terminate_routes_to_blocked(monkeypatch, tmp_path):
    # item 9 (#770 v2 r3 — POST-terminate raise, pod GONE per the liveness probe):
    # _failover_wedged_runpod terminates the wedged pod BEFORE the re-provision, so
    # a raise AFTER that point must NOT degrade to "alert" (which would post no
    # durable record and could not be retried next tick — the terminated pod is gone
    # from the RUNNING-only snapshot). The except branch PROBES get_pod_by_name; a
    # None (pod gone) confirms the raise was post-terminate, so it mirrors the
    # poller's caller defense (backend_poll ~1864-1880): convert the raise to a
    # ("blocked", terminal) outcome carrying failure_class=infra
    # reason=runpod_wedge_failover_error so the caller records it durably. Still
    # fail-LOUD (does not crash the watcher tick) and no double-action.
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, raises=RuntimeError("router exploded"))
    # The pod is GONE (post-terminate raise) -> the liveness probe returns None.
    monkeypatch.setattr(asw, "get_pod_by_name", lambda name: None)
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "blocked"
    assert terminal is not None
    assert terminal["status"] == "dead"
    assert terminal["failure_class"] == "infra"
    assert terminal["reason"] == "runpod_wedge_failover_error"
    # The raised exception type+message is carried in the log_tail for a human.
    assert "RuntimeError" in terminal["log_tail_excerpt"]
    assert "router exploded" in terminal["log_tail_excerpt"]


def test_wedge_failover_preterminate_raise_pod_alive_degrades_to_alert(monkeypatch, tmp_path):
    # item 9b (#770 v2 r3 BLOCKER watcher-failover-preterminate-raise-falsely-blocks-
    # live-pod): _failover_wedged_runpod has fallible PRE-terminate steps (the
    # _runpod_wedge_already_handled lease check; _wedged_run_inputs_on_hf ->
    # huggingface_hub.list_repo_files) that can raise BEFORE terminate_pod, leaving
    # the pod RUNNING+billing. Mapping that raise to "blocked" would post a FALSE
    # terminal record (claiming the pod terminated) AND clear the wedge clock. The
    # except branch PROBES get_pod_by_name; a non-None (pod still ALIVE) confirms the
    # raise was pre-terminate, so it degrades to "alert" (no terminal_json) — the
    # clock is preserved for a next-tick retry, never a false terminal record.
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, raises=RuntimeError("HF list_repo_files blip pre-terminate"))
    # The pod is still ALIVE (pre-terminate raise) -> the liveness probe returns it.
    monkeypatch.setattr(asw, "get_pod_by_name", lambda name: _wedged_info())
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "alert"
    assert terminal is None  # no terminal record on the alert degrade


def test_wedge_failover_preterminate_raise_probe_raises_degrades_to_alert(monkeypatch, tmp_path):
    # item 9c (#770 v2 r3): the liveness probe itself can raise (network/transport
    # on get_pod_by_name). The pod's terminate state is then UNCERTAIN -> bias SAFE:
    # degrade to "alert" (preserve the clock) rather than post a possibly-false
    # terminal "blocked" record.
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, raises=RuntimeError("router exploded"))

    def _probe_boom(name):
        raise RuntimeError("RunPod API transport error")

    monkeypatch.setattr(asw, "get_pod_by_name", _probe_boom)
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "alert"
    assert terminal is None


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
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "already-handled"
    assert terminal == {"status": "dead", "reason": "runpod_wedge_already_handled"}


def test_wedge_failover_no_capacity_terminal(monkeypatch, tmp_path):
    # item 11: terminated but RunPod unavailable for the re-provision ->
    # "no-capacity" (the capacity-retry pass re-drives; the watcher does NOT
    # re-arm here).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, returns={"status": "dead", "reason": "no_compute_available"})
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "no-capacity"
    # The terminal JSON is propagated so the caller can mirror the poller's
    # epm:failure (failure_class + reason) — see test_wedge_no_capacity_*.
    assert terminal == {"status": "dead", "reason": "no_compute_available"}


def test_wedge_failover_inputs_unverified_blocks(monkeypatch, tmp_path):
    # item 12: a PARTIAL cell -> _failover_wedged_runpod refuses to terminate and
    # returns reason=runpod_wedge_inputs_unverified -> the helper returns
    # "blocked" (the terminate did NOT happen; a human resolves, halt-crit #2).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(
        monkeypatch, returns={"status": "dead", "reason": "runpod_wedge_inputs_unverified"}
    )
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "blocked"
    assert terminal == {"status": "dead", "reason": "runpod_wedge_inputs_unverified"}


def test_wedge_failover_dry_run_no_side_effects(monkeypatch, tmp_path):
    # item 13: dry_run=True -> the helper returns "failover" WITHOUT calling
    # _failover_wedged_runpod (no real terminate / re-provision).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=True)
    assert outcome == "failover"
    assert terminal is None  # dry-run short-circuits BEFORE any recovery dict
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
    # terminal reason backend_poll can emit maps to the "blocked" outcome AND
    # carries its terminal JSON (so the caller can mirror the poller's epm:failure
    # with the exact reason — pinned by test_wedge_blocked_emits_failure_*).
    _stub_handle_sidecar(monkeypatch, tmp_path)
    _stub_failover_fn(monkeypatch, returns={"status": "dead", "reason": reason})
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "blocked"
    assert terminal == {"status": "dead", "reason": reason}


def test_wedge_blocked_outcome_not_terminated_by_watcher_clock_cleared(
    isolated_registry, monkeypatch
):
    # SR2: on outcome == "blocked" the WATCHER itself does NOT reversibly stop the
    # pod (the terminate decision lives entirely inside _failover_wedged_runpod) —
    # it mirrors the poller's path on a terminal infra JSON (epm:failure +
    # set-status blocked, CRITICAL #1) and clears the wedge clock (the human is the
    # resolver, so a re-stamp that re-fires next tick is wrong). The end-to-end
    # dispatch is exercised via the stubbed _wedge_failover returning "blocked".
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
        failover_terminal={
            "status": "dead",
            "failure_class": "infra",
            "reason": "runpod_wedge_inputs_unverified",
        },
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert failovers == [692]  # the failover recovery WAS consulted
    assert stops == []  # but the watcher itself NEVER reversibly stops the pod
    # A terminal infra block is recorded as epm:failure + status:blocked (NOT a
    # plain wedge-failover progress note).
    tags = [t[1] for t in posts]
    assert "FAILURE" in tags
    assert "BLOCKED" in tags
    assert "wedge-failover" not in tags
    # The wedge clock IS cleared (the human resolves the terminal block; a
    # re-stamp would re-fire the failover every tick).
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    assert loaded["wedge_missed"] == 0
    # The pod-incarnation first_seen GC anchor survives.
    assert loaded["first_seen"] == now - (K + 50.0)


# ===========================================================================
# 8. Round-2 review fixes (#770 r2): the terminal-infra-JSON redrive contract
#    (CRITICAL #1), the sidecar-binding fresh-pod defense (CRITICAL #2), and the
#    per-reason "blocked" marker text (CONCERN #3).
# ===========================================================================


def _matured_confirming_state(now):
    """Persist the wedge state one tick BEFORE the confirming transition (so the
    next _process_pod tick at threshold=2 fires the terminate-failover action)."""
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


def _patch_real_marker_recorders(monkeypatch, *, outcome, terminal):
    """Patch the wedge arm's I/O EXCEPT _post_failure_marker / _set_status_blocked
    (which are RECORDED with their real call args, so the failure-note shape can
    be parsed) and _wedge_failover (stubbed to return (outcome, terminal)).
    Returns (failure_notes, blocked_calls, progress_labels)."""
    failure_notes: list[str] = []
    blocked_calls: list[int] = []
    progress_labels: list[str] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: (outcome, terminal),
    )
    monkeypatch.setattr(
        asw,
        "_post_failure_marker",
        lambda issue, note, dry_run: failure_notes.append(note) or True,
    )
    monkeypatch.setattr(
        asw,
        "_set_status_blocked",
        lambda issue, dry_run: blocked_calls.append(issue) or True,
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: progress_labels.append(label),
    )
    return failure_notes, blocked_calls, progress_labels


def test_wedge_no_capacity_emits_failure_marker_and_blocks_redrivable(
    isolated_registry, monkeypatch
):
    # CRITICAL #1: the no-capacity terminal JSON (terminated, RunPod unavailable)
    # is recorded as epm:failure v1 (failure_class: infra reason: no_compute_available)
    # AND the task is set to status:blocked — so the capacity-retry pass re-drives
    # it. Asserts (a) _post_failure_marker fired, (b) _set_status_blocked fired,
    # (c) the posted note PARSES into a transient-capacity block (the predicate the
    # capacity-retry pass actually keys on returns True).
    now = 1_000_000.0
    _matured_confirming_state(now)
    failure_notes, blocked_calls, progress_labels = _patch_real_marker_recorders(
        monkeypatch,
        outcome="no-capacity",
        terminal={"status": "dead", "failure_class": "infra", "reason": "no_compute_available"},
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    # (a) + (b): the marker and the status change both fired.
    assert len(failure_notes) == 1
    assert blocked_calls == [692]
    # A no-capacity terminal is NOT a plain progress note.
    assert "wedge-failover" not in progress_labels
    # (c) the posted note PARSES as a transient-capacity block via the REAL parser
    # + classifier the capacity-retry pass uses (so a re-drive WOULD fire).
    fc, reason = asw._parse_failure_fields(failure_notes[0])
    assert fc == "infra"
    assert reason == "no_compute_available"
    synthetic_marker = {"kind": "epm:failure v1", "note": failure_notes[0], "ts": None}
    retriable, parsed_reason, _block_ts = asw._is_transient_capacity_block([synthetic_marker])
    assert retriable is True
    assert parsed_reason == "no_compute_available"
    # The wedge clock is cleared (recorded as a terminal failure now).
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None


def test_wedge_blocked_emits_failure_marker_and_blocks_not_redrivable(
    isolated_registry, monkeypatch
):
    # CRITICAL #1 companion: a "blocked" terminal JSON (a non-capacity infra
    # reason) ALSO emits epm:failure + set-status blocked — but its reason is NOT
    # in TRANSIENT_CAPACITY_REASONS, so the capacity-retry classifier leaves it
    # parked for a human (retriable False), mirroring the poller's own path.
    now = 1_000_000.0
    _matured_confirming_state(now)
    failure_notes, blocked_calls, _progress = _patch_real_marker_recorders(
        monkeypatch,
        outcome="blocked",
        terminal={
            "status": "dead",
            "failure_class": "infra",
            "reason": "sidecar_persistence_failed",
        },
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert len(failure_notes) == 1
    assert blocked_calls == [692]
    synthetic_marker = {"kind": "epm:failure v1", "note": failure_notes[0], "ts": None}
    retriable, parsed_reason, _ = asw._is_transient_capacity_block([synthetic_marker])
    assert retriable is False  # a human resolves; the capacity-retry pass never re-drives it
    assert parsed_reason == "sidecar_persistence_failed"


def test_wedge_failover_raise_after_terminate_routes_to_blocked_redrivable(
    isolated_registry, monkeypatch
):
    # #770 v2 r2 BLOCKER (watcher-failover-raise-after-terminate-strands-task):
    # a POST-terminate raise from _failover_wedged_runpod must NOT degrade to
    # "alert" (which would strand the run — no epm:failure, no status:blocked, and
    # the terminated pod is gone from the RUNNING-only snapshot so the next tick
    # never re-enters _process_wedged_pod). End-to-end through _process_pod: the
    # raise is converted to a ("blocked", terminal) outcome that routes through the
    # durable epm:failure + set-status blocked retry helper, so the run is recorded
    # (NOT stranded) and a human inspects the raise.
    now = 1_000_000.0
    _matured_confirming_state(now)
    # Stub _failover_wedged_runpod (NOT _wedge_failover) to RAISE — so the real
    # _wedge_failover except-Exception branch synthesizes the terminal record and
    # the whole _process_pod -> _handle_wedge_failover_outcome path is exercised.
    import backend_poll as bp

    def _fake_failover(*, issue, handle, result, sidecar):
        raise RuntimeError("router exploded post-terminate")

    monkeypatch.setattr(bp, "_failover_wedged_runpod", _fake_failover)
    # #770 v2 r3: the except branch probes get_pod_by_name to distinguish a
    # POST-terminate raise (pod GONE -> "blocked") from a PRE-terminate one (pod
    # ALIVE -> "alert"). This test exercises the POST-terminate path, so the pod is
    # gone -> the probe returns None.
    monkeypatch.setattr(asw, "get_pod_by_name", lambda name: None)
    # Reconstruct a handle naming the wedged pod (so the sidecar-binding defense
    # passes and the failover call is actually reached) — _stub_handle_sidecar
    # writes a sidecar + handle whose pod_name == _wedged_info().name == "pod-692".
    sidecar_dir = isolated_registry / "_sidecar"
    sidecar_dir.mkdir()
    _stub_handle_sidecar(monkeypatch, tmp_path=sidecar_dir)
    failure_notes: list[str] = []
    blocked_calls: list[int] = []
    progress_labels: list[str] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw, "_post_failure_marker", lambda issue, note, dry_run: failure_notes.append(note) or True
    )
    monkeypatch.setattr(
        asw, "_set_status_blocked", lambda issue, dry_run: blocked_calls.append(issue) or True
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: progress_labels.append(label),
    )

    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    # (a) the durable terminal record both fired (NOT a plain progress note).
    assert len(failure_notes) == 1
    assert blocked_calls == [692]
    assert "wedge-failover" not in progress_labels
    # (b) the marker parses to failure_class=infra reason=runpod_wedge_failover_error
    # via the REAL parser the capacity-retry pass uses.
    fc, reason = asw._parse_failure_fields(failure_notes[0])
    assert fc == "infra"
    assert reason == "runpod_wedge_failover_error"
    # (c) NOT re-drivable — runpod_wedge_failover_error is not a transient-capacity
    # reason, so the capacity-retry pass parks it for a human (never re-drives
    # doomed-broken code).
    synthetic_marker = {"kind": "epm:failure v1", "note": failure_notes[0], "ts": None}
    retriable, parsed_reason, _ = asw._is_transient_capacity_block([synthetic_marker])
    assert retriable is False
    assert parsed_reason == "runpod_wedge_failover_error"
    # (d) the raised exception is named in the note for the human.
    assert "router exploded post-terminate" in failure_notes[0]
    # (e) the wedge clock is cleared (a terminal failure is recorded — the episode
    # is resolved from the watcher's vantage; not re-fired next tick).
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None


def test_wedge_failover_preterminate_raise_does_not_falsely_block_live_pod(
    isolated_registry, monkeypatch
):
    # #770 v2 r3 BLOCKER (watcher-failover-preterminate-raise-falsely-blocks-live-pod):
    # a PRE-terminate raise from _failover_wedged_runpod (an HF list_repo_files blip
    # / lease-check error BEFORE terminate_pod) leaves the wedged pod RUNNING+billing.
    # Mapping that raise to ("blocked", runpod_wedge_failover_error) would post a FALSE
    # durable record (the marker text claims the pod was "likely terminated") AND clear
    # the wedge clock, while the pod keeps billing — and reason
    # runpod_wedge_failover_error is not re-drivable, so capacity-retry never re-drives
    # it. End-to-end through _process_pod: the except branch PROBES get_pod_by_name; a
    # non-None (pod still ALIVE) confirms the raise was pre-terminate, so _wedge_failover
    # returns ("alert", None) and the whole path degrades to ALERT — no epm:failure, no
    # status:blocked, and the wedge clock is PRESERVED so the next tick re-detects the
    # still-RUNNING wedge and re-matures it.
    now = 1_000_000.0
    _matured_confirming_state(now)
    import backend_poll as bp

    def _fake_failover(*, issue, handle, result, sidecar):
        raise RuntimeError("HF list_repo_files blip pre-terminate")

    monkeypatch.setattr(bp, "_failover_wedged_runpod", _fake_failover)
    # The pod is STILL ALIVE (pre-terminate raise) -> the liveness probe returns it.
    monkeypatch.setattr(asw, "get_pod_by_name", lambda name: _wedged_info())
    sidecar_dir = isolated_registry / "_sidecar"
    sidecar_dir.mkdir()
    _stub_handle_sidecar(monkeypatch, tmp_path=sidecar_dir)
    failure_notes: list[str] = []
    blocked_calls: list[int] = []
    progress_labels: list[str] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw, "_post_failure_marker", lambda issue, note, dry_run: failure_notes.append(note) or True
    )
    monkeypatch.setattr(
        asw, "_set_status_blocked", lambda issue, dry_run: blocked_calls.append(issue) or True
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: progress_labels.append(label),
    )

    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    # (a) NO durable terminal record — neither epm:failure nor status:blocked fired
    # (the pod is alive; a terminal record would be false).
    assert failure_notes == []
    assert blocked_calls == []
    # (b) it degraded to the ALERT path (a wedge-alert progress note, NOT a
    # wedge-failover terminal note).
    assert "wedge-alert" in progress_labels
    assert "wedge-failover" not in progress_labels
    # (c) the wedge clock is PRESERVED (non-None) so the next tick re-detects the
    # still-RUNNING wedge and re-matures it — never silently stranded.
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is not None


@pytest.mark.parametrize(
    ("reason", "claims_not_terminated"),
    [
        # PRE-terminate: the PARTIAL-cell refusal happens BEFORE the terminate.
        ("runpod_wedge_inputs_unverified", True),
        # POST-terminate: the wedged pod WAS terminated; the re-provision failed.
        ("sidecar_persistence_failed", False),
        ("runpod_wedge_relaunch_spec_missing", False),
    ],
)
def test_wedge_blocked_marker_text_terminate_state_by_reason(
    isolated_registry, monkeypatch, reason, claims_not_terminated
):
    # CONCERN #3: the grouped "blocked" marker must NOT claim "did NOT terminate"
    # for the POST-terminate reasons (sidecar_persistence_failed,
    # runpod_wedge_relaunch_spec_missing) — those terminated the pod. Only the
    # PRE-terminate inputs_unverified reason says the pod is still running.
    now = 1_000_000.0
    _matured_confirming_state(now)
    failure_notes, _blocked, _progress = _patch_real_marker_recorders(
        monkeypatch,
        outcome="blocked",
        terminal={"status": "dead", "failure_class": "infra", "reason": reason},
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert len(failure_notes) == 1
    note = failure_notes[0].lower()
    if claims_not_terminated:
        assert "did not terminate" in note
        assert "still running" in note  # the pod is still billing until a human acts
    else:
        # POST-terminate: must NOT claim it did not terminate; must state it WAS.
        assert "did not terminate" not in note
        assert "terminated the wedged pod" in note
    # The reason is named in the note either way (mirrors the poller's path).
    assert reason in failure_notes[0]


def test_wedge_failover_sidecar_pod_name_mismatch_is_already_handled(monkeypatch, tmp_path):
    # CRITICAL #2: between _wedge_inputs_safe's read and _wedge_failover's re-read,
    # a revived poller could re-point the sidecar at a FRESH, HEALTHY pod. The
    # bounded-once lease inside _failover_wedged_runpod is keyed on the FRESH
    # handle, so it would NOT catch this. Defense: the helper asserts the
    # freshly-read handle.pod_name == info.name; a mismatch => "already-handled"
    # and _failover_wedged_runpod is NEVER called (never terminate the fresh pod).
    import explore_persona_space.backends.issue_dispatch as idp

    sidecar = tmp_path / "issue-692-handle.json"
    sidecar.write_text("{}")
    monkeypatch.setattr(idp, "resolve_handle_sidecar_path", lambda issue: (sidecar, [sidecar]))

    class _FreshHandle:
        # The sidecar now names a DIFFERENT (fresh) pod than the wedged one.
        pod_name = "pod-692-FRESH"
        job_id = "j692-fresh"

    monkeypatch.setattr(idp, "read_handle_sidecar", lambda path: _FreshHandle())
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})

    # info.name is the WEDGED pod the watcher observed ("pod-692").
    outcome, terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "already-handled"  # never terminate the fresh pod
    assert terminal is None
    assert calls == []  # _failover_wedged_runpod NEVER called against the fresh handle


def test_wedge_failover_sidecar_pod_name_match_proceeds(monkeypatch, tmp_path):
    # CRITICAL #2 (positive control): when the freshly-read handle still names the
    # WEDGED pod (the normal case), the helper proceeds to call
    # _failover_wedged_runpod — the binding defense does not block the real path.
    _stub_handle_sidecar(monkeypatch, tmp_path)  # _Handle.pod_name == "pod-692" == info.name
    calls = _stub_failover_fn(monkeypatch, returns={"status": "running"})
    outcome, _terminal = asw._wedge_failover(692, _wedged_info(), "1.10h", dry_run=False)
    assert outcome == "failover"
    assert len(calls) == 1  # the matching pod_name proceeds to the recovery


# ===========================================================================
# 9. Round-3 review fix (#770 r3): the terminal-recording-best-effort-before-clear
#    BLOCKER — gate the wedge-clock clear on BOTH the epm:failure marker AND the
#    set-status blocked actually landing (a transient task.py / flock / network
#    failure must NOT clear the clock, or the failure record is lost AND the next
#    tick treats the pod as freshly-wedged).
# ===========================================================================


@pytest.mark.parametrize(
    ("marker_ok", "blocked_ok"),
    [
        # _set_status_blocked fails (transient flock contention) — the brief's
        # primary mode.
        (True, False),
        # _post_failure_marker fails (the sister mode).
        (False, True),
        # Both fail.
        (False, False),
    ],
)
def test_wedge_terminal_recording_partial_does_not_clear_clock(
    isolated_registry, monkeypatch, marker_ok, blocked_ok
):
    # BLOCKER (wedge-terminal-recording-best-effort-before-clear, gated in r3,
    # made DURABLE in v2): on the no-capacity terminal outcome the watcher posts
    # epm:failure AND set-status blocked, THEN clears the wedge clock. Under the
    # v2 strategy pivot each write is wrapped in a bounded synchronous in-tick
    # retry (_retry_durable_write), so a write that returns a FIXED False on every
    # call means the retry budget is EXHAUSTED (_WEDGE_RECORD_RETRY_ATTEMPTS
    # attempts all swallowed). The invariant this test pins is UNCHANGED — a
    # persistently-False (non-transient) write must NOT clear the wedge clock,
    # because clearing it would both lose the failure record AND let the next tick
    # re-wedge the pod (defeating bounded-once from the watcher side), AND the
    # capacity-retry pass (which needs BOTH status:blocked + a parseable
    # epm:failure) never re-drives the re-drivable no_compute_available block. v2
    # additionally asserts the EXHAUSTED-retry attempt counts (vs r3's single shot)
    # and patches time.sleep so the parametrized full-failure matrix does not
    # really sleep ~3s per cell.
    now = 1_000_000.0
    _matured_confirming_state(now)
    prior = asw._load_pod_safety_state(692)
    assert prior["wedge_first_seen"] == now - (K + 50.0)  # the clock is live pre-fire

    sleeps: list[float] = []
    monkeypatch.setattr(asw.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: (
            "no-capacity",
            {"status": "dead", "failure_class": "infra", "reason": "no_compute_available"},
        ),
    )
    # Simulate the non-transient write failure: a swallowed marker/status write
    # returns a FIXED False on EVERY call (the round-3 bool contract), exactly as a
    # persistent task.py / flock / disk failure would — so the v2 retry budget is
    # exhausted.
    marker_attempts: list[int] = []
    blocked_attempts: list[int] = []
    monkeypatch.setattr(
        asw,
        "_post_failure_marker",
        lambda issue, note, dry_run: marker_attempts.append(1) or marker_ok,
    )
    monkeypatch.setattr(
        asw,
        "_set_status_blocked",
        lambda issue, dry_run: blocked_attempts.append(1) or blocked_ok,
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: None,
    )

    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    # The wedge clock is INTACT (NOT cleared to None) — the durable record did not
    # land after the bounded retries, so the episode is held.
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == now - (K + 50.0)
    assert loaded["wedge_first_seen"] is not None
    # The pod-incarnation GC anchor is untouched.
    assert loaded["first_seen"] == now - (K + 50.0)

    # v2: a write that returns a FIXED False is retried to EXHAUSTION
    # (_WEDGE_RECORD_RETRY_ATTEMPTS attempts); a write that succeeds first-try is
    # attempted exactly once. The marker write runs first; the blocked write runs
    # unconditionally after it (the caller retries each independently).
    expected_marker = 1 if marker_ok else asw._WEDGE_RECORD_RETRY_ATTEMPTS
    expected_blocked = 1 if blocked_ok else asw._WEDGE_RECORD_RETRY_ATTEMPTS
    assert len(marker_attempts) == expected_marker
    assert len(blocked_attempts) == expected_blocked
    # No real sleeping in the test; one backoff between attempts that remain (no
    # trailing sleep on the last failure). A first-try success contributes 0 sleeps.
    expected_sleeps = (0 if marker_ok else asw._WEDGE_RECORD_RETRY_ATTEMPTS - 1) + (
        0 if blocked_ok else asw._WEDGE_RECORD_RETRY_ATTEMPTS - 1
    )
    assert len(sleeps) == expected_sleeps


def test_wedge_terminal_recording_both_succeed_clears_clock(isolated_registry, monkeypatch):
    # Happy-path companion to the partial-failure parametrization above: when BOTH
    # the epm:failure marker AND set-status blocked land on the FIRST call, the
    # wedge clock IS cleared (the terminal failure is durably recorded) and NO
    # backoff sleep fires. Re-confirms the success gate after the v2 retry change
    # (the time.sleep recorder asserting zero calls doubles as a no-spurious-
    # backoff-on-the-happy-path check).
    now = 1_000_000.0
    _matured_confirming_state(now)
    sleeps: list[float] = []
    monkeypatch.setattr(asw.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: (
            "no-capacity",
            {"status": "dead", "failure_class": "infra", "reason": "no_compute_available"},
        ),
    )
    monkeypatch.setattr(asw, "_post_failure_marker", lambda issue, note, dry_run: True)
    monkeypatch.setattr(asw, "_set_status_blocked", lambda issue, dry_run: True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: None,
    )

    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None  # both landed -> clock cleared
    assert sleeps == []  # both succeeded first-try -> no backoff sleep


# ===========================================================================
# 10. Round-1 of the v2 STRATEGY PIVOT (#770 v2): the bounded synchronous
#     in-tick retry. r3 gated the wedge-clock clear on both durable writes
#     landing and promised "the next tick retries" on a partial write — but the
#     next tick can NEVER re-enter the wedge arm for a pod the failover already
#     terminated (it is gone from _running_managed_issue_pods(), RUNNING-only).
#     v2 makes the two task.py writes DURABLE by retrying them synchronously
#     in-tick with bounded exponential backoff before the function returns.
# ===========================================================================


def test_wedge_terminal_record_retries_then_succeeds(isolated_registry, monkeypatch):
    # THE FIX (§5.1): a TRANSIENT marker-write failure (False on the first call,
    # True on the second — a one-shot flock/network blip that recovers) is retried
    # IN-TICK and the durable record DOES land within the bounded window. Because
    # both writes ultimately succeed, the wedge clock IS cleared (the episode
    # resolves) — the exact gap the pivot closes, where r3 would have stranded the
    # task on the swallowed first write.
    now = 1_000_000.0
    _matured_confirming_state(now)

    sleeps: list[float] = []
    monkeypatch.setattr(asw.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: (
            "no-capacity",
            {"status": "dead", "failure_class": "infra", "reason": "no_compute_available"},
        ),
    )

    # _post_failure_marker: False on attempt 1, True on attempt 2 (recovers).
    marker_returns = iter([False, True])
    marker_attempts: list[int] = []
    monkeypatch.setattr(
        asw,
        "_post_failure_marker",
        lambda issue, note, dry_run: marker_attempts.append(1) or next(marker_returns),
    )
    blocked_attempts: list[int] = []
    monkeypatch.setattr(
        asw,
        "_set_status_blocked",
        lambda issue, dry_run: blocked_attempts.append(1) or True,
    )
    # No exhaustion alert should fire on the recover path.
    progress_labels: list[str] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: progress_labels.append(label),
    )

    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    # The marker write was attempted >= 2 times (failed once, then succeeded);
    # set-status blocked landed first-try.
    assert len(marker_attempts) == 2
    assert len(blocked_attempts) == 1
    # Exactly one backoff sleep (between the two marker attempts); none for the
    # first-try blocked write.
    assert len(sleeps) == 1
    assert sleeps[0] == asw._WEDGE_RECORD_RETRY_BASE_S  # 1.0s, the first backoff
    # Both writes ultimately landed -> the wedge clock IS cleared (episode resolved).
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    # No exhaustion alert fired (the durable record landed within the window).
    assert "wedge-failover" not in progress_labels


def test_wedge_terminal_record_exhausts_retries_keeps_clock(isolated_registry, monkeypatch):
    # EXHAUSTED RETRIES (§5.2): _post_failure_marker returns False on EVERY call (a
    # non-transient failure). The marker write is attempted exactly
    # _WEDGE_RECORD_RETRY_ATTEMPTS times, time.sleep fires exactly
    # _WEDGE_RECORD_RETRY_ATTEMPTS - 1 times (no trailing sleep on the last
    # failure), the wedge clock is INTACT (NOT cleared), the pod-incarnation
    # first_seen GC anchor is untouched, and a loud stderr alert + an
    # epm:progress wedge-failover exhaustion alert fire (the human signals). This
    # pins the acceptable residual: a genuinely non-transient failure does not
    # silently strand.
    now = 1_000_000.0
    _matured_confirming_state(now)

    sleeps: list[float] = []
    monkeypatch.setattr(asw.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda issue, info, wedged_h, dry_run: (
            "no-capacity",
            {"status": "dead", "failure_class": "infra", "reason": "no_compute_available"},
        ),
    )
    marker_attempts: list[int] = []
    monkeypatch.setattr(
        asw,
        "_post_failure_marker",
        lambda issue, note, dry_run: marker_attempts.append(1) or False,
    )
    # set-status blocked succeeds (the marker is what exhausts); both must succeed
    # for the clock to clear, so the marker exhaustion alone holds the clock.
    monkeypatch.setattr(asw, "_set_status_blocked", lambda issue, dry_run: True)
    progress_labels: list[str] = []
    progress_notes: list[str] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: (
            progress_labels.append(label),
            progress_notes.append(note),
        ),
    )

    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    # The marker write was attempted exactly _WEDGE_RECORD_RETRY_ATTEMPTS times.
    assert len(marker_attempts) == asw._WEDGE_RECORD_RETRY_ATTEMPTS
    # time.sleep fired exactly _WEDGE_RECORD_RETRY_ATTEMPTS - 1 times (no trailing
    # sleep on the last failure). set-status blocked succeeded first-try (0 sleeps).
    assert len(sleeps) == asw._WEDGE_RECORD_RETRY_ATTEMPTS - 1
    # The wedge clock is INTACT (NOT cleared) — the durable record did not land.
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == now - (K + 50.0)
    assert loaded["wedge_first_seen"] is not None
    # The pod-incarnation GC anchor is untouched.
    assert loaded["first_seen"] == now - (K + 50.0)
    # The exhaustion alert fired (the human signal).
    assert "wedge-failover" in progress_labels
    assert any("EXHAUSTED" in n or "FAILED to durably record" in n for n in progress_notes)


def test_wedge_terminated_pod_absent_from_running_set_no_redrive_next_tick(monkeypatch):
    # CODEX r3 REACHABILITY PIN (§5.3): the reason the in-tick retry is necessary.
    # _process_wedged_pod is re-entered ONLY for pods in
    # _running_managed_issue_pods(), which filters desired_status != "RUNNING".
    # Once _failover_wedged_runpod TERMINATES the wedged pod, it leaves the RUNNING
    # set, so the r3 "retry next tick" can NEVER fire for it. Pin that the
    # RUNNING-only filter EXCLUDES a terminated managed pod for the same issue, so
    # a future edit that reintroduces a "retry next tick" assumption fails here.
    running = PodInfo(
        pod_id="p692-RUNNING",
        name="pod-692",
        desired_status="RUNNING",
        ssh_host=None,
        ssh_port=None,
    )
    terminated = PodInfo(
        pod_id="p692-TERMINATED",
        name="pod-692",
        desired_status="TERMINATED",
        ssh_host=None,
        ssh_port=None,
    )
    exited = PodInfo(
        pod_id="p692-EXITED",
        name="pod-692",
        desired_status="EXITED",
        ssh_host=None,
        ssh_port=None,
    )
    monkeypatch.setattr(asw, "list_team_pods", lambda: [running, terminated, exited])

    out = asw._running_managed_issue_pods()
    assert out is not None
    # Only the RUNNING managed pod survives; the TERMINATED + EXITED ones (the
    # post-failover states) are filtered out, so _process_wedged_pod is never
    # re-entered for the terminated pod.
    pod_ids = [pod_id for (_issue, pod_id, _name, _info) in out]
    assert pod_ids == ["p692-RUNNING"]
    assert "p692-TERMINATED" not in pod_ids
    assert "p692-EXITED" not in pod_ids


# ===========================================================================
# 11. The #1667 owner-liveness guard: decide_pod_wedge owner axis, the
#     tri-state _wedge_owner_live probe (#1586 incident replay), the
#     alert-owner-live defer block (once-per-episode marker, clock
#     preserved), the kill switch, the env window, and the self-defer-loop
#     sentinel pin.
# ===========================================================================


def test_wedge_decide_owner_live_demotes_to_alert():
    # Durability pin (#1667): confirmed + keep_running=False + inputs_ok=True
    # but a LIVE owning session -> the terminate demotes to the defer action.
    assert _decide(keep_running=False, inputs_ok=True, owner_live=True) == (
        "alert-owner-live",
        0,
    )


def test_wedge_decide_owner_unknown_demotes_to_alert():
    # The owner axis extends the MF2 only-the-literal-False-acts invariant
    # (the test_wedge_decision_invariant_terminate_only_on_literal_false
    # sweep pattern): "unknown" AND any garbage value demote; terminate fires
    # ONLY on the literal False.
    for owner_live in (True, "unknown", None, 0, 1, "live", ""):
        action, counter = _decide(keep_running=False, inputs_ok=True, owner_live=owner_live)
        assert action == "alert-owner-live", owner_live
        assert counter == 0, owner_live
        assert action != "terminate-failover", owner_live
    # The full (keep_running x inputs_ok x owner_live) grid: terminate ONLY at
    # the one literal-False + True + literal-False cell.
    for keep_running in (True, False, "unknown"):
        for inputs_ok in (True, False):
            for owner_live in (True, False, "unknown"):
                action, _ = _decide(
                    keep_running=keep_running, inputs_ok=inputs_ok, owner_live=owner_live
                )
                if keep_running is False and inputs_ok and owner_live is False:
                    assert action == "terminate-failover"
                else:
                    assert action != "terminate-failover", (keep_running, inputs_ok, owner_live)


def test_wedge_decide_owner_absent_still_terminates():
    # The terminate-still-fires-when-owner-dead pin: the #664 dead-everything
    # case is preserved (owner_live resolved to the literal False).
    assert _decide(keep_running=False, inputs_ok=True, owner_live=False) == (
        "terminate-failover",
        0,
    )


def test_wedge_decide_owner_default_backcompat():
    # Param omitted -> default False -> terminate unchanged (pins kill-switch
    # equivalence at the pure layer: the kill-switch path passes the same
    # literal False every pre-#1667 call site implied).
    assert _decide(keep_running=False, inputs_ok=True) == ("terminate-failover", 0)


# Content-thinned REAL event rows from tasks/*/1586/events.jsonl (the #1586
# incident this guard closes): the crash-fix-round epm:failure + the
# stage-dispatch breadcrumb, plus the 05:42:00Z wedge-failover WATCHER row —
# included so its exclusion from leg-1 recency is demonstrated by the code
# path (_latest_nonwatcher_event_ts), not by fixture omission.
_ROWS_1586 = [
    {
        "kind": "epm:progress",
        "ts": "2026-07-24T05:16:00Z",
        "by": "poll_pipeline",
        "note": "phase transition: p7_margin -> p8_capture",
    },
    {
        "kind": "epm:failure",
        "ts": "2026-07-24T05:18:59Z",
        "by": "unknown",
        "note": "failure_class: code phase: p8_capture (fu resume 6, fix_sha 6e329e7c)",
    },
    {
        "kind": "epm:progress",
        "ts": "2026-07-24T05:19:13Z",
        "by": "unknown",
        "note": "stage-dispatch stage=followup-implementing round=10 "
        "subagent=experiment-implementer label=caveat-fix",
    },
    {
        "kind": "epm:progress",
        "ts": "2026-07-24T05:42:00Z",
        "by": "autonomous_session_watch",
        "note": asw._WEDGE_FAILOVER_NOTE_SENTINEL
        + " TERMINATED + re-provisioned a FRESH pod for issue #1586",
    },
]


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _fixture_1586_events(monkeypatch):
    """Route the REAL _task_events -> _latest_nonwatcher_event_ts path through
    the #1586 fixture: the subprocess boundary (task.py list-markers --json)
    is the ONLY thing faked, exactly as _fake_task_view does."""

    class _Out:
        returncode = 0
        stdout = json.dumps(_ROWS_1586)

    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: _Out())


def test_wedge_owner_live_1586_timeline(monkeypatch):
    # Incident replay of _wedge_owner_live against the real #1586 rows.
    monkeypatch.delenv("EPM_DISABLE_WEDGE_OWNER_GUARD", raising=False)
    monkeypatch.delenv("EPM_WEDGE_OWNER_RECENT_H", raising=False)
    _fixture_1586_events(monkeypatch)

    # LIVE arm: at the ~05:33Z decision instant the newest non-watcher marker
    # (stage-dispatch, 05:19:13Z) is ~14.5 min old << 2h -> leg 1 reads True
    # (the demote fires on its own motivating incident). Leg 2 is
    # short-circuited (returning "absent" here must NOT flip the verdict).
    monkeypatch.setattr(
        asw, "_keep_running_owner_state", lambda issue, now, min_idle_s: ("absent", {})
    )
    now_live = _epoch("2026-07-24T05:33:43Z")
    assert asw._wedge_owner_live(1586, now_live) is True

    # DEAD-owner arm: at 07:30:00Z every NON-watcher row is > 2h old
    # (05:19:13Z -> gap 7847s >= 7200) while the 05:42:00Z WATCHER failover
    # row is only 6480s old — if the sentinel row counted, leg 1 would read
    # LIVE and the wedge would defer to its own marker forever (the
    # self-defer loop). Exclusion by the code path + leg 2 "absent" -> the
    # literal False (terminate may proceed — the #664 dead-everything case).
    now_dead = _epoch("2026-07-24T07:30:00Z")
    assert asw._wedge_owner_live(1586, now_dead) is False

    # Unresolvable leg 2 ("unknown" owner state) with the same stale markers
    # -> "unknown", never False (uncertainty must not enable terminate).
    monkeypatch.setattr(
        asw, "_keep_running_owner_state", lambda issue, now, min_idle_s: ("unknown", {})
    )
    assert asw._wedge_owner_live(1586, now_dead) == "unknown"


def test_wedge_owner_live_tsless_events_and_exception_unknown(monkeypatch):
    # ts-less events (leg 1 unresolvable) -> "unknown" even when leg 2 reads
    # wedged/absent; and ANY probe exception -> "unknown" (loud, never False).
    monkeypatch.delenv("EPM_DISABLE_WEDGE_OWNER_GUARD", raising=False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress"}])
    monkeypatch.setattr(
        asw, "_keep_running_owner_state", lambda issue, now, min_idle_s: ("absent", {})
    )
    assert asw._wedge_owner_live(1586, time.time()) == "unknown"
    # Exception path: the events read raises -> "unknown".
    monkeypatch.setattr(
        asw, "_task_events", lambda issue: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert asw._wedge_owner_live(1586, time.time()) == "unknown"


@pytest.mark.parametrize("owner_live", [True, "unknown"])
def test_wedge_owner_defer_marker_posted_once_per_episode(
    isolated_registry, monkeypatch, capsys, owner_live
):
    # Integration through _process_wedged_pod: the confirming tick with a
    # live/"unknown" owner posts ONE defer marker (sentinel + pod_id +
    # terminate recipe in the note), preserves the wedge clock, and later
    # confirmed ticks do NOT repost. The wedge-ALERT path is NOT consumed:
    # no wedge-alert marker fires and wedge_alerted stays False, so a LATER
    # gated-off tick still gets its once-per-episode alert (the
    # "alert-owner-live return skips nothing pre-existing" pin — this branch
    # was previously terminate-only, so no pre-#1667 marker fired on it).
    now = 1_000_000.0
    onset = now - (K + 50.0)
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=onset,
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": onset, "pod_id": "p692"},
    )
    stops, _posts, failovers = _patch_wedge_io(
        monkeypatch,
        status="running",
        keep_running=False,
        inputs_ok=True,
        owner_live=owner_live,
    )
    notes: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: notes.append((issue, label, note)),
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert failovers == []  # the terminate was DEFERRED
    assert stops == []
    defer_notes = [n for _i, label, n in notes if label == "wedge-owner-defer"]
    assert len(defer_notes) == 1
    assert asw._WEDGE_OWNER_DEFER_NOTE_SENTINEL in defer_notes[0]
    assert "p692" in defer_notes[0]
    assert "pod.py terminate --issue 692 --yes" in defer_notes[0]
    assert f"owner_live={owner_live}" in defer_notes[0]
    # The stderr defer line fires on BOTH the True and the "unknown" branch.
    assert "WEDGE-OWNER-DEFER issue #692" in capsys.readouterr().err
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == onset  # CLOCK PRESERVED (acceptance #4)
    assert loaded["wedge_owner_defer_noted"] is True
    assert loaded["wedge_alerted"] is False  # the wedge-alert dedup is NOT consumed
    assert not any(label == "wedge-alert" for _i, label, _n in notes)
    # Tick 2 (counter was reset to 0 -> re-accumulates) + tick 3 (re-confirmed
    # -> the probe re-runs, still live) -> NO second defer marker.
    asw._process_pod(692, "p692", _wedged_info(), now + 600, dry_run=False, threshold=2)
    asw._process_pod(692, "p692", _wedged_info(), now + 1200, dry_run=False, threshold=2)
    assert len([n for _i, label, n in notes if label == "wedge-owner-defer"]) == 1
    assert failovers == []


def test_wedge_owner_defer_rearms_after_episode_reset(isolated_registry, monkeypatch):
    # Episode-reset re-arm: after a defer episode ends (the heal-path
    # _clear_wedge_state clear), a FRESH matured episode re-posts the defer
    # marker — and an MF4 pod_id-mismatch save resets the dedup flag too
    # (pinned independent of the _CARRY-vs-kwarg persistence choice).
    now = 1_000_000.0
    onset = now - (K + 50.0)

    def _seed(pod_id: str) -> None:
        asw._save_pod_safety_state(
            692,
            pod_id,
            missed=0,
            alerted=False,
            last_progress_ts=None,
            wedge_first_seen=onset,
            wedge_missed=1,
            wedge_alerted=False,
            wedge_owner_defer_noted=False,
            prev={"first_seen": onset, "pod_id": pod_id},
        )

    _seed("p692")
    _stops, posts, _fo = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True, owner_live=True
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert [label for _i, label in posts].count("wedge-owner-defer") == 1
    assert asw._load_pod_safety_state(692)["wedge_owner_defer_noted"] is True
    # Heal path: the port re-appears -> _clear_wedge_state ends the episode
    # and clears the defer dedup with the other wedge fields.
    asw._clear_wedge_state(692, "p692")
    assert asw._load_pod_safety_state(692)["wedge_owner_defer_noted"] is False
    # A fresh matured episode re-fires the defer marker.
    _seed("p692")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert [label for _i, label in posts].count("wedge-owner-defer") == 2
    # MF4: a stale noted flag under an OLD pod_id never shields the NEW
    # incarnation — the mismatch save persists the reset flag.
    asw._save_pod_safety_state(
        692,
        "p_OLD",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=onset,
        wedge_missed=5,
        wedge_alerted=True,
        wedge_owner_defer_noted=True,
        prev={"first_seen": onset, "pod_id": "p_OLD"},
    )
    asw._process_pod(692, "p_NEW", _wedged_info(pod_id="p_NEW"), now, dry_run=False, threshold=2)
    assert asw._load_pod_safety_state(692)["wedge_owner_defer_noted"] is False


def test_wedge_owner_defer_then_owner_dead_terminates(isolated_registry, monkeypatch):
    # Defer tick, then the owner quiets (probe flips to the literal False):
    # the re-confirmation ticks dispatch _wedge_failover WITHOUT restarting
    # the K-floor maturation (the clock is never re-stamped from zero).
    now = 1_000_000.0
    onset = now - (K + 50.0)
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=onset,
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": onset, "pod_id": "p692"},
    )
    owner = {"live": True}
    _stops, _posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True
    )
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: owner["live"])
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert failovers == []  # deferred to the live owner
    assert asw._load_pod_safety_state(692)["wedge_first_seen"] == onset
    # Owner activity quiets.
    owner["live"] = False
    # Re-confirmation tick 1: counter 0 -> 1 < threshold -> keep (clock intact).
    asw._process_pod(692, "p692", _wedged_info(), now + 600, dry_run=False, threshold=2)
    assert failovers == []
    assert asw._load_pod_safety_state(692)["wedge_first_seen"] == onset  # never restarted
    # Re-confirmation tick 2: confirmed again + owner dead -> terminate fires.
    asw._process_pod(692, "p692", _wedged_info(), now + 1200, dry_run=False, threshold=2)
    assert failovers == [692]


def test_wedge_owner_guard_kill_switch(isolated_registry, monkeypatch):
    # EPM_DISABLE_WEDGE_OWNER_GUARD=1 -> byte-equivalent pre-#1667 terminate:
    # the probe legs (events read, owner-state resolver) are NEVER invoked
    # (raise-if-called) and the terminate proceeds.
    monkeypatch.setenv("EPM_DISABLE_WEDGE_OWNER_GUARD", "1")
    now = 1_000_000.0
    onset = now - (K + 50.0)
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=onset,
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": onset, "pod_id": "p692"},
    )
    _stops, _posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True
    )
    # The REAL probe runs (un-stub it), but its legs must never be touched.
    monkeypatch.setattr(asw, "_wedge_owner_live", _REAL_OWNER_LIVE)

    def _raise(*_a, **_k):
        raise AssertionError("probe leg invoked despite the kill switch")

    monkeypatch.setattr(asw, "_task_events", _raise)
    monkeypatch.setattr(asw, "_latest_nonwatcher_event_ts", _raise)
    monkeypatch.setattr(asw, "_keep_running_owner_state", _raise)
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert failovers == [692]  # terminate proceeded exactly as pre-#1667


def test_wedge_owner_recent_window_env_and_malformed_fallback(monkeypatch):
    # EPM_WEDGE_OWNER_RECENT_H honored (HOURS); malformed / non-positive /
    # empty fall back to the 7200s default (never a kill switch).
    monkeypatch.delenv("EPM_WEDGE_OWNER_RECENT_H", raising=False)
    assert asw.WEDGE_OWNER_RECENT_S == 2 * 3600
    assert asw._wedge_owner_recent_s() == 7200.0
    monkeypatch.setenv("EPM_WEDGE_OWNER_RECENT_H", "6")
    assert asw._wedge_owner_recent_s() == 6 * 3600.0
    monkeypatch.setenv("EPM_WEDGE_OWNER_RECENT_H", "0.5")
    assert asw._wedge_owner_recent_s() == 1800.0
    for bad in ("garbage", "", "0", "-5"):
        monkeypatch.setenv("EPM_WEDGE_OWNER_RECENT_H", bad)
        assert asw._wedge_owner_recent_s() == 7200.0
    # Kill-switch reader: default enabled; truthy values disable.
    monkeypatch.delenv("EPM_DISABLE_WEDGE_OWNER_GUARD", raising=False)
    assert asw._wedge_owner_guard_enabled() is True
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("EPM_DISABLE_WEDGE_OWNER_GUARD", truthy)
        assert asw._wedge_owner_guard_enabled() is False


def test_wedge_owner_defer_sentinel_in_watcher_note_sentinels():
    # The self-defer-loop pin: the defer marker is epm:progress; if it counted
    # as non-watcher activity, leg 1 of _wedge_owner_live would read the
    # watcher's OWN marker as owner liveness and defer forever. Membership in
    # _WATCHER_NOTE_SENTINELS excludes it from BOTH recency filters.
    assert asw._WEDGE_OWNER_DEFER_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    row = {
        "kind": "epm:progress",
        "ts": "2026-07-24T05:42:00Z",
        "note": f"{asw._WEDGE_OWNER_DEFER_NOTE_SENTINEL} WEDGE FAILOVER DEFERRED TO LIVE OWNER",
        "by": "autonomous_session_watch",
    }
    assert asw._latest_nonwatcher_event_ts([row]) is None
    assert asw._latest_progress_ts([row]) is None


def test_wedge_owner_probe_lazy(isolated_registry, monkeypatch):
    # The probe is NEVER invoked below eligibility: below confirmation, when
    # keep_running is True/"unknown", or when inputs_ok=False (the MF2 lazy
    # pattern — the events read + daemon /list are paid only for a matured,
    # otherwise terminate-eligible wedge).
    now = 1_000_000.0

    def _run(status_kw):
        _patch_wedge_io(monkeypatch, **status_kw)
        monkeypatch.setattr(
            asw,
            "_wedge_owner_live",
            lambda issue, now: (_ for _ in ()).throw(
                AssertionError("owner probe invoked below eligibility")
            ),
        )
        asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)

    def _seed_matured():
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

    # (a) Below the K floor (fresh onset this tick): probe never runs.
    _run(dict(status="running", keep_running=False, inputs_ok=True))
    # (b) Past K but unconfirmed (miss 0 -> 1 < threshold): probe never runs.
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
    _run(dict(status="running", keep_running=False, inputs_ok=True))
    # (c) Confirmed + keep_running=True: probe never runs.
    _seed_matured()
    _run(dict(status="running", keep_running=True, inputs_ok=True))
    # (d) Confirmed + keep_running="unknown": probe never runs.
    _seed_matured()
    _run(dict(status="running", keep_running="unknown", inputs_ok=True))
    # (e) Confirmed + untagged + inputs_ok=False: probe never runs.
    _seed_matured()
    _run(dict(status="running", keep_running=False, inputs_ok=False))


def test_wedge_owner_defer_dry_run_no_writes(isolated_registry, monkeypatch):
    # Dry-run on a DEFERRING tick (the alert-owner-live branch): no marker
    # subprocess fires (the REAL _post_progress_marker body dry-run-prints
    # only) and no state file is written (mirrors the keep_running_owner
    # file's dry-run contract).
    now = 1_000_000.0
    onset = now - (K + 50.0)
    asw._save_pod_safety_state(
        692,
        "p692",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        wedge_first_seen=onset,
        wedge_missed=1,
        wedge_alerted=False,
        prev={"first_seen": onset, "pod_id": "p692"},
    )
    seeded = asw._load_pod_safety_state(692)
    _stops, posts, failovers = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True, owner_live=True
    )
    # Restore the REAL marker helper body; its subprocess boundary raises if
    # ever touched (dry-run must return before the subprocess).
    monkeypatch.setattr(asw, "_post_progress_marker", _REAL_POST_PROGRESS)
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("marker subprocess fired under dry_run")
        ),
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=True, threshold=2)
    assert failovers == []
    assert posts == []  # the stub was restored; nothing recorded there either
    assert asw._load_pod_safety_state(692) == seeded  # no state write under dry_run


# ===========================================================================
# 13. #1668 D10 (T13): the watcher's durable wedge-failover note is TRUTHFUL
#     on the D9 probe-success return — the REAL _wedge_failover mapping
#     (structural current_phase key) + the REAL _handle_wedge_failover_outcome
#     note composer run end-to-end; only backend_poll._failover_wedged_runpod
#     (the shared recovery, via _stub_failover_fn) and the I/O seams are
#     stubbed. Constants come from backend_poll (never literals).
# ===========================================================================


def test_wedge_probe_alive_outcome_posts_truthful_note_and_failover_note_unchanged(
    isolated_registry, monkeypatch, tmp_path
):
    """T13: leg (a) — the D9 probe-success running return (current_phase ==
    RUNPOD_WORKLOAD_OBSERVED_PHASE) posts a note that claims NEITHER a
    terminate NOR a re-provision (nothing happened to the pod), names the
    probe, and clears the wedge-episode state exactly as the success branch
    does today. Leg (b) — the relaunch-success running return
    (current_phase == FRESH_POD_PHASE) still posts the established
    "TERMINATED + re-provisioned" note (no regression on true-failover
    narration)."""
    now = 1_000_000.0
    posts: list[tuple[int, str, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: True)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)  # #1667: no owner
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: (_ for _ in ()).throw(
            AssertionError("watcher must never _stop_pod")
        ),
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, note, label)),
    )
    _stub_handle_sidecar(monkeypatch, tmp_path)

    # ── leg (a): D9 probe-success shape → the "probe-alive" truthful note ──
    _matured_confirming_state(now)
    _stub_failover_fn(
        monkeypatch,
        returns={
            "status": "running",
            "current_phase": bp.RUNPOD_WORKLOAD_OBSERVED_PHASE,
            "log_tail_excerpt": "no-port API read contradicted by live SSH",
        },
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    wedge_posts = [(i, n) for i, n, label in posts if label == "wedge-failover"]
    assert len(wedge_posts) == 1
    note = wedge_posts[0][1]
    assert "TERMINATED" not in note  # nothing was terminated — no false durable record
    assert "re-provisioned" not in note  # nothing was re-provisioned
    assert "probe" in note  # names the liveness probe as the reason
    # The wedge-episode state clears exactly as the success branch does today.
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None
    assert loaded["wedge_missed"] == 0
    assert loaded["wedge_alerted"] is False

    # ── leg (b): relaunch-success shape → the terminate note still posts ──
    posts.clear()
    _matured_confirming_state(now)
    _stub_failover_fn(
        monkeypatch,
        returns={
            "status": "running",
            "current_phase": bp.RUNPOD_NOPORT_WEDGE_FAILOVER_FRESH_POD_PHASE,
            "log_tail_excerpt": "terminated + re-provisioned fresh pod",
        },
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    wedge_posts = [(i, n) for i, n, label in posts if label == "wedge-failover"]
    assert len(wedge_posts) == 1
    note_b = wedge_posts[0][1]
    assert "TERMINATED + re-provisioned" in note_b  # the true-failover narration is unchanged
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] is None  # state clears on the failover outcome too
