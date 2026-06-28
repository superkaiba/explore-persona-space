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


def test_wedge_boundary_confirm_transition():
    # MF5: the action transitions EXACTLY at wedge_missed + 1 == threshold.
    # threshold=3: wedge_missed=1 -> new=2 < 3 -> KEEP; wedge_missed=2 -> new=3 -> act.
    assert _decide(wedged_for=K + 1, wedge_missed=1, threshold=3, inputs_ok=True) == ("keep", 2)
    assert _decide(wedged_for=K + 1, wedge_missed=2, threshold=3, inputs_ok=True) == ("stop", 0)


def test_wedge_stop_confirmed_inputs_safe_tag_absent():
    # Confirmed past K + inputs verified on HF + keep_running is the literal
    # False -> the reversible STOP fires.
    assert _decide(keep_running=False, inputs_ok=True) == ("stop", 0)


def test_wedge_alert_inputs_unverified():
    # Confirmed + inputs NOT verified on HF -> ALERT-only (never strand work).
    assert _decide(keep_running=False, inputs_ok=False) == ("alert", 0)


def test_wedge_alert_keep_running_true():
    # Confirmed + the keep-running tag is present -> ALERT-only (tag exemption).
    # Even with inputs_ok=True the stop is suppressed.
    assert _decide(keep_running=True, inputs_ok=True) == ("alert", 0)


def test_wedge_alert_keep_running_unknown():
    # MF2 closure: confirmed + inputs_ok=True but the keep-running read FAILED
    # ("unknown") -> ALERT, NOT stop. A persistent tag-read failure must never
    # silently override a (possibly present) keep-running tag on a live-work pod.
    assert _decide(keep_running="unknown", inputs_ok=True) == ("alert", 0)


def test_wedge_decision_invariant_stop_only_on_literal_false():
    # The decision invariant (MF2): ("stop", _) is returned ONLY when
    # keep_running is the literal False AND inputs_ok is True. Sweep the cross
    # product and assert no other combination stops.
    for keep_running in (True, "unknown"):
        for inputs_ok in (True, False):
            action, _ = _decide(keep_running=keep_running, inputs_ok=inputs_ok)
            assert action != "stop", (keep_running, inputs_ok)
    # Only False + True stops.
    assert _decide(keep_running=False, inputs_ok=True)[0] == "stop"
    assert _decide(keep_running=False, inputs_ok=False)[0] == "alert"


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
):
    """Monkeypatch the wedge arm's I/O: task status, the tri-state keep-running
    read, the inputs-on-HF gate, and the action helpers. Returns the recorders
    ``(stops, posts)``."""
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: keep_running)
    monkeypatch.setattr(asw, "_wedge_inputs_safe", lambda issue: inputs_ok)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    # Guard: a non-wedge fall-through path must not touch the status-class I/O
    # in these wedge tests (they all pass a wedged info + non-DONE status).
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    return stops, posts


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
    stops, _posts = _patch_wedge_io(monkeypatch, status="running")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []  # the K floor measures the no-port episode, not pod uptime
    loaded = asw._load_pod_safety_state(692)
    assert loaded["wedge_first_seen"] == now  # stamped at ONSET this tick


def test_wedge_first_tick_zero_keeps(isolated_registry, monkeypatch):
    # The first wedged tick of a fresh incarnation -> wedged_for=0 -> KEEP.
    now = 1_000_000.0
    stops, posts = _patch_wedge_io(monkeypatch, status="approved")
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
    stops, _posts = _patch_wedge_io(monkeypatch, status="running")
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
    stops, posts = _patch_wedge_io(monkeypatch, status="running")
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert posts == []
    assert asw._load_pod_safety_state(692)["wedge_missed"] == 1


def test_wedge_confirming_transition_stops(isolated_registry, monkeypatch):
    # The confirming tick (wedge_missed 1 -> new 2 == threshold), inputs_ok=True,
    # keep_running=False -> STOP fires exactly here (reversible _stop_pod).
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
    stops, posts = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == [692]
    assert "wedge-stop" in [label for _i, label in posts]
    # The wedge fields are cleared on the stop, the first_seen GC anchor survives.
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
    stops, posts = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=False
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
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
    stops, posts = _patch_wedge_io(monkeypatch, status="running", keep_running=True, inputs_ok=True)
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
    assert "wedge-alert" in [label for _i, label in posts]


def test_wedge_keep_running_unknown_alerts_never_stops(isolated_registry, monkeypatch):
    # MF2 closure end-to-end: confirmed + inputs_ok=True + keep_running="unknown"
    # -> ALERT, and _stop_pod is NEVER called.
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
    stops, posts = _patch_wedge_io(
        monkeypatch, status="running", keep_running="unknown", inputs_ok=True
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    assert stops == []
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
    stops, posts = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=False
    )
    asw._process_pod(692, "p692", _wedged_info(), now, dry_run=False, threshold=2)
    asw._process_pod(692, "p692", _wedged_info(), now + 600, dry_run=False, threshold=2)
    wedge_alerts = [label for _i, label in posts if label == "wedge-alert"]
    assert len(wedge_alerts) == 1  # once per episode, not per tick
    assert stops == []  # the gated-off (inputs-unverified) path never stops
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
    stops, _posts = _patch_wedge_io(
        monkeypatch, status="running", keep_running=False, inputs_ok=True
    )
    # The live pod has a NEW pod_id with no port (freshly wedged this tick).
    asw._process_pod(692, "p_NEW", _wedged_info(pod_id="p_NEW"), now, dry_run=False, threshold=2)
    assert stops == []  # stale wedge state from the old pod must not stop the new one
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
