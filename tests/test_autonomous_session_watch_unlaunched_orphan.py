"""Unit tests for the #1519 ALERT-ONLY unlaunched-orphan pod-safety arm.

The #1481 incident this arm closes: ``pod_lifecycle provision`` deliberately
detaches (``setsid``, #573); a backgrounded wait-for-capacity re-run outlived
its requesting orchestrator turn and delivered pod-1481 (8xH100, ~$32/hr)
with no owner. No ``epm:run-launched`` (or any pre-launch signal) was ever
posted; the task sat at status ``running`` posting fresh markers from its GCP
lanes, so every existing watcher arm read the pod as healthy. It idled ~2h
until a manual burn probe found it, and its name blocked a later RunPod
fallback provision. Covers:

* the durability pin replaying the #1481 timeline shape end-to-end through
  ``_process_pod`` — exactly ONE marker + ONE push per pod incarnation
  (``unlaunched_orphan_noted`` dedup), never a stop/terminate;
* the additive-only contract on BOTH sides: a FIRING alert on an ACTIVE task
  still lets the status-class decision run afterward (and the later save
  carries the noted flag), and a DONE-status task is a NON-fire control whose
  canonical auto-stop still acts;
* the pure predicate's legs (status class / bare name / noted / created_ts /
  grace / launch-signal recency incl. the skew window) + the empty-events
  fail-toward-keep hardening + the lazy keep-running shield;
* the env-override reader + the ``_save_pod_safety_state`` pod_id-keyed carry
  + the in-memory mirror (the #1490 r2 clobber lesson).

Follows ``tests/test_autonomous_session_watch_orphan_pod.py`` conventions:
PodInfo fixtures, the patched state dir, ``task.py`` reads monkeypatched, no
network, no real marker posts.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

GRACE = asw.UNLAUNCHED_ORPHAN_GRACE_SEC  # 3600s (env-overridable at call time)
SKEW = asw.UNLAUNCHED_ORPHAN_LAUNCH_SKEW_SEC  # 900s (fixed)


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the per-pod state dir at a tmp dir (mirrors the orphan-pod suite)."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _iso(epoch: float) -> str:
    """Canonical task-event / RunPod ``createdAt`` timestamp shape."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _info(
    pod_id: str = "p1481",
    name: str = "pod-1481",
    created_at: str | None = None,
    gpu_count: int | None = 8,
    gpu_type_id: str | None = "NVIDIA H100 80GB HBM3",
) -> PodInfo:
    """A RUNNING pod with a public port (so the #692 wedge arm never handles it)."""
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status="RUNNING",
        gpu_count=gpu_count,
        gpu_type_id=gpu_type_id,
        ssh_host="1.2.3.4",
        ssh_port=22001,
        created_at=created_at,
    )


def _events_1481(now: float) -> list[dict]:
    """The #1481 timeline shape: fresh NON-watcher progress rows (the task's
    GCP lanes kept posting, so ``_status_class`` reads pod-active-fresh) PLUS
    launch-ADJACENT rows whose kinds are deliberately NOT in the evidence set
    — and ZERO rows of any ``_POD_FOLLOWUP_SIGNAL_KINDS`` kind.

    Durability pin (binding concern 3): the adjacent rows are NEWER than
    ``created_ts - SKEW`` for the 2h-old pod the fire tests use, so if anyone
    ever widens the evidence set to ``epm:cluster-launched`` /
    ``epm:backend-selected`` the predicate flips to keep and the fire tests
    FAIL LOUD — exactly the widening that would have suppressed the
    motivating incident's alert.
    """
    return [
        {
            "kind": "epm:progress",
            "ts": _iso(now - 42 * 60),
            "note": "run-status: gcp lanes healthy, gpu-idle advisory",
            "by": "poll_pipeline",
        },
        {
            "kind": "epm:results",
            "ts": _iso(now - 90 * 60),
            "note": "partial results landed",
            "by": "poll_pipeline",
        },
        {
            "kind": "epm:cluster-launched",
            "ts": _iso(now - 100 * 60),
            "note": "gcp cluster up",
            "by": "dispatch",
        },
        {
            "kind": "epm:backend-selected",
            "ts": _iso(now - 115 * 60),
            "note": "chosen_kind=gcp reason=auto",
            "by": "dispatch",
        },
    ]


@pytest.fixture
def marker_recorder(monkeypatch):
    posts: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label=None: posts.append((issue, note, label)),
    )
    return posts


@pytest.fixture
def push_recorder(monkeypatch):
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append(msg) or True)
    return pushes


@pytest.fixture
def stop_recorder(monkeypatch):
    stops: list[int] = []
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    return stops


@pytest.fixture
def active_1481_task(monkeypatch):
    """The #1481 fixture: RUNNING status, the incident-shaped events, no shields.

    ``_latest_progress_ts`` is deliberately NOT monkeypatched — the real
    filter computes pod-active-fresh from the fixture's fresh non-watcher
    rows, exactly the incident's every-existing-arm-reads-healthy property.
    """
    now = time.time()
    events = _events_1481(now)
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_task_events", lambda issue: events)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    return now, events


def _unlaunched_posts(posts):
    return [p for p in posts if asw._UNLAUNCHED_ORPHAN_NOTE_SENTINEL in p[1]]


def _fire_kwargs(now: float, **overrides) -> dict:
    """decide_unlaunched_orphan kwargs that ALERT unless overridden."""
    kwargs = {
        "status": "running",
        "pod_name": "pod-1481",
        "issue": 1481,
        "created_ts": now - 2 * 3600,
        "latest_launch_signal_ts": None,
        "noted": False,
        "now": now,
        "grace_sec": GRACE,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 1. The durability pin: the #1481 shape fires (end-to-end, once per pod)
# ---------------------------------------------------------------------------


def test_unlaunched_orphan_fires_on_1481_shape(
    isolated_registry, marker_recorder, push_recorder, stop_recorder, active_1481_task
):
    """Durability pin: RUNNING bare pod-1481, created 2h ago, events = fresh
    progress rows + launch-ADJACENT rows but ZERO launch-signal kinds, no
    keep-running -> exactly ONE marker + ONE push across 3 consecutive ticks
    (noted dedup); NO stop/terminate ever."""
    now, events = active_1481_task
    # Fixture sanity (binding concern 3): the launch-adjacent rows are present,
    # newer than created_ts - SKEW, and NOT in the evidence set — widening the
    # evidence set to them would suppress the #1481 alert and fail this test.
    created = now - 2 * 3600
    kinds = {e["kind"] for e in events}
    assert {"epm:cluster-launched", "epm:backend-selected"} <= kinds
    assert not (kinds & asw._POD_FOLLOWUP_SIGNAL_KINDS)
    adjacent_ts = [
        asw._parse_event_ts(e["ts"])
        for e in events
        if e["kind"] in {"epm:cluster-launched", "epm:backend-selected"}
    ]
    assert all(ts is not None and ts > created - SKEW for ts in adjacent_ts)

    info = _info(created_at=_iso(created))
    for tick in range(3):
        asw._process_pod(1481, "p1481", info, now + tick * 600, dry_run=False, threshold=2)

    fired = _unlaunched_posts(marker_recorder)
    assert len(fired) == 1  # exactly ONE marker across 3 ticks
    issue, note, label = fired[0]
    assert issue == 1481
    assert label == "unlaunched-orphan"
    assert "pod-1481" in note
    assert "epm:run-launched" in note
    assert "pod.py terminate --issue 1481" in note
    assert "keep-running" in note
    assert "est. $" in note  # 8x known GPU type -> a nonzero rate estimate
    assert len(push_recorder) == 1  # exactly ONE push across 3 ticks
    assert "UNLAUNCHED ORPHAN" in push_recorder[0]
    assert stop_recorder == []  # alert-only: never a stop
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert state["unlaunched_orphan_noted"] is True


def test_additive_status_class_still_runs_after_firing_alert(
    isolated_registry, marker_recorder, push_recorder, stop_recorder, active_1481_task
):
    """Binding concern 2: on an ACTIVE fixture where the arm FIRES, the
    status-class decision still runs afterward — its keep-branch save (which
    passes prev=the mirrored in-memory snapshot) persists last_progress_ts
    AND carries unlaunched_orphan_noted forward instead of clobbering it."""
    now, _events = active_1481_task
    info = _info(created_at=_iso(now - 2 * 3600))
    asw._process_pod(1481, "p1481", info, now, dry_run=False, threshold=2)
    assert len(_unlaunched_posts(marker_recorder)) == 1  # the arm fired...
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    # ...and the status-class keep-save ran AFTER it: the arm's own save wrote
    # last_progress_ts=None (no prior state), so a non-None value here proves
    # the later status-class save executed — and its None-carry preserved the
    # just-persisted flag (the #1490 r2 lesson applied to this arm).
    assert state["last_progress_ts"] is not None
    assert state["unlaunched_orphan_noted"] is True
    assert stop_recorder == []


def test_done_status_non_fire_control_autostop_still_acts(
    isolated_registry, marker_recorder, push_recorder, stop_recorder, monkeypatch
):
    """NON-fire control (binding concern 2): the arm cannot fire on a DONE
    status (POD_ACTIVE leg), and the canonical escaped-pod auto-stop still
    acts on the same tick — the arm is invisible to the existing decision."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_events", lambda issue: _events_1481(now))
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: None)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    # missed=1 -> new_missed=2 == threshold -> the auto-stop fires this tick.
    payload = {
        "pod_id": "p1481",
        "missed": 1,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7200,
    }
    (isolated_registry / "pod-safety-1481.json").write_text(json.dumps(payload))

    asw._process_pod(
        1481, "p1481", _info(created_at=_iso(now - 2 * 3600)), now, dry_run=False, threshold=2
    )
    assert _unlaunched_posts(marker_recorder) == []  # DONE status: never fires
    assert push_recorder == []
    assert stop_recorder == [1481]  # the canonical auto-stop still ran


# ---------------------------------------------------------------------------
# 2. Pure-predicate legs (decide_unlaunched_orphan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    ["completed", "awaiting_promotion", "archived", "on_hold", "blocked", "planning", None],
)
def test_decide_keep_on_non_active_status(status):
    """DONE/parked/blocked/unknown statuses are out of the #1519 scope."""
    now = time.time()
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, status=status)) == "keep"


@pytest.mark.parametrize("status", sorted(asw.POD_ACTIVE))
def test_decide_alert_on_every_pod_active_status(status):
    """Every POD_ACTIVE status fires when all other legs are orphan-shaped."""
    now = time.time()
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, status=status)) == "alert"


def test_decide_keep_on_suffixed_pod_name():
    """A suffixed follow-up pod (pod-1481-b) is out of scope (bare name only)."""
    now = time.time()
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, pod_name="pod-1481-b")) == "keep"


def test_decide_keep_when_noted():
    """The once-per-incarnation dedup flag suppresses a re-alert."""
    now = time.time()
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, noted=True)) == "keep"


def test_decide_keep_on_created_ts_none():
    """An unparseable/missing createdAt fails toward keep, never a crash."""
    now = time.time()
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, created_ts=None)) == "keep"


def test_decide_grace_window_boundary():
    """Within the grace window -> keep (healthy provision/bootstrap/dispatch);
    at/over the grace -> alert (the age comparator is `< grace_sec`)."""
    now = time.time()
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, created_ts=now - GRACE + 60)) == "keep"
    assert asw.decide_unlaunched_orphan(**_fire_kwargs(now, created_ts=now - GRACE)) == "alert"


def test_decide_keep_on_launch_signal_newer_than_created_at():
    """A healthy launched pod (signal after creation) never alerts."""
    now = time.time()
    created = now - 2 * 3600
    assert (
        asw.decide_unlaunched_orphan(
            **_fire_kwargs(now, created_ts=created, latest_launch_signal_ts=created + 300)
        )
        == "keep"
    )


def test_decide_keep_on_signal_within_skew_before_created_at():
    """A signal posted at/just before provision completion (the CLAUDE.md
    'in any case before launch' ordering) still counts as launched."""
    now = time.time()
    created = now - 2 * 3600
    assert (
        asw.decide_unlaunched_orphan(
            **_fire_kwargs(now, created_ts=created, latest_launch_signal_ts=created - 600)
        )
        == "keep"
    )


def test_decide_alert_on_signal_older_than_skew():
    """A signal older than created_ts - skew belongs to a PREDECESSOR pod and
    does NOT shield a fresh replacement pod — the orphan signature."""
    now = time.time()
    created = now - 2 * 3600
    assert (
        asw.decide_unlaunched_orphan(
            **_fire_kwargs(now, created_ts=created, latest_launch_signal_ts=created - 7200)
        )
        == "alert"
    )


# ---------------------------------------------------------------------------
# 3. Caller hardening + shields
# ---------------------------------------------------------------------------


def test_keep_on_empty_events(isolated_registry, marker_recorder, push_recorder, monkeypatch):
    """Empty-events hardening (binding concern 1): events == [] on an ACTIVE
    task means the read FAILED or the task is pathological (every real task
    carries >=1 epm: marker) -> treated as UNREADABLE -> keep, never as
    positive 'no launch signal' evidence."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    prev_state: dict = {}
    fired = asw._maybe_flag_unlaunched_orphan_pod(
        1481, _info(created_at=_iso(now - 2 * 3600)), "running", [], now, prev_state, False
    )
    assert fired is False
    assert _unlaunched_posts(marker_recorder) == []
    assert push_recorder == []
    assert "unlaunched_orphan_noted" not in prev_state  # no mirror on a keep


def test_keep_on_keep_running_tag(isolated_registry, marker_recorder, push_recorder, monkeypatch):
    """The lazy keep-running shield (re-checked at fire time only) suppresses
    the alert — a deliberate pre-launch hold is the tag's documented use."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    fired = asw._maybe_flag_unlaunched_orphan_pod(
        1481,
        _info(created_at=_iso(now - 2 * 3600)),
        "running",
        _events_1481(now),
        now,
        {},
        False,
    )
    assert fired is False
    assert _unlaunched_posts(marker_recorder) == []
    assert push_recorder == []


def test_keep_on_unparseable_created_at_and_fractional_parses(
    isolated_registry, marker_recorder, push_recorder, monkeypatch
):
    """created_at None or garbage -> keep (fail-toward-keep); a
    fractional-seconds ISO createdAt PARSES (pinning the _parse_event_ts
    reuse) and fires."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    events = _events_1481(now)
    for bad in (None, "not-a-timestamp"):
        fired = asw._maybe_flag_unlaunched_orphan_pod(
            1481, _info(created_at=bad), "running", events, now, {}, False
        )
        assert fired is False
    assert _unlaunched_posts(marker_recorder) == []
    # Fractional-seconds shape (RunPod emits both) parses and fires.
    frac = datetime.fromtimestamp(now - 2 * 3600, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.123456Z")
    assert asw._parse_event_ts(frac) is not None
    fired = asw._maybe_flag_unlaunched_orphan_pod(
        1481,
        _info(created_at=frac),
        "running",
        events,
        now,
        {},
        True,  # dry_run: no state write
    )
    assert fired is True
    assert len(_unlaunched_posts(marker_recorder)) == 1


def test_rate_estimate_degrades_to_spec_only_on_zero_gpus(
    isolated_registry, marker_recorder, push_recorder, monkeypatch
):
    """estimate_pod_hourly_rate returns 0.0 for gpu_count None/0 (never
    raises) -> the note renders spec-only, with no 'est. $' fragment."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    fired = asw._maybe_flag_unlaunched_orphan_pod(
        1481,
        _info(created_at=_iso(now - 2 * 3600), gpu_count=None, gpu_type_id=None),
        "running",
        _events_1481(now),
        now,
        {},
        True,
    )
    assert fired is True
    ((_, note, _),) = _unlaunched_posts(marker_recorder)
    assert "est. $" not in note  # spec-only wording on a 0.0 rate
    assert "?xunknown-gpu" in note


# ---------------------------------------------------------------------------
# 4. Dedup / incarnation / state-carry mechanics
# ---------------------------------------------------------------------------


def test_realerts_on_new_pod_incarnation(
    isolated_registry, marker_recorder, push_recorder, stop_recorder, active_1481_task
):
    """The dedup flag is per pod INCARNATION: a stored unlaunched_orphan_noted
    under a DIFFERENT pod_id does not suppress the fresh incarnation's alert,
    and the fresh incarnation then dedups normally."""
    now, _events = active_1481_task
    payload = {
        "pod_id": "p_OLD",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7200,
        "unlaunched_orphan_noted": True,
    }
    (isolated_registry / "pod-safety-1481.json").write_text(json.dumps(payload))

    info = _info(pod_id="p_NEW", created_at=_iso(now - 2 * 3600))
    asw._process_pod(1481, "p_NEW", info, now, dry_run=False, threshold=2)
    assert len(_unlaunched_posts(marker_recorder)) == 1
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert state["pod_id"] == "p_NEW"
    assert state["unlaunched_orphan_noted"] is True

    asw._process_pod(1481, "p_NEW", info, now + 600, dry_run=False, threshold=2)
    assert len(_unlaunched_posts(marker_recorder)) == 1  # dedup on tick 2
    assert stop_recorder == []


def test_mirror_survives_status_class_save_on_new_incarnation(
    isolated_registry, marker_recorder, push_recorder, monkeypatch
):
    """The #1490 r2 clobber lesson, applied to this arm: after the arm fires
    on a NEW incarnation, the in-memory prev_state mirrors BOTH the flag and
    the new pod_id — so a later same-tick status-class save (flag left at its
    None carry, prev=the mirrored snapshot) carries the just-persisted flag
    forward instead of recomputing same_pod=False off the stale OLD pod_id
    and clobbering it back to False."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    prev_state = {
        "pod_id": "p_OLD",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7200,
        "unlaunched_orphan_noted": True,
    }
    fired = asw._maybe_flag_unlaunched_orphan_pod(
        1481,
        _info(pod_id="p_NEW", created_at=_iso(now - 2 * 3600)),
        "running",
        _events_1481(now),
        now,
        prev_state,
        False,
    )
    assert fired is True
    assert prev_state["pod_id"] == "p_NEW"  # the mirror
    assert prev_state["unlaunched_orphan_noted"] is True

    # The status-class arm's later same-tick save (keep-shaped).
    asw._save_pod_safety_state(
        1481, "p_NEW", missed=0, alerted=False, last_progress_ts=None, prev=prev_state
    )
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert state["unlaunched_orphan_noted"] is True  # pre-mirror: clobbered to False
    assert state["pod_id"] == "p_NEW"


def test_save_state_carry_and_pod_id_reset(isolated_registry):
    """_save_pod_safety_state: None carries unlaunched_orphan_noted forward
    on the SAME pod_id; a save under a NEW pod_id resets it to False (re-arms
    a fresh incarnation) — byte-parallel to orphan_gcp_noted."""
    asw._save_pod_safety_state(
        1481, "pA", missed=0, alerted=False, last_progress_ts=None, unlaunched_orphan_noted=True
    )
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert state["unlaunched_orphan_noted"] is True

    asw._save_pod_safety_state(
        1481, "pA", missed=1, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert state["unlaunched_orphan_noted"] is True  # None-carry, same pod

    asw._save_pod_safety_state(
        1481, "pB", missed=0, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert state["unlaunched_orphan_noted"] is False  # new incarnation re-arms


def test_dry_run_alerts_but_persists_nothing(
    isolated_registry, marker_recorder, push_recorder, stop_recorder, active_1481_task
):
    """Dry-run: the alert goes through _post_progress_marker/_telegram_push
    (which no-op real sends under dry_run in production) but NO state write."""
    now, _events = active_1481_task
    payload = {
        "pod_id": "p1481",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7200,
    }
    (isolated_registry / "pod-safety-1481.json").write_text(json.dumps(payload))
    asw._process_pod(
        1481, "p1481", _info(created_at=_iso(now - 2 * 3600)), now, dry_run=True, threshold=2
    )
    assert len(_unlaunched_posts(marker_recorder)) == 1
    state = json.loads((isolated_registry / "pod-safety-1481.json").read_text())
    assert "unlaunched_orphan_noted" not in state  # the seeded file was never rewritten
    assert stop_recorder == []


# ---------------------------------------------------------------------------
# 5. Env-override reader + registry / invariant pins
# ---------------------------------------------------------------------------


def test_grace_env_override_and_bad_values_fall_back(monkeypatch):
    """EPM_UNLAUNCHED_ORPHAN_GRACE_SEC: a positive int overrides at CALL time;
    missing / garbage / zero / negative fall back to the default (never a
    kill switch)."""
    monkeypatch.delenv("EPM_UNLAUNCHED_ORPHAN_GRACE_SEC", raising=False)
    assert asw._unlaunched_orphan_grace_sec() == GRACE
    monkeypatch.setenv("EPM_UNLAUNCHED_ORPHAN_GRACE_SEC", "7200")
    assert asw._unlaunched_orphan_grace_sec() == 7200
    for bad in ("garbage", "", "0", "-5"):
        monkeypatch.setenv("EPM_UNLAUNCHED_ORPHAN_GRACE_SEC", bad)
        assert asw._unlaunched_orphan_grace_sec() == GRACE


def test_watcher_sentinel_registered():
    """The alert rides epm:progress, so its sentinel MUST be excluded from
    'real progress' — otherwise the alert would reset the staleness clocks
    the pass measures (the load-bearing _WATCHER_NOTE_SENTINELS membership)."""
    assert asw._UNLAUNCHED_ORPHAN_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_arm_never_stops_source_pin():
    """Alert-only invariant: neither the pure fn nor the caller references any
    stop/terminate call site (the behavioral half is the stop_recorder
    assertions in the end-to-end tests; the remediation COMMAND string in the
    note text names `pod.py terminate` for the human, which is not a call)."""
    src = inspect.getsource(asw._maybe_flag_unlaunched_orphan_pod) + inspect.getsource(
        asw.decide_unlaunched_orphan
    )
    for banned in ("_stop_pod", "terminate_pod", "_failover", "stop_pod("):
        assert banned not in src
