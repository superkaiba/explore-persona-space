"""Unit tests for the #1997 bounded-diagnosis-window pod-safety arm.

The #1739 incident this arm closes: a RunPod fallback-rung pod whose workload
start FAILED (`runpod_workload_start_failed`) is deliberately left RUNNING for
SSH diagnosis (the RunPod-as-diagnosis-lane doctrine) — but the reason token
is deliberately NOT watcher-re-drivable, so the task parks ``blocked`` while
the pod bills UNBOUNDED behind the contract (the #1519 arm only alerts, 4x
per incident). The arm keys on POSITIVE handle-sidecar evidence (the #954
partial handle: ``workload_executed: false`` + ``workload_start_error``) and,
past ``EPS_RUNPOD_DIAGNOSIS_TTL_HOURS`` (default 6h) with no keep-running tag
and no live owner, issues a REVERSIBLE ``pod.py stop`` (volume + /workspace
logs preserved — crash forensics survive; NEVER a terminate). Covers:

* plan test a1 — stop fires end-to-end through ``_process_pod`` when ALL
  predicates hold (sentinel-led marker + push + noted state);
* plan test a2 — every guard fails toward keep (within TTL / no sidecar /
  workload_executed True / no start_error / keep-running tag incl. the
  tri-state "unknown" read / live+unknown owner / already noted / stale or
  unparseable sidecar / schema-drift sidecar / sidecar naming a different
  pod);
* plan test a3 — ``_stop_pod`` invoked exactly once, a monkeypatched
  destruction path is NEVER called, the marker LEADS with the sentinel, and
  the sidecar is constructed via the REAL ``_build_handle`` field names
  (real ``RunHandle`` + ``write_handle_sidecar`` + a source pin on the
  literal keys — the plan §8 drift risk);
* plan test a4 — once-per-incarnation dedup (+ the new-incarnation re-arm);
* the env-override reader, the ``_save_pod_safety_state`` pod_id-keyed
  carry, the ``_WATCHER_NOTE_SENTINELS`` membership pin, the dry-run
  no-mutation contract, the stop-failure retry posture, and a
  no-destruction source pin over the three new functions.

Follows ``tests/test_autonomous_session_watch_unlaunched_orphan.py``
conventions: PodInfo fixtures, the patched state dir, ``task.py`` reads
monkeypatched, no network, no real marker posts.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import runpod_api  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

import explore_persona_space.backends.issue_dispatch as idp  # noqa: E402
import explore_persona_space.backends.runpod as runpod_backend  # noqa: E402
from explore_persona_space.backends.base import RunHandle  # noqa: E402

TTL_SEC = asw.DIAGNOSIS_WINDOW_TTL_HOURS * 3600.0  # 6h default (env-overridable)
SENTINEL = asw._DIAGNOSIS_WINDOW_STOP_NOTE_SENTINEL
START_ERROR = (
    "RunPodWorkloadStartError: workload launcher exited rc=1 before writing a fresh pidfile"
)


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the per-pod state dir at a tmp dir (mirrors the sibling suites)."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _info(
    pod_id: str = "p1997",
    name: str = "pod-1997",
    desired_status: str = "RUNNING",
) -> PodInfo:
    """A RUNNING pod with a public port (so the #692 wedge arm never handles it)."""
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status=desired_status,
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host="1.2.3.4",
        ssh_port=22001,
        created_at=None,
    )


def _write_sidecar(
    tmp_path,
    monkeypatch,
    *,
    backend: str = "runpod",
    pod_name: str = "pod-1997",
    workload_executed=False,
    workload_start_error: str | None = START_ERROR,
    raw_extra: dict | None = None,
    garbage: bool = False,
    exists: bool = True,
) -> Path:
    """Write a handle sidecar via the REAL serializer + point the resolver at it.

    Field names deliberately mirror ``RunPodBackend.launch``'s ``_build_handle``
    failure path (the plan §8 drift risk): ``workload_executed`` +
    ``workload_start_error`` live in ``extra``; ``backend`` / ``pod_name`` are
    the typed handle fields. ``raw_extra`` overrides the extra dict wholesale
    (schema-drift cases); ``garbage=True`` writes unparseable bytes.
    The resolver monkeypatch targets the SOURCE module (the watcher imports it
    function-locally at call time, so the patched attr is what it sees).
    """
    path = tmp_path / "issue-1997-handle.json"
    if garbage:
        path.write_text("{ not json")
    elif exists:
        if raw_extra is not None:
            extra: dict = dict(raw_extra)
        else:
            extra = {"workload_executed": workload_executed}
            if workload_start_error is not None:
                extra["workload_start_error"] = workload_start_error
        handle = RunHandle(
            backend=backend,
            cluster=None,
            job_id="",
            pod_name=pod_name,
            scratch_dir="/workspace",
            log_path="/workspace/logs/issue-1997-workload.log",
            extra=extra,
        )
        idp.write_handle_sidecar(handle, path)
    monkeypatch.setattr(
        idp,
        "resolve_handle_sidecar_path",
        lambda issue, explicit=None, lane_suffix=None: (path, [path]),
    )
    return path


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
    """Record ``_stop_pod`` calls, honoring its dry-run contract (False, no-op)."""
    stops: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        asw,
        "_stop_pod",
        lambda issue, dry_run: stops.append((issue, dry_run)) or (not dry_run),
    )
    return stops


@pytest.fixture
def terminate_guards(monkeypatch):
    """Trap every destruction route the watcher could conceivably reach."""
    calls: list[str] = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pod_id: calls.append(f"terminate_pod:{pod_id}")
    )
    monkeypatch.setattr(
        asw,
        "_wedge_failover",
        lambda *a, **kw: calls.append("_wedge_failover") or ("alert", None),
    )
    return calls


@pytest.fixture
def shields_clear(monkeypatch):
    """Both tri-state shields resolved to the literal False (stop-permitting)."""
    monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: False)
    # For the other arms _process_pod runs through on end-to-end calls.
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)


@pytest.fixture
def blocked_task(monkeypatch):
    """The #1739 task shape: parked ``blocked`` after runpod_workload_start_failed."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_status", lambda issue: "blocked")
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda issue: [
            {"kind": "epm:failure", "ts": "2026-08-01T22:40:00Z", "note": "start failed"}
        ],
    )
    return now


def _seed_state(registry: Path, now: float, *, age_h: float = 7.0, **extra) -> dict:
    """Seed the pod-safety state file with a matured incarnation clock."""
    payload = {
        "pod_id": "p1997",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - age_h * 3600,
        **extra,
    }
    (registry / "pod-safety-1997.json").write_text(json.dumps(payload))
    return payload


def _diagnosis_posts(posts):
    return [p for p in posts if SENTINEL in p[1]]


# ---------------------------------------------------------------------------
# a1 — stop fires end-to-end when ALL predicates hold
# ---------------------------------------------------------------------------


def test_stop_after_ttl_with_sidecar_evidence(
    isolated_registry,
    marker_recorder,
    push_recorder,
    stop_recorder,
    terminate_guards,
    shields_clear,
    blocked_task,
    tmp_path,
    monkeypatch,
):
    """RUNNING pod-1997, sidecar workload_executed=false + start_error, 7h-old
    incarnation clock (> 6h TTL), no shields -> ONE reversible stop + ONE
    sentinel-led marker + ONE push; state notes the incarnation."""
    now = blocked_task
    _write_sidecar(tmp_path, monkeypatch)
    _seed_state(isolated_registry, now)

    asw._process_pod(1997, "p1997", _info(), now, dry_run=False, threshold=2)

    assert stop_recorder == [(1997, False)]
    fired = _diagnosis_posts(marker_recorder)
    assert len(fired) == 1
    issue, note, label = fired[0]
    assert issue == 1997
    assert label == "diagnosis-window-stop"
    assert note.startswith(SENTINEL)  # the sentinel LEADS the note
    assert "pod-1997" in note
    assert "6.0h" in note  # the elapsed-vs-TTL wording names the TTL
    assert "7.0h" in note  # ...and the elapsed hours
    assert "EPS_RUNPOD_DIAGNOSIS_TTL_HOURS" in note
    assert "volume + /workspace logs PRESERVED" in note  # the reversibility guarantee
    assert f"pod.py resume --issue {issue}" in note  # the reopen command
    assert START_ERROR[:120] in note  # the truncated workload_start_error
    assert len(push_recorder) == 1
    assert "diagnosis-window STOP" in push_recorder[0]
    assert terminate_guards == []  # never a destruction path
    state = json.loads((isolated_registry / "pod-safety-1997.json").read_text())
    assert state["diagnosis_stop_noted"] is True
    assert state["pod_id"] == "p1997"


# ---------------------------------------------------------------------------
# a2 — every guard fails toward keep
# ---------------------------------------------------------------------------


KEEP_CASES = [
    "within-ttl",
    "no-sidecar",
    "workload-executed-true",
    "no-start-error",
    "keep-running-tag",
    "keep-running-unknown",
    "live-owner",
    "owner-unknown",
    "already-noted",
    "sidecar-unparseable",
    "sidecar-schema-drift",
    "sidecar-names-different-pod",
]


@pytest.mark.parametrize("case", KEEP_CASES)
def test_keep_cases(
    case,
    isolated_registry,
    marker_recorder,
    push_recorder,
    stop_recorder,
    terminate_guards,
    shields_clear,
    tmp_path,
    monkeypatch,
):
    """Every guard fails toward keep: no stop, no marker, no push, no mirror."""
    now = time.time()
    prev_state = {
        "pod_id": "p1997",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7 * 3600,
    }
    sidecar_kwargs: dict = {}
    if case == "within-ttl":
        prev_state["first_seen"] = now - 3600  # 1h < 6h TTL
    elif case == "no-sidecar":
        sidecar_kwargs = {"exists": False}
    elif case == "workload-executed-true":
        # A workload that LAUNCHED then died stays with the poller/failure
        # paths + #1519 alerts (the stated non-goal) — even with an error.
        sidecar_kwargs = {"workload_executed": True}
    elif case == "no-start-error":
        sidecar_kwargs = {"workload_start_error": None}
    elif case == "keep-running-tag":
        monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: True)
    elif case == "keep-running-unknown":
        # A FAILED tag read must never silently override a possible user tag.
        monkeypatch.setattr(asw, "_wedge_keep_running", lambda issue: "unknown")
    elif case == "live-owner":
        monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: True)
    elif case == "owner-unknown":
        # Only the LITERAL False permits the stop (the #1667 convention).
        monkeypatch.setattr(asw, "_wedge_owner_live", lambda issue, now: "unknown")
    elif case == "already-noted":
        prev_state["diagnosis_stop_noted"] = True
    elif case == "sidecar-unparseable":
        # A sidecar that exists but is stale/unreadable reads as keep (the
        # critic round-1 concern): any read/parse failure -> no evidence.
        sidecar_kwargs = {"garbage": True}
    elif case == "sidecar-schema-drift":
        # A renamed/re-typed field silently DISARMS the arm (inert, never
        # destructive): a non-bool workload_executed reads None -> keep.
        sidecar_kwargs = {"raw_extra": {"workload_executed": "false"}}
    elif case == "sidecar-names-different-pod":
        # A re-pointed / suffixed-sibling handle never licenses acting on
        # THIS pod (the #770-r2 sidecar-binding lesson).
        sidecar_kwargs = {"pod_name": "pod-1997-b"}

    _write_sidecar(tmp_path, monkeypatch, **sidecar_kwargs)

    fired = asw._maybe_stop_diagnosis_window_pod(1997, _info(), now, prev_state, False)

    assert fired is False
    assert stop_recorder == []
    assert _diagnosis_posts(marker_recorder) == []
    assert push_recorder == []
    assert terminate_guards == []
    if case != "already-noted":
        assert "diagnosis_stop_noted" not in prev_state  # no mirror on a keep
    assert not (isolated_registry / "pod-safety-1997.json").exists()  # no state write


def test_decide_pure_fn_boundaries():
    """Pure-fn legs the arm-level cases cannot isolate: non-RUNNING pod
    status, the TTL boundary (at-TTL fires, the < comparator keeps), and a
    non-runpod handle backend."""
    now = time.time()
    kwargs = {
        "pod_status": "RUNNING",
        "pod_name": "pod-1997",
        "handle_backend": "runpod",
        "workload_executed": False,
        "workload_start_error": START_ERROR,
        "handle_pod_name": "pod-1997",
        "first_seen_ts": now - 7 * 3600,
        "keep_running": False,
        "owner_live": False,
        "noted": False,
        "now": now,
        "ttl_sec": TTL_SEC,
    }
    assert asw.decide_diagnosis_window_stop(**kwargs) == "stop"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "pod_status": "EXITED"}) == "keep"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "pod_status": None}) == "keep"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "handle_backend": "gcp"}) == "keep"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "handle_backend": None}) == "keep"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "workload_executed": None}) == "keep"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "first_seen_ts": None}) == "keep"
    # Boundary: strictly within the TTL keeps; exactly at the TTL fires.
    assert (
        asw.decide_diagnosis_window_stop(**{**kwargs, "first_seen_ts": now - TTL_SEC + 60})
        == "keep"
    )
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "first_seen_ts": now - TTL_SEC}) == "stop"
    # Only the literal False owner read permits the stop.
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "owner_live": "unknown"}) == "keep"
    assert asw.decide_diagnosis_window_stop(**{**kwargs, "owner_live": True}) == "keep"


# ---------------------------------------------------------------------------
# a3 — stop, never a destruction path; real _build_handle field names
# ---------------------------------------------------------------------------


def test_stop_arm_calls_stop_never_terminate(
    isolated_registry,
    marker_recorder,
    push_recorder,
    stop_recorder,
    terminate_guards,
    shields_clear,
    tmp_path,
    monkeypatch,
):
    """``_stop_pod`` exactly once; the monkeypatched destruction routes are
    NEVER called; the marker leads with the sentinel. The sidecar is built
    through the REAL RunHandle serializer with the REAL failure-path field
    names (workload_executed / workload_start_error in extra)."""
    now = time.time()
    prev_state = {
        "pod_id": "p1997",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7 * 3600,
    }
    _write_sidecar(tmp_path, monkeypatch)

    fired = asw._maybe_stop_diagnosis_window_pod(1997, _info(), now, prev_state, False)

    assert fired is True
    assert stop_recorder == [(1997, False)]  # exactly once
    assert terminate_guards == []  # no destruction route ever invoked
    ((_, note, _),) = _diagnosis_posts(marker_recorder)
    assert note.startswith(SENTINEL)
    # The in-memory mirror (flag + pod_id — the #1490 r2 lesson).
    assert prev_state["diagnosis_stop_noted"] is True
    assert prev_state["pod_id"] == "p1997"


def test_sidecar_field_names_match_build_handle_source():
    """Plan §8 drift pin: the arm's evidence keys are the LITERAL field names
    ``_build_handle`` writes on the #954 failure path. A rename in
    ``backends/runpod.py`` fails THIS test (loud) while the arm itself
    degrades inertly (fail-toward-keep)."""
    src = inspect.getsource(runpod_backend)
    assert '"workload_executed": workload_executed' in src
    assert 'extra["workload_start_error"] = workload_start_error' in src
    evidence_src = inspect.getsource(asw._diagnosis_sidecar_evidence)
    assert '"workload_executed"' in evidence_src
    assert '"workload_start_error"' in evidence_src


def test_arm_source_never_references_destruction():
    """No-destruction source pin over the three new functions: the arm's only
    action is the reversible stop helper — no terminate mutation, no wedge
    failover route, no kill-approval bypass."""
    src = (
        inspect.getsource(asw._maybe_stop_diagnosis_window_pod)
        + inspect.getsource(asw._diagnosis_sidecar_evidence)
        + inspect.getsource(asw.decide_diagnosis_window_stop)
    )
    for banned in ("terminate_pod", "_failover", "kill_approval", "EPS_ALLOW"):
        assert banned not in src, banned


# ---------------------------------------------------------------------------
# a4 — once-per-incarnation dedup (+ new-incarnation re-arm)
# ---------------------------------------------------------------------------


def test_once_per_incarnation_dedup(
    isolated_registry,
    marker_recorder,
    push_recorder,
    stop_recorder,
    terminate_guards,
    shields_clear,
    blocked_task,
    tmp_path,
    monkeypatch,
):
    """A second tick on the SAME incarnation (pod still listed RUNNING — a
    slow-to-take stop) keeps: exactly ONE stop + ONE marker across 3 ticks."""
    now = blocked_task
    _write_sidecar(tmp_path, monkeypatch)
    _seed_state(isolated_registry, now)

    for tick in range(3):
        asw._process_pod(1997, "p1997", _info(), now + tick * 600, dry_run=False, threshold=2)

    assert stop_recorder == [(1997, False)]  # one stop across 3 ticks
    assert len(_diagnosis_posts(marker_recorder)) == 1
    assert len(push_recorder) == 1
    assert terminate_guards == []


def test_new_incarnation_rearms(
    isolated_registry,
    marker_recorder,
    stop_recorder,
    push_recorder,
    terminate_guards,
    shields_clear,
    tmp_path,
    monkeypatch,
):
    """The dedup flag is pod_id-keyed: a stored noted flag under a DIFFERENT
    pod_id does not shield a fresh incarnation."""
    now = time.time()
    prev_state = {
        "pod_id": "p_OLD",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7 * 3600,
        "diagnosis_stop_noted": True,
    }
    _write_sidecar(tmp_path, monkeypatch)
    fired = asw._maybe_stop_diagnosis_window_pod(
        1997, _info(pod_id="p_NEW"), now, prev_state, False
    )
    assert fired is True
    assert stop_recorder == [(1997, False)]
    state = json.loads((isolated_registry / "pod-safety-1997.json").read_text())
    assert state["pod_id"] == "p_NEW"
    assert state["diagnosis_stop_noted"] is True


def test_save_state_carry_and_pod_id_reset(isolated_registry):
    """_save_pod_safety_state: None carries diagnosis_stop_noted forward on
    the SAME pod_id; a save under a NEW pod_id resets it to False — byte-
    parallel to unlaunched_orphan_noted."""
    asw._save_pod_safety_state(
        1997, "pA", missed=0, alerted=False, last_progress_ts=None, diagnosis_stop_noted=True
    )
    state = json.loads((isolated_registry / "pod-safety-1997.json").read_text())
    assert state["diagnosis_stop_noted"] is True

    asw._save_pod_safety_state(
        1997, "pA", missed=1, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1997.json").read_text())
    assert state["diagnosis_stop_noted"] is True  # None-carry, same pod

    asw._save_pod_safety_state(
        1997, "pB", missed=0, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1997.json").read_text())
    assert state["diagnosis_stop_noted"] is False  # new incarnation re-arms


# ---------------------------------------------------------------------------
# Dry-run / stop-failure / env knob / registry pins
# ---------------------------------------------------------------------------


def test_dry_run_would_stop_but_mutates_nothing(
    isolated_registry,
    marker_recorder,
    push_recorder,
    stop_recorder,
    terminate_guards,
    shields_clear,
    tmp_path,
    monkeypatch,
):
    """Dry-run: the would-stop log line only (via _stop_pod's own dry-run
    contract) — no marker, no push, no state write; returns True so the dry
    run predicts the production early-return."""
    now = time.time()
    prev_state = {
        "pod_id": "p1997",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7 * 3600,
    }
    _write_sidecar(tmp_path, monkeypatch)
    fired = asw._maybe_stop_diagnosis_window_pod(1997, _info(), now, prev_state, True)
    assert fired is True
    assert stop_recorder == [(1997, True)]  # reached _stop_pod, which no-ops on dry-run
    assert _diagnosis_posts(marker_recorder) == []
    assert push_recorder == []
    assert "diagnosis_stop_noted" not in prev_state
    assert not (isolated_registry / "pod-safety-1997.json").exists()


def test_stop_failure_does_not_note_and_retries_next_tick(
    isolated_registry,
    marker_recorder,
    push_recorder,
    terminate_guards,
    shields_clear,
    tmp_path,
    monkeypatch,
):
    """A REAL stop failure (pod.py stop rc != 0) does NOT note — the next
    tick retries (the #1155 retryable-episode posture); no marker either."""
    now = time.time()
    stops: list[int] = []
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or False)
    prev_state = {
        "pod_id": "p1997",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - 7 * 3600,
    }
    _write_sidecar(tmp_path, monkeypatch)
    fired = asw._maybe_stop_diagnosis_window_pod(1997, _info(), now, prev_state, False)
    assert fired is False
    assert stops == [1997]
    assert _diagnosis_posts(marker_recorder) == []
    assert "diagnosis_stop_noted" not in prev_state
    # Next tick: the un-noted state retries the stop.
    fired = asw._maybe_stop_diagnosis_window_pod(1997, _info(), now + 600, prev_state, False)
    assert fired is False
    assert stops == [1997, 1997]


def test_ttl_env_override_and_bad_values_fall_back(monkeypatch):
    """EPS_RUNPOD_DIAGNOSIS_TTL_HOURS: a positive FLOAT (hours) overrides at
    CALL time; missing / garbage / zero / negative fall back to the 6.0h
    default (never a kill switch)."""
    monkeypatch.delenv("EPS_RUNPOD_DIAGNOSIS_TTL_HOURS", raising=False)
    assert asw._diagnosis_window_ttl_sec() == pytest.approx(6.0 * 3600.0)
    monkeypatch.setenv("EPS_RUNPOD_DIAGNOSIS_TTL_HOURS", "2.5")
    assert asw._diagnosis_window_ttl_sec() == pytest.approx(2.5 * 3600.0)
    for bad in ("garbage", "", "0", "-5"):
        monkeypatch.setenv("EPS_RUNPOD_DIAGNOSIS_TTL_HOURS", bad)
        assert asw._diagnosis_window_ttl_sec() == pytest.approx(6.0 * 3600.0)


def test_watcher_sentinel_registered():
    """The stop marker rides epm:progress, so its sentinel MUST be excluded
    from 'real progress' — otherwise the stop marker would reset the very
    staleness clocks the pass measures."""
    assert asw._DIAGNOSIS_WINDOW_STOP_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_fresh_state_has_no_clock_and_keeps(
    isolated_registry,
    marker_recorder,
    stop_recorder,
    terminate_guards,
    shields_clear,
    tmp_path,
    monkeypatch,
):
    """A fresh incarnation (no first_seen clock yet) keeps — the age
    accumulates from the pod-safety pass's first save, so `pod.py resume`
    re-opens a FRESH TTL window (the state GC reset the clock)."""
    now = time.time()
    _write_sidecar(tmp_path, monkeypatch)
    fired = asw._maybe_stop_diagnosis_window_pod(1997, _info(), now, {}, False)
    assert fired is False
    assert stop_recorder == []
    assert _diagnosis_posts(marker_recorder) == []
