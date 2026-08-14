"""Unit tests for the #2283 pod-safety OWNER-FENCE DEFER arm.

The gap this arm closes: the #2277 terminate guard refuses to DESTROY a pod
carrying an unexpired owner fence without an owner-matching PASS, but the
watcher's pod-safety pass could still auto-STOP the same pod ~20 min after a
non-owner (or the owner itself, pre-park) drove the task to a DONE status —
interrupting a live owner's run through the softer verb. Covers:

* the pure ``decide_pod_safety`` extension (defer branch, precedence below
  the keep-running / follow-up shields, the DEFER-ONLY hard invariant —
  byte-identical decision table at the False defaults, mechanically swept
  over the full input cross-product);
* the tri-state ``_pod_owner_fence_active`` wrapper AGAINST THE REAL
  ``pod_lifecycle.owner_fence_state`` reader chain (production-body
  coverage: no fence-reader stubs — fixture events lists drive the real
  window/token/PASS parsing), incl. the kill switch and the loud fail-open
  ``None`` on a reader failure;
* ``_escaped_pod_exemptions`` eligibility: STRICTLY ``status in
  AUTO_STOP_DONE`` (a user-paused ``on_hold`` task NEVER defers — the merged
  ``"auto-stop-done"`` class covers it, blocker 1) + laziness (no fence read
  while a cheaper shield holds);
* the #2277 semantics end-to-end through ``_process_pod``: owner-matched
  PASS / expired fence / ``fence_until=none`` clear / malformed fence /
  pod-scoping / the run-launched evidence-window reset all decline the
  defer and the stop proceeds exactly as pre-#2283;
* episode state in the pod_id-keyed ``fd_pod`` sub-dict (blocker 2 — the
  ``kr_pod``/``nr_pod`` contract): once-per-episode marker, 24h push
  re-alert, the cumulative ceiling exhaust-and-re-arm (once-per-episode
  ceiling marker), evaluated-and-inactive CLEAR vs not-evaluated CARRY,
  sibling-pod-save survival, GC on pod departure;
* the registration-inertness of BOTH watcher markers (an ``epm:progress``
  note binding the pod with a fence token would itself REGISTER/CLEAR the
  owner's fence — the self-defeat hazard), sentinel anti-liveness
  membership, env-knob defaults/clamps, dry-run purity, and the real
  sidecar-appender body.

``test_pod_safety_unexpired_fence_defers_stop`` is the fires-pre-fix test:
on pre-#2283 HEAD the fence tokens are ignored and the pod is stopped on
the second tick.

Follows ``tests/test_autonomous_session_watch_keep_running_owner.py``
conventions: PodInfo fixtures, the patched state dir, ``task.py`` reads
monkeypatched, no network, no real marker posts.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import pod_lifecycle as pl  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

ISSUE = 92283  # fictitious: no real task / run-handle sidecar can collide
POD = "pod-92283"
POD_ID = "fd92283a"
OWNER = "sess-owner-1"
FUTURE_FENCE = "2099-01-01T00:00:00Z"
PAST_FENCE = "2020-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the per-pod state dir at a tmp dir (mirrors the sibling suites)."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _iso(epoch: float) -> str:
    """Canonical task-event timestamp shape."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _info(
    pod_id: str = POD_ID,
    name: str = POD,
    created_at: str | None = None,
) -> PodInfo:
    """A RUNNING pod with a public port (so the #692 wedge arm never handles it)."""
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status="RUNNING",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host="1.2.3.4",
        ssh_port=22001,
        created_at=created_at,
    )


def _launch_event(ts: str, *, pod: str = POD, owner: str | None = OWNER, fence: str | None):
    """An ``epm:run-launched`` naming the pod in structured position (#1961),
    optionally registering ``owner=`` / ``fence_until=`` tokens (#2277)."""
    note = f"pod={pod}"
    if owner is not None:
        note += f" owner={owner}"
    if fence is not None:
        note += f" fence_until={fence}"
    note += " run launched on 1x H100"
    return {"kind": "epm:run-launched", "ts": ts, "note": note, "by": "experimenter"}


def _hb_event(ts: str, *, pod: str = POD, fence: str):
    """An ``epm:progress`` heartbeat re-registering / clearing the fence."""
    return {
        "kind": "epm:progress",
        "ts": ts,
        "note": f"pod={pod} fence_until={fence} long-phase-heartbeat",
        "by": "orchestrator",
    }


def _pass_event(ts: str, *, pod: str = POD, owner: str | None = OWNER):
    """A pod-bound ``epm:upload-verification`` PASS carrying the owner token."""
    note = f"Verdict: PASS — inline-round verification; prefixes: x pod={pod}"
    if owner is not None:
        note += f" owner={owner}"
    note += " outroot=none"
    return {"kind": "epm:upload-verification", "ts": ts, "note": note, "by": "upload-verifier"}


def _done_event(ts: str, *, status: str = "completed"):
    return {
        "kind": "epm:status-changed",
        "ts": ts,
        "note": f"verifying -> {status}",
        "by": "task-workflow",
    }


@pytest.fixture
def fence_rig(monkeypatch, isolated_registry):
    """The #2283 replay rig: RUNNING primary pod on a DONE-status task whose
    launch registered an owner + an unexpired fence; no keep-running tag, no
    live follow-up, no vetoes. The fence READ goes through the REAL
    ``pod_lifecycle.owner_fence_state`` chain over the fixture events list —
    only the task.py / marker / push / sidecar / stop boundaries are faked.

    ``_post_progress_marker`` APPENDS each posted marker to the live events
    list (ts = the rig clock), so the registration-inertness of the watcher's
    own markers is genuinely exercised across ticks (a pod-binding note would
    CLEAR the fence and flip later ticks to stop)."""
    t0 = time.time()
    clock = {"now": t0}
    holders = {"status": "completed", "keep_running": False, "followup": False}
    events = [
        _launch_event(_iso(t0 - 72 * 3600), fence=FUTURE_FENCE),
        _done_event(_iso(t0 - 3600)),
    ]
    posts: list[tuple[int, str, str | None, bool]] = []
    pushes: list[tuple[str, bool]] = []
    stops: list[int] = []
    sidecar: list[tuple[dict, bool]] = []

    def _post_stub(issue, note, dry_run, label=None):
        posts.append((issue, note, label, dry_run))
        events.append(
            {
                "kind": "epm:progress",
                "ts": _iso(clock["now"]),
                "note": note,
                "by": "autonomous_session_watch",
            }
        )

    monkeypatch.setattr(asw, "_task_status", lambda issue: holders["status"])
    monkeypatch.setattr(asw, "_task_events", lambda issue: events)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: holders["keep_running"])
    monkeypatch.setattr(
        asw, "_task_followup_active", lambda issue, events=None, **_kw: holders["followup"]
    )
    monkeypatch.setattr(asw, "_post_progress_marker", _post_stub)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append((msg, dry_run)))
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    # raising=False: the attribute does not exist on pre-#2283 HEAD, and the
    # fires-pre-fix test must fail on BEHAVIOR (the stop firing), never on a
    # fixture AttributeError.
    monkeypatch.setattr(
        asw,
        "_append_pod_owner_fence_event",
        lambda payload, dry_run: sidecar.append((payload, dry_run)),
        raising=False,
    )

    def tick(
        offset_s: float = 0.0,
        *,
        dry_run: bool = False,
        threshold: int = 2,
        pod_id: str = POD_ID,
        name: str = POD,
        running_pod_ids: set[str] | None = None,
    ) -> None:
        clock["now"] = t0 + offset_s
        asw._process_pod(
            ISSUE,
            pod_id,
            _info(pod_id=pod_id, name=name, created_at=_iso(t0 - 72 * 3600)),
            clock["now"],
            dry_run=dry_run,
            threshold=threshold,
            issue_running_pod_ids=running_pod_ids if running_pod_ids is not None else {pod_id},
        )

    return SimpleNamespace(
        t0=t0,
        clock=clock,
        holders=holders,
        events=events,
        posts=posts,
        pushes=pushes,
        stops=stops,
        sidecar=sidecar,
        state_path=isolated_registry / f"pod-safety-{ISSUE}.json",
        registry=isolated_registry,
        tick=tick,
    )


def _defer_posts(posts):
    return [p for p in posts if asw._FENCE_DEFER_NOTE_SENTINEL in p[1]]


def _ceiling_posts(posts):
    return [p for p in posts if asw._FENCE_CEILING_NOTE_SENTINEL in p[1]]


def _autostop_posts(posts):
    return [p for p in posts if asw._AUTOSTOP_NOTE_SENTINEL in p[1]]


def _fd_entry(rig, pod_id: str = POD_ID) -> dict | None:
    state = json.loads(rig.state_path.read_text())
    fd = state.get("fd_pod")
    assert isinstance(fd, dict), state
    return fd.get(pod_id)


def _pre2283_decide_oracle(status_class, missed, alerted, threshold, keep_running, followup):
    """The pre-#2283 decide_pod_safety decision table, verbatim."""
    if status_class == "auto-stop-done":
        if keep_running:
            return ("keep-running-skip", 0)
        if followup:
            return ("followup-skip", 0)
        new_missed = missed + 1
        if new_missed >= threshold:
            return ("stop", 0)
        return ("keep", new_missed)
    if status_class == "pod-active-stale" and not alerted:
        return ("alert", 0)
    return ("keep", 0)


# ---------------------------------------------------------------------------
# 1. Pure predicate (decide_pod_safety) — the defer branch + DEFER-ONLY sweep
# ---------------------------------------------------------------------------


def test_decide_fence_defer_fires_on_active_unexhausted_fence():
    assert asw.decide_pod_safety(
        status_class="auto-stop-done",
        missed=1,
        stale=False,
        alerted=False,
        fence_active=True,
        fence_defer_exhausted=False,
    ) == ("fence-defer", 0)


def test_decide_keep_running_precedes_fence_defer():
    assert asw.decide_pod_safety(
        status_class="auto-stop-done",
        missed=0,
        stale=False,
        alerted=False,
        keep_running=True,
        fence_active=True,
    ) == ("keep-running-skip", 0)


def test_decide_followup_precedes_fence_defer():
    assert asw.decide_pod_safety(
        status_class="auto-stop-done",
        missed=0,
        stale=False,
        alerted=False,
        followup_active=True,
        fence_active=True,
    ) == ("followup-skip", 0)


def test_decide_exhausted_fence_resumes_accumulation():
    """An exhausted ceiling bypasses the defer branch: ordinary accumulation."""
    assert asw.decide_pod_safety(
        status_class="auto-stop-done",
        missed=0,
        stale=False,
        alerted=False,
        fence_active=True,
        fence_defer_exhausted=True,
    ) == ("keep", 1)
    assert asw.decide_pod_safety(
        status_class="auto-stop-done",
        missed=1,
        stale=False,
        alerted=False,
        fence_active=True,
        fence_defer_exhausted=True,
    ) == ("stop", 0)


def test_decide_defaults_byte_identical_to_pre2283_table():
    """With the fence params at their False defaults, decide_pod_safety is
    byte-identical to the pre-#2283 decision table (back-compat contract)."""
    classes = ["auto-stop-done", "pod-active-stale", "pod-active-fresh", "other"]
    for sc, missed, alerted, kr, fu in itertools.product(
        classes, range(4), (False, True), (False, True), (False, True)
    ):
        expected = _pre2283_decide_oracle(sc, missed, alerted, 2, kr, fu)
        got = asw.decide_pod_safety(
            status_class=sc,
            missed=missed,
            stale=sc == "pod-active-stale",
            alerted=alerted,
            keep_running=kr,
            followup_active=fu,
        )
        assert got == expected, (sc, missed, alerted, kr, fu)


def test_decide_fence_arm_never_accelerates_a_stop():
    """DEFER-ONLY hard invariant, swept over the full input cross-product:
    the ONLY deviation from the pre-#2283 table is stop/keep -> fence-defer
    on the eligible (active, unexhausted, unshielded, auto-stop-done) cell —
    never a stop where the old table kept, never a higher miss count."""
    classes = ["auto-stop-done", "pod-active-stale", "pod-active-fresh", "other"]
    for sc, missed, alerted, kr, fu, fa, fx in itertools.product(
        classes, range(4), (False, True), (False, True), (False, True), (False, True), (False, True)
    ):
        oracle = _pre2283_decide_oracle(sc, missed, alerted, 2, kr, fu)
        got = asw.decide_pod_safety(
            status_class=sc,
            missed=missed,
            stale=sc == "pod-active-stale",
            alerted=alerted,
            keep_running=kr,
            followup_active=fu,
            fence_active=fa,
            fence_defer_exhausted=fx,
        )
        if got != oracle:
            # The one licensed deviation: a defer where the old table would
            # have kept-toward-stop or stopped.
            assert got == ("fence-defer", 0), (sc, missed, alerted, kr, fu, fa, fx, got)
            assert sc == "auto-stop-done" and not kr and not fu and fa and not fx
            assert oracle[0] in {"keep", "stop"}
        # Never accelerate: a stop only where the oracle already stopped.
        if got[0] == "stop":
            assert oracle[0] == "stop"
        assert got[1] <= oracle[1] or got == oracle


# ---------------------------------------------------------------------------
# 2. Env knobs
# ---------------------------------------------------------------------------


def test_pod_fence_defer_max_s_default_overrides_and_floor(monkeypatch):
    monkeypatch.delenv("EPM_POD_FENCE_DEFER_MAX_H", raising=False)
    assert asw._pod_fence_defer_max_s() == 24.0 * 3600.0
    monkeypatch.setenv("EPM_POD_FENCE_DEFER_MAX_H", "6")
    assert asw._pod_fence_defer_max_s() == 6.0 * 3600.0
    # Positive sub-floor value clamps UP to the 1h floor.
    monkeypatch.setenv("EPM_POD_FENCE_DEFER_MAX_H", "0.25")
    assert asw._pod_fence_defer_max_s() == 1.0 * 3600.0
    # Malformed / non-positive / non-finite all fall back to the default.
    for bad in ("garbage", "0", "-3", "inf", "nan"):
        monkeypatch.setenv("EPM_POD_FENCE_DEFER_MAX_H", bad)
        assert asw._pod_fence_defer_max_s() == 24.0 * 3600.0, bad


def test_pod_fence_defer_enabled_kill_switch(monkeypatch):
    monkeypatch.delenv("EPM_DISABLE_POD_FENCE_DEFER", raising=False)
    assert asw._pod_fence_defer_enabled() is True
    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv("EPM_DISABLE_POD_FENCE_DEFER", truthy)
        assert asw._pod_fence_defer_enabled() is False, truthy
    monkeypatch.setenv("EPM_DISABLE_POD_FENCE_DEFER", "0")
    assert asw._pod_fence_defer_enabled() is True


# ---------------------------------------------------------------------------
# 3. Tri-state wrapper against the REAL pod_lifecycle reader chain
# ---------------------------------------------------------------------------


def test_wrapper_true_on_unexpired_unwaived_fence():
    now = time.time()
    events = [_launch_event(_iso(now - 3600), fence=FUTURE_FENCE)]
    assert asw._pod_owner_fence_active(POD, events, now) is True


@pytest.mark.parametrize(
    "events_factory",
    [
        # no tokens at all
        lambda now: [_launch_event(_iso(now - 3600), owner=None, fence=None)],
        # expired fence
        lambda now: [_launch_event(_iso(now - 3600), fence=PAST_FENCE)],
        # fence cleared by a later fence_until=none heartbeat
        lambda now: [
            _launch_event(_iso(now - 3600), fence=FUTURE_FENCE),
            _hb_event(_iso(now - 600), fence="none"),
        ],
        # owner-matched pod-bound PASS discharges the fence
        lambda now: [
            _launch_event(_iso(now - 3600), fence=FUTURE_FENCE),
            _pass_event(_iso(now - 300), owner=OWNER),
        ],
    ],
    ids=["no-tokens", "expired", "cleared-none", "owner-matched-pass"],
)
def test_wrapper_false_when_fence_evaluates_inactive(events_factory):
    now = time.time()
    assert asw._pod_owner_fence_active(POD, events_factory(now), now) is False


def test_wrapper_malformed_fence_reads_false_with_warn(capsys):
    """A malformed fence_until is the pod_lifecycle fail-open (WARN + absent)
    — the wrapper reads it as an EVALUATED False, never a defer."""
    now = time.time()
    events = [_launch_event(_iso(now - 3600), fence="not-a-date")]
    assert asw._pod_owner_fence_active(POD, events, now) is False
    assert "unparseable fence_until" in capsys.readouterr().err


def test_wrapper_none_on_kill_switch(monkeypatch):
    monkeypatch.setenv("EPM_DISABLE_POD_FENCE_DEFER", "1")
    now = time.time()
    events = [_launch_event(_iso(now - 3600), fence=FUTURE_FENCE)]
    assert asw._pod_owner_fence_active(POD, events, now) is None


def test_wrapper_none_and_warns_on_reader_failure(monkeypatch, capsys):
    """Any reader-chain exception -> one loud WARN + None (fail-open: no
    defer, no episode clear) — never a crash into the watcher pass."""

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic reader failure")

    monkeypatch.setattr(pl, "owner_fence_state", _boom)
    now = time.time()
    events = [_launch_event(_iso(now - 3600), fence=FUTURE_FENCE)]
    assert asw._pod_owner_fence_active(POD, events, now) is None
    err = capsys.readouterr().err
    assert "owner-fence read FAILED" in err
    assert "RuntimeError" in err


# ---------------------------------------------------------------------------
# 4. _escaped_pod_exemptions eligibility (blocker 1 + laziness)
# ---------------------------------------------------------------------------


def test_exemptions_fence_eligibility_keyed_on_auto_stop_done_set(monkeypatch):
    """Eligibility is STRICTLY ``status in AUTO_STOP_DONE`` — the merged
    ``"auto-stop-done"`` class also covers user-paused ``on_hold``
    (AUTO_STOP_PAUSED), which must read None (not evaluated), never a
    defer-capable True (blocker 1)."""
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, **_kw: False)
    now = time.time()
    events = [_launch_event(_iso(now - 3600), fence=FUTURE_FENCE)]
    for status in sorted(asw.AUTO_STOP_DONE):
        kr, fu, fence_read = asw._escaped_pod_exemptions(
            ISSUE, "auto-stop-done", events, pod_name=POD, now=now, status=status
        )
        assert (kr, fu, fence_read) == (False, False, True), status
    # on_hold: same merged status class, NEVER evaluated.
    assert "on_hold" not in asw.AUTO_STOP_DONE  # the merged-class premise
    kr, fu, fence_read = asw._escaped_pod_exemptions(
        ISSUE, "auto-stop-done", events, pod_name=POD, now=now, status="on_hold"
    )
    assert (kr, fu, fence_read) == (False, False, None)
    # A hypothetical unknown status inside the merged class: also None.
    assert (
        asw._escaped_pod_exemptions(
            ISSUE, "auto-stop-done", events, pod_name=POD, now=now, status="mystery_new_status"
        )[2]
        is None
    )


def test_exemptions_fence_read_is_lazy_behind_cheaper_shields(monkeypatch):
    """No fence read while keep-running / follow-up holds, off the auto-stop
    class, or with missing inputs — the wrapper must not even be called."""
    calls: list[str] = []
    monkeypatch.setattr(
        asw, "_pod_owner_fence_active", lambda pod, events, now: calls.append(pod) or True
    )
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, **_kw: False)
    now = time.time()
    assert asw._escaped_pod_exemptions(
        ISSUE, "auto-stop-done", [], pod_name=POD, now=now, status="completed"
    ) == (True, False, None)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, **_kw: True)
    assert asw._escaped_pod_exemptions(
        ISSUE, "auto-stop-done", [], pod_name=POD, now=now, status="completed"
    ) == (False, True, None)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, **_kw: False)
    # Off the auto-stop class: nothing is even consulted.
    assert asw._escaped_pod_exemptions(
        ISSUE, "pod-active-fresh", [], pod_name=POD, now=now, status="running"
    ) == (False, False, None)
    # Missing pod_name / now: no fence claim without the inputs.
    assert asw._escaped_pod_exemptions(ISSUE, "auto-stop-done", [], status="completed")[2] is None
    assert calls == []  # the wrapper was never reached


# ---------------------------------------------------------------------------
# 5. End-to-end through _process_pod (the fires-pre-fix test first)
# ---------------------------------------------------------------------------


def test_pod_safety_unexpired_fence_defers_stop(fence_rig):
    """FIRES-PRE-FIX: on pre-#2283 HEAD the fence tokens are ignored, the
    miss counter accumulates, and the pod is STOPPED on the second tick.
    Post-fix: the stop is deferred every tick, ONE defer marker is posted,
    and the episode onset is stamped in the fd_pod entry."""
    fence_rig.tick(0)
    fence_rig.tick(600)
    fence_rig.tick(1200)
    assert fence_rig.stops == []  # pre-fix: [ISSUE] (stopped on tick 2)
    assert len(_defer_posts(fence_rig.posts)) == 1
    assert _autostop_posts(fence_rig.posts) == []
    entry = _fd_entry(fence_rig)
    assert entry is not None
    assert entry["noted"] is True
    assert entry["first_ts"] == pytest.approx(fence_rig.t0)
    state = json.loads(fence_rig.state_path.read_text())
    assert state["missed"] == 0  # defer resets the accumulation


def test_pod_safety_no_fence_tokens_decisions_unchanged(fence_rig):
    """Token-less events: the arm is invisible and the stop fires exactly as
    pre-#2283 (evidence-gated back-compat)."""
    fence_rig.events[:] = [
        _launch_event(_iso(fence_rig.t0 - 72 * 3600), owner=None, fence=None),
        _done_event(_iso(fence_rig.t0 - 3600)),
    ]
    fence_rig.tick(0)
    assert fence_rig.stops == []  # miss 1 of 2
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert len(_autostop_posts(fence_rig.posts)) == 1
    assert _defer_posts(fence_rig.posts) == []
    assert not fence_rig.state_path.exists()  # stop clears the state file


def test_pod_safety_owner_matched_pass_does_not_defer(fence_rig):
    """The owner's own pod-bound PASS discharges the fence: verified-done
    teardown resumes (the #2277 owner-teardown parity)."""
    fence_rig.events.insert(1, _pass_event(_iso(fence_rig.t0 - 1800), owner=OWNER))
    fence_rig.tick(0)
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert _defer_posts(fence_rig.posts) == []


def test_pod_safety_expired_fence_does_not_defer(fence_rig):
    fence_rig.events[0] = _launch_event(_iso(fence_rig.t0 - 72 * 3600), fence=PAST_FENCE)
    fence_rig.tick(0)
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert _defer_posts(fence_rig.posts) == []


def test_pod_safety_fence_cleared_by_none_does_not_defer(fence_rig):
    fence_rig.events.insert(1, _hb_event(_iso(fence_rig.t0 - 1800), fence="none"))
    fence_rig.tick(0)
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert _defer_posts(fence_rig.posts) == []


def test_pod_safety_malformed_fence_does_not_defer(fence_rig, capsys):
    """A malformed fence is the pod_lifecycle fail-open (WARN + absent): the
    stop proceeds — garbage evidence must never shield a pod."""
    fence_rig.events[0] = _launch_event(_iso(fence_rig.t0 - 72 * 3600), fence="2099-99-99")
    fence_rig.tick(0)
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert _defer_posts(fence_rig.posts) == []
    assert "unparseable fence_until" in capsys.readouterr().err


def test_pod_safety_user_paused_on_hold_never_defers(fence_rig):
    """Blocker 1: a USER PAUSE (on_hold) is never deferred by a self-posted
    fence — the pause-window escaped-pod stop proceeds at threshold."""
    fence_rig.holders["status"] = "on_hold"
    fence_rig.tick(0)
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert _defer_posts(fence_rig.posts) == []


def test_pod_safety_window_reset_at_new_run_launched_drops_stale_fence(fence_rig):
    """A relaunch naming the pod WITHOUT fence tokens resets the #2277
    evidence window: the prior incarnation's fence never binds the fresh
    pod, so the stop proceeds."""
    fence_rig.events.insert(
        1, _launch_event(_iso(fence_rig.t0 - 24 * 3600), owner=None, fence=None)
    )
    fence_rig.tick(0)
    fence_rig.tick(600)
    assert fence_rig.stops == [ISSUE]
    assert _defer_posts(fence_rig.posts) == []


def test_pod_safety_fence_is_pod_scoped(fence_rig):
    """The fence binds pod-92283 only: a sibling pod on the SAME issue is
    stopped exactly as before, while the fenced pod defers."""
    sib_id, sib_name = "sib92283b", f"{POD}-b"
    both = {POD_ID, sib_id}
    fence_rig.tick(0, running_pod_ids=both)  # fenced pod: defer
    fence_rig.tick(60, pod_id=sib_id, name=sib_name, running_pod_ids=both)  # sib: miss 1
    fence_rig.tick(600, pod_id=sib_id, name=sib_name, running_pod_ids=both)  # sib: stop
    assert fence_rig.stops == [ISSUE]
    assert len(_defer_posts(fence_rig.posts)) == 1


def test_pod_safety_fence_defer_never_stops_or_terminates(fence_rig, monkeypatch):
    """DEFER-ONLY hard invariant on the live path: while the fence holds
    (inside the ceiling), no tick stops, terminates, or shells out."""

    def _no_shell(*a, **kw):  # pragma: no cover - defense
        raise AssertionError(f"fence-defer tick must not shell out: {a!r}")

    monkeypatch.setattr(asw.subprocess, "run", _no_shell)
    for off in (0, 600, 1200, 1800, 3600):
        fence_rig.tick(off)
    assert fence_rig.stops == []
    assert _autostop_posts(fence_rig.posts) == []


# ---------------------------------------------------------------------------
# 6. Episode state: dedup, re-alert TTL, ceiling, clear-vs-carry, siblings, GC
# ---------------------------------------------------------------------------


def test_pod_safety_defer_marker_deduped_and_push_realerts(fence_rig, monkeypatch):
    """ONE marker per episode; push + sidecar re-fire on the 24h TTL. The
    ceiling is raised out of the way so the re-alert path is isolated."""
    monkeypatch.setenv("EPM_POD_FENCE_DEFER_MAX_H", "1000")
    fence_rig.tick(0)
    assert len(_defer_posts(fence_rig.posts)) == 1
    assert len(fence_rig.pushes) == 1
    assert len(fence_rig.sidecar) == 1
    fence_rig.tick(3600)  # inside the TTL: no re-emission
    assert len(_defer_posts(fence_rig.posts)) == 1
    assert len(fence_rig.pushes) == 1
    assert len(fence_rig.sidecar) == 1
    fence_rig.tick(25 * 3600)  # past the 24h TTL: push + sidecar, NO 2nd marker
    assert len(_defer_posts(fence_rig.posts)) == 1
    assert len(fence_rig.pushes) == 2
    assert len(fence_rig.sidecar) == 2
    assert fence_rig.sidecar[1][0]["action"] == "defer-re-alert"


def test_pod_safety_ceiling_exhausts_and_rearms(fence_rig):
    """A continuously-refreshed fence cannot shield forever: past the 24h
    cumulative ceiling the ONCE-per-episode ceiling escalation fires and
    ordinary accumulation resumes -> stop two ticks later."""
    fence_rig.tick(0)  # defer; first_ts = t0
    assert fence_rig.stops == []
    fence_rig.tick(25 * 3600)  # exhausted: ceiling marker + miss 1
    assert len(_ceiling_posts(fence_rig.posts)) == 1
    assert fence_rig.stops == []
    ceiling_rows = [p for p, _dry in fence_rig.sidecar if p.get("kind") == "fence-ceiling"]
    assert len(ceiling_rows) == 1
    assert ceiling_rows[0]["deferred_h"] == pytest.approx(25.0, abs=0.1)
    entry = _fd_entry(fence_rig)
    assert entry["ceiling_noted"] is True
    assert entry["first_ts"] == pytest.approx(fence_rig.t0)  # onset preserved
    fence_rig.tick(25 * 3600 + 600)  # miss 2 -> STOP; ceiling marker NOT re-posted
    assert fence_rig.stops == [ISSUE]
    assert len(_ceiling_posts(fence_rig.posts)) == 1


def test_pod_safety_episode_clears_on_evaluated_inactive_read(fence_rig):
    """An EVALUATED-and-inactive read (here: the owner's PASS landing after
    the defer) CLEARS the fd entry to fresh defaults, so a LATER fence opens
    a fresh episode with a fresh ceiling clock."""
    fence_rig.tick(0)
    assert _fd_entry(fence_rig)["noted"] is True
    fence_rig.events.append(_pass_event(_iso(fence_rig.t0 + 300), owner=OWNER))
    fence_rig.tick(600)  # fence discharged -> accumulation resumes (miss 1)
    assert fence_rig.stops == []
    entry = _fd_entry(fence_rig)
    assert entry == {
        "first_ts": None,
        "noted": False,
        "last_push_ts": None,
        "ceiling_noted": False,
    }
    fence_rig.tick(1200)
    assert fence_rig.stops == [ISSUE]


def test_pod_safety_not_evaluated_read_carries_episode(fence_rig, monkeypatch):
    """A NOT-EVALUATED read (kill switch here; reader failure is the same
    None) never defers — the stop path resumes as pre-#2283 — but the
    episode entry is CARRIED verbatim, never cleared."""
    fence_rig.tick(0)  # defer; episode open
    monkeypatch.setenv("EPM_DISABLE_POD_FENCE_DEFER", "1")
    fence_rig.tick(600)  # None: accumulate miss 1, entry carried
    assert fence_rig.stops == []
    entry = _fd_entry(fence_rig)
    assert entry["noted"] is True
    assert entry["first_ts"] == pytest.approx(fence_rig.t0)
    fence_rig.tick(1200)  # miss 2 -> stop (kill switch = pre-#2283 behavior)
    assert fence_rig.stops == [ISSUE]


def test_pod_safety_fd_entry_survives_sibling_pod_save_and_gcs_on_departure(fence_rig):
    """Blocker 2: fd_pod is pod_id-keyed on the kr_pod/nr_pod contract — a
    sibling pod's save forward-carries a fenced pod's entry verbatim, and a
    departed pod's entry is GC'd by a defer-save whose keep-set excludes it."""
    sib_id, sib_name = "sib92283b", f"{POD}-b"
    # Fence the sibling too (its own launch event names it).
    fence_rig.events.insert(
        1, _launch_event(_iso(fence_rig.t0 - 71 * 3600), pod=sib_name, fence=FUTURE_FENCE)
    )
    both = {POD_ID, sib_id}
    fence_rig.tick(0, running_pod_ids=both)  # pod A defers; entry A
    fence_rig.tick(60, pod_id=sib_id, name=sib_name, running_pod_ids=both)  # pod B defers
    state = json.loads(fence_rig.state_path.read_text())
    assert set(state["fd_pod"]) == both
    # A's entry survived B's save verbatim (onset + noted intact).
    assert state["fd_pod"][POD_ID]["first_ts"] == pytest.approx(fence_rig.t0)
    assert state["fd_pod"][POD_ID]["noted"] is True
    assert state["fd_pod"][sib_id]["first_ts"] == pytest.approx(fence_rig.t0 + 60)
    # Pod A departs: B's next defer-save GCs A's entry.
    fence_rig.tick(600, pod_id=sib_id, name=sib_name, running_pod_ids={sib_id})
    state = json.loads(fence_rig.state_path.read_text())
    assert set(state["fd_pod"]) == {sib_id}


# ---------------------------------------------------------------------------
# 7. Registration-inertness, sentinels, dry-run, real appender body
# ---------------------------------------------------------------------------


def test_fence_defer_marker_is_registration_inert(fence_rig):
    """The defer marker is an epm:progress note (a REGISTRATION kind) that
    quotes the release recipe (`fence_until=none`): it must NEVER bind the
    pod in structured position, or the watcher's own marker would CLEAR the
    owner's fence and flip the next tick to stop (the self-defeat hazard)."""
    fence_rig.tick(0)
    note = _defer_posts(fence_rig.posts)[0][1]
    assert "fence_until=none" in note  # the release recipe IS quoted...
    assert not pl._note_names_pod(note, POD)  # ...but never binds the pod
    # The marker was appended to events by the rig: the REAL reader chain
    # still sees the original fence intact afterwards.
    st = pl.owner_fence_state(fence_rig.events, POD, _dt.datetime.now(_dt.UTC))
    assert st.fence_unexpired is True
    assert st.owner_registered == OWNER
    # And the next ticks still defer (no stop, no second marker).
    fence_rig.tick(600)
    assert fence_rig.stops == []
    assert len(_defer_posts(fence_rig.posts)) == 1


def test_fence_ceiling_marker_is_registration_inert(fence_rig):
    fence_rig.tick(0)
    fence_rig.tick(25 * 3600)
    note = _ceiling_posts(fence_rig.posts)[0][1]
    assert not pl._note_names_pod(note, POD)
    st = pl.owner_fence_state(fence_rig.events, POD, _dt.datetime.now(_dt.UTC))
    assert st.fence_unexpired is True  # the ceiling marker cleared nothing


def test_fence_sentinels_are_anti_liveness_members():
    assert asw._FENCE_DEFER_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._FENCE_CEILING_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_pod_safety_fence_defer_dry_run_no_mutation(fence_rig):
    """Dry-run defer tick: log line only — no state file, and every recorded
    emission carries the dry_run flag (the stubs record; the real helpers
    no-op under dry_run)."""
    fence_rig.tick(0, dry_run=True)
    assert not fence_rig.state_path.exists()
    assert fence_rig.stops == []
    assert all(dry for _i, _n, _l, dry in fence_rig.posts)
    assert all(dry for _m, dry in fence_rig.pushes)
    assert all(dry for _p, dry in fence_rig.sidecar)


def test_fence_sidecar_real_appender_body(tmp_path, monkeypatch):
    """Production-body test for _append_pod_owner_fence_event (the one seam
    the rig stubs): real writes land one JSON line with ts + the default
    kind, a payload kind overrides, and dry_run writes nothing."""
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)
    dest = tmp_path / ".claude" / "cache" / "pod-owner-fence-events.jsonl"
    asw._append_pod_owner_fence_event({"issue": ISSUE, "pod_id": POD_ID}, dry_run=True)
    assert not dest.exists()
    asw._append_pod_owner_fence_event({"issue": ISSUE, "pod_id": POD_ID}, dry_run=False)
    asw._append_pod_owner_fence_event({"issue": ISSUE, "kind": "fence-ceiling"}, dry_run=False)
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["fence-defer", "fence-ceiling"]
    assert rows[0]["issue"] == ISSUE
    assert rows[0]["ts"]  # stamped


def test_owner_fence_state_parity_with_guard_read(fence_rig):
    """The watcher consumes the SAME owner_fence_state the terminate guard
    runs: the rig's baseline events read blocks_teardown=True (unexpired,
    unwaived) through the public pod_lifecycle entrypoint directly."""
    st = pl.owner_fence_state(fence_rig.events, POD, _dt.datetime.now(_dt.UTC))
    assert st.blocks_teardown is True
    assert st.owner_registered == OWNER
    assert st.pass_owner is None


def test_named_residuals_and_consequence_are_disclosed():
    """Plan §4.4 promised four disclosures; every behavior below is
    implemented + pinned above, so the only way they regress is silently
    losing their prose. Rule paragraph: the wedge fall-through consequence +
    the lapse-recycle residual. Arm docstring: lapse-recycle, the
    CARRY-across-handover decision, and the copied-token trust model."""
    repo = Path(__file__).resolve().parents[1]
    rule = (repo / ".claude" / "rules" / "background-automation.md").read_text()
    arm = asw._handle_fence_defer_action.__doc__ or ""
    for token in ("wedge fall-through", "lapse-recycle"):
        assert token in rule, f"{token!r} missing from the background-automation arm paragraph"
    for token in ("lapse-recycle", "CARRY across a long shield handover", "copied-token"):
        assert token in arm, f"{token!r} missing from _handle_fence_defer_action.__doc__"
