"""Decision-matrix + I/O-wrapper tests for the autonomous-session watcher.

Two passes are pinned here:

1. **Respawn pass.** A wrong RESPAWN launches a duplicate session -> a duplicate
   pod -> real spend, so the pure :func:`decide` gate is pinned exhaustively.
2. **Pod-safety pass.** The CONSERVATIVE + ALERT redesign (2026-06-05): the STOP
   trigger is task STATUS, not session-cwd liveness. Two regressions the prior
   round missed are pinned explicitly:
     * Bug A — a real ``pod-<N>`` pod must be RECOGNIZED end-to-end (the old
       ``epm-issue-<N>``-only regex matched no live pod, so the pass was dead
       code).
     * Bug B — a LIVE interactive session (cwd = repo root, NOT the worktree)
       must NOT cause a stop (the old cwd-liveness stop trigger misread it as
       dead and would have killed healthy pods).
"""

import sys
from pathlib import Path

import pytest

# scripts/ holds autonomous_session_watch.py (and its spawn_session import).
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402
from autonomous_session_watch import (  # noqa: E402
    ACTIVE,
    ALERT_STALE_HOURS,
    AUTO_STOP_DONE,
    PARK,
    POD_ACTIVE,
    TERMINAL,
    decide,
    decide_pod_safety,
)


@pytest.mark.parametrize("status", sorted(TERMINAL))
@pytest.mark.parametrize("alive", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_terminal_always_deletes(status, alive, missed):
    # A finished run is dropped no matter what — never re-spawned.
    assert decide(status, alive, missed) == ("delete", 0)


@pytest.mark.parametrize("status", sorted(PARK))
@pytest.mark.parametrize("alive", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_park_keeps_and_resets(status, alive, missed):
    # Parked tasks (waiting on the user / a gate) are never re-spawned, and the
    # miss counter resets so a later flip to ACTIVE starts clean.
    assert decide(status, alive, missed) == ("keep", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_active_alive_keeps(status):
    assert decide(status, alive=True, missed=3) == ("keep", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_active_dead_needs_two_misses_before_respawn(status):
    # First dead check only increments; respawn fires on the SECOND consecutive
    # miss (default threshold 2) — guards a transient daemon-list glitch.
    assert decide(status, alive=False, missed=0, threshold=2) == ("keep", 1)
    assert decide(status, alive=False, missed=1, threshold=2) == ("respawn", 0)


def test_threshold_one_respawns_immediately():
    assert decide("running", alive=False, missed=0, threshold=1) == ("respawn", 0)


def test_higher_threshold_delays_respawn():
    assert decide("running", alive=False, missed=1, threshold=3) == ("keep", 2)
    assert decide("running", alive=False, missed=2, threshold=3) == ("respawn", 0)


def test_unknown_status_is_inert():
    # A renamed/unexpected status must never spawn; keep the entry untouched so
    # a human notices rather than silently dropping or double-spawning.
    assert decide("some_new_status", alive=False, missed=4) == ("keep", 4)
    assert decide("some_new_status", alive=True, missed=0) == ("keep", 0)


def test_status_sets_are_disjoint_and_cover_enum():
    # The three sets must not overlap (an overlap would make decide order-
    # dependent) and must cover the canonical task status enum.
    assert ACTIVE.isdisjoint(PARK)
    assert ACTIVE.isdisjoint(TERMINAL)
    assert PARK.isdisjoint(TERMINAL)
    canonical = {
        "proposed", "clarifying", "planning", "plan_pending", "approved",
        "running", "verifying", "interpreting", "reviewing",
        "awaiting_promotion", "completed", "blocked", "archived",
    }  # fmt: skip
    assert canonical <= (ACTIVE | PARK | TERMINAL)


def test_register_writes_atomic_entry(tmp_path, monkeypatch):
    # The crash-recovery invariant depends on this write succeeding; pin its
    # shape (missed must start at 0; the recorded id is what the watcher checks).
    import json

    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    spawn_session._register_autonomous_session(207, "sess-abc", "/repo", 7.0)
    entry = json.loads((tmp_path / "issue-207.json").read_text())
    assert entry["issue"] == 207
    assert entry["happy_session_id"] == "sess-abc"
    assert entry["auto_approve_gpu_hours"] == 7.0
    assert entry["missed"] == 0
    # No leftover temp file from the atomic write.
    assert not list(tmp_path.glob("*.tmp"))


def test_register_raises_on_unwritable_dir(tmp_path, monkeypatch):
    # A registration failure MUST raise (not swallow) so spawn_session can stop
    # the just-spawned session — an untracked live --auto session would risk a
    # duplicate re-spawn by the watcher.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a dir")
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", blocker / "sub")
    with pytest.raises(OSError):
        spawn_session._register_autonomous_session(1, "x", "/repo", 24.0)


# ─── pod-safety decision matrix ──────────────────────────────────────────────
# decide_pod_safety is keyed on the task STATUS CLASS, not session liveness.
# Stopping is reversible (pod.py stop preserves the volume), but a wrong stop
# still interrupts a live experiment, so the gate is pinned exhaustively.


@pytest.mark.parametrize("missed", [0, 1, 5])
def test_pod_safety_done_needs_two_misses_before_stop(missed):
    # A DONE task's RUNNING pod is an escaped pod. First check only increments;
    # the stop fires on the SECOND consecutive miss (default threshold 2) —
    # guards a transient API/status glitch.
    assert decide_pod_safety(
        status_class="auto-stop-done", missed=0, stale=False, alerted=False, threshold=2
    ) == ("keep", 1)
    assert decide_pod_safety(
        status_class="auto-stop-done", missed=1, stale=False, alerted=False, threshold=2
    ) == ("stop", 0)


def test_pod_safety_done_threshold_one_stops_immediately():
    assert decide_pod_safety(
        status_class="auto-stop-done", missed=0, stale=False, alerted=False, threshold=1
    ) == ("stop", 0)


def test_pod_safety_done_higher_threshold_delays_stop():
    assert decide_pod_safety(
        status_class="auto-stop-done", missed=1, stale=False, alerted=False, threshold=3
    ) == ("keep", 2)
    assert decide_pod_safety(
        status_class="auto-stop-done", missed=2, stale=False, alerted=False, threshold=3
    ) == ("stop", 0)


def test_pod_safety_stale_pod_active_alerts_not_stops():
    # The mid-run-death case: a pod-active task gone stale gets an ALERT, never a
    # stop. A false alert is a cheap nudge; a false stop kills a healthy run.
    assert decide_pod_safety(
        status_class="pod-active-stale", missed=0, stale=True, alerted=False
    ) == ("alert", 0)


def test_pod_safety_stale_already_alerted_stays_quiet():
    # Dedup: once alerted this episode, stay quiet (don't re-alert every tick).
    assert decide_pod_safety(
        status_class="pod-active-stale", missed=0, stale=True, alerted=True
    ) == ("keep", 0)


def test_pod_safety_fresh_pod_active_keeps():
    # A healthy mid-run pod (recent progress) is left strictly alone.
    assert decide_pod_safety(
        status_class="pod-active-fresh", missed=1, stale=False, alerted=False
    ) == ("keep", 0)


@pytest.mark.parametrize("missed", [0, 1, 5])
@pytest.mark.parametrize("alerted", [True, False])
def test_pod_safety_other_status_never_acts(missed, alerted):
    # blocked / followups_running / unknown statuses are classified "other":
    # never stopped, never alerted, miss counter reset.
    assert decide_pod_safety(status_class="other", missed=missed, stale=False, alerted=alerted) == (
        "keep",
        0,
    )


def test_pod_safety_shares_default_threshold_with_decide():
    # Both passes use the same 2-consecutive-miss default, so a single transient
    # glitch never acts in either pass.
    import inspect

    from autonomous_session_watch import decide as _decide

    assert (
        inspect.signature(decide_pod_safety).parameters["threshold"].default
        == inspect.signature(_decide).parameters["threshold"].default
        == 2
    )


def test_status_class_sets_disjoint():
    # A status must not be both "auto-stop" and "pod-active" — that would make
    # the classifier order-dependent.
    assert AUTO_STOP_DONE.isdisjoint(POD_ACTIVE)
    # blocked is deliberately in NEITHER (kept, alert-only-if-stale).
    assert "blocked" not in AUTO_STOP_DONE
    assert "blocked" not in POD_ACTIVE


# ─── _status_class classifier ────────────────────────────────────────────────


def test_status_class_done_statuses():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    for s in sorted(AUTO_STOP_DONE):
        assert asw._status_class(s, latest_progress_ts=now, now=now) == "auto-stop-done"


def test_status_class_pod_active_fresh_vs_stale():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    fresh = now - 3600  # 1h ago, under the 6h cap
    stale = now - (ALERT_STALE_HOURS + 1) * 3600
    assert asw._status_class("running", latest_progress_ts=fresh, now=now) == "pod-active-fresh"
    assert asw._status_class("running", latest_progress_ts=stale, now=now) == "pod-active-stale"


def test_status_class_pod_active_no_progress_is_stale():
    # A pod-active task with NO real progress marker at all is itself a signal.
    import autonomous_session_watch as asw

    assert (
        asw._status_class("verifying", latest_progress_ts=None, now=1_000_000.0)
        == "pod-active-stale"
    )


def test_status_class_none_and_blocked_are_other():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    assert asw._status_class(None, latest_progress_ts=now, now=now) == "other"
    assert asw._status_class("blocked", latest_progress_ts=None, now=now) == "other"


# ─── _latest_progress_ts (real-progress filter) ──────────────────────────────


def test_latest_progress_ts_picks_newest_real_marker():
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-06-05T10:00:00Z", "note": "step 100"},
        {"kind": "epm:results", "ts": "2026-06-05T12:00:00Z", "note": "done"},
        {"kind": "epm:clarify", "ts": "2026-06-05T13:00:00Z", "note": "n/a"},  # not progress
    ]
    ts = asw._latest_progress_ts(events)
    # Newest PROGRESS marker is the 12:00 results, not the 13:00 clarify.
    assert ts == asw._parse_event_ts("2026-06-05T12:00:00Z")


def test_latest_progress_ts_excludes_watchers_own_alert():
    # The watcher posts its stale-alert as epm:progress; it must NOT count as
    # real progress, or the alert would reset the staleness clock it measures.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-06-05T10:00:00Z", "note": "step 100"},
        {
            "kind": "epm:progress",
            "ts": "2026-06-05T18:00:00Z",
            "note": f"{asw._ALERT_NOTE_SENTINEL} STALE pod-active task ...",
        },
    ]
    ts = asw._latest_progress_ts(events)
    # The 18:00 event is the watcher's own alert -> ignored; newest real
    # progress stays the 10:00 step.
    assert ts == asw._parse_event_ts("2026-06-05T10:00:00Z")


def test_latest_progress_ts_none_when_no_progress():
    import autonomous_session_watch as asw

    assert asw._latest_progress_ts([]) is None
    assert asw._latest_progress_ts([{"kind": "epm:clarify", "ts": "2026-06-05T10:00:00Z"}]) is None


# ─── pod-safety I/O wrapper tests ────────────────────────────────────────────


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point AUTONOMOUS_REGISTRY_DIR at a tmp dir, in BOTH spawn_session (the
    canonical home) and autonomous_session_watch (which re-exports it via the
    `from spawn_session import` block). Both names refer to the same Path
    object at import time, so each must be patched independently."""
    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _write_state(
    reg_dir, issue, pod_id, missed, first_seen, *, alerted=False, last_progress_ts=None
):
    import json

    (reg_dir / f"pod-safety-{issue}.json").write_text(
        json.dumps(
            {
                "pod_id": pod_id,
                "missed": missed,
                "alerted": alerted,
                "last_progress_ts": last_progress_ts,
                "first_seen": first_seen,
            }
        )
    )


# ── Bug A regression: a real `pod-<N>` name is recognized end-to-end ──────────


def test_running_managed_pods_recognizes_canonical_pod_name(monkeypatch):
    # The whole point of the fix: a live pod named `pod-489` (canonical) MUST be
    # recognized. The old `epm-issue-<N>`-only regex returned [] here -> dead
    # code. Reuses the canonical pod_lifecycle helpers via the live API list.
    import autonomous_session_watch as asw
    from runpod_api import PodInfo

    monkeypatch.setattr(
        asw,
        "list_team_pods",
        lambda: [
            PodInfo(pod_id="p489", name="pod-489", desired_status="RUNNING"),
            PodInfo(pod_id="p444", name="pod-444", desired_status="RUNNING"),
            PodInfo(pod_id="pold", name="epm-issue-377", desired_status="RUNNING"),  # legacy too
            PodInfo(pod_id="pexit", name="pod-100", desired_status="EXITED"),  # not RUNNING
            PodInfo(pod_id="punm", name="some-random-pod", desired_status="RUNNING"),  # unmanaged
        ],
    )
    got = sorted(asw._running_managed_issue_pods())
    # pod-444, pod-489, and the legacy epm-issue-377 are recognized; the EXITED
    # and unmanaged ones are excluded.
    assert got == [(377, "pold"), (444, "p444"), (489, "p489")]


def test_running_managed_pods_api_error_returns_empty(monkeypatch):
    import autonomous_session_watch as asw

    def boom():
        raise RuntimeError("transport down")

    monkeypatch.setattr(asw, "list_team_pods", boom)
    assert asw._running_managed_issue_pods() == []


# ── Bug B regression: a LIVE interactive session must NOT trigger a stop ──────


def test_live_interactive_session_does_not_cause_stop(isolated_registry, monkeypatch):
    # An interactive `/issue 489` session is spawned with cwd = REPO ROOT (the
    # worktree doesn't exist yet at spawn), so cwd-liveness reports it as dead.
    # Under the OLD design that misread would STOP the pod. Under the new design
    # the STOP trigger is task STATUS — a `running` (pod-active) task with fresh
    # progress is KEPT regardless of any cwd signal.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    # Fresh progress 1h ago -> pod-active-fresh -> keep.
    monkeypatch.setattr(
        asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "2026-06-05T10:00:00Z"}]
    )
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: now - 3600)  # 1h ago, fresh
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)

    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    assert stops == []  # a live interactive session's pod is never stopped here


# ── auto-stop on a DONE task's RUNNING pod ────────────────────────────────────


def test_auto_stop_fires_on_done_task_second_miss(isolated_registry, monkeypatch):
    # A `completed` task with a still-RUNNING pod is an escaped pod. Tick 1
    # increments to missed=1 (no stop), tick 2 hits threshold and stops ONCE,
    # then the state is cleared.
    import json

    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    state_path = isolated_registry / "pod-safety-489.json"
    assert stops == []
    assert json.loads(state_path.read_text())["missed"] == 1

    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == [489]
    assert posts == [(489, "auto-stop")]
    assert not state_path.exists()  # cleared after stop


@pytest.mark.parametrize("status", ["awaiting_promotion", "archived", "completed"])
def test_auto_stop_fires_for_all_done_statuses(isolated_registry, monkeypatch, status):
    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(7, "p7")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.pod_safety_pass(dry_run=False, threshold=1, now=now)  # threshold=1 -> stop immediately
    assert stops == [7]


@pytest.mark.parametrize("status", ["blocked", "followups_running"])
def test_no_auto_stop_for_blocked_or_followups(isolated_registry, monkeypatch, status):
    # blocked (may be under investigation) and followups_running (parent pod may
    # be in use) are KEPT, never auto-stopped.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(7, "p7")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.pod_safety_pass(dry_run=False, threshold=1, now=now)
    assert stops == []


# ── alert (not stop) on a stale pod-active task ───────────────────────────────


def test_alert_fires_on_stale_pod_active_and_does_not_stop(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    # No real progress for well over the stale cap.
    stale_ts = now - (ALERT_STALE_HOURS + 2) * 3600
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "old"}])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: stale_ts)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    assert stops == []  # NEVER stop a mid-run pod
    assert posts == [(489, "alert")]
    # alerted flag is persisted so the next tick stays quiet.
    state = json.loads((isolated_registry / "pod-safety-489.json").read_text())
    assert state["alerted"] is True


def test_alert_dedups_across_ticks(isolated_registry, monkeypatch):
    # Two consecutive stale ticks -> exactly ONE alert (dedup via the alerted
    # flag), no stop.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "verifying")
    stale_ts = now - (ALERT_STALE_HOURS + 2) * 3600
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "old"}])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: stale_ts)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    assert posts == [(489, "alert")]  # exactly one, despite two stale ticks
    assert stops == []


def test_alert_re_fires_after_progress_advances(isolated_registry, monkeypatch):
    # If real progress advances after an alert, the alerted flag clears so a NEW
    # staleness episode can alert again.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "x"}])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: None)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # Tick 1: stale at old_ts -> alert.
    old_ts = now - (ALERT_STALE_HOURS + 2) * 3600
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: old_ts)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    # Tick 2: progress advanced to ~now (fresh) -> keep, alerted cleared.
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: now - 60)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    # Tick 3: stale again at a newer-but-still-stale ts -> alert AGAIN.
    later_stale = now + 24 * 3600  # advance the clock a day...
    monkeypatch.setattr(
        asw, "_latest_progress_ts", lambda events: later_stale - (ALERT_STALE_HOURS + 2) * 3600
    )
    asw.pod_safety_pass(dry_run=False, threshold=2, now=later_stale)

    assert posts == [(489, "alert"), (489, "alert")]


def test_no_alert_on_fresh_pod_active(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "x"}])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: now - 600)  # 10 min ago, fresh
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    assert posts == []
    assert stops == []


# ── fail-closed: API error -> no action ───────────────────────────────────────


def test_pod_safety_pass_api_error_does_not_stop(isolated_registry, monkeypatch):
    # When `_running_managed_issue_pods` returns [] (transport error or
    # genuinely no pods), `pod_safety_pass` MUST NOT call `_stop_pod`. Fail-
    # closed invariant for the destructive action.
    import autonomous_session_watch as asw

    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)

    asw.pod_safety_pass(dry_run=False, threshold=2)

    assert stops == []


# ── orphan-state GC ──────────────────────────────────────────────────────────


def test_gc_orphan_removes_state_for_pod_not_in_running_set(isolated_registry):
    # The bug this guards: a pod that left RUNNING by manual stop/terminate /
    # self-EXIT / TTL never gets its miss-state file cleared by the per-pod
    # loop, so a re-used issue-N pod inherits a stale missed=1.
    import time as t

    import autonomous_session_watch as asw

    _write_state(isolated_registry, 137, "abc123", missed=1, first_seen=t.time())
    _write_state(isolated_registry, 99, "def456", missed=0, first_seen=t.time())

    cleared = asw._gc_orphan_pod_safety_state(running_issues={99}, dry_run=False)

    assert cleared == [137]
    assert not (isolated_registry / "pod-safety-137.json").exists()
    assert (isolated_registry / "pod-safety-99.json").exists()  # still in running set


def test_gc_orphan_age_backstop_drops_stale_file(isolated_registry):
    # Secondary backstop: a state file older than POD_SAFETY_STATE_MAX_AGE_S is
    # dropped on the not-in-running path even if the API is flaky.
    import autonomous_session_watch as asw

    very_old = 0.0  # 1970 — definitely past the 7-day cap
    _write_state(isolated_registry, 200, "old-pod", missed=1, first_seen=very_old)

    cleared = asw._gc_orphan_pod_safety_state(running_issues=set(), dry_run=False)

    assert cleared == [200]
    assert not (isolated_registry / "pod-safety-200.json").exists()


def test_gc_orphan_dry_run_does_not_delete(isolated_registry):
    import time as t

    import autonomous_session_watch as asw

    _write_state(isolated_registry, 50, "x", missed=2, first_seen=t.time())
    cleared = asw._gc_orphan_pod_safety_state(running_issues=set(), dry_run=True)
    assert cleared == [50]
    assert (isolated_registry / "pod-safety-50.json").exists()  # NOT deleted


def test_gc_orphan_ignores_garbled_filenames(isolated_registry):
    # A hand-debug file like `pod-safety-foo.json` (non-int issue) is left
    # alone — not the GC's business.
    (isolated_registry / "pod-safety-foo.json").write_text('{"junk": true}')

    import autonomous_session_watch as asw

    cleared = asw._gc_orphan_pod_safety_state(running_issues=set(), dry_run=False)

    assert cleared == []
    assert (isolated_registry / "pod-safety-foo.json").exists()


def test_pod_safety_pass_gc_runs_even_with_no_running_pods(isolated_registry, monkeypatch):
    # GC must fire BEFORE the `if not running: return` early-out; otherwise a
    # tick where every managed pod has vanished would never clean up its state.
    import autonomous_session_watch as asw

    _write_state(isolated_registry, 99, "gone", missed=1, first_seen=__import__("time").time())
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [])

    asw.pod_safety_pass(dry_run=False, threshold=2)

    assert not (isolated_registry / "pod-safety-99.json").exists()


# ── daemon-reachability gates ONLY the respawn pass ──────────────────────────


def test_main_daemon_unreachable_still_runs_pod_safety(isolated_registry, monkeypatch):
    # The pod-safety pass reasons about task status + the live pod list, neither
    # of which needs the Happy daemon. So a daemon outage must NOT skip it
    # (unlike the old design, which gated BOTH passes on the daemon).
    import autonomous_session_watch as asw

    pod_safety_calls: list[tuple] = []
    respawn_entry_calls: list[tuple] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))
    monkeypatch.setattr(asw, "_process_entry", lambda *a, **kw: respawn_entry_calls.append((a, kw)))

    rc = asw.main([])

    assert rc == 0
    assert len(pod_safety_calls) == 1  # pod-safety RAN despite the outage
    assert respawn_entry_calls == []  # respawn pass skipped (no entries processed)


def test_main_daemon_reachable_runs_both_passes(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    pod_safety_calls: list[tuple] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_load_session_meta", lambda: {})
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))

    rc = asw.main([])

    assert rc == 0
    assert len(pod_safety_calls) == 1


# ── state-store round-trip ────────────────────────────────────────────────────


def test_save_pod_safety_state_carries_first_seen_forward(isolated_registry):
    import json

    import autonomous_session_watch as asw

    asw._save_pod_safety_state(
        7, "pod-7", missed=1, alerted=False, last_progress_ts=42.0, prev={"first_seen": 1234.0}
    )
    payload = json.loads((isolated_registry / "pod-safety-7.json").read_text())
    assert payload == {
        "pod_id": "pod-7",
        "missed": 1,
        "alerted": False,
        "last_progress_ts": 42.0,
        "first_seen": 1234.0,
    }

    # On a second save (passing the previous payload), first_seen must persist.
    asw._save_pod_safety_state(
        7, "pod-7", missed=2, alerted=True, last_progress_ts=99.0, prev=payload
    )
    payload2 = json.loads((isolated_registry / "pod-safety-7.json").read_text())
    assert payload2["first_seen"] == 1234.0
    assert payload2["missed"] == 2
    assert payload2["alerted"] is True
    assert payload2["last_progress_ts"] == 99.0
