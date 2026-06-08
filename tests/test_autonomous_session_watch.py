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
    STALLED_MAX_RESPAWNS,
    STALLED_WINDOW_S,
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
    # dependent) and must EXACTLY equal the authoritative runtime enum
    # `task_workflow.STATUSES` — no missing status (a fall-through would
    # silently classify as unknown→keep) and no phantom member (a name the
    # runtime can never produce, like the prior `clarifying` in PARK that
    # the reviewer caught). Mirrors the pod-safety pass's
    # `test_status_classes_subset_of_authoritative_enum`.
    from explore_persona_space.task_workflow import STATUSES

    enum = set(STATUSES)
    assert ACTIVE.isdisjoint(PARK)
    assert ACTIVE.isdisjoint(TERMINAL)
    assert PARK.isdisjoint(TERMINAL)
    assert enum == ACTIVE | PARK | TERMINAL, (
        f"session-pass classification disagrees with runtime STATUSES: "
        f"missing={enum - (ACTIVE | PARK | TERMINAL)}, "
        f"phantom={(ACTIVE | PARK | TERMINAL) - enum}"
    )


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
    # blocked / interpreting / reviewing / unknown statuses are classified
    # "other": never stopped, never alerted, miss counter reset.
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


def test_status_classes_subset_of_authoritative_enum():
    # Every status named by AUTO_STOP_DONE / POD_ACTIVE MUST exist in the
    # authoritative runtime enum task_workflow.STATUSES — otherwise the member
    # is a phantom that can never match what `_task_status` returns (the prior
    # round shipped `cancelled` / `uploading` / `followups_running` as phantoms,
    # silently making the auto-stop / no-auto-stop guarantees vacuous). This
    # pin catches that whole class of bug.
    from explore_persona_space.task_workflow import STATUSES

    enum = set(STATUSES)
    assert enum >= AUTO_STOP_DONE, f"phantom AUTO_STOP_DONE members: {AUTO_STOP_DONE - enum}"
    assert enum >= POD_ACTIVE, f"phantom POD_ACTIVE members: {POD_ACTIVE - enum}"


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


@pytest.mark.parametrize("status", ["blocked", "interpreting", "reviewing"])
def test_no_auto_stop_for_other_class_statuses(isolated_registry, monkeypatch, status):
    # blocked (may be under investigation), interpreting / reviewing (those
    # stages don't drive pods — interp/review reads from WandB/HF, so a
    # RUNNING pod observed there classifies as "other" and is kept until the
    # task reaches awaiting_promotion). All real runtime statuses — never
    # auto-stopped.
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


def test_alert_re_fires_after_none_then_first_progress_then_stale(isolated_registry, monkeypatch):
    # The None->first-progress->stale-again path. A pod alerted while it had
    # ZERO real progress markers (latest_progress_ts=None), then posts its
    # first real epm:progress (the prev_progress baseline transitions
    # None->float, so the `progressed` check is False — it requires BOTH sides
    # non-None), then goes stale again. Under the must-fix #2 patch, the
    # alerted flag clears because the task is currently pod-active-fresh, so
    # the new staleness episode re-alerts. Without the (b) clause this test
    # would never see a second alert.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "x"}])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # Tick 1: pod-active with NO real progress yet -> classified pod-active-
    # stale (None path), alert fires, prev_progress=None persisted.
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: None)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    # Tick 2: the FIRST real progress marker just landed (fresh, 1 min ago).
    # status_class flips to pod-active-fresh; under the (b) clause, alerted
    # clears. prev_progress baseline saved at the fresh ts.
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: now - 60)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)

    # Tick 3: time advances a day; the (still-only) progress marker is now
    # stale again. Without must-fix #2 the second alert would never fire here.
    later = now + 24 * 3600
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: now - 60)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=later)

    assert stops == []
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


# ─── stalled-detector decision matrix ────────────────────────────────────────
# decide_session_stalled is the pure decision for the Phase-2 auto-respawn
# path. Pin the (respawn / exhausted / alert / keep) action selection
# exhaustively — a wrong respawn duplicates a session, a wrong alert misses
# a real bug, and a wrong cap exhaustion silently strands a run.


def test_session_stalled_missing_self_report_is_keep():
    # No self-report file at all (interactive session, or autonomous that
    # hasn't ticked yet) -> never alert / respawn.
    from autonomous_session_watch import decide_session_stalled

    action, missed = decide_session_stalled(
        self_report_age_s=None,
        marker_progress_age_s=None,
        has_pod=False,
        missed=5,
        alerted=False,
        respawn_eligible=True,
        respawn_count=0,
    )
    assert action == "keep"
    assert missed == 0


def test_session_stalled_fresh_self_report_resets_miss_counter():
    from autonomous_session_watch import decide_session_stalled

    fresh = 60.0  # 1 min ago, well under the window
    action, missed = decide_session_stalled(
        self_report_age_s=fresh,
        marker_progress_age_s=None,
        has_pod=True,
        missed=3,
        alerted=True,
        respawn_eligible=True,
    )
    assert action == "keep"
    assert missed == 0


def test_session_stalled_requires_both_signals_stale():
    # Self-report stale but marker-progress FRESH -> keep (bg chain still posting).
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    fresh_marker = 60.0
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=fresh_marker,
        has_pod=True,
        missed=0,
        alerted=False,
        respawn_eligible=True,
    )
    assert action == "keep"


def test_session_stalled_needs_two_misses_before_acting():
    # First stale check only increments (1); second consecutive stale check
    # triggers the recovery action. Guards a transient self-report-write race.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    a1, m1 = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=0,
        alerted=False,
        respawn_eligible=True,
        respawn_count=0,
        threshold=2,
    )
    assert (a1, m1) == ("keep", 1)
    a2, m2 = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        respawn_count=0,
        threshold=2,
    )
    assert (a2, m2) == ("respawn", 0)


def test_session_stalled_respawn_eligible_returns_respawn():
    # respawn_eligible=True + count below cap -> respawn.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        respawn_count=0,
        threshold=2,
    )
    assert action == "respawn"


def test_session_stalled_respawn_just_below_cap_still_respawns():
    # Boundary case (reviewer Minor #5): the LAST allowed respawn must still
    # fire. `respawn_count == max - 1` means we've issued `max - 1` respawns
    # and are about to issue the `max`-th — that's `<` `max`, so the
    # comparison must allow it. An off-by-one here (`>` vs `>=`) would
    # silently cut the budget by 1.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        respawn_count=STALLED_MAX_RESPAWNS - 1,
        threshold=2,
    )
    assert action == "respawn"


def test_session_stalled_respawn_at_cap_returns_exhausted():
    # respawn_eligible=True but respawn_count == max -> exhausted (don't loop).
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        respawn_count=STALLED_MAX_RESPAWNS,
        threshold=2,
    )
    assert action == "exhausted"


def test_session_stalled_respawn_above_cap_returns_exhausted():
    # Defensive: if respawn_count drifts > max (e.g. cap lowered between
    # ticks), still classify as exhausted rather than respawning.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        respawn_count=STALLED_MAX_RESPAWNS + 5,
        threshold=2,
    )
    assert action == "exhausted"


def test_session_stalled_not_eligible_returns_alert():
    # respawn_eligible=False (non-ACTIVE status OR daemon unreachable) ->
    # alert-only, regardless of how many respawns have happened.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=False,
        respawn_count=0,
        threshold=2,
    )
    assert action == "alert"


def test_session_stalled_already_alerted_is_keep():
    # Dedup: once the alert flag has been set this episode, stay quiet
    # until self-report-ts advancement clears it (caller's responsibility).
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=True,
        respawn_eligible=True,
        respawn_count=0,
    )
    assert action == "keep"


def test_session_stalled_marker_absent_treated_as_stale():
    # No real progress markers at all is itself a stale signal — a pod-
    # active autonomous session that's never posted progress is suspicious.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=None,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        respawn_count=0,
        threshold=2,
    )
    assert action == "respawn"


# ─── stalled-detector I/O wrapper tests ──────────────────────────────────────
# These exercise _process_stalled_session: the ACTIVE-only gating, daemon-
# down fallback, crash-loop cap, and the stop-then-spawn ordering.


def _write_autonomous_entry(reg_dir, issue, session_id, cap=12.0):
    """Helper: write an autonomous-registry entry matching spawn_session's
    layout so `_process_stalled_session` can load it."""
    import json
    import time as _t

    (reg_dir / f"issue-{issue}.json").write_text(
        json.dumps(
            {
                "issue": issue,
                "happy_session_id": session_id,
                "cwd": "/repo",
                "auto_approve_gpu_hours": cap,
                "spawned_at": _t.time(),
                "missed": 0,
            }
        )
    )


def _patch_stale_signals(monkeypatch, asw, *, status: str, age_s: float | None = None):
    """Helper: monkeypatch the I/O helpers so a session reads as stale.

    Returns the value `age_s` used (the caller can assert it). Patches:
    - `_task_status` -> the given status (ACTIVE / PARK / TERMINAL).
    - `_self_report_age_seconds` -> (`age_s`, "ts-iso") so the self-report
      is parsed as that many seconds old (default = past the staleness window).
    - `_task_events` / `_latest_progress_ts` -> a single stale event past the window.
    - `_running_managed_issue_pods` -> no managed pods.
    """
    if age_s is None:
        age_s = STALLED_WINDOW_S + 60
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (age_s, "ts-old"))
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "old"}])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: 0.0)
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [])
    return age_s


@pytest.fixture
def stalled_recorder(monkeypatch):
    """Capture every recovery side-effect (stop / spawn / marker) without
    actually executing them, and inject them into autonomous_session_watch."""
    import autonomous_session_watch as asw

    stops: list[str] = []
    spawns: list[tuple[int, float]] = []
    markers: list[tuple[int, str]] = []

    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or True)
    monkeypatch.setattr(
        asw,
        "_respawn_stalled_session",
        lambda issue, cap, dry_run: spawns.append((issue, cap)) or True,
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label)),
    )
    return stops, spawns, markers


def test_stalled_active_status_auto_respawns_after_two_misses(
    isolated_registry, monkeypatch, stalled_recorder
):
    # The fix this round is for: an ACTIVE-status stalled session auto-
    # respawns (stop-then-spawn) instead of alerting only.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 518, "sess-518", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1: increments to missed=1, no action.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []

    # Tick 2: threshold met, ACTIVE + daemon_reachable -> respawn.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-518"]
    assert spawns == [(518, 24.0)]
    assert markers == [(518, "session-auto-respawn")]


def test_stalled_park_status_falls_back_to_alert(isolated_registry, monkeypatch, stalled_recorder):
    # A `plan_pending` / `blocked` / `awaiting_promotion` etc. is a gate
    # the session is legitimately parked at — never auto-respawn there.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 600, "sess-600")
    _patch_stale_signals(monkeypatch, asw, status="plan_pending")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    # Threshold met but PARK status -> alert, not respawn.
    assert stops == [] and spawns == []
    assert markers == [(600, "session-stalled-alert")]


def test_stalled_terminal_status_falls_back_to_alert(
    isolated_registry, monkeypatch, stalled_recorder
):
    # A `completed` / `archived` / `awaiting_promotion` task is terminal —
    # never auto-respawn. The GC pass reaps the registry entry shortly after;
    # this protects the tick between status flip and GC.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 700, "sess-700")
    _patch_stale_signals(monkeypatch, asw, status="awaiting_promotion")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    assert stops == [] and spawns == []
    assert markers == [(700, "session-stalled-alert")]


def test_stalled_daemon_down_falls_back_to_alert(isolated_registry, monkeypatch, stalled_recorder):
    # Daemon outage: detection still runs, but stop+spawn would fail
    # mid-flight (the local daemon RPC isn't answering), so degrade to
    # alert-only this tick. Mirrors the crash-recovery pass.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 800, "sess-800")
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)

    assert stops == [] and spawns == []
    assert markers == [(800, "session-stalled-alert")]


def test_stalled_crash_loop_cap_exhausts_after_max_respawns(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Bound: respawn at most STALLED_MAX_RESPAWNS times per episode. Once
    # exhausted, post the loud one-time marker and stop respawning until
    # real progress advances and clears the cap.
    import json

    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 900, "sess-900")
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Drive the episode forward: each "respawn" needs two stale ticks
    # (1st increments to missed=1, 2nd fires the action). After each
    # respawn the state is persisted with respawn_count++. The cap is
    # hit when respawn_count reaches STALLED_MAX_RESPAWNS, then the
    # next two-tick cycle posts the exhausted marker.
    for _ in range(STALLED_MAX_RESPAWNS):
        # tick A: missed -> 1
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
        # tick B: respawn fires; bumps respawn_count, resets missed
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    # Sanity: every respawn was issued.
    assert len(spawns) == STALLED_MAX_RESPAWNS
    assert len(stops) == STALLED_MAX_RESPAWNS

    # On-disk respawn_count is at the cap; the alerted flag was reset
    # after each respawn so the next episode could fire.
    state = json.loads((isolated_registry / "stalled-900.json").read_text())
    assert state["respawn_count"] == STALLED_MAX_RESPAWNS

    # Two more stale ticks -> exhausted marker, NOT another respawn.
    pre_spawn_count = len(spawns)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert len(spawns) == pre_spawn_count  # no further respawn
    assert (900, "session-auto-respawn-exhausted") in markers

    # On-disk exhausted flag is set so the next tick stays quiet.
    state2 = json.loads((isolated_registry / "stalled-900.json").read_text())
    assert state2["exhausted"] is True


def test_stalled_real_progress_resets_respawn_cap(isolated_registry, monkeypatch, stalled_recorder):
    # The cap is per-EPISODE: if the session resumes self-reporting (the
    # self_report_ts advances), the count must reset so a future episode
    # can re-respawn from scratch. Without this, a session that hit the
    # cap once would never auto-recover again.
    import autonomous_session_watch as asw

    _stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 950, "sess-950")
    now = 1_000_000.0

    # Episode 1: drive one full respawn.
    _patch_stale_signals(monkeypatch, asw, status="running")
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert len(spawns) == 1

    # Self-report advances (new ts AND fresh age) -> alerted + respawn_count clear.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (1.0, "ts-NEW"))
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    # No new respawn this tick (signals are fresh, just persists the reset).

    # Episode 2: another stale stretch with a still-newer ts -> can respawn again
    # from scratch, NOT exhausted.
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (STALLED_WINDOW_S + 60, "ts-NEWER"),
    )
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    # Two episodes -> two respawns; cap was NOT reached.
    assert len(spawns) == 2
    assert (950, "session-auto-respawn-exhausted") not in markers


def test_stalled_stop_failure_skips_spawn(isolated_registry, monkeypatch, stalled_recorder):
    # If `_stop_session` returns False (stop RPC failed), we MUST NOT spawn
    # a fresh session — that would leave two `--auto` sessions racing on
    # the same issue. respawn_count must NOT be bumped (we never actually
    # respawned), so the cap is unaffected.
    import json

    import autonomous_session_watch as asw

    _stops, spawns, markers = stalled_recorder
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: False)
    _write_autonomous_entry(isolated_registry, 960, "sess-960")
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    assert spawns == []  # never spawned
    # No respawn-success marker; no exhausted marker either.
    assert all(label != "session-auto-respawn" for _i, label in markers)

    state = json.loads((isolated_registry / "stalled-960.json").read_text())
    assert state["respawn_count"] == 0
    assert state["exhausted"] is False


def test_stalled_missing_session_id_declines_respawn(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Safety regression (reviewer Major #1): if the registry entry has no
    # usable `happy_session_id` (None, missing, or non-str), the stop
    # precondition cannot be verified, so we MUST NOT spawn — otherwise
    # two `--auto` sessions would race on the same issue and double the
    # pod cost. Stop is not called either (nothing to stop); the tick
    # persists state and waits for the next entry-read to pick up a
    # rewritten id.
    import json

    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    # Write an autonomous-registry entry with happy_session_id=None.
    import time as _t

    (isolated_registry / "issue-970.json").write_text(
        json.dumps(
            {
                "issue": 970,
                "happy_session_id": None,
                "cwd": "/repo",
                "auto_approve_gpu_hours": 12.0,
                "spawned_at": _t.time(),
                "missed": 0,
            }
        )
    )
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Two stale ticks: threshold met, status ACTIVE, daemon reachable, so
    # the decision says "respawn" — but the actor must DECLINE because sid
    # is unusable.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    assert spawns == []  # NEVER spawned without a verified stop
    assert stops == []  # nothing to stop in the first place
    # No respawn-success marker fired.
    assert all(label != "session-auto-respawn" for _i, label in markers)

    state = json.loads((isolated_registry / "stalled-970.json").read_text())
    assert state["respawn_count"] == 0  # cap unaffected
    assert state["exhausted"] is False


def test_stalled_main_passes_daemon_flag(isolated_registry, monkeypatch):
    # The stalled-detector must reuse the same daemon_reachable result that
    # the crash-recovery pass probed, so a daemon flap mid-tick can't make
    # them disagree. Verify main() threads it through.
    import autonomous_session_watch as asw

    captured_kwargs: dict = {}
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: None)

    def _record_stalled(*a, **kw):
        captured_kwargs.update(kw)

    monkeypatch.setattr(asw, "stalled_session_pass", _record_stalled)

    rc = asw.main([])

    assert rc == 0
    assert captured_kwargs.get("daemon_reachable") is False


def test_save_stalled_state_carries_first_seen_and_respawn_fields(isolated_registry):
    # State-store round-trip for the new fields: respawn_count + exhausted
    # are persisted and first_seen carries forward across saves (mirrors
    # the pod-safety-state contract).
    import json

    import autonomous_session_watch as asw

    asw._save_stalled_state(
        7,
        "sess-7",
        missed=1,
        alerted=False,
        last_self_report_ts="ts-1",
        respawn_count=2,
        exhausted=False,
        prev={"first_seen": 1234.0},
    )
    payload = json.loads((isolated_registry / "stalled-7.json").read_text())
    assert payload == {
        "happy_session_id": "sess-7",
        "missed": 1,
        "alerted": False,
        "respawn_count": 2,
        "exhausted": False,
        "last_self_report_ts": "ts-1",
        "first_seen": 1234.0,
    }

    asw._save_stalled_state(
        7,
        "sess-7",
        missed=0,
        alerted=True,
        last_self_report_ts="ts-2",
        respawn_count=3,
        exhausted=True,
        prev=payload,
    )
    payload2 = json.loads((isolated_registry / "stalled-7.json").read_text())
    assert payload2["first_seen"] == 1234.0  # carried forward
    assert payload2["respawn_count"] == 3
    assert payload2["exhausted"] is True
    assert payload2["alerted"] is True
