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
    ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
    ORPHAN_SPAWN_GRACE_S,
    ORPHAN_STALENESS_S_DEFAULT,
    PARK,
    POD_ACTIVE,
    STALLED_MAX_RESPAWNS,
    STALLED_WINDOW_S,
    TERMINAL,
    decide,
    decide_orphan,
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


@pytest.mark.parametrize("missed", [0, 1, 5])
def test_pod_safety_keep_running_tag_skips_stop(missed):
    # A DONE task with the keep-running tag is NEVER stopped, even past the
    # miss threshold, and the miss counter resets to 0 — so removing the tag
    # later re-arms a fresh >=threshold-checks accumulation (#530 regression).
    assert decide_pod_safety(
        status_class="auto-stop-done",
        missed=missed,
        stale=False,
        alerted=False,
        threshold=2,
        keep_running=True,
    ) == ("keep-running-skip", 0)


def test_pod_safety_keep_running_does_not_suppress_alert():
    # The tag only exempts the auto-stop arm. A stale pod-active task still
    # alerts (alerts never stop anything, so there is nothing to exempt).
    assert decide_pod_safety(
        status_class="pod-active-stale",
        missed=0,
        stale=True,
        alerted=False,
        keep_running=True,
    ) == ("alert", 0)


@pytest.mark.parametrize("missed", [0, 1, 5])
def test_pod_safety_followup_active_skips_stop(missed):
    # The #477 regression: a promoted task with a fresh `epm:run-launched`
    # (newer than the latest done-transition) is a live inline follow-up.
    # The auto-stop is SKIPPED with the miss counter reset, so when the
    # follow-up finishes (predicate flips False) the auto-stop re-arms with a
    # fresh >=threshold-checks accumulation. Same semantics as keep-running.
    assert decide_pod_safety(
        status_class="auto-stop-done",
        missed=missed,
        stale=False,
        alerted=False,
        threshold=2,
        followup_active=True,
    ) == ("followup-skip", 0)


def test_pod_safety_keep_running_beats_followup_active():
    # Precedence: an explicit user-set keep-running tag wins over the
    # inferred-from-events follow-up predicate. The user signal is stronger
    # and predictable from the dashboard; the inferred one is best-effort.
    assert decide_pod_safety(
        status_class="auto-stop-done",
        missed=5,
        stale=False,
        alerted=False,
        threshold=2,
        keep_running=True,
        followup_active=True,
    ) == ("keep-running-skip", 0)


def test_task_followup_active_predicate():
    # _task_followup_active compares the latest `epm:run-launched` ts vs the
    # latest of `epm:promoted` / `epm:status-changed`. Truthy iff there is a
    # run-launched newer than every done-transition.
    import autonomous_session_watch as asw

    # No run-launched at all -> False.
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:status-changed", "ts": "2026-06-10T00:00:00Z", "note": ""},
                {"kind": "epm:promoted", "ts": "2026-06-10T00:00:01Z", "note": ""},
            ],
        )
        is False
    )
    # No done-transition (defensive case — caller has already verified DONE
    # status, so this is unreachable in practice) -> False conservatively.
    assert (
        asw._task_followup_active(
            0,
            events=[{"kind": "epm:run-launched", "ts": "2026-06-10T03:00:00Z", "note": ""}],
        )
        is False
    )
    # run-launched OLDER than done-transition -> False (the run-launched
    # belongs to the experiment that produced the now-completed task).
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:run-launched", "ts": "2026-06-09T20:00:00Z", "note": ""},
                {"kind": "epm:promoted", "ts": "2026-06-10T00:00:00Z", "note": ""},
            ],
        )
        is False
    )
    # run-launched NEWER than done-transition -> True (a legitimate inline
    # follow-up).
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:promoted", "ts": "2026-06-10T00:00:00Z", "note": ""},
                {"kind": "epm:run-launched", "ts": "2026-06-10T03:00:00Z", "note": ""},
            ],
        )
        is True
    )
    # Compares against the LATEST done-transition (not the earliest).
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:promoted", "ts": "2026-06-10T00:00:00Z", "note": ""},
                {"kind": "epm:run-launched", "ts": "2026-06-10T03:00:00Z", "note": ""},
                # A SECOND done-transition (e.g. follow-up finished) after the
                # run-launched -> predicate flips False (follow-up is done).
                {"kind": "epm:status-changed", "ts": "2026-06-10T05:00:00Z", "note": ""},
            ],
        )
        is False
    )


def test_pod_safety_followup_active_only_on_auto_stop_arm():
    # The followup_active predicate is consulted ONLY when status_class is
    # auto-stop-done. A pod-active-stale task still alerts (alerts never stop
    # anything; nothing to exempt). A pod-active-fresh task keeps as usual.
    assert decide_pod_safety(
        status_class="pod-active-stale",
        missed=0,
        stale=True,
        alerted=False,
        followup_active=True,
    ) == ("alert", 0)
    assert decide_pod_safety(
        status_class="pod-active-fresh",
        missed=0,
        stale=False,
        alerted=False,
        followup_active=True,
    ) == ("keep", 0)
    assert decide_pod_safety(
        status_class="other",
        missed=0,
        stale=False,
        alerted=False,
        followup_active=True,
    ) == ("keep", 0)


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
    # silently making the auto-stop / no-auto-stop guarantees vacuous;
    # `followups_running` was later un-phantomed on 2026-06-10 — it joined the
    # runtime enum and POD_ACTIVE for the same-issue follow-up loop). This
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
    # and unmanaged ones are excluded. The third element is the pod NAME,
    # threaded out so callers (e.g. the #488 stale-port self-heal in
    # ``_handle_stalled_alert``) can address the pod by name without a
    # second ``list_team_pods`` round-trip.
    assert got == [
        (377, "pold", "epm-issue-377"),
        (444, "p444", "pod-444"),
        (489, "p489", "pod-489"),
    ]


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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(7, "p7", "pod-7")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.pod_safety_pass(dry_run=False, threshold=1, now=now)  # threshold=1 -> stop immediately
    assert stops == [7]


def test_keep_running_tag_skips_stop_and_notes_once(isolated_registry, monkeypatch):
    # The #530 regression: a keep-running-tagged task at awaiting_promotion
    # (a user-directed follow-up still using the pod) must NOT be auto-stopped.
    # The skip posts ONE marker per pod incarnation, not one per 20-min tick.
    import json

    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(530, "p530", "pod-530")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "awaiting_promotion")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # threshold=1 would stop an untagged pod on the FIRST tick; three ticks
    # with the tag -> zero stops, exactly one keep-running-skip marker.
    for _ in range(3):
        asw.pod_safety_pass(dry_run=False, threshold=1, now=now)

    assert stops == []
    assert posts == [(530, "keep-running-skip")]
    state = json.loads((isolated_registry / "pod-safety-530.json").read_text())
    assert state["keep_running_noted"] is True
    assert state["missed"] == 0


def test_keep_running_tag_removal_re_arms_auto_stop(isolated_registry, monkeypatch):
    # Removing the tag re-arms the normal >=2-checks accumulation: the next
    # two no-tag ticks stop the pod (fresh count — the tagged ticks did not
    # accumulate misses).
    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(530, "p530", "pod-530")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "awaiting_promotion")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    # Two tagged ticks: no stop, no miss accumulation.
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == []

    # Tag removed: tick 1 only increments (missed 0->1), tick 2 stops.
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == []
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == [530]


def test_inline_followup_run_launched_skips_stop(isolated_registry, monkeypatch):
    # The #477 regression end-to-end: a completed/promoted task whose
    # events.jsonl shows an `epm:run-launched` NEWER than the latest
    # done-transition (`epm:status-changed` to a DONE status, or
    # `epm:promoted`) is a live user-approved inline follow-up. The
    # auto-stop is SKIPPED with exactly ONE follow-up exemption marker per
    # incarnation; the keep-running tag is NOT required.
    import json

    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    # Events: status-changed-to-completed at t=0, then a follow-up
    # run-launched 1h later. The follow-up predicate compares the latest
    # run-launched ts vs the latest done-transition ts.
    events = [
        {"kind": "epm:status-changed", "ts": "2026-06-10T00:00:00Z", "note": "-> completed"},
        {"kind": "epm:promoted", "ts": "2026-06-10T00:00:01Z", "note": "promoted as useful"},
        {"kind": "epm:run-launched", "ts": "2026-06-10T03:12:08Z", "note": "pod=pod-477"},
    ]
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(477, "p477", "pod-477")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: events)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # threshold=1 would stop an unprotected pod on the FIRST tick; three
    # ticks with an active follow-up -> zero stops, exactly one followup-skip
    # marker (dedup via `followup_noted`).
    for _ in range(3):
        asw.pod_safety_pass(dry_run=False, threshold=1, now=now)

    assert stops == []
    assert posts == [(477, "followup-skip")]
    state = json.loads((isolated_registry / "pod-safety-477.json").read_text())
    assert state["followup_noted"] is True
    assert state["missed"] == 0


def test_inline_followup_after_completion_re_arms_auto_stop(isolated_registry, monkeypatch):
    # When the follow-up finishes (the next `epm:status-changed` /
    # `epm:promoted` lands AFTER the latest `epm:run-launched`), the
    # follow-up predicate flips False and the auto-stop re-arms with a fresh
    # >=2-checks accumulation — mirrors the keep-running tag-removal path.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    # Phase 1: follow-up launched after promotion.
    active_events = [
        {"kind": "epm:promoted", "ts": "2026-06-10T00:00:00Z", "note": "promoted as useful"},
        {"kind": "epm:run-launched", "ts": "2026-06-10T03:00:00Z", "note": "pod=pod-477"},
    ]
    # Phase 2: follow-up done — next done-transition lands AFTER the
    # run-launched, so the predicate flips False.
    finished_events = [
        *active_events,
        {"kind": "epm:status-changed", "ts": "2026-06-10T05:00:00Z", "note": "followup done"},
    ]
    state = {"events": active_events}
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(477, "p477", "pod-477")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: state["events"])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    # Two ticks while the follow-up is live: no stop, no miss accumulation.
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == []

    # Follow-up finished: tick 1 only increments (missed 0->1), tick 2 stops.
    state["events"] = finished_events
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == []
    asw.pod_safety_pass(dry_run=False, threshold=2, now=now)
    assert stops == [477]


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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(7, "p7", "pod-7")])
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(489, "p489", "pod-489")])
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
    vm_disk_calls: list[tuple] = []
    orphan_calls: list[tuple] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))
    monkeypatch.setattr(asw, "_process_entry", lambda *a, **kw: respawn_entry_calls.append((a, kw)))
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: vm_disk_calls.append((a, kw)))
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: orphan_calls.append((a, kw)))

    rc = asw.main([])

    assert rc == 0
    assert len(pod_safety_calls) == 1  # pod-safety RAN despite the outage
    assert respawn_entry_calls == []  # respawn pass skipped (no entries processed)
    assert len(vm_disk_calls) == 1  # vm-disk pass runs unconditionally (daemon-free)
    # The orphan sweep is invoked unconditionally but self-gates on the
    # daemon flag (it would mass-respawn on an outage otherwise).
    assert len(orphan_calls) == 1
    assert orphan_calls[0][1]["daemon_reachable"] is False


def test_main_daemon_reachable_runs_both_passes(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    pod_safety_calls: list[tuple] = []
    orphan_calls: list[tuple] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: orphan_calls.append((a, kw)))

    rc = asw.main([])

    assert rc == 0
    assert len(pod_safety_calls) == 1
    assert len(orphan_calls) == 1
    assert orphan_calls[0][1]["daemon_reachable"] is True
    assert orphan_calls[0][1]["live_ids"] == set()


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
        "keep_running_noted": False,
        "followup_noted": False,
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

    # keep_running_noted carries forward from prev when not explicitly passed,
    # and an explicit value overrides.
    asw._save_pod_safety_state(
        7,
        "pod-7",
        missed=0,
        alerted=False,
        last_progress_ts=99.0,
        keep_running_noted=True,
        prev=payload2,
    )
    payload3 = json.loads((isolated_registry / "pod-safety-7.json").read_text())
    assert payload3["keep_running_noted"] is True
    asw._save_pod_safety_state(
        7, "pod-7", missed=0, alerted=False, last_progress_ts=99.0, prev=payload3
    )
    payload4 = json.loads((isolated_registry / "pod-safety-7.json").read_text())
    assert payload4["keep_running_noted"] is True  # carried forward


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


def test_session_stalled_already_alerted_escalates_to_respawn_when_eligible():
    # Regression for incident #506 (2026-06-08): a Phase-1 alert set
    # alerted=True ~11h before respawn became eligible, and the prior
    # `if alerted: return keep` short-circuit then suppressed the
    # respawn on every subsequent tick for 10+ hours while the 8xH200
    # pod idle-burned ~$460. The `alerted` flag must dedup REPEAT
    # ALERTS only — it must not gate off the stronger respawn action
    # once respawn becomes eligible. Previously this test asserted
    # `action == "keep"` (encoded the bug); now it asserts the correct
    # escalation. The dedup-of-repeat-alerts case (alerted + NOT
    # eligible) is pinned by
    # `test_session_stalled_already_alerted_eligibility_false_stays_keep`
    # below.
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
    assert action == "respawn"


def test_session_stalled_already_alerted_eligibility_false_stays_keep():
    # Dedup-of-repeat-alerts: alerted + respawn NOT eligible (non-ACTIVE
    # status, or daemon unreachable this tick) -> stay quiet. The prior
    # alert already deduped; a respawn would crash on the missing
    # prerequisite. This was the original intent of
    # `test_session_stalled_already_alerted_is_keep` before the
    # incident-#506 regression test re-purposed that name.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=True,
        respawn_eligible=False,
        respawn_count=0,
    )
    assert action == "keep"


def test_session_stalled_already_alerted_at_cap_stays_keep():
    # Exhausted-cap respected from the alerted branch: if respawn_count
    # is already at the cap, the new escalation path must NOT resurrect
    # a respawn. Stay quiet — the caller's `exhausted` flag dedups the
    # loud one-time exhausted marker separately.
    from autonomous_session_watch import decide_session_stalled

    stale = STALLED_WINDOW_S + 60
    action, _ = decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=True,
        respawn_eligible=True,
        respawn_count=STALLED_MAX_RESPAWNS,
        threshold=2,
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
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: None)

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
    # ``refresh_attempted`` (default False) is the #488 stale-port self-heal
    # dedup flag added 2026-06-09; see ``_handle_stalled_alert`` +
    # ``_refresh_pods_conf_from_api``. Schema-shape coverage stays exhaustive.
    assert payload == {
        "happy_session_id": "sess-7",
        "missed": 1,
        "alerted": False,
        "respawn_count": 2,
        "exhausted": False,
        "refresh_attempted": False,
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


# ── #488 stale-port self-heal in the stalled-detector ALERT branch ───────────


def test_stalled_alert_fires_refresh_from_api_when_has_pod(
    isolated_registry, monkeypatch, stalled_recorder
):
    """When the stalled-detector hits the ALERT branch (respawn ineligible —
    either non-ACTIVE status OR daemon unreachable) AND the issue has a
    RUNNING managed pod whose name we know, ``_handle_stalled_alert`` MUST
    fire ``pod.py config --refresh-from-api <pod_name>`` once — the #488
    stale-port self-heal that closes the gap between "polling chain dies on
    a stale port" and "manual refresh-from-api command exists." The alert
    marker also still fires."""
    import autonomous_session_watch as asw

    _stops, _spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 488, "sess-488")
    # Use a PARK status so we land on the ALERT branch (respawn ineligible).
    # The pod is still RUNNING despite the park — that's exactly the #488
    # shape: the user-park happened while a pod was alive.
    _patch_stale_signals(monkeypatch, asw, status="plan_pending")
    # Override the pods stub to have a RUNNING managed pod for issue 488.
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(488, "p488", "pod-488")])
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        asw,
        "_refresh_pods_conf_from_api",
        lambda pod_name, dry_run: refresh_calls.append(pod_name) or True,
    )
    now = 1_000_000.0

    # Tick 1: increments to missed=1, no action.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert refresh_calls == []

    # Tick 2: threshold met -> ALERT branch fires (plan_pending is parked,
    # so respawn ineligible) AND the refresh-from-api auto-heal fires.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    assert refresh_calls == ["pod-488"]
    assert (488, "session-stalled-alert") in markers


def test_stalled_alert_skips_refresh_when_no_pod(isolated_registry, monkeypatch, stalled_recorder):
    """The #488 refresh auto-heal only fires when the issue HAS a RUNNING
    managed pod. A stalled session with no pod has no SSH endpoint to
    refresh — firing the auto-heal would be wasted work."""
    import autonomous_session_watch as asw

    _stops, _spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 600, "sess-600")
    _patch_stale_signals(monkeypatch, asw, status="plan_pending")
    # No managed pods (the default _patch_stale_signals behavior).
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        asw,
        "_refresh_pods_conf_from_api",
        lambda pod_name, dry_run: refresh_calls.append(pod_name) or True,
    )
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    assert refresh_calls == []
    assert (600, "session-stalled-alert") in markers


def test_stalled_alert_refresh_dedups_within_episode(
    isolated_registry, monkeypatch, stalled_recorder
):
    """``refresh_attempted`` dedups: a stalled episode that triggers
    multiple alert ticks fires refresh-from-api at most ONCE — the
    same dedup shape ``alerted`` uses for the loud marker."""
    import autonomous_session_watch as asw

    _stops, _spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 488, "sess-488")
    _patch_stale_signals(monkeypatch, asw, status="plan_pending")
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(488, "p488", "pod-488")])
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        asw,
        "_refresh_pods_conf_from_api",
        lambda pod_name, dry_run: refresh_calls.append(pod_name) or True,
    )
    now = 1_000_000.0

    # Tick 1: missed=1, no action.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    # Tick 2: alert fires + refresh fires (first time this episode).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    # Tick 3: still stalled, but refresh_attempted=True -> NO second refresh.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    assert refresh_calls == ["pod-488"]  # exactly once


def test_stalled_alert_refresh_re_fires_after_self_report_advances(
    isolated_registry, monkeypatch, stalled_recorder
):
    """When the session resumes self-reporting (episode over), the
    ``refresh_attempted`` flag clears alongside ``alerted`` /
    ``respawn_count`` / ``exhausted``, so a subsequent staleness episode
    can re-fire the refresh-from-api auto-heal — same shape as the
    alert-dedup re-arm."""
    import autonomous_session_watch as asw

    _stops, _spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 488, "sess-488")
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(488, "p488", "pod-488")])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "plan_pending")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [{"kind": "epm:progress", "ts": "old"}])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: 0.0)
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        asw,
        "_refresh_pods_conf_from_api",
        lambda pod_name, dry_run: refresh_calls.append(pod_name) or True,
    )

    # First episode: stale at ts-old.
    monkeypatch.setattr(
        asw, "_self_report_age_seconds", lambda issue, now: (STALLED_WINDOW_S + 60, "ts-1")
    )
    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)
    assert refresh_calls == ["pod-488"]

    # Self-report ADVANCES (session resumed) -> episode ends, flags clear.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (0.0, "ts-2"))
    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)

    # Second episode: stale again with a NEWER ts that's still old. The
    # refresh_attempted flag must have cleared, so the new staleness episode
    # re-fires the auto-heal once.
    monkeypatch.setattr(
        asw, "_self_report_age_seconds", lambda issue, now: (STALLED_WINDOW_S + 60, "ts-3")
    )
    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)

    assert refresh_calls == ["pod-488", "pod-488"]


def test_refresh_pods_conf_from_api_fail_soft_on_nonzero_exit(monkeypatch):
    """``_refresh_pods_conf_from_api`` returns False (does NOT raise) on a
    non-zero exit from ``pod.py config --refresh-from-api``. The watcher
    pass must never crash on the auto-heal — fail-soft contract."""
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **kw: type(
            "R", (), {"returncode": 2, "stdout": "", "stderr": "ERROR: pod not found"}
        )(),
    )
    assert asw._refresh_pods_conf_from_api("pod-488", dry_run=False) is False


def test_refresh_pods_conf_from_api_fail_soft_on_oserror(monkeypatch):
    """A subprocess OSError on the refresh call also returns False instead of
    propagating. Same fail-soft contract."""
    import autonomous_session_watch as asw

    def _boom(*a, **kw):
        raise OSError("uv not found")

    monkeypatch.setattr(asw.subprocess, "run", _boom)
    assert asw._refresh_pods_conf_from_api("pod-488", dry_run=False) is False


def test_refresh_pods_conf_from_api_dry_run_does_not_invoke(monkeypatch):
    """Dry-run mode logs the call but never invokes subprocess.run — same
    contract as ``_stop_pod`` / ``_post_progress_marker``."""
    import autonomous_session_watch as asw

    called: list[bool] = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **kw: called.append(True))
    result = asw._refresh_pods_conf_from_api("pod-488", dry_run=True)
    assert result is False
    assert called == []


# ─── stalled-detector: manual (`manual-issue-<N>.json`) ALERT-ONLY coverage ──
#
# #505 round-2 orphaning (2026-06-10): a dead bare-`spawn-issue` session at an
# ACTIVE status orphaned silently because the stalled pass only globbed
# `issue-*.json`. Manual registrations now get the SAME staleness detection in
# ALERT-ONLY mode — never a respawn (user-driven sessions are the user's to
# restart), and never double-processing when an autonomous entry covers the
# same issue.


def _write_manual_entry(reg_dir, issue, session_id):
    """Helper: write a manual-registry entry matching spawn_session's
    `_register_manual_session` layout."""
    import json
    import time as _t

    (reg_dir / f"manual-issue-{issue}.json").write_text(
        json.dumps(
            {
                "issue": issue,
                "happy_session_id": session_id,
                "cwd": "/repo",
                "spawned_at": _t.time(),
                "mode": "manual",
            }
        )
    )


def test_stalled_manual_entry_alerts_never_respawns(
    isolated_registry, monkeypatch, stalled_recorder
):
    # ACTIVE status + reachable daemon would make an AUTONOMOUS entry
    # respawn-eligible; a manual entry must still get ALERT-ONLY.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_manual_entry(isolated_registry, 505, "sess-505-manual")
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1: missed -> 1, no action (the 2-miss guard applies to manual too).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []

    # Tick 2: threshold met -> ALERT, never a stop/spawn.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == []
    assert markers == [(505, "session-stalled-alert")]

    # Ticks 3+4: alerted episode dedups, and eligibility stays False for
    # manual entries so the alert never escalates to a respawn (contrast
    # the autonomous escalation pinned by
    # test_session_stalled_already_alerted_escalates_to_respawn_when_eligible).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == []
    assert markers == [(505, "session-stalled-alert")]


def test_stalled_manual_entry_skipped_when_autonomous_entry_exists(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Both registrations for the same issue share stalled-<N>.json; the
    # manual one must be skipped or one tick would double-increment the
    # 2-miss guard. Autonomous behavior must be exactly as without the
    # manual sibling: respawn on the SECOND stale tick, no stalled-alert.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 510, "sess-510", cap=24.0)
    _write_manual_entry(isolated_registry, 510, "sess-510-manual")
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1: autonomous missed -> 1; manual skipped (no double increment,
    # so nothing fires on this tick).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []

    # Tick 2: the autonomous respawn fires once; still no stalled-alert
    # from the manual sibling.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-510"]
    assert spawns == [(510, 24.0)]
    assert markers == [(510, "session-auto-respawn")]


def test_stalled_manual_entry_without_self_report_is_skipped(
    isolated_registry, monkeypatch, stalled_recorder
):
    # A bare manual session that never started self-reporting (spawned but
    # never driven) must not alert — a missing self-report means this pass
    # doesn't apply (decide_session_stalled case 1).
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_manual_entry(isolated_registry, 506, "sess-506-manual")
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))

    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=1_000_000.0, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []


# ── vm-disk headroom pass (task #552 incident, 2026-06-10) ───────────────────


def test_decide_vm_disk_levels():
    import autonomous_session_watch as asw

    gib = 2**30
    assert asw.decide_vm_disk(25 * gib, alerted=False, last_reclaim_ts=None, now=0.0) == (
        "ok",
        False,
        False,
    )
    assert asw.decide_vm_disk(12 * gib, alerted=False, last_reclaim_ts=None, now=0.0) == (
        "low",
        True,
        False,  # low-but-not-critical never reclaims
    )
    assert asw.decide_vm_disk(4 * gib, alerted=False, last_reclaim_ts=None, now=0.0) == (
        "critical",
        True,
        True,
    )


def test_decide_vm_disk_alert_dedups_within_episode():
    import autonomous_session_watch as asw

    level, do_alert, _ = asw.decide_vm_disk(12 * 2**30, alerted=True, last_reclaim_ts=None, now=0.0)
    assert (level, do_alert) == ("low", False)


def test_decide_vm_disk_reclaim_rearms_after_window():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    # Within the re-arm window: no second reclaim (no hot-loop pruning).
    _, _, do_reclaim = asw.decide_vm_disk(
        4 * 2**30, alerted=True, last_reclaim_ts=now - 60.0, now=now
    )
    assert do_reclaim is False
    # Past the window: re-fires (junk re-accumulated during a long episode).
    _, _, do_reclaim = asw.decide_vm_disk(
        4 * 2**30, alerted=True, last_reclaim_ts=now - asw.VM_DISK_RECLAIM_REARM_S, now=now
    )
    assert do_reclaim is True


def test_vm_disk_sentinel_excluded_from_real_progress():
    # The vm-disk alert is posted as epm:progress on a task; it must NOT reset
    # that task's real-progress staleness clock (same contract as every other
    # watcher-posted note).
    import autonomous_session_watch as asw

    assert asw._VM_DISK_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_vm_disk_pass_ok_clears_episode_state(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    (isolated_registry / "vm-disk.json").write_text(
        json.dumps({"alerted": True, "last_reclaim_ts": None, "first_seen": 1.0})
    )
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 100 * 2**30)
    asw.vm_disk_pass(dry_run=False, now=1_000_000.0)
    assert not (isolated_registry / "vm-disk.json").exists()


def test_vm_disk_pass_alert_posts_marker_once_per_episode(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    (isolated_registry / "issue-552.json").write_text(json.dumps({"issue": 552}))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 12 * 2**30)  # low, not critical
    markers: list[tuple[int, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label)),
    )
    prunes: list[bool] = []
    monkeypatch.setattr(asw, "_vm_reclaim_uv_cache", lambda dry_run: prunes.append(True))

    asw.vm_disk_pass(dry_run=False, now=1_000_000.0)
    asw.vm_disk_pass(dry_run=False, now=1_000_600.0)  # next tick: deduped

    assert markers == [(552, "vm-disk-low")]
    assert prunes == []  # low-but-not-critical never reclaims


def test_vm_disk_pass_fallback_event_when_no_active_issue(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 12 * 2**30)
    asw.vm_disk_pass(dry_run=False, now=1_000_000.0)

    lines = (isolated_registry / "vm-disk-events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["kind"] == "vm-disk-low"
    assert asw._VM_DISK_NOTE_SENTINEL in event["note"]


def test_vm_disk_pass_critical_runs_reclaims_with_rearm(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 4 * 2**30)  # critical
    prunes: list[bool] = []
    sweeps: list[float] = []
    monkeypatch.setattr(asw, "_vm_reclaim_uv_cache", lambda dry_run: prunes.append(True))
    monkeypatch.setattr(
        asw, "_sweep_stale_claude_tmp", lambda now, dry_run: (sweeps.append(now), 0)[1]
    )

    now = 1_000_000.0
    asw.vm_disk_pass(dry_run=False, now=now)
    asw.vm_disk_pass(dry_run=False, now=now + 600.0)  # within re-arm window: no churn
    asw.vm_disk_pass(dry_run=False, now=now + asw.VM_DISK_RECLAIM_REARM_S + 600.0)

    assert len(prunes) == 2  # first tick + post-window re-fire
    assert len(sweeps) == 2


def test_vm_disk_pass_dry_run_mutates_nothing(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 4 * 2**30)  # critical
    prune_cmds: list[bool] = []
    monkeypatch.setattr(asw, "_sweep_stale_claude_tmp", lambda now, dry_run: 0)
    monkeypatch.setattr(
        asw,
        "subprocess",
        type("S", (), {"run": staticmethod(lambda *a, **kw: prune_cmds.append(True))}),
    )

    asw.vm_disk_pass(dry_run=True, now=1_000_000.0)

    assert prune_cmds == []  # uv cache prune not actually invoked
    assert not (isolated_registry / "vm-disk.json").exists()  # no state saved
    assert not (isolated_registry / "vm-disk-events.jsonl").exists()  # no event written


# ─── orphan sweep (registration-independent safety net) ─────────────────────
#
# Pins the 2026-06-10 #472/#518 incident class: an ACTIVE-status task with no
# live REGISTERED session must be recovered even when no registry entry exists
# at all (#472: entry deleted at a TERMINAL park, task revived by a same-issue
# follow-up driven by an unregistered session that then died). A wrong RESPAWN
# costs a duplicate session, so the pure decide_orphan gate is pinned
# exhaustively like decide / decide_pod_safety.

_STALE = ORPHAN_STALENESS_S_DEFAULT + 60.0  # comfortably past the threshold


@pytest.mark.parametrize("status", [*sorted(PARK | TERMINAL), None, "some_new_status"])
def test_orphan_non_active_clears(status):
    # Only ACTIVE statuses are orphanable; everything else clears state.
    assert decide_orphan(status, False, False, None, _STALE, missed=3) == ("clear", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_orphan_mapped_alive_clears(status):
    # A live registered session (autonomous OR manual id) ends the episode,
    # regardless of marker staleness or accumulated misses.
    assert decide_orphan(status, True, False, None, _STALE, missed=2) == ("clear", 0)


def test_orphan_fresh_registration_keeps():
    # A registry entry written within the spawn-grace window means a recovery
    # is in flight (same-tick respawn by another pass, or a manual recovery
    # whose session id hasn't reached the daemon's live set yet).
    assert decide_orphan("running", False, False, ORPHAN_SPAWN_GRACE_S - 1, _STALE, missed=1) == (
        "keep",
        0,
    )


def test_orphan_fresh_markers_keep():
    # Real progress within the staleness window: something is driving the
    # task even if we can't map a session to it — don't double-spawn.
    assert decide_orphan(
        "running", False, False, None, ORPHAN_STALENESS_S_DEFAULT - 60.0, missed=1
    ) == ("keep", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_orphan_needs_two_misses_before_respawn(status):
    # Mirrors the respawn pass's 2-miss guard: first stale observation only
    # accumulates; the respawn fires on the SECOND consecutive miss.
    assert decide_orphan(status, False, False, None, _STALE, missed=0) == ("keep", 1)
    assert decide_orphan(status, False, False, None, _STALE, missed=1) == ("respawn", 0)


def test_orphan_no_marker_at_all_counts_as_stale():
    # An ACTIVE task with zero real progress markers is itself the signal
    # (mirrors the pod-safety None-is-stale rule).
    assert decide_orphan("running", False, False, None, None, missed=1) == ("respawn", 0)


def test_orphan_manual_only_alerts_never_respawns():
    # A task whose only registration is MANUAL is user-driven: never
    # auto-respawn (#505 round-2 orphaning); alert loudly instead.
    assert decide_orphan("running", False, True, None, _STALE, missed=1) == ("alert", 2)


def test_orphan_daily_cap_exhausted_alerts():
    assert decide_orphan(
        "running",
        False,
        False,
        None,
        _STALE,
        missed=1,
        respawns_today=ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
    ) == ("alert", 2)


def test_orphan_threshold_one_respawns_immediately():
    assert decide_orphan("running", False, False, None, _STALE, missed=0, threshold=1) == (
        "respawn",
        0,
    )


def test_orphan_sentinels_excluded_from_real_progress():
    # The sweep's own markers must never reset the staleness clock they
    # measure — pin their membership in the shared exclusion set.
    from autonomous_session_watch import (
        _ORPHAN_ALERT_NOTE_SENTINEL,
        _ORPHAN_RESPAWN_NOTE_SENTINEL,
        _WATCHER_NOTE_SENTINELS,
    )

    assert _ORPHAN_RESPAWN_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS
    assert _ORPHAN_ALERT_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS


def test_orphan_state_roundtrip_and_clear(isolated_registry):
    import autonomous_session_watch as asw

    asw._save_orphan_state(
        472, missed=1, alerted=True, respawn_day="2026-06-10", respawns_today=2, prev=None
    )
    state = asw._load_orphan_state(472)
    assert state["missed"] == 1
    assert state["alerted"] is True
    assert state["respawn_day"] == "2026-06-10"
    assert state["respawns_today"] == 2
    assert isinstance(state["first_seen"], float)
    asw._clear_orphan_state(472)
    assert asw._load_orphan_state(472) == {}


def test_session_alive_ignores_worktree_cwd_zombies(isolated_registry):
    # The 2026-06-10 #518 regression: a superseded driver generation parked in
    # the issue worktree must NOT count as "alive" for the registered entry.
    # Liveness is recorded-id OR manual-registration-id only.
    import json

    import autonomous_session_watch as asw

    entry = {"issue": 518, "happy_session_id": "dead-sid"}
    assert asw._session_alive(entry, live_ids={"zombie-other-sid"}) is False
    # A live MANUAL replacement session keeps the issue alive (no duplicate
    # respawn next to a user-driven session).
    (isolated_registry / "manual-issue-518.json").write_text(
        json.dumps({"issue": 518, "happy_session_id": "manual-sid", "mode": "manual"})
    )
    assert asw._session_alive(entry, live_ids={"manual-sid"}) is True
    assert asw._session_alive(entry, live_ids={"dead-sid-x"}) is False
    # The recorded autonomous id itself still counts, of course.
    assert asw._session_alive(entry, live_ids={"dead-sid"}) is True


# ─── session-reconcile pass (sessions-vs-status; 2026-06-10 disk incident) ───
# A wrong STOP kills a session the user may still want (hence alert-only
# default + the DONE-status gate + the 2-miss guard), while a missing stop
# re-opens the incident class (idle sessions pinning worktrees + holding
# deleted-file handles). Both directions are pinned here.


def test_session_reconcile_done_set_is_terminal_for_gc():
    # The DONE set is deliberately shared with the GC's terminal set:
    # completed/archived only. awaiting_promotion and blocked are excluded —
    # the user may be live-parked there (promotion gate / investigation).
    from autonomous_session_watch import SESSION_RECONCILE_DONE, TERMINAL_FOR_GC

    assert SESSION_RECONCILE_DONE == TERMINAL_FOR_GC == {"completed", "archived"}
    assert "awaiting_promotion" not in SESSION_RECONCILE_DONE
    assert "blocked" not in SESSION_RECONCILE_DONE


@pytest.mark.parametrize(
    "status",
    [
        None,
        "proposed",
        "planning",
        "plan_pending",
        "approved",
        "running",
        "verifying",
        "interpreting",
        "reviewing",
        "awaiting_promotion",
        "blocked",
    ],
)
@pytest.mark.parametrize("idle", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_session_reconcile_non_done_always_clears(status, idle, missed):
    # Any non-terminal status (including the user-parked awaiting_promotion /
    # blocked and an unreadable None) clears the episode — never an action,
    # even with autostop armed and a huge miss count.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(status, idle, missed, alerted=True, autostop=True) == (
        "clear",
        0,
    )


@pytest.mark.parametrize("status", ["completed", "archived"])
def test_session_reconcile_fresh_activity_clears(status):
    # A DONE task with recent activity (e.g. it JUST completed) keeps its
    # session — the idle window is the post-completion grace period.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(status, False, 5, alerted=True, autostop=True) == ("clear", 0)


def test_session_reconcile_two_miss_guard_then_alert():
    # Alert-only default: tick 1 accumulates, tick 2 alerts ONCE, later ticks
    # stay quiet (dedup) while the miss count keeps growing so a later
    # autostop-enable fires immediately.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile("completed", True, 0, alerted=False) == ("keep", 1)
    assert decide_session_reconcile("completed", True, 1, alerted=False) == ("alert", 2)
    assert decide_session_reconcile("completed", True, 2, alerted=True) == ("keep", 3)


def test_session_reconcile_autostop_stops_at_threshold():
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile("completed", True, 0, alerted=False, autostop=True) == (
        "keep",
        1,
    )
    assert decide_session_reconcile("completed", True, 1, alerted=False, autostop=True) == (
        "stop",
        0,
    )


def test_session_reconcile_autostop_enable_mid_episode_escalates():
    # The #506 lesson: an already-alerted episode must escalate to the
    # stronger action the moment it becomes eligible — flipping
    # EPM_SESSION_RECONCILE_AUTOSTOP=1 mid-episode stops on the NEXT tick
    # without re-accumulating the miss guard.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile("completed", True, 2, alerted=True, autostop=True) == (
        "stop",
        0,
    )


def test_session_reconcile_keep_running_skips_and_beats_followup():
    # The explicit user tag wins (same precedence as decide_pod_safety) and
    # resets the miss counter so tag removal re-arms a fresh accumulation.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(
        "completed", True, 5, alerted=False, autostop=True, keep_running=True
    ) == ("keep-running-skip", 0)
    assert decide_session_reconcile(
        "completed",
        True,
        5,
        alerted=False,
        autostop=True,
        keep_running=True,
        followup_active=True,
    ) == ("keep-running-skip", 0)


def test_session_reconcile_followup_active_skips():
    # A live inline follow-up (epm:run-launched newer than the done
    # transition) means the session is the follow-up's driver — never stop
    # it, even if the follow-up itself is quiet past the idle window.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(
        "completed", True, 5, alerted=False, autostop=True, followup_active=True
    ) == ("followup-skip", 0)


def test_map_sessions_registry_beats_cwd_and_unmapped_skipped():
    # Registered mapping wins over the worktree-cwd inference; sessions with
    # NEITHER (the PM session at repo root, other-project chat sessions,
    # missing path, non-str sid) are skipped entirely — they can never be
    # acted on by the pass.
    from autonomous_session_watch import _map_sessions_to_issues

    live = {"reg-sid", "zombie-sid", "pm-sid", "goat-sid", "no-path-sid", None}
    registry_map = {"reg-sid": 489}
    paths = {
        # Registered session sitting in a DIFFERENT issue's worktree: the
        # registry mapping must win.
        "reg-sid": "/home/t/explore-persona-space/.claude/worktrees/issue-999",
        "zombie-sid": "/home/t/explore-persona-space/.claude/worktrees/issue-518",
        "pm-sid": "/home/t/explore-persona-space",
        "goat-sid": "/home/t/my-goat",
        # no-path-sid deliberately absent.
    }
    assert _map_sessions_to_issues(live, registry_map, paths) == {
        489: {"reg-sid"},
        518: {"zombie-sid"},
    }


def test_session_reconcile_sentinels_are_filtered_from_progress():
    # Both new watcher-posted markers land as epm:progress on the very task
    # whose inactivity they measure — they MUST be excluded from the
    # real-progress clock or the alert would end the episode it reports.
    import autonomous_session_watch as asw

    events = [
        {
            "kind": "epm:progress",
            "ts": "2026-06-10T10:00:00Z",
            "note": f"{asw._SESSION_RECONCILE_ALERT_NOTE_SENTINEL} IDLE session(s) ...",
        },
        {
            "kind": "epm:progress",
            "ts": "2026-06-10T11:00:00Z",
            "note": f"{asw._SESSION_RECONCILE_STOP_NOTE_SENTINEL} auto-stopped ...",
        },
    ]
    assert asw._latest_progress_ts(events) is None
    assert asw._SESSION_RECONCILE_ALERT_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._SESSION_RECONCILE_STOP_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def _patch_session_reconcile_io(monkeypatch, *, status, events=None, self_report=(None, None)):
    """Common monkeypatching for the session-reconcile I/O wrapper tests:
    task reads + the daemon-derived maps, leaving state files + decisions
    real. Returns the (stops, posts) recorders."""
    import autonomous_session_watch as asw

    stops: list[str] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_events", lambda issue: list(events or []))
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: self_report)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    monkeypatch.setattr(asw, "_load_session_issue_map", lambda: {"sid-a": 42, "sid-b": 42})
    monkeypatch.setattr(asw, "_load_session_meta", lambda: {})
    return stops, posts


def test_session_reconcile_alert_only_default_posts_once_never_stops(
    isolated_registry, monkeypatch
):
    # Default posture (env flag unset): tick 1 accumulates, tick 2 posts ONE
    # alert marker, tick 3 stays quiet. No session is ever stopped.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed")
    now = 1_000_000.0
    live = {"sid-a", "sid-b"}

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    state_path = isolated_registry / "session-reconcile-42.json"
    assert json.loads(state_path.read_text())["missed"] == 1
    assert stops == [] and posts == []

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert posts == [(42, "session-reconcile-alert")]
    state = json.loads(state_path.read_text())
    assert state["alerted"] is True and state["missed"] == 2
    assert state["sids"] == ["sid-a", "sid-b"]
    assert stops == []

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert posts == [(42, "session-reconcile-alert")]  # dedup: still exactly one
    assert stops == []


def test_session_reconcile_autostop_stops_all_sessions_and_clears(isolated_registry, monkeypatch):
    # With EPM_SESSION_RECONCILE_AUTOSTOP=1: tick 1 accumulates, tick 2 stops
    # EVERY live mapped session, posts the stop marker, clears the state.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", "1")
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed")
    now = 1_000_000.0
    live = {"sid-a", "sid-b"}

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert stops == []

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert sorted(stops) == ["sid-a", "sid-b"]
    assert posts == [(42, "session-reconcile-stop")]
    assert not (isolated_registry / "session-reconcile-42.json").exists()


@pytest.mark.parametrize("status", ["awaiting_promotion", "blocked", "running"])
def test_session_reconcile_never_acts_on_non_done_status(isolated_registry, monkeypatch, status):
    # awaiting_promotion (live-parked for promotion), blocked (under
    # investigation), and any ACTIVE status are untouchable — no stop, no
    # marker, no state accumulation, even with autostop armed.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", "1")
    stops, posts = _patch_session_reconcile_io(monkeypatch, status=status)
    for _ in range(3):
        asw.session_reconcile_pass(
            False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=1_000_000.0
        )
    assert stops == [] and posts == []
    assert not (isolated_registry / "session-reconcile-42.json").exists()


def test_session_reconcile_fresh_completion_keeps_session(isolated_registry, monkeypatch):
    # A task that completed 1h ago is inside the idle grace window: its
    # session is kept and any prior episode state is cleared.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", "1")
    ts = "2026-06-10T10:00:00Z"
    now = asw._parse_event_ts(ts) + 3600  # 1h after the completion marker
    events = [{"kind": "epm:status-changed", "ts": ts, "note": "-> completed"}]
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed", events=events)
    # Pre-existing episode state from an earlier (now-recovered) episode.
    asw._save_session_reconcile_state(42, missed=1, alerted=True, sids=["sid-a"])

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=now)
    assert stops == [] and posts == []
    assert not (isolated_registry / "session-reconcile-42.json").exists()  # cleared


def test_session_reconcile_keep_running_tag_skips_stop(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", "1")
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    for _ in range(3):
        asw.session_reconcile_pass(
            False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=1_000_000.0
        )
    assert stops == [] and posts == []
    state = json.loads((isolated_registry / "session-reconcile-42.json").read_text())
    assert state["missed"] == 0  # tag removal re-arms a fresh accumulation


def test_session_reconcile_gc_drops_state_without_mapped_session(isolated_registry, monkeypatch):
    # When the sessions died / were stopped by any path, the per-issue state
    # is reaped so a later session on the same issue starts a fresh episode.
    import autonomous_session_watch as asw

    asw._save_session_reconcile_state(42, missed=1, alerted=True, sids=["sid-a"])
    asw._save_session_reconcile_state(99, missed=1, alerted=False, sids=["sid-z"])
    cleared = asw._gc_orphan_session_reconcile_state({42}, dry_run=False, now=1_000_000.0)
    assert cleared == [99]
    assert (isolated_registry / "session-reconcile-42.json").exists()
    assert not (isolated_registry / "session-reconcile-99.json").exists()
    # Dry-run never deletes.
    cleared = asw._gc_orphan_session_reconcile_state(set(), dry_run=True, now=1_000_000.0)
    assert cleared == [42]
    assert (isolated_registry / "session-reconcile-42.json").exists()


def test_session_reconcile_pass_daemon_unreachable_skips(isolated_registry, monkeypatch):
    # Session liveness is unknowable during a daemon outage, and the stop
    # action POSTs to the daemon — the whole pass must no-op.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", "1")
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed")
    asw._save_session_reconcile_state(42, missed=5, alerted=True, sids=["sid-a"])
    asw.session_reconcile_pass(False, 2, daemon_reachable=False, live_ids=None, now=1_000_000.0)
    assert stops == [] and posts == []
    # State untouched (no GC either — liveness unknown).
    assert (isolated_registry / "session-reconcile-42.json").exists()
