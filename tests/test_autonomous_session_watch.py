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
    INFRA_DRAIN_BACKOFF_S_DEFAULT,
    INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT,
    INFRA_DRAIN_OCCUPIED_STATUSES,
    INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES,
    ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
    ORPHAN_SPAWN_GRACE_S,
    ORPHAN_STALENESS_S_DEFAULT,
    PARK,
    POD_ACTIVE,
    STALLED_MAX_RESPAWNS,
    STALLED_WINDOW_S,
    TERMINAL,
    decide,
    decide_infra_drain,
    decide_orphan,
    decide_pod_safety,
    parse_infra_drain_queue,
)

from explore_persona_space.task_workflow import STATUSES  # noqa: E402


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


def test_task_followup_active_user_chat_scope_marker():
    # refs #573: a USER-CHAT inline follow-up posts `epm:followup-scope`
    # BEFORE re-invoking /issue, so the pod can be provisioned before any
    # `epm:run-launched` lands. The widened predicate must treat a fresh
    # followup-scope (or free-analysis-followup-run) as a live follow-up.
    import autonomous_session_watch as asw

    # followup-scope NEWER than done-transition, NO run-launched -> True.
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:promoted", "ts": "2026-06-10T00:00:00Z", "note": ""},
                {"kind": "epm:followup-scope", "ts": "2026-06-10T03:00:00Z", "note": ""},
            ],
        )
        is True
    )
    # free-analysis-followup-run NEWER than done-transition -> True.
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:status-changed", "ts": "2026-06-10T00:00:00Z", "note": ""},
                {
                    "kind": "epm:free-analysis-followup-run",
                    "ts": "2026-06-10T01:00:00Z",
                    "note": "",
                },
            ],
        )
        is True
    )
    # followup-scope OLDER than the latest done-transition -> False (that
    # follow-up round already settled; the auto-stop re-arms).
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:followup-scope", "ts": "2026-06-09T20:00:00Z", "note": ""},
                {"kind": "epm:status-changed", "ts": "2026-06-10T00:00:00Z", "note": ""},
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


def test_running_managed_pods_api_error_returns_none(monkeypatch):
    # A FAILED snapshot must be distinguishable from "genuinely no pods":
    # None, not []. The pod-safety state GC keys off this — it must not reap
    # episode state (dedup flags, miss counters) on a transport-error tick.
    import autonomous_session_watch as asw

    def boom():
        raise RuntimeError("transport down")

    monkeypatch.setattr(asw, "list_team_pods", boom)
    assert asw._running_managed_issue_pods() is None


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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [(7, "p7", "pod-7")])
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(530, "p530", "pod-530")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(530, "p530", "pod-530")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(477, "p477", "pod-477")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(477, "p477", "pod-477")]
    )
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [(7, "p7", "pod-7")])
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(489, "p489", "pod-489")]
    )
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


@pytest.mark.parametrize("snapshot", [None, []], ids=["failed-snapshot", "genuinely-empty"])
def test_pod_safety_pass_api_error_does_not_stop(isolated_registry, monkeypatch, snapshot):
    # Whether the snapshot FAILED (None, transport error) or is genuinely
    # empty ([]), `pod_safety_pass` MUST NOT call `_stop_pod`. Fail-closed
    # invariant for the destructive action.
    import autonomous_session_watch as asw

    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: snapshot)
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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])

    asw.pod_safety_pass(dry_run=False, threshold=2)

    assert not (isolated_registry / "pod-safety-99.json").exists()


def test_pod_safety_pass_failed_snapshot_does_not_gc_state(isolated_registry, monkeypatch):
    # A transport-error tick (snapshot=None) must NOT reap pod-safety state:
    # the GC cannot tell "snapshot failed" from "every pod left RUNNING", and
    # reaping resets not just the fail-safe 2-miss counters but the
    # once-per-episode dedup flags (`alerted` etc.), so every API hiccup
    # would re-arm duplicate markers.
    import json
    import time as t

    import autonomous_session_watch as asw

    _write_state(isolated_registry, 99, "p99", missed=1, first_seen=t.time(), alerted=True)
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: None)

    asw.pod_safety_pass(dry_run=False, threshold=2)

    state_path = isolated_registry / "pod-safety-99.json"
    assert state_path.exists()  # NOT reaped on the failed snapshot
    payload = json.loads(state_path.read_text())
    assert payload["alerted"] is True  # once-per-episode dedup flag survives
    assert payload["missed"] == 1  # miss counter survives too


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
    zombie_calls: list[tuple] = []
    idle_calls: list[tuple] = []
    snapshot = [{"happySessionId": "sid-shared", "pid": 12345}]
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_children", lambda: snapshot)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: orphan_calls.append((a, kw)))
    # Patched so the unit test never RPCs the real daemon / scans real /proc /
    # spawns task.py subprocesses for whatever sessions are live on the VM.
    monkeypatch.setattr(asw, "zombie_wrapper_pass", lambda *a, **kw: zombie_calls.append((a, kw)))
    monkeypatch.setattr(asw, "idle_unmapped_pass", lambda *a, **kw: idle_calls.append((a, kw)))

    rc = asw.main([])

    assert rc == 0
    assert len(pod_safety_calls) == 1
    assert len(orphan_calls) == 1
    assert orphan_calls[0][1]["daemon_reachable"] is True
    assert orphan_calls[0][1]["live_ids"] == set()
    # The two reaper passes share ONE /list snapshot (the same object),
    # fetched once in main() — not one RPC per pass.
    assert zombie_calls[0][1]["children"] is snapshot
    assert idle_calls[0][1]["children"] is snapshot


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
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    # Neutralize the in-flight-provision exemption (refs #573): these tests
    # use REAL issue numbers, and the probe reads the live VM's /proc + the
    # repo's .claude/cache/poll-pipeline-<N>.json — both nondeterministic
    # here. Tests of the exemption itself re-patch this explicitly.
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
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
    # Neutralize the in-flight-provision exemption (refs #573): the probe
    # reads the live VM's /proc + the repo's real poll-pipeline state files,
    # which is nondeterministic under the fake `now` these tests use.
    # Exemption-specific tests re-patch this explicitly.
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
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


def test_stalled_exemption_live_provision_blocks_respawn(
    isolated_registry, monkeypatch, stalled_recorder
):
    # refs #573: a session whose bg-Bash chain is blocked on a live
    # `pod.py provision --wait-for-capacity` is NOT stalled — #534's
    # auto-respawn killed an in-flight provision 3x (~8h lost). When the
    # in-flight-provision probe returns a reason, the stalled detector
    # must neither accumulate misses nor stop/spawn/post markers.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 518, "sess-518", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(
        asw,
        "_provision_in_flight_reason",
        lambda issue, now: f"live pod provision/resume process (pid 4242) for issue #{issue}",
    )
    now = 1_000_000.0

    for _ in range(4):  # well past the 2-miss threshold
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []


def test_provision_in_flight_reason_fresh_poll_state(monkeypatch, tmp_path):
    # Signal 2: a fresh poll-pipeline-<N>.json mtime exempts the session even
    # without a live provision process; a stale one does not.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_find_provision_process", lambda issue: None)
    monkeypatch.setattr(asw, "_POLL_STATE_DIR", tmp_path)
    state = tmp_path / "poll-pipeline-77.json"
    state.write_text("{}")
    mtime = state.stat().st_mtime

    fresh = asw._provision_in_flight_reason(77, now=mtime + 60.0)
    assert fresh is not None and "poll-pipeline" in fresh

    stale = asw._provision_in_flight_reason(77, now=mtime + asw.STALLED_WINDOW_S + 60.0)
    assert stale is None

    # Missing file -> no exemption.
    assert asw._provision_in_flight_reason(78, now=mtime) is None


def test_provision_in_flight_reason_live_process(monkeypatch, tmp_path):
    # Signal 1: a live provision/resume process wins regardless of poll state.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_POLL_STATE_DIR", tmp_path)  # no state files
    monkeypatch.setattr(asw, "_find_provision_process", lambda issue: 4242)
    reason = asw._provision_in_flight_reason(77, now=1_000_000.0)
    assert reason is not None and "pid 4242" in reason


def test_find_provision_process_matches_own_argv():
    # End-to-end /proc scan against THIS test process: temporarily nothing
    # matches (this pytest process has no pod.py provision argv), so the
    # scan returns None without raising.
    import autonomous_session_watch as asw

    assert asw._find_provision_process(999_999_999) is None


def test_stalled_pass_failed_pod_snapshot_degrades_to_empty(
    isolated_registry, monkeypatch, stalled_recorder
):
    # A FAILED pod snapshot (None) degrades to "no pods" for the stalled
    # detector — identical decision inputs (has_pod=False) to today's
    # empty-set fallback, so the pass neither crashes nor changes outcome.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 518, "sess-518", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: None)
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
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
    # ``_refresh_pods_conf_from_api``. ``followups_child_alerted`` (default
    # False) is the dedup flag for the followups_running-parent-waiting-on-
    # open-child suppression alert added 2026-06-11 (#533); see
    # ``_followups_awaiting_child_reason``. Schema-shape coverage stays
    # exhaustive.
    assert payload == {
        "happy_session_id": "sess-7",
        "missed": 1,
        "alerted": False,
        "respawn_count": 2,
        "exhausted": False,
        "refresh_attempted": False,
        "followups_child_alerted": False,
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(488, "p488", "pod-488")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(488, "p488", "pod-488")]
    )
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [(488, "p488", "pod-488")]
    )
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
    assert asw.decide_vm_disk(
        25 * gib, alerted=False, last_reclaim_ts=None, last_audit_ts=None, now=0.0
    ) == ("ok", False, False, False)
    assert asw.decide_vm_disk(
        17 * gib, alerted=False, last_reclaim_ts=None, last_audit_ts=None, now=0.0
    ) == (
        "low",
        True,
        False,  # low-but-not-critical never runs the cache reclaims...
        True,  # ...but the worktree audit fires at the advisory threshold
    )
    assert asw.decide_vm_disk(
        4 * gib, alerted=False, last_reclaim_ts=None, last_audit_ts=None, now=0.0
    ) == ("critical", True, True, True)


def test_decide_vm_disk_critical_threshold_is_15_gib_default():
    # 12 GiB sat in the old "low" band (8 GiB critical); after the 2026-06-11
    # incident (17 GiB -> 1.2 GiB within hours) the default critical threshold
    # is 15 GiB (env EPM_VM_DISK_CRITICAL_GIB).
    import autonomous_session_watch as asw

    level, _, _, _ = asw.decide_vm_disk(
        12 * 2**30, alerted=False, last_reclaim_ts=None, last_audit_ts=None, now=0.0
    )
    assert level == "critical"


def test_env_gib_bytes_fail_soft(monkeypatch):
    # A garbled / non-positive / inf / nan knob falls back to the default
    # instead of crashing the watcher at import (int(inf * 2**30) raises).
    import autonomous_session_watch as asw

    for bad in ("garbled", "-3", "0", "inf", "nan", ""):
        monkeypatch.setenv("EPM_TEST_GIB", bad)
        assert asw._env_gib_bytes("EPM_TEST_GIB", 15) == 15 * 2**30, bad
    monkeypatch.setenv("EPM_TEST_GIB", "10")
    assert asw._env_gib_bytes("EPM_TEST_GIB", 15) == 10 * 2**30


def test_decide_vm_disk_alert_dedups_within_episode():
    import autonomous_session_watch as asw

    level, do_alert, _, _ = asw.decide_vm_disk(
        17 * 2**30, alerted=True, last_reclaim_ts=None, last_audit_ts=None, now=0.0
    )
    assert (level, do_alert) == ("low", False)


def test_decide_vm_disk_reclaim_rearms_after_window():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    # Within the re-arm window: no second reclaim (no hot-loop pruning).
    _, _, do_reclaim, _ = asw.decide_vm_disk(
        4 * 2**30, alerted=True, last_reclaim_ts=now - 60.0, last_audit_ts=None, now=now
    )
    assert do_reclaim is False
    # Past the window: re-fires (junk re-accumulated during a long episode).
    _, _, do_reclaim, _ = asw.decide_vm_disk(
        4 * 2**30,
        alerted=True,
        last_reclaim_ts=now - asw.VM_DISK_RECLAIM_REARM_S,
        last_audit_ts=None,
        now=now,
    )
    assert do_reclaim is True


def test_decide_vm_disk_audit_rearms_after_window():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    # Within the re-arm window: no second audit (no hot-loop sweeping).
    _, _, _, do_audit = asw.decide_vm_disk(
        17 * 2**30, alerted=True, last_reclaim_ts=None, last_audit_ts=now - 60.0, now=now
    )
    assert do_audit is False
    # Past the window: re-fires (catches a worktree whose holder process died
    # after the first audit, during a long episode).
    _, _, _, do_audit = asw.decide_vm_disk(
        17 * 2**30,
        alerted=True,
        last_reclaim_ts=None,
        last_audit_ts=now - asw.VM_DISK_RECLAIM_REARM_S,
        now=now,
    )
    assert do_audit is True


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


def test_vm_disk_pass_boundary_flap_does_not_rerun_audit(isolated_registry, monkeypatch):
    # Episode-flap churn (code-review Minor 2 on the auto-remediation fix):
    # free space oscillating around the 20 GiB advisory boundary must NOT
    # re-fire the worktree audit (or the once-per-episode alert) on each
    # fresh dip. Recovery INSIDE the hysteresis band (alert <= free <
    # alert + VM_DISK_CLEAR_HYSTERESIS_BYTES) keeps the episode state, so a
    # re-dip within the re-arm window sees the prior last_audit_ts +
    # alerted flag.
    import autonomous_session_watch as asw

    free = {"v": 19 * 2**30}  # low, not critical
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: free["v"])
    audits: list[bool] = []
    monkeypatch.setattr(
        asw,
        "_vm_remediate_worktrees",
        lambda dry_run: (audits.append(True), "worktree-audit rc=0: ok")[1],
    )
    notes: list[str] = []
    monkeypatch.setattr(
        asw, "_append_vm_disk_fallback_event", lambda note, dry_run: notes.append(note)
    )

    now = 1_000_000.0
    asw.vm_disk_pass(dry_run=False, now=now)  # dip: audit + alert fire
    free["v"] = 21 * 2**30  # recover INTO the band (20 <= free < 22 GiB)
    asw.vm_disk_pass(dry_run=False, now=now + 600.0)
    assert (isolated_registry / "vm-disk.json").exists()  # state kept, not cleared
    free["v"] = 19 * 2**30  # re-dip well within the 6h re-arm window
    asw.vm_disk_pass(dry_run=False, now=now + 1_200.0)

    assert len(audits) == 1  # the flap did NOT re-run the audit
    assert len(notes) == 1  # ...and did not re-alert (same episode)


def test_vm_disk_pass_decisive_recovery_clears_state_and_rearms(isolated_registry, monkeypatch):
    # At or above alert + hysteresis (~22 GiB) the episode IS over: the state
    # clears, so a later dip is a genuinely fresh episode (a new disk
    # consumer) and the audit + alert correctly fire again.
    import autonomous_session_watch as asw

    free = {"v": 19 * 2**30}  # low, not critical
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: free["v"])
    audits: list[bool] = []
    monkeypatch.setattr(
        asw,
        "_vm_remediate_worktrees",
        lambda dry_run: (audits.append(True), "worktree-audit rc=0: ok")[1],
    )
    notes: list[str] = []
    monkeypatch.setattr(
        asw, "_append_vm_disk_fallback_event", lambda note, dry_run: notes.append(note)
    )

    now = 1_000_000.0
    asw.vm_disk_pass(dry_run=False, now=now)
    free["v"] = 23 * 2**30  # decisive recovery: above alert + hysteresis
    asw.vm_disk_pass(dry_run=False, now=now + 600.0)
    assert not (isolated_registry / "vm-disk.json").exists()
    free["v"] = 19 * 2**30
    asw.vm_disk_pass(dry_run=False, now=now + 1_200.0)

    assert len(audits) == 2  # fresh episode re-runs the audit
    assert len(notes) == 2  # ...and re-alerts


def test_vm_disk_pass_alert_posts_marker_once_per_episode(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    (isolated_registry / "issue-552.json").write_text(json.dumps({"issue": 552}))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 17 * 2**30)  # low, not critical
    markers: list[tuple[int, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label)),
    )
    prunes: list[bool] = []
    monkeypatch.setattr(asw, "_vm_reclaim_uv_cache", lambda dry_run: prunes.append(True))
    monkeypatch.setattr(asw, "_vm_remediate_worktrees", lambda dry_run: "worktree-audit rc=0: ok")

    asw.vm_disk_pass(dry_run=False, now=1_000_000.0)
    asw.vm_disk_pass(dry_run=False, now=1_000_600.0)  # next tick: deduped

    assert markers == [(552, "vm-disk-low")]
    assert prunes == []  # low-but-not-critical never runs the cache reclaims


def test_vm_disk_pass_low_runs_worktree_audit_and_notes_remediation(isolated_registry, monkeypatch):
    # The 2026-06-11 incident class: advisory fired at 17 GiB but the
    # remediation that frees the big space (worktree_audit.py --apply) was
    # only on a once-daily cron; / hit 100% within hours. The pass now runs
    # the audit itself at the ADVISORY threshold and the marker note records
    # what was done, not just that disk was low.
    import json

    import autonomous_session_watch as asw

    (isolated_registry / "issue-552.json").write_text(json.dumps({"issue": 552}))
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 17 * 2**30)  # low, not critical
    notes: list[str] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: notes.append(note),
    )
    audits: list[bool] = []
    monkeypatch.setattr(
        asw,
        "_vm_remediate_worktrees",
        lambda dry_run: (audits.append(True), "worktree-audit rc=0: removed 15")[1],
    )

    now = 1_000_000.0
    asw.vm_disk_pass(dry_run=False, now=now)
    asw.vm_disk_pass(dry_run=False, now=now + 600.0)  # within re-arm window: no churn
    asw.vm_disk_pass(dry_run=False, now=now + asw.VM_DISK_RECLAIM_REARM_S + 600.0)

    assert len(audits) == 2  # first tick + post-window re-fire
    assert len(notes) == 1  # alert still once per episode
    assert "[auto-remediation:" in notes[0]
    assert "worktree-audit rc=0: removed 15" in notes[0]


def test_vm_disk_pass_fallback_event_when_no_active_issue(isolated_registry, monkeypatch):
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 17 * 2**30)
    monkeypatch.setattr(asw, "_vm_remediate_worktrees", lambda dry_run: "worktree-audit rc=0: ok")
    asw.vm_disk_pass(dry_run=False, now=1_000_000.0)

    lines = (isolated_registry / "vm-disk-events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["kind"] == "vm-disk-low"
    assert asw._VM_DISK_NOTE_SENTINEL in event["note"]


def test_vm_disk_pass_critical_runs_reclaims_with_rearm(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 4 * 2**30)  # critical
    wandb_cleanups: list[bool] = []
    prunes: list[bool] = []
    npm_cleans: list[bool] = []
    hf_evictions: list[float] = []
    sweeps: list[float] = []
    monkeypatch.setattr(
        asw,
        "_vm_reclaim_wandb_cache",
        lambda dry_run: (wandb_cleanups.append(True), "wandb-artifacts rc=0: ok")[1],
    )
    monkeypatch.setattr(
        asw, "_vm_reclaim_uv_cache", lambda dry_run: (prunes.append(True), "uv-cache rc=0: ok")[1]
    )
    monkeypatch.setattr(
        asw,
        "_vm_reclaim_npm_cache",
        lambda dry_run: (npm_cleans.append(True), "npm-cache rc=0: ok")[1],
    )
    monkeypatch.setattr(
        asw,
        "_vm_reclaim_hf_hub_cache",
        lambda now, dry_run: (hf_evictions.append(now), "hf-hub-ttl: nothing stale")[1],
    )
    monkeypatch.setattr(asw, "_vm_remediate_worktrees", lambda dry_run: "worktree-audit rc=0: ok")
    monkeypatch.setattr(
        asw, "_sweep_stale_claude_tmp", lambda now, dry_run: (sweeps.append(now), 0)[1]
    )

    now = 1_000_000.0
    asw.vm_disk_pass(dry_run=False, now=now)
    asw.vm_disk_pass(dry_run=False, now=now + 600.0)  # within re-arm window: no churn
    asw.vm_disk_pass(dry_run=False, now=now + asw.VM_DISK_RECLAIM_REARM_S + 600.0)

    assert len(wandb_cleanups) == 2  # the wandb artifact cache rides the critical arm
    assert len(prunes) == 2  # first tick + post-window re-fire
    assert len(npm_cleans) == 2  # npm cache clean rides the same critical arm
    assert len(hf_evictions) == 2  # ...as does the HF hub TTL eviction
    assert len(sweeps) == 2


def test_vm_disk_critical_note_carries_per_step_reclaim_summaries(isolated_registry, monkeypatch):
    # The 2026-06-11 episode's marker said only "cache reclaims ran" while the
    # reclaims freed ~0 GB and 17.6 GB (wandb) + 41.5 GB (HF hub) sat
    # untouched — the note must name each step and what it did.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 4 * 2**30)  # critical
    monkeypatch.setattr(
        asw, "_vm_reclaim_wandb_cache", lambda dry_run: "wandb-artifacts rc=0: reclaimed"
    )
    monkeypatch.setattr(
        asw, "_vm_reclaim_uv_cache", lambda dry_run: "uv-cache skipped (lock contention / timeout)"
    )
    monkeypatch.setattr(asw, "_vm_reclaim_npm_cache", lambda dry_run: "npm-cache rc=0: ok")
    monkeypatch.setattr(
        asw,
        "_vm_reclaim_hf_hub_cache",
        lambda now, dry_run: "hf-hub-ttl: evicted 3 revision(s), freed 20.1G",
    )
    monkeypatch.setattr(asw, "_vm_remediate_worktrees", lambda dry_run: "worktree-audit rc=0: ok")
    monkeypatch.setattr(asw, "_sweep_stale_claude_tmp", lambda now, dry_run: 1)

    asw.vm_disk_pass(dry_run=False, now=1_000_000.0)

    lines = (isolated_registry / "vm-disk-events.jsonl").read_text().strip().splitlines()
    note = json.loads(lines[0])["note"]
    for expected in (
        "wandb-artifacts rc=0: reclaimed",
        "uv-cache skipped (lock contention / timeout)",
        "npm-cache rc=0: ok",
        "hf-hub-ttl: evicted 3 revision(s), freed 20.1G",
        "swept 1 stale /tmp/claude-* tree(s)",
    ):
        assert expected in note


def test_hf_stale_revisions_ttl_selection():
    # Pure selector cut for the HF hub TTL eviction: only revisions that are
    # old (last_modified > TTL), unread (newest blob atime > TTL), AND
    # (detached OR in a repo idle > TTL) qualify. The actively re-downloaded
    # dataset repo, an in-flight download, and a sha-pinned (ref-less)
    # adapter that is still being READ must never be selected.
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    now = 1_000_000_000.0
    old = now - asw.VM_DISK_HF_TTL_S - 60.0
    fresh = now - 60.0

    def rev(commit_hash, refs, mtime, atime):
        # SimpleNamespace is unhashable, so the fakes use tuples where the
        # real HFCacheInfo carries frozensets — the selector only iterates.
        return SimpleNamespace(
            commit_hash=commit_hash,
            refs=refs,
            last_modified=mtime,
            files=(SimpleNamespace(blob_last_accessed=atime),),
        )

    kept_active_refd = rev("a", {"main"}, old, old)
    evict_active_detached = rev("b", set(), old, old)
    kept_active_inflight = rev("c", set(), fresh, fresh)
    kept_active_pinned_read = rev("f", set(), old, fresh)  # sha-pinned, actively read
    active_repo = SimpleNamespace(
        last_accessed=fresh,
        revisions=(
            kept_active_refd,
            evict_active_detached,
            kept_active_inflight,
            kept_active_pinned_read,
        ),
    )

    evict_idle_refd = rev("d", {"main"}, old, old)
    kept_idle_fresh = rev("e", {"main"}, fresh, fresh)
    idle_repo = SimpleNamespace(last_accessed=old, revisions=(evict_idle_refd, kept_idle_fresh))

    cache_info = SimpleNamespace(repos=(active_repo, idle_repo))
    stale = asw._hf_stale_revisions(cache_info, now)

    assert {r.commit_hash for r in stale} == {"b", "d"}


def test_hf_rev_last_accessed_empty_files_falls_back_to_mtime():
    # A revision with no files reads as its last_modified — it never looks
    # fresher than it is.
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    rev = SimpleNamespace(last_modified=123.0, files=())
    assert asw._hf_rev_last_accessed(rev) == 123.0


def test_vm_reclaim_hf_hub_cache_times_out_fail_soft(monkeypatch):
    # The HF scan+evict is the only IN-PROCESS remediation step; a hung
    # scan_cache_dir() must be cut at the wall-clock bound (daemon-thread
    # join) and reported as a fail-soft skip — never a stalled watcher tick.
    import time as _time
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    def slow_scan(*_a, **_k):
        _time.sleep(5.0)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(scan_cache_dir=slow_scan))
    monkeypatch.setattr(asw, "VM_DISK_HF_RECLAIM_TIMEOUT_S", 0.05)

    t0 = _time.monotonic()
    summary = asw._vm_reclaim_hf_hub_cache(now=1_000_000.0, dry_run=False)
    assert _time.monotonic() - t0 < 2.0  # returned at the bound, not after the 5s sleep
    assert "timed out" in summary
    assert "fail-soft" in summary


def test_vm_reclaim_hf_hub_cache_evicts_through_bounded_worker(monkeypatch):
    # Normal path through the bounded worker: scan -> stale cut -> delete
    # strategy executed -> summary carries the count + freed size.
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    executed: list[bool] = []
    strategy = SimpleNamespace(
        expected_freed_size_str="20.1G", execute=lambda: executed.append(True)
    )
    cache_info = SimpleNamespace(delete_revisions=lambda *_hashes: strategy)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(scan_cache_dir=lambda: cache_info)
    )
    monkeypatch.setattr(
        asw, "_hf_stale_revisions", lambda *_a, **_k: [SimpleNamespace(commit_hash="abc")]
    )

    summary = asw._vm_reclaim_hf_hub_cache(now=1_000_000.0, dry_run=False)
    assert executed == [True]
    assert summary == "hf-hub-ttl: evicted 1 revision(s), freed 20.1G"


def test_vm_run_remediations_annotates_per_step_freed_delta(isolated_registry, monkeypatch):
    # A step that actually buys space gets a "(+X.X GiB)" annotation in its
    # note line; steps whose before/after delta sits under the 128 MiB noise
    # floor stay bare.
    import autonomous_session_watch as asw

    free_values = [10 * 2**30, 13 * 2**30]  # step 1: before 10 GiB, after 13 GiB

    def fake_free():
        return free_values.pop(0) if free_values else 13 * 2**30

    monkeypatch.setattr(asw, "_vm_free_bytes", fake_free)
    monkeypatch.setattr(asw, "_vm_reclaim_wandb_cache", lambda dry_run: "wandb-artifacts rc=0: ok")
    monkeypatch.setattr(asw, "_vm_reclaim_uv_cache", lambda dry_run: "uv-cache rc=0: ok")
    monkeypatch.setattr(asw, "_vm_reclaim_npm_cache", lambda dry_run: "npm-cache rc=0: ok")
    monkeypatch.setattr(
        asw, "_vm_reclaim_hf_hub_cache", lambda now, dry_run: "hf-hub-ttl: nothing stale"
    )
    monkeypatch.setattr(asw, "_sweep_stale_claude_tmp", lambda now, dry_run: 0)

    now = 1_000_000.0
    remediation, new_reclaim_ts, _ = asw._vm_run_remediations(
        do_audit=False,
        do_reclaim=True,
        last_reclaim_ts=None,
        last_audit_ts=None,
        now=now,
        dry_run=False,
    )

    assert remediation[0] == "wandb-artifacts rc=0: ok (+3.0 GiB)"
    assert remediation[1] == "uv-cache rc=0: ok"  # zero delta: no annotation
    assert new_reclaim_ts == now


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

    assert prune_cmds == []  # uv prune / npm clean / worktree audit not actually invoked
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


# ─── followups_running parent-waiting-on-open-child exemption (incident #533) ─


def _make_step_completed_event(
    step: str = "10", exit_kind: str = "parked", ts: str = "2026-06-11T13:45:41Z"
) -> dict:
    """Construct a minimal valid epm:step-completed event row (matches the
    shape `scripts/post_step_completed.py` writes — top-level ``step`` and
    ``exit_kind`` fields the helper reads)."""
    return {
        "ts": ts,
        "kind": "epm:step-completed",
        "version": 1,
        "by": "task_state shim",
        "note": (
            f"<!-- epm:step-completed v1 -->\n## Step Completed\n\n"
            f"step: {step}\nexit_kind: {exit_kind}\n"
            f"<!-- /epm:step-completed -->"
        ),
        "step": step,
        "exit_kind": exit_kind,
    }


def test_followups_awaiting_child_reason_fires_on_canonical_533_shape(monkeypatch):
    # Canonical #533 shape (2026-06-11): status=followups_running, no live pod,
    # latest step-completed step=10 exit_kind=parked, one child at
    # awaiting_promotion (user gate). Exemption MUST fire.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw,
        "_task_children",
        lambda issue: [
            {"id": 546, "status": "awaiting_promotion"},
            {"id": 547, "status": "archived"},  # terminal — does NOT count
        ],
    )
    reason = asw._followups_awaiting_child_reason(
        533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
    )
    assert reason is not None
    assert "#546" in reason
    assert "#547" not in reason  # terminal child must NOT be listed
    assert "followups_running" in reason


@pytest.mark.parametrize(
    "status", ["running", "interpreting", "approved", "verifying", "reviewing"]
)
def test_followups_awaiting_child_reason_inert_off_followups_running(monkeypatch, status):
    # ANY non-followups_running ACTIVE status is inert — the exemption is
    # narrowly scoped to the parent-waiting-on-child case (incident #533).
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 1, "status": "awaiting_promotion"}]
    )
    reason = asw._followups_awaiting_child_reason(
        533,
        status=status,
        has_pod=False,
        events=[_make_step_completed_event()],
    )
    assert reason is None


def test_followups_awaiting_child_reason_inert_when_has_pod(monkeypatch):
    # A live pod means a same-issue follow-up round is in flight — keep
    # respawn coverage. Even with all other preconditions met, the
    # exemption MUST decline.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    reason = asw._followups_awaiting_child_reason(
        533,
        status="followups_running",
        has_pod=True,
        events=[_make_step_completed_event()],
    )
    assert reason is None


def test_followups_awaiting_child_reason_inert_when_all_children_terminal(monkeypatch):
    # All children at completed/archived — the parent CAN advance (Step 10
    # will flip it to completed on the next /issue tick). Respawn-eligible.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw,
        "_task_children",
        lambda issue: [
            {"id": 546, "status": "completed"},
            {"id": 547, "status": "archived"},
        ],
    )
    reason = asw._followups_awaiting_child_reason(
        533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
    )
    assert reason is None


def test_followups_awaiting_child_reason_inert_when_no_children(monkeypatch):
    # A followups_running parent with NO children is in a different shape
    # (legitimately re-driving its own follow-up cycle) — never apply the
    # parent-waiting suppression.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_children", lambda issue: [])
    reason = asw._followups_awaiting_child_reason(
        533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
    )
    assert reason is None


@pytest.mark.parametrize(
    ("step", "exit_kind"),
    [
        ("9a-bis", "parked"),  # earlier step — parent still has work
        ("10", "clean"),  # step 10 ran to completion; not a parked wait
        ("4b", "clean"),
    ],
)
def test_followups_awaiting_child_reason_inert_off_step10_parked(monkeypatch, step, exit_kind):
    # Only the step=10 exit_kind=parked shape is the children-wait state.
    # Earlier steps OR a clean step-10 exit do NOT trigger the exemption.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    reason = asw._followups_awaiting_child_reason(
        533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event(step=step, exit_kind=exit_kind)],
    )
    assert reason is None


def test_followups_awaiting_child_reason_inert_when_no_step_completed(monkeypatch):
    # A fresh task with no step-completed markers at all (e.g. a parent
    # whose /issue has never reached Step 10) is not in the parked-wait
    # shape — never suppress.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    reason = asw._followups_awaiting_child_reason(
        533,
        status="followups_running",
        has_pod=False,
        events=[
            {
                "ts": "2026-06-11T10:53:51Z",
                "kind": "epm:merged",
                "note": "branch merged",
            }
        ],
    )
    assert reason is None


def test_followups_awaiting_child_sentinel_in_watcher_filter():
    # The suppression's own alert marker must NEVER reset the staleness
    # clock it is measuring — pin the sentinel into the shared exclusion
    # set, mirroring every other watcher-posted marker.
    from autonomous_session_watch import (
        _FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL,
        _WATCHER_NOTE_SENTINELS,
    )

    assert _FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS


def test_apply_stalled_followups_exemption_rewrites_respawn_to_keep(monkeypatch):
    # The stalled-detector helper: an `action="respawn"` that meets the
    # exemption MUST become `action="keep"` with `new_missed=0`, and the
    # one-time alert MUST be posted on the first call only (dedup'd via
    # `followups_child_alerted`).
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    action, new_missed, child_alerted = asw._apply_stalled_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action, new_missed, child_alerted) == ("keep", 0, True)
    assert len(posted) == 1  # alert posted once
    assert posted[0][0] == 533
    assert "Respawn suppressed" in posted[0][1]
    assert posted[0][2] == "followups-awaiting-child"

    # Second call within the same episode: `followups_child_alerted=True`
    # carried forward — the alert MUST NOT re-post.
    posted.clear()
    action2, new_missed2, child_alerted2 = asw._apply_stalled_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
        action="respawn",
        new_missed=2,
        followups_child_alerted=True,
        dry_run=False,
    )
    assert (action2, new_missed2, child_alerted2) == ("keep", 0, True)
    assert posted == []  # dedup'd


def test_apply_stalled_followups_exemption_no_op_on_healthy_path(monkeypatch):
    # The exemption helper MUST be a no-op when action=="keep" AND
    # new_missed==0 — otherwise the healthy-session hot path would pay
    # `task.py list-children` every tick.
    import autonomous_session_watch as asw

    def _boom(issue):
        raise AssertionError("_task_children must not be called on the healthy path")

    monkeypatch.setattr(asw, "_task_children", _boom)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda *a, **kw: pytest.fail("must not post on healthy path"),
    )
    action, new_missed, child_alerted = asw._apply_stalled_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
        action="keep",
        new_missed=0,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action, new_missed, child_alerted) == ("keep", 0, False)


def test_check_orphan_followups_exemption_rewrites_respawn(monkeypatch):
    # Orphan-sweep helper: action="respawn" + exemption preconditions met
    # becomes action="followups-awaiting-child" + reason string.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    action, reason = asw._check_orphan_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=[_make_step_completed_event()],
        action="respawn",
    )
    assert action == "followups-awaiting-child"
    assert reason is not None
    assert "#546" in reason


def test_check_orphan_followups_exemption_inert_on_non_respawn(monkeypatch):
    # Helper MUST short-circuit when action != "respawn" so the
    # task.py list-children subprocess is not paid on alert / keep / clear
    # branches.
    import autonomous_session_watch as asw

    def _boom(issue):
        raise AssertionError("_task_children must not be called when action != respawn")

    monkeypatch.setattr(asw, "_task_children", _boom)
    for action in ("keep", "clear", "alert"):
        new_action, reason = asw._check_orphan_followups_exemption(
            issue=533,
            status="followups_running",
            has_pod=False,
            events=[_make_step_completed_event()],
            action=action,
        )
        assert new_action == action
        assert reason is None


def test_handle_orphan_followups_awaiting_child_posts_once_and_skips_budget(
    isolated_registry, monkeypatch
):
    # The orphan handler MUST (a) post the one-time alert dedup'd via
    # followups_child_alerted; (b) persist state WITHOUT incrementing
    # respawns_today (the suppression does not consume the daily budget).
    import autonomous_session_watch as asw

    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )

    # First call: alert MUST post; state MUST record followups_child_alerted=True
    # AND respawns_today UNCHANGED (input=0).
    asw._handle_orphan_followups_awaiting_child(
        issue=533,
        reason="followups_running parent waiting on open child(ren) #546",
        followups_child_alerted=False,
        new_missed=2,
        alerted=False,
        respawn_day="2026-06-11",
        respawns_today=0,
        state={},
        dry_run=False,
    )
    assert len(posted) == 1
    assert posted[0][2] == "followups-awaiting-child"
    state = asw._load_orphan_state(533)
    assert state["followups_child_alerted"] is True
    assert state["respawns_today"] == 0  # NOT incremented

    # Second call within the same episode: dedup'd — alert MUST NOT
    # re-post; respawns_today STILL not incremented.
    posted.clear()
    asw._handle_orphan_followups_awaiting_child(
        issue=533,
        reason="followups_running parent waiting on open child(ren) #546",
        followups_child_alerted=True,
        new_missed=3,
        alerted=False,
        respawn_day="2026-06-11",
        respawns_today=0,
        state=state,
        dry_run=False,
    )
    assert posted == []
    state2 = asw._load_orphan_state(533)
    assert state2["respawns_today"] == 0


# ─── round-complete re-park (incident #533 freeze, 2026-06-11→12) ────────────


def _make_followup_scope_event(ts: str = "2026-06-11T09:00:00Z") -> dict:
    """Minimal epm:followup-scope row — marks a same-issue round START."""
    return {
        "ts": ts,
        "kind": "epm:followup-scope",
        "version": 1,
        "note": "followup_label: bare-word-install-step-grid\nsource: user-chat",
    }


def _make_followup_run_event(ts: str = "2026-06-11T10:55:00Z") -> dict:
    """Minimal epm:same-issue-followup-run row — the round's completion
    (idempotency) record, posted AFTER the designed re-park."""
    return {
        "ts": ts,
        "kind": "epm:same-issue-followup-run",
        "version": 1,
        "note": "followup_label: bare-word-install-step-grid\nsource: user-chat\nround: 1",
    }


def test_followup_round_complete_reason_fires_on_533_round_end_shape():
    # The #533 freeze shape: round started (followup-scope), round-end
    # step-completed (9a-bis, parked) NEWER than the scope — the designed
    # re-park never ran. The predicate MUST fire for 9a-bis AND for the
    # step-10 parks the respawned sessions posted.
    import autonomous_session_watch as asw

    for step in ("9a-bis", "10"):
        reason = asw._followup_round_complete_reason(
            [
                _make_followup_scope_event("2026-06-11T09:00:00Z"),
                _make_step_completed_event(step=step, ts="2026-06-11T10:54:12Z"),
            ]
        )
        assert reason is not None, step
        assert "designed re-park" in reason


def test_followup_round_complete_reason_inert_without_scope_marker():
    # No epm:followup-scope on record = the legacy children-in-flight shape
    # (or a plain parent run) — NEVER re-park off step-completed alone.
    import autonomous_session_watch as asw

    assert asw._followup_round_complete_reason([_make_step_completed_event()]) is None
    assert asw._followup_round_complete_reason([]) is None


def test_followup_round_complete_reason_inert_while_round_in_flight():
    # Scope NEWER than every round-end signal = the round is still running
    # (the scope marker resets the clock at each round start). Keep the
    # normal respawn coverage.
    import autonomous_session_watch as asw

    reason = asw._followup_round_complete_reason(
        [
            _make_step_completed_event(step="9a-bis", ts="2026-06-11T08:00:00Z"),
            _make_followup_scope_event("2026-06-11T09:00:00Z"),
        ]
    )
    assert reason is None


def test_followup_round_complete_reason_inert_on_mid_round_park():
    # A mid-round park (e.g. step 2c over-cap plan approval, held in place
    # at followups_running) is NOT round-end — re-parking there would
    # abandon an unapproved round. Same for a clean (non-parked) exit.
    import autonomous_session_watch as asw

    for step, exit_kind in (("2c", "parked"), ("9a-bis", "clean"), ("10", "clean")):
        reason = asw._followup_round_complete_reason(
            [
                _make_followup_scope_event("2026-06-11T09:00:00Z"),
                _make_step_completed_event(
                    step=step, exit_kind=exit_kind, ts="2026-06-11T10:54:12Z"
                ),
            ]
        )
        assert reason is None, (step, exit_kind)


def test_followup_round_complete_reason_inert_on_recorded_round():
    # Mixed-history legacy shape: a properly completed-and-RECORDED past
    # round (scope T1 -> run marker T2 > T1), then the task later returns
    # to followups_running via the legacy children-in-flight transition and
    # posts a children-wait step-10 park (T3 > T2). The recorded round
    # means the re-park already happened (designed step-3 -> step-4
    # ordering) — the predicate MUST stay inert and defer to the
    # awaiting-child suppression, never yank a promoted children-waiting
    # parent to awaiting_promotion. Also self-disarms the predicate after
    # the watcher's own re-park (which posts the run marker itself).
    import autonomous_session_watch as asw

    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_followup_run_event("2026-06-11T10:55:00Z"),
        _make_step_completed_event(step="10", ts="2026-06-12T08:00:00Z"),
    ]
    assert asw._followup_round_complete_reason(events) is None


def test_scope_note_field_parses_latest_scope():
    # _scope_note_field reads `<field>: <value>` lines off the LATEST
    # scope marker's note; missing field / no scope -> None.
    import autonomous_session_watch as asw

    events = [
        {
            "ts": "2026-06-10T09:00:00Z",
            "kind": "epm:followup-scope",
            "note": "followup_label: old-round\nsource: proposer-9b",
        },
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
    ]
    assert asw._scope_note_field(events, "followup_label") == "bare-word-install-step-grid"
    assert asw._scope_note_field(events, "source") == "user-chat"
    assert asw._scope_note_field(events, "gpu_hours_estimate") is None
    assert asw._scope_note_field([], "followup_label") is None


def test_post_followup_run_marker_posts_matching_label(monkeypatch):
    # On a successful re-park the watcher posts the round's completion
    # marker so the scope is RUN for /issue Step 0 routing — label + source
    # parsed from the scope, round = 1 + existing run-marker count.
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    ok = asw._post_followup_run_marker(
        533, [_make_followup_scope_event("2026-06-11T09:00:00Z")], dry_run=False
    )
    assert ok is True
    assert len(calls) == 1
    assert "post-marker" in calls[0]
    assert "epm:same-issue-followup-run" in calls[0]
    note = calls[0][-1]
    assert "followup_label: bare-word-install-step-grid" in note
    assert "source: user-chat" in note
    assert "round: 1" in note


def test_post_followup_run_marker_fails_soft_without_label(monkeypatch):
    # No parseable followup_label -> no marker posted, returns False
    # (fail-soft: the re-park already happened; the failure is logged).
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("must not shell out without a label"),
    )
    scope_no_label = {
        "ts": "2026-06-11T09:00:00Z",
        "kind": "epm:followup-scope",
        "note": "malformed scope note",
    }
    assert asw._post_followup_run_marker(533, [scope_no_label], dry_run=False) is False


def test_repark_completed_followup_round_dry_run_never_mutates(monkeypatch):
    # dry_run classifies only: no subprocess, no marker.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("dry-run must not shell out"),
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda *a, **kw: pytest.fail("dry-run must not post a marker"),
    )
    assert (
        asw._repark_completed_followup_round(
            533, "round complete", [_make_followup_scope_event()], dry_run=True
        )
        is True
    )


def test_repark_completed_followup_round_executes_set_status(monkeypatch):
    # Live mode: shells `task.py set-status <N> awaiting_promotion` from
    # PROJECT_ROOT, then posts the round's epm:same-issue-followup-run
    # completion marker (closing the scope for Step 0 routing) and the
    # sentinel-stamped progress marker.
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    assert (
        asw._repark_completed_followup_round(
            533, "round complete", [_make_followup_scope_event()], dry_run=False
        )
        is True
    )
    assert len(calls) == 2
    assert calls[0][-3:] == ["set-status", "533", "awaiting_promotion"]
    assert "epm:same-issue-followup-run" in calls[1]
    assert "followup_label: bare-word-install-step-grid" in calls[1][-1]
    assert len(posted) == 1
    assert posted[0][0] == 533
    assert asw._FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL in posted[0][1]
    assert posted[0][2] == "followup-round-repark"


def test_repark_completed_followup_round_failure_returns_false(monkeypatch):
    # A failed set-status (rc != 0) returns False and posts NO marker
    # (neither the run marker nor the progress marker) — callers fall back
    # to the pre-existing handling.
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda cmd, **kw: _subprocess.CompletedProcess(cmd, 1, stdout="", stderr="guard refused"),
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda *a, **kw: pytest.fail("must not post a marker on failure"),
    )
    monkeypatch.setattr(
        asw,
        "_post_followup_run_marker",
        lambda *a, **kw: pytest.fail("must not post the run marker on failure"),
    )
    assert (
        asw._repark_completed_followup_round(
            533, "round complete", [_make_followup_scope_event()], dry_run=False
        )
        is False
    )


def test_apply_stalled_followups_exemption_reparks_completed_round(monkeypatch):
    # Stalled pass: a completed round stranded at followups_running is
    # RE-PARKED (action rewritten to keep, no miss accumulation) WITHOUT
    # consulting children — the re-park probe runs before the
    # awaiting-child suppression and short-circuits it.
    import autonomous_session_watch as asw

    def _boom(issue):
        raise AssertionError("_task_children must not be consulted on the re-park path")

    monkeypatch.setattr(asw, "_task_children", _boom)
    reparked: list[tuple[int, str]] = []
    monkeypatch.setattr(
        asw,
        "_repark_completed_followup_round",
        lambda issue, reason, events, dry_run: (reparked.append((issue, reason)), True)[1],
    )
    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
    ]
    action, new_missed, child_alerted = asw._apply_stalled_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=events,
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action, new_missed, child_alerted) == ("keep", 0, False)
    assert len(reparked) == 1
    assert reparked[0][0] == 533


def test_apply_stalled_followups_exemption_falls_back_when_repark_fails(monkeypatch):
    # A FAILED re-park must fall through to the pre-existing handling (here:
    # the awaiting-child suppression, since an open child exists) — never
    # worse than the old behavior.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    monkeypatch.setattr(
        asw, "_repark_completed_followup_round", lambda issue, reason, events, dry_run: False
    )
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_step_completed_event(step="10", ts="2026-06-11T12:45:25Z"),
    ]
    action, new_missed, child_alerted = asw._apply_stalled_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=events,
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action, new_missed, child_alerted) == ("keep", 0, True)
    assert len(posted) == 1  # the awaiting-child alert, not the repark marker
    assert posted[0][2] == "followups-awaiting-child"


def test_check_orphan_followups_exemption_returns_repark_action(monkeypatch):
    # Orphan pass: a completed round stranded at followups_running rewrites
    # respawn -> "followup-round-repark" (mutation deferred to the handler;
    # the probe stays read-only) without consulting children.
    import autonomous_session_watch as asw

    def _boom(issue):
        raise AssertionError("_task_children must not be consulted on the re-park path")

    monkeypatch.setattr(asw, "_task_children", _boom)
    action, reason = asw._check_orphan_followups_exemption(
        issue=533,
        status="followups_running",
        has_pod=False,
        events=[
            _make_followup_scope_event("2026-06-11T09:00:00Z"),
            _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
        ],
        action="respawn",
    )
    assert action == "followup-round-repark"
    assert reason is not None
    assert "designed re-park" in reason


def test_handle_orphan_followup_round_repark_state(isolated_registry, monkeypatch):
    # Orphan handler: success resets the miss counter; failure persists
    # `new_missed` as-is (0 from decide_orphan's respawn decision in
    # production — the pass re-probes and retries once staleness
    # re-accumulates to the respawn action). The daily respawn budget is
    # never consumed.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_repark_completed_followup_round", lambda issue, reason, events, dry_run: True
    )
    asw._handle_orphan_followup_round_repark(
        issue=533,
        reason="round complete",
        events=[_make_followup_scope_event()],
        new_missed=3,
        alerted=False,
        respawn_day="2026-06-12",
        respawns_today=1,
        followups_child_alerted=False,
        state={},
        dry_run=False,
    )
    state = asw._load_orphan_state(533)
    assert state["missed"] == 0
    assert state["respawns_today"] == 1  # NOT incremented

    monkeypatch.setattr(
        asw, "_repark_completed_followup_round", lambda issue, reason, events, dry_run: False
    )
    asw._handle_orphan_followup_round_repark(
        issue=533,
        reason="round complete",
        events=[_make_followup_scope_event()],
        new_missed=0,  # the production value from decide_orphan's ("respawn", 0)
        alerted=False,
        respawn_day="2026-06-12",
        respawns_today=1,
        followups_child_alerted=False,
        state=state,
        dry_run=False,
    )
    state2 = asw._load_orphan_state(533)
    assert state2["missed"] == 0  # persisted as-is; respawn budget untouched
    assert state2["respawns_today"] == 1


def test_followup_round_repark_sentinel_in_watcher_filter():
    # The re-park marker must NEVER reset the staleness clock it is
    # measured against — pin the sentinel into the shared exclusion set.
    from autonomous_session_watch import (
        _FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL,
        _WATCHER_NOTE_SENTINELS,
    )

    assert _FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS


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
# A wrong STOP kills a session the user may still want (hence the
# parked/terminal-status gate + the followup/pod/keep-running skips + the
# 2-miss guard), while a missing stop re-opens the incident class (idle
# sessions pinning worktrees + holding deleted-file handles + ~0.5-0.6GB RSS
# each). Both directions are pinned here.


def test_session_reconcile_done_set_is_pod_auto_stop_set():
    # The DONE set shares the pod-safety auto-stop set: awaiting_promotion /
    # completed / archived (2026-06-10 user request: "stop the happy sessions
    # once they reach awaiting promotion"). followups_running (a same-issue
    # follow-up round is executing) and blocked (under investigation) are
    # excluded — the session may be legitimately live there.
    from autonomous_session_watch import AUTO_STOP_DONE, SESSION_RECONCILE_DONE

    assert SESSION_RECONCILE_DONE == AUTO_STOP_DONE
    assert {"completed", "awaiting_promotion", "archived"} == SESSION_RECONCILE_DONE
    assert "followups_running" not in SESSION_RECONCILE_DONE
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
        "followups_running",
        "blocked",
    ],
)
@pytest.mark.parametrize("idle", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_session_reconcile_non_done_always_clears(status, idle, missed):
    # Any non-parked status (including the follow-up-executing
    # followups_running, the user-parked blocked, and an unreadable None)
    # clears the episode — never an action, even with autostop armed and a
    # huge miss count.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(status, idle, missed, alerted=True, autostop=True) == (
        "clear",
        0,
    )


@pytest.mark.parametrize("status", ["completed", "archived", "awaiting_promotion"])
def test_session_reconcile_fresh_activity_clears(status):
    # A DONE task with recent activity (e.g. it JUST parked) keeps its
    # session — the idle window is the post-park grace period.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(status, False, 5, alerted=True, autostop=True) == ("clear", 0)


def test_session_reconcile_two_miss_guard_then_alert():
    # Alert-only fallback (EPM_SESSION_RECONCILE_AUTOSTOP=0): tick 1
    # accumulates, tick 2 alerts ONCE, later ticks stay quiet (dedup) while
    # the miss count keeps growing so a later autostop re-enable fires
    # immediately.
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
    # A live inline follow-up (a follow-up signal marker newer than the done
    # transition) means the session is the follow-up's driver — never stop
    # it, even if the follow-up itself is quiet past the idle window.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(
        "completed", True, 5, alerted=False, autostop=True, followup_active=True
    ) == ("followup-skip", 0)


def test_session_reconcile_pod_running_skips():
    # A RUNNING managed pod on the issue means work may still be in flight
    # that the markers haven't surfaced yet — skip + reset the miss counter.
    # Precedence: keep_running and followup_active are checked first.
    from autonomous_session_watch import decide_session_reconcile

    assert decide_session_reconcile(
        "awaiting_promotion", True, 5, alerted=False, autostop=True, pod_running=True
    ) == ("pod-skip", 0)
    assert decide_session_reconcile(
        "completed",
        True,
        5,
        alerted=False,
        autostop=True,
        followup_active=True,
        pod_running=True,
    ) == ("followup-skip", 0)


def test_session_reconcile_autostop_default_enabled(monkeypatch):
    # Auto-stop is the DEFAULT (2026-06-10 user request, superseding the
    # same-day alert-only decision). Only an explicit falsy env value
    # disables it; the legacy arming values stay backwards-compatible.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    assert asw._session_reconcile_autostop_enabled() is True
    for off in ("0", "false", "no", " FALSE "):
        monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", off)
        assert asw._session_reconcile_autostop_enabled() is False
    for on in ("1", "true", "yes", ""):
        monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", on)
        assert asw._session_reconcile_autostop_enabled() is True


def test_session_idle_s_env_override(monkeypatch):
    # Default 2h; EPM_SESSION_RECONCILE_IDLE_S overrides; garbled /
    # non-positive values fall back to the default instead of crashing.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_IDLE_S", raising=False)
    assert asw._session_idle_s() == asw.SESSION_IDLE_S == 2 * 3600
    monkeypatch.setenv("EPM_SESSION_RECONCILE_IDLE_S", "7200.5")
    assert asw._session_idle_s() == 7200.5
    for bad in ("garbage", "0", "-5"):
        monkeypatch.setenv("EPM_SESSION_RECONCILE_IDLE_S", bad)
        assert asw._session_idle_s() == asw.SESSION_IDLE_S


def test_session_followup_predicate_expanded_kinds():
    # The session sweep's follow-up inference is wider than pod-safety's:
    # followup-scope / free-analysis-followup-run count as follow-up signals
    # (the request may predate any run-launched), and pod-terminated /
    # step-completed count as done-transitions (a round wrapping up).
    import autonomous_session_watch as asw

    def ev(kind, ts):
        return {"kind": kind, "ts": ts, "note": ""}

    # followup-scope NEWER than the done-transition -> active (the window
    # between a user posting the scope and a session picking it up).
    events = [
        ev("epm:status-changed", "2026-06-10T10:00:00Z"),
        ev("epm:followup-scope", "2026-06-10T11:00:00Z"),
    ]
    assert asw._task_session_followup_active(0, events=events) is True

    # free-analysis-followup-run newer than the transition -> active.
    events = [
        ev("epm:promoted", "2026-06-10T10:00:00Z"),
        ev("epm:free-analysis-followup-run", "2026-06-10T10:30:00Z"),
    ]
    assert asw._task_session_followup_active(0, events=events) is True

    # pod-terminated NEWER than every follow-up signal -> the follow-up
    # provably finished; inactive.
    events = [
        ev("epm:status-changed", "2026-06-10T08:00:00Z"),
        ev("epm:run-launched", "2026-06-10T09:00:00Z"),
        ev("epm:pod-terminated", "2026-06-10T12:00:00Z"),
    ]
    assert asw._task_session_followup_active(0, events=events) is False

    # No follow-up signal at all / no done-transition -> conservative False.
    assert asw._task_session_followup_active(0, events=[]) is False
    assert (
        asw._task_session_followup_active(
            0, events=[ev("epm:run-launched", "2026-06-10T09:00:00Z")]
        )
        is False
    )


def test_latest_nonwatcher_event_ts_counts_any_kind_but_filters_sentinels():
    # The idle clock counts markers of ANY kind (a parked task's
    # followup-scope / interp-critique / workflow-fix markers are all
    # evidence of activity) but never the watcher's own posts.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:interp-critique", "ts": "2026-06-10T10:00:00Z", "note": "round 1"},
        {
            "kind": "epm:progress",
            "ts": "2026-06-10T12:00:00Z",
            "note": f"{asw._SESSION_RECONCILE_ALERT_NOTE_SENTINEL} IDLE session(s) ...",
        },
    ]
    # The non-progress-kind marker counts; the newer watcher alert does not.
    assert asw._latest_nonwatcher_event_ts(events) == asw._parse_event_ts("2026-06-10T10:00:00Z")
    assert asw._latest_nonwatcher_event_ts([]) is None


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


def _patch_session_reconcile_io(
    monkeypatch, *, status, events=None, self_report=(None, None), pods=(), patch_pods=True
):
    """Common monkeypatching for the session-reconcile I/O wrapper tests:
    task reads + the daemon-derived maps + the RunPod snapshot, leaving
    state files + decisions real. Returns the (stops, posts) recorders.
    ``patch_pods=False`` leaves the real :func:`_running_managed_issue_pods`
    in place (for tests exercising its caller-label threading)."""
    import autonomous_session_watch as asw

    stops: list[str] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_events", lambda issue: list(events or []))
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: self_report)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    if patch_pods:
        monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: list(pods))
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    monkeypatch.setattr(asw, "_load_session_issue_map", lambda: {"sid-a": 42, "sid-b": 42})
    monkeypatch.setattr(asw, "_load_session_meta", lambda: {})
    return stops, posts


def test_session_reconcile_alert_only_optout_posts_once_never_stops(isolated_registry, monkeypatch):
    # Opt-out posture (EPM_SESSION_RECONCILE_AUTOSTOP=0): tick 1 accumulates,
    # tick 2 posts ONE alert marker, tick 3 stays quiet. No session is ever
    # stopped.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_RECONCILE_AUTOSTOP", "0")
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


@pytest.mark.parametrize("status", ["completed", "awaiting_promotion"])
def test_session_reconcile_default_autostop_stops_all_sessions_and_clears(
    isolated_registry, monkeypatch, status
):
    # DEFAULT posture (env unset, 2026-06-10 user request): tick 1
    # accumulates, tick 2 stops EVERY live mapped session and posts the stop
    # marker. The state is NOT cleared on the daemon ACK — the ACKed sids are
    # recorded in `stopped_at` and verified actually-gone on the NEXT tick,
    # where the live-session-keyed GC reaps the state (the verified-gone
    # path). awaiting_promotion is covered (the request's headline case:
    # sessions idling at the promotion park).
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    stops, posts = _patch_session_reconcile_io(monkeypatch, status=status)
    now = 1_000_000.0
    live = {"sid-a", "sid-b"}
    state_path = isolated_registry / "session-reconcile-42.json"

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert stops == []

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert sorted(stops) == ["sid-a", "sid-b"]
    assert posts == [(42, "session-reconcile-stop")]
    state = json.loads(state_path.read_text())
    assert sorted(state["stopped_at"]) == ["sid-a", "sid-b"]  # ACK recorded, awaiting verification

    # Tick 3: the daemon actually killed both -> no live mapped session ->
    # the GC reaps the state file. No second stop, no extra marker.
    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=set(), now=now)
    assert not state_path.exists()
    assert sorted(stops) == ["sid-a", "sid-b"]
    assert posts == [(42, "session-reconcile-stop")]


def test_session_reconcile_running_pod_blocks_stop(isolated_registry, monkeypatch):
    # A RUNNING managed pod for the issue blocks the stop and resets the
    # miss counter — even at the default auto-stop posture.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    stops, posts = _patch_session_reconcile_io(
        monkeypatch, status="awaiting_promotion", pods=[(42, "pod-id-x", "pod-42")]
    )
    for _ in range(3):
        asw.session_reconcile_pass(
            False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=1_000_000.0
        )
    assert stops == [] and posts == []
    state = json.loads((isolated_registry / "session-reconcile-42.json").read_text())
    assert state["missed"] == 0  # pod leaving the RUNNING set re-arms a fresh accumulation


def test_session_reconcile_followup_scope_blocks_stop(isolated_registry, monkeypatch):
    # The headline near-miss from the 2026-06-10 manual sweep: a parked task
    # with a follow-up REQUEST (epm:followup-scope newer than the latest
    # done-transition) keeps its session even when the markers are idle past
    # the grace window.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    base = asw._parse_event_ts("2026-06-10T00:00:00Z")
    now = base + 30 * 3600
    events = [
        {"kind": "epm:status-changed", "ts": "2026-06-10T00:00:00Z", "note": "-> parked"},
        {"kind": "epm:followup-scope", "ts": "2026-06-10T01:00:00Z", "note": "user followup"},
    ]
    stops, posts = _patch_session_reconcile_io(
        monkeypatch, status="awaiting_promotion", events=events
    )
    for _ in range(3):
        asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=now)
    assert stops == [] and posts == []


@pytest.mark.parametrize("status", ["followups_running", "blocked", "running"])
def test_session_reconcile_never_acts_on_non_done_status(isolated_registry, monkeypatch, status):
    # followups_running (a same-issue follow-up round is executing), blocked
    # (under investigation), and any ACTIVE status are untouchable — no stop,
    # no marker, no state accumulation, even at the default auto-stop posture.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
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


# ── caller-label attribution on the shared RUNNING-pod helper ─────────────────


def test_running_managed_pods_warning_carries_caller_label(monkeypatch, capsys):
    # The transport-error warning is attributed to the INVOKING pass: the
    # stalled-detector and session-reconcile passes reuse this pod-safety
    # helper, and a `pod-safety:`-prefixed warning from those passes sent
    # cron-log triage to the wrong pass.
    import autonomous_session_watch as asw

    def boom():
        raise RuntimeError("transport down")

    monkeypatch.setattr(asw, "list_team_pods", boom)
    assert asw._running_managed_issue_pods() is None
    assert "pod-safety: list_team_pods failed" in capsys.readouterr().err
    assert asw._running_managed_issue_pods(caller="session-reconcile") is None
    assert "session-reconcile: list_team_pods failed" in capsys.readouterr().err


def test_session_reconcile_pass_threads_caller_label(isolated_registry, monkeypatch, capsys):
    # End-to-end: the session-reconcile pass calls the shared helper with its
    # own caller label, so a transport error during THIS pass is attributed
    # to session-reconcile in the cron log, not to pod-safety.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    _patch_session_reconcile_io(monkeypatch, status="completed", patch_pods=False)

    def boom():
        raise RuntimeError("transport down")

    monkeypatch.setattr(asw, "list_team_pods", boom)
    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=1_000_000.0)
    err = capsys.readouterr().err
    assert "session-reconcile: list_team_pods failed" in err
    assert "pod-safety:" not in err


def test_session_reconcile_failed_pod_snapshot_degrades_to_empty(isolated_registry, monkeypatch):
    # A FAILED pod snapshot (None) degrades to the empty set for session-
    # reconcile — same decision inputs as today's empty-set fallback: the
    # tick still counts the miss (the idle grace + 2-miss guard remain the
    # safety margins) and nothing is stopped or posted on tick 1.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed", patch_pods=False)
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: None)

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=1_000_000.0)

    state_path = isolated_registry / "session-reconcile-42.json"
    assert json.loads(state_path.read_text())["missed"] == 1
    assert stops == [] and posts == []


# ── next-tick stop verification (daemon ACK != kill) ──────────────────────────


def test_session_reconcile_ack_without_kill_retries_once_then_alerts(
    isolated_registry, monkeypatch, capsys
):
    # Alive-after-stop: the daemon ACKs the stop but the session never leaves
    # the live set. The first zombie tick loudly retries the stop ONCE; the
    # next tick posts the one-time stop-failed marker; later ticks stay
    # quiet. The episode state is never cleared while the zombie lives.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed")
    now = 1_000_000.0
    live = {"sid-a"}
    state_path = isolated_registry / "session-reconcile-42.json"

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)  # miss 1
    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)  # stop ACK
    assert stops == ["sid-a"]
    assert posts == [(42, "session-reconcile-stop")]
    capsys.readouterr()  # drain

    # Tick 3: sid-a STILL alive -> loud stderr log + exactly one retry.
    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert stops == ["sid-a", "sid-a"]
    assert "STOP-VERIFY FAILED issue #42" in capsys.readouterr().err
    state = json.loads(state_path.read_text())
    assert state["stop_retried"] is True and state["stop_failed_alerted"] is False

    # Tick 4: STILL alive after the retry -> one-time loud marker, no 3rd stop.
    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert stops == ["sid-a", "sid-a"]
    assert posts[-1] == (42, "session-reconcile-stop-failed")
    state = json.loads(state_path.read_text())
    assert state["stop_failed_alerted"] is True

    # Tick 5: dedup — no new stop, no new marker; state kept for triage.
    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids=live, now=now)
    assert stops == ["sid-a", "sid-a"]
    assert posts.count((42, "session-reconcile-stop-failed")) == 1
    assert state_path.exists()


def test_session_reconcile_state_backcompat_missing_stop_fields(isolated_registry, monkeypatch):
    # A state file written BEFORE the stop-verification fields existed (no
    # stopped_at / stop_retried / stop_failed_alerted keys) must behave like
    # an in-flight pre-upgrade episode: the missing keys read back as
    # empty/false and the normal decision path proceeds unchanged.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_SESSION_RECONCILE_AUTOSTOP", raising=False)
    stops, posts = _patch_session_reconcile_io(monkeypatch, status="completed")
    legacy = {"missed": 1, "alerted": False, "sids": ["sid-a"], "first_seen": 999_000.0}
    (isolated_registry / "session-reconcile-42.json").write_text(json.dumps(legacy))

    asw.session_reconcile_pass(False, 2, daemon_reachable=True, live_ids={"sid-a"}, now=1_000_000.0)
    assert stops == ["sid-a"]  # missed 1 -> 2 hits the threshold; the stop proceeds
    assert posts == [(42, "session-reconcile-stop")]


# ─── zombie-wrapper pass (dead inner Claude; 2026-06-11 zombie sweep) ─────────
#
# 25 finished-issue sessions with NO inner Claude process showed as "running"
# indefinitely because they had lost their issue mapping — invisible to the
# session-reconcile pass. The zombie pass keys on "no Claude process anywhere
# under the daemon-reported wrapper pid", regardless of mapping, with the
# conservative 2-checks + 2h-grace design (a live wrapper revives its inner
# Claude IN PLACE on the next phone message, so a no-Claude snapshot alone can
# be a healthy idle session).


def test_zombie_decide_mapped_active_status_clears():
    # An issue-mapped session at any ACTIVE/blocked/plan_pending (or
    # unreadable) status is out of scope — other passes own those states.
    import autonomous_session_watch as asw

    for status in [*sorted(asw.ZOMBIE_STATUS_EXCLUDE), None]:
        assert asw.decide_zombie_wrapper(status, True, False, 5, 99_999.0, False) == ("clear", 0), (
            status
        )


def test_zombie_decide_exclude_set_covers_required_statuses():
    # The hard-requirement exclusion list, pinned verbatim: running, verifying,
    # interpreting, reviewing, followups_running, blocked, planning,
    # plan_pending, approved.
    import autonomous_session_watch as asw

    required = {
        "running",
        "verifying",
        "interpreting",
        "reviewing",
        "followups_running",
        "blocked",
        "planning",
        "plan_pending",
        "approved",
    }
    assert required == asw.ZOMBIE_STATUS_EXCLUDE


def test_zombie_decide_claude_present_clears():
    # A Claude process anywhere in the wrapper's tree ends the episode — even
    # for unmapped sessions deep into an accumulation.
    import autonomous_session_watch as asw

    assert asw.decide_zombie_wrapper(None, False, True, 5, 99_999.0, True) == ("clear", 0)
    assert asw.decide_zombie_wrapper("completed", True, True, 1, 0.0, False) == ("clear", 0)


def test_zombie_decide_two_miss_guard_and_grace_both_required():
    # Stop needs BOTH >= threshold consecutive misses AND >= grace since the
    # first miss: miss 1 keeps; miss 2 inside the grace window keeps; miss 2
    # past the grace window stops.
    import autonomous_session_watch as asw

    grace = asw.ZOMBIE_WRAPPER_GRACE_S
    assert asw.decide_zombie_wrapper("completed", True, False, 0, 0.0, False) == ("keep", 1)
    assert asw.decide_zombie_wrapper("completed", True, False, 1, grace - 1, False) == ("keep", 2)
    assert asw.decide_zombie_wrapper("completed", True, False, 1, grace + 1, False) == ("stop", 0)
    # Unmapped sessions (the 2026-06-11 zombie class) follow the same ladder,
    # status ignored.
    assert asw.decide_zombie_wrapper(None, False, False, 1, grace + 1, False) == ("stop", 0)


def test_zombie_decide_kill_switch_alerts_once_then_quiet():
    # reap_enabled=False (EPM_ZOMBIE_WRAPPER_REAP=0): one alert per episode,
    # then quiet keeps; the count keeps accumulating so a later re-enable
    # stops on the next tick.
    import autonomous_session_watch as asw

    grace = asw.ZOMBIE_WRAPPER_GRACE_S
    assert asw.decide_zombie_wrapper(
        None, False, False, 1, grace + 1, False, reap_enabled=False
    ) == ("alert", 2)
    assert asw.decide_zombie_wrapper(
        None, False, False, 2, grace + 1, True, reap_enabled=False
    ) == ("keep", 3)
    assert asw.decide_zombie_wrapper(None, False, False, 2, grace + 1, True, reap_enabled=True) == (
        "stop",
        0,
    )


def test_zombie_sentinels_registered_and_filtered():
    # All three zombie sentinels must be in the watcher-note exclusion set so
    # the pass's own markers never reset the staleness clocks they measure.
    import autonomous_session_watch as asw

    for sentinel in (
        asw._ZOMBIE_WRAPPER_STOP_NOTE_SENTINEL,
        asw._ZOMBIE_WRAPPER_ALERT_NOTE_SENTINEL,
        asw._ZOMBIE_WRAPPER_STOP_FAILED_NOTE_SENTINEL,
    ):
        assert sentinel in asw._WATCHER_NOTE_SENTINELS
        events = [{"kind": "epm:progress", "ts": "2026-06-11T10:00:00Z", "note": sentinel + " x"}]
        assert asw._latest_progress_ts(events) is None


# ── zombie-wrapper pass-level (I/O wrapper) tests ─────────────────────────────

# Synthetic EPS repo root for the pass-level session tests. Both patch helpers
# pin asw.PROJECT_ROOT to this path, so the passes' EPS-cwd prefix check and
# the issue inference are cwd-independent. It must NOT end in
# `.claude/worktrees/issue-<N>`: the passes infer an issue from the session
# cwd via spawn_session._WORKTREE_ISSUE_RE, and the previous constant — the
# REAL spawn_session.PROJECT_ROOT — resolves to the issue worktree when the
# suite runs inside one (the /issue Step 9c test gate), which mapped the
# "unmapped" fake sessions to a live task whose excluded/unreadable status
# flipped the decision to "clear" (task #580 incident, 2026-06-12).
_Z_ROOT = "/synthetic-eps-checkout/explore-persona-space"


def _patch_zombie_io(
    monkeypatch,
    *,
    children,
    meta,
    status=None,
    has_claude=False,
    registry=None,
    pm_sids=frozenset(),
):
    """Common monkeypatching for the zombie-wrapper I/O tests: daemon children
    + session metadata + task status + the /proc walk, leaving state files and
    decisions real. Pins asw.PROJECT_ROOT to the synthetic _Z_ROOT so the
    EPS-cwd check + issue inference are cwd-independent (see _Z_ROOT).
    Returns the (stops, posts, fallback) recorders."""
    import autonomous_session_watch as asw

    stops: list[str] = []
    posts: list[tuple[int, str]] = []
    fallback: list[str] = []
    monkeypatch.setattr(asw, "PROJECT_ROOT", Path(_Z_ROOT))
    monkeypatch.setattr(asw, "_live_children", lambda: list(children))
    monkeypatch.setattr(asw, "_load_session_meta", lambda: dict(meta))
    monkeypatch.setattr(asw, "_load_session_issue_map", lambda: dict(registry or {}))
    monkeypatch.setattr(asw, "_load_pm_session_ids", lambda: set(pm_sids))
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_proc_children_map", lambda: {})
    monkeypatch.setattr(asw, "_has_claude_descendant", lambda pid, cm=None: has_claude)
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    monkeypatch.setattr(
        asw, "_append_zombie_fallback_event", lambda note, dry_run: fallback.append(note)
    )
    return stops, posts, fallback


def test_zombie_pass_stop_fires_after_threshold_and_grace(isolated_registry, monkeypatch):
    # The headline behavior: an unmapped repo-root EPS session with no Claude
    # descendant accumulates a miss on tick 1, and is stopped on tick 2 once
    # the grace window has also elapsed. The record lands in the fallback
    # events file (no issue to carry a marker).
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_ZOMBIE_WRAPPER_REAP", raising=False)
    children = [{"happySessionId": "sid-z", "pid": 4242}]
    meta = {"sid-z": {"path": _Z_ROOT}}
    stops, posts, fallback = _patch_zombie_io(monkeypatch, children=children, meta=meta)
    state_path = isolated_registry / "zombie-wrapper-sid-z.json"
    t0 = 1_000_000.0

    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=t0)
    state = json.loads(state_path.read_text())
    assert state["missed"] == 1 and state["first_miss_ts"] == t0
    assert stops == [] and fallback == []

    t1 = t0 + asw.ZOMBIE_WRAPPER_GRACE_S + 60
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=t1)
    assert stops == ["sid-z"]
    assert len(fallback) == 1 and posts == []  # unmapped -> fallback, not a marker
    state = json.loads(state_path.read_text())
    assert state["stopped_at"] == t1  # ACK recorded for next-tick verification


def test_zombie_pass_mapped_done_task_posts_marker(isolated_registry, monkeypatch):
    # A worktree-cwd session (issue inferred) at a DONE status gets the same
    # ladder, with the stop recorded as a marker on the issue.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_ZOMBIE_WRAPPER_REAP", raising=False)
    children = [{"happySessionId": "sid-w", "pid": 77}]
    meta = {"sid-w": {"path": f"{_Z_ROOT}/.claude/worktrees/issue-99"}}
    stops, posts, fallback = _patch_zombie_io(
        monkeypatch, children=children, meta=meta, status="awaiting_promotion"
    )
    t0 = 1_000_000.0
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=t0)
    asw.zombie_wrapper_pass(
        False, 2, daemon_reachable=True, now=t0 + asw.ZOMBIE_WRAPPER_GRACE_S + 60
    )
    assert stops == ["sid-w"]
    assert posts == [(99, "zombie-wrapper-stop")] and fallback == []


def test_zombie_pass_claude_present_clears_state(isolated_registry, monkeypatch):
    # A session whose tree has a Claude process clears any accumulated state.
    import json

    import autonomous_session_watch as asw

    children = [{"happySessionId": "sid-h", "pid": 11}]
    meta = {"sid-h": {"path": _Z_ROOT}}
    stops, posts, fallback = _patch_zombie_io(
        monkeypatch, children=children, meta=meta, has_claude=True
    )
    state_path = isolated_registry / "zombie-wrapper-sid-h.json"
    state_path.write_text(json.dumps({"missed": 1, "alerted": False, "first_miss_ts": 999_000.0}))
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=1_000_000.0)
    assert not state_path.exists()
    assert stops == [] and posts == [] and fallback == []


def test_zombie_pass_pm_and_non_eps_sessions_never_touched(isolated_registry, monkeypatch):
    # PM-registered sids and non-EPS cwds are skipped before any state is
    # even created — they can never accumulate toward a stop.
    import autonomous_session_watch as asw

    children = [
        {"happySessionId": "sid-pm", "pid": 1},
        {"happySessionId": "sid-other", "pid": 2},
        {"happySessionId": "sid-nometa", "pid": 3},
    ]
    meta = {
        "sid-pm": {"path": _Z_ROOT},
        "sid-other": {"path": "/home/thomasjiralerspong/my-goat"},
        # sid-nometa: no metadata at all -> EPS-ness unknown -> skipped
    }
    stops, posts, fallback = _patch_zombie_io(
        monkeypatch, children=children, meta=meta, pm_sids={"sid-pm"}
    )
    t0 = 1_000_000.0
    for now in (t0, t0 + asw.ZOMBIE_WRAPPER_GRACE_S + 60):
        asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and posts == [] and fallback == []
    assert not list(isolated_registry.glob("zombie-wrapper-*.json"))


def test_zombie_pass_mapped_active_status_excluded(isolated_registry, monkeypatch):
    # A registry-mapped session whose task is ACTIVE is never stopped, even
    # with no Claude descendant for far longer than the grace window.
    import autonomous_session_watch as asw

    children = [{"happySessionId": "sid-a", "pid": 5}]
    meta = {"sid-a": {"path": _Z_ROOT}}
    stops, posts, fallback = _patch_zombie_io(
        monkeypatch, children=children, meta=meta, status="running", registry={"sid-a": 7}
    )
    t0 = 1_000_000.0
    for now in (t0, t0 + 10 * asw.ZOMBIE_WRAPPER_GRACE_S):
        asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and posts == [] and fallback == []


def test_zombie_pass_kill_switch_alert_only(isolated_registry, monkeypatch):
    # EPM_ZOMBIE_WRAPPER_REAP=0: one alert per episode, never a stop.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_ZOMBIE_WRAPPER_REAP", "0")
    children = [{"happySessionId": "sid-k", "pid": 9}]
    meta = {"sid-k": {"path": f"{_Z_ROOT}/.claude/worktrees/issue-55"}}
    stops, posts, _fallback = _patch_zombie_io(
        monkeypatch, children=children, meta=meta, status="completed"
    )
    t0 = 1_000_000.0
    later = t0 + asw.ZOMBIE_WRAPPER_GRACE_S + 60
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=t0)
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later)
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later + 600)
    assert stops == []
    assert posts == [(55, "zombie-wrapper-alert")]  # exactly one per episode


def test_zombie_pass_stop_verification_retry_then_alert(isolated_registry, monkeypatch, capsys):
    # ACK != kill: a session still live after its ACKed stop gets ONE retry,
    # then ONE loud record, then quiet — the state is kept for triage and
    # reaped only when the session actually leaves the live set.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_ZOMBIE_WRAPPER_REAP", raising=False)
    children = [{"happySessionId": "sid-v", "pid": 13}]
    meta = {"sid-v": {"path": _Z_ROOT}}
    stops, _posts, fallback = _patch_zombie_io(monkeypatch, children=children, meta=meta)
    state_path = isolated_registry / "zombie-wrapper-sid-v.json"
    t0 = 1_000_000.0
    later = t0 + asw.ZOMBIE_WRAPPER_GRACE_S + 60

    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=t0)  # miss 1
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later)  # stop ACK
    assert stops == ["sid-v"] and len(fallback) == 1
    capsys.readouterr()

    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later + 600)  # retry
    assert stops == ["sid-v", "sid-v"]
    assert "ZOMBIE STOP-VERIFY FAILED" in capsys.readouterr().err

    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later + 1200)  # loud record
    assert stops == ["sid-v", "sid-v"]
    assert len(fallback) == 2  # stop + stop-failed records

    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later + 1800)  # quiet
    assert stops == ["sid-v", "sid-v"] and len(fallback) == 2
    assert state_path.exists()

    # The session finally dies -> the live-session-keyed GC reaps the state.
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=True, now=later + 2400)
    assert not state_path.exists()


def test_zombie_pass_daemon_unreachable_skips(isolated_registry, monkeypatch):
    # Daemon-gated: liveness + the stop RPC both need the daemon.
    import autonomous_session_watch as asw

    stops, posts, fallback = _patch_zombie_io(
        monkeypatch, children=[{"happySessionId": "sid-x", "pid": 1}], meta={}
    )
    asw.zombie_wrapper_pass(False, 2, daemon_reachable=False, now=1_000_000.0)
    assert stops == [] and posts == [] and fallback == []
    assert not list(isolated_registry.glob("zombie-wrapper-*.json"))


def test_pm_session_registry_roundtrip_dedup_and_cap(isolated_registry):
    # spawn-pm / register-pm append to pm-session.json: deduped, newest last,
    # bounded — and the watcher-facing loader returns the set.
    _ = isolated_registry  # patches AUTONOMOUS_REGISTRY_DIR in both modules

    spawn_session._register_pm_session("sid-1")
    spawn_session._register_pm_session("sid-2")
    spawn_session._register_pm_session("sid-1")  # re-register moves to newest, no dup
    assert spawn_session._load_pm_session_ids_ordered() == ["sid-2", "sid-1"]
    assert spawn_session._load_pm_session_ids() == {"sid-1", "sid-2"}
    for i in range(30):
        spawn_session._register_pm_session(f"gen-{i}")
    ordered = spawn_session._load_pm_session_ids_ordered()
    assert len(ordered) == spawn_session._PM_SESSION_MAX_IDS
    assert ordered[-1] == "gen-29"


def test_pm_session_loader_empty_on_missing_or_garbled(isolated_registry):
    # A missing or garbled registry must degrade to "no PM exclusion", never
    # crash the watcher pass that consumes it.
    assert spawn_session._load_pm_session_ids() == set()
    (isolated_registry / spawn_session.PM_SESSION_BASENAME).write_text("not json")
    assert spawn_session._load_pm_session_ids() == set()


def test_zombie_pass_dry_run_mutates_nothing(isolated_registry, monkeypatch):
    # Dry-run discipline: with an episode seeded AT the stop point
    # (threshold met, grace elapsed), a dry-run tick must not stop, must not
    # record anywhere, and must leave the state file byte-for-byte untouched.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_ZOMBIE_WRAPPER_REAP", raising=False)
    children = [{"happySessionId": "sid-d", "pid": 21}]
    meta = {"sid-d": {"path": _Z_ROOT}}
    stops, posts, fallback = _patch_zombie_io(monkeypatch, children=children, meta=meta)
    # The shared fake _stop_session ignores dry_run; this test pins dry-run
    # discipline, so mirror the REAL helper's contract (returns False without
    # acting when dry_run=True).
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: (not dry_run) and (stops.append(sid) or True)
    )
    t0 = 1_000_000.0
    state_path = isolated_registry / "zombie-wrapper-sid-d.json"
    seeded = json.dumps({"missed": 1, "alerted": False, "first_miss_ts": t0})
    state_path.write_text(seeded)

    later = t0 + asw.ZOMBIE_WRAPPER_GRACE_S + 60
    asw.zombie_wrapper_pass(True, 2, daemon_reachable=True, now=later)
    assert stops == [] and posts == [] and fallback == []
    assert state_path.read_text() == seeded  # untouched, not even rewritten

    # _stop_session itself honours dry_run (returns False without stopping),
    # so even the real helper could not have acted; here we additionally pin
    # that the pass never persisted a stopped_at / incremented miss count.
    state = json.loads(state_path.read_text())
    assert "stopped_at" not in state


# ─── idle-unmapped-session pass (live-but-idle Claude; 2026-06-12 VM lag) ─────
#
# 25 unmapped sessions sat idle 19-43h each with a LIVE inner Claude plus ~8
# MCP children (~23 GB RSS total). The zombie pass needs a DEAD inner Claude
# and the session-reconcile pass needs an issue mapping, so this class was
# structurally invisible to both. The idle-unmapped pass keys on the resolved
# Claude transcript's mtime, with hard never-touch guards (PM, non-EPS,
# mapped, controlling TTY, unresolvable signal) all pinned below.


def test_idle_unmapped_decide_mapped_or_tty_clears():
    # Issue-mapped sessions belong to the reconcile/zombie passes; a TTY
    # means a terminal Thomas may be sitting at. Both end the episode, even
    # deep into an accumulation.
    import autonomous_session_watch as asw

    over = asw.UNMAPPED_IDLE_REAP_S + 60
    assert asw.decide_idle_unmapped(True, False, over, 5, True) == ("clear", 0)
    assert asw.decide_idle_unmapped(False, True, over, 5, True) == ("clear", 0)
    assert asw.decide_idle_unmapped(True, True, None, 3, False) == ("clear", 0)


def test_idle_unmapped_decide_missing_signal_skips_frozen():
    # The fail-toward-keep contract: an unavailable idleness signal neither
    # accumulates toward a stop NOR erases a real episode — the count is
    # FROZEN exactly as it was.
    import autonomous_session_watch as asw

    assert asw.decide_idle_unmapped(False, False, None, 0, False) == ("skip", 0)
    assert asw.decide_idle_unmapped(False, False, None, 1, False) == ("skip", 1)
    assert asw.decide_idle_unmapped(False, False, None, 5, True) == ("skip", 5)


def test_idle_unmapped_decide_recent_activity_clears():
    # Any transcript write inside the reap window ends the episode.
    import autonomous_session_watch as asw

    window = asw.UNMAPPED_IDLE_REAP_S
    assert asw.decide_idle_unmapped(False, False, 0.0, 1, False) == ("clear", 0)
    assert asw.decide_idle_unmapped(False, False, window - 1, 1, False) == ("clear", 0)


def test_idle_unmapped_decide_two_miss_guard():
    # Stop needs >= threshold consecutive over-window checks: check 1 keeps,
    # check 2 stops (at the default threshold of 2).
    import autonomous_session_watch as asw

    over = asw.UNMAPPED_IDLE_REAP_S + 60
    assert asw.decide_idle_unmapped(False, False, over, 0, False) == ("keep", 1)
    assert asw.decide_idle_unmapped(False, False, over, 1, False) == ("stop", 0)
    # A custom window threads through.
    assert asw.decide_idle_unmapped(False, False, 100.0, 1, False, idle_reap_s=50.0) == (
        "stop",
        0,
    )
    assert asw.decide_idle_unmapped(False, False, 100.0, 1, False, idle_reap_s=200.0) == (
        "clear",
        0,
    )


def test_idle_unmapped_decide_kill_switch_alerts_once_then_quiet():
    # reap_enabled=False (EPM_UNMAPPED_IDLE_REAP=0): one alert per episode,
    # then quiet keeps; the count keeps accumulating so a later re-enable
    # stops on the next tick.
    import autonomous_session_watch as asw

    over = asw.UNMAPPED_IDLE_REAP_S + 60
    assert asw.decide_idle_unmapped(False, False, over, 1, False, reap_enabled=False) == (
        "alert",
        2,
    )
    assert asw.decide_idle_unmapped(False, False, over, 2, True, reap_enabled=False) == (
        "keep",
        3,
    )
    assert asw.decide_idle_unmapped(False, False, over, 2, True, reap_enabled=True) == (
        "stop",
        0,
    )


def test_idle_unmapped_env_helpers(monkeypatch):
    # EPM_UNMAPPED_IDLE_REAP_S: positive number wins; garbled / non-positive
    # falls back. EPM_UNMAPPED_IDLE_REAP: only explicit falsy disables.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    assert asw._unmapped_idle_reap_s() == asw.UNMAPPED_IDLE_REAP_S
    monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP_S", "3600")
    assert asw._unmapped_idle_reap_s() == 3600.0
    monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP_S", "garbled")
    assert asw._unmapped_idle_reap_s() == asw.UNMAPPED_IDLE_REAP_S
    monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP_S", "-5")
    assert asw._unmapped_idle_reap_s() == asw.UNMAPPED_IDLE_REAP_S

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    assert asw._unmapped_idle_reap_enabled() is True
    for falsy in ("0", "false", "no", " FALSE "):
        monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP", falsy)
        assert asw._unmapped_idle_reap_enabled() is False, falsy
    monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP", "1")
    assert asw._unmapped_idle_reap_enabled() is True


def test_idle_unmapped_sentinels_registered_and_filtered():
    # All three idle-unmapped sentinels must be in the watcher-note exclusion
    # set so a hypothetical task-carried note never resets a staleness clock.
    import autonomous_session_watch as asw

    for sentinel in (
        asw._IDLE_UNMAPPED_STOP_NOTE_SENTINEL,
        asw._IDLE_UNMAPPED_ALERT_NOTE_SENTINEL,
        asw._IDLE_UNMAPPED_STOP_FAILED_NOTE_SENTINEL,
    ):
        assert sentinel in asw._WATCHER_NOTE_SENTINELS
        events = [{"kind": "epm:progress", "ts": "2026-06-12T10:00:00Z", "note": sentinel + " x"}]
        assert asw._latest_progress_ts(events) is None


# ── idle-unmapped pass-level (I/O wrapper) tests ──────────────────────────────


def _patch_idle_io(
    monkeypatch,
    *,
    children,
    meta,
    idle_age=None,
    signal_reason="transcript unresolvable",
    has_tty=False,
    registry=None,
    pm_sids=frozenset(),
):
    """Common monkeypatching for the idle-unmapped I/O tests: daemon children
    + session metadata + the TTY probe + the transcript-idle signal, leaving
    state files and decisions real. Pins asw.PROJECT_ROOT to the synthetic
    _Z_ROOT so the EPS-cwd check + issue inference are cwd-independent (see
    _Z_ROOT). Returns the (stops, records) recorders."""
    import autonomous_session_watch as asw

    stops: list[str] = []
    records: list[str] = []
    monkeypatch.setattr(asw, "PROJECT_ROOT", Path(_Z_ROOT))
    monkeypatch.setattr(asw, "_live_children", lambda: list(children))
    monkeypatch.setattr(asw, "_load_session_meta", lambda: dict(meta))
    monkeypatch.setattr(asw, "_load_session_issue_map", lambda: dict(registry or {}))
    monkeypatch.setattr(asw, "_load_pm_session_ids", lambda: set(pm_sids))
    monkeypatch.setattr(asw, "_wrapper_has_controlling_tty", lambda pid: has_tty)
    monkeypatch.setattr(
        asw,
        "_transcript_idle_age_s",
        lambda pid, now: (idle_age, None if idle_age is not None else signal_reason),
    )
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or True)
    monkeypatch.setattr(
        asw, "_append_idle_unmapped_event", lambda note, dry_run: records.append(note)
    )
    return stops, records


def test_idle_unmapped_pass_stop_fires_after_threshold(isolated_registry, monkeypatch):
    # The headline behavior: an unmapped repo-root EPS session over the idle
    # window accumulates a miss on tick 1 and is stopped on tick 2. The
    # record lands in the fallback events file (no issue to carry a marker).
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    children = [{"happySessionId": "sid-i", "pid": 4242}]
    meta = {"sid-i": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=over)
    state_path = isolated_registry / "idle-unmapped-sid-i.json"
    t0 = 1_000_000.0

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)
    state = json.loads(state_path.read_text())
    assert state["missed"] == 1 and state["first_over_ts"] == t0
    assert stops == [] and records == []

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == ["sid-i"]
    assert len(records) == 1 and "auto-stopped idle unmapped" in records[0]
    state = json.loads(state_path.read_text())
    assert state["stopped_at"] == t0 + 600  # ACK recorded for next-tick verification


def test_idle_unmapped_pass_never_touch_set(isolated_registry, monkeypatch):
    # PM-registered sids, non-EPS cwds, no-metadata sids, registry-mapped
    # sids, and worktree-cwd-inferred sids are all out of scope. The mapped
    # ones get their stale state cleared rather than accumulated.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    children = [
        {"happySessionId": "sid-pm", "pid": 1},
        {"happySessionId": "sid-other", "pid": 2},
        {"happySessionId": "sid-nometa", "pid": 3},
        {"happySessionId": "sid-reg", "pid": 4},
        {"happySessionId": "sid-wt", "pid": 5},
    ]
    meta = {
        "sid-pm": {"path": _Z_ROOT},
        "sid-other": {"path": "/home/thomasjiralerspong/my-goat"},
        # sid-nometa: no metadata at all -> EPS-ness unknown -> skipped
        "sid-reg": {"path": _Z_ROOT},
        "sid-wt": {"path": f"{_Z_ROOT}/.claude/worktrees/issue-99"},
    }
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=over,
        registry={"sid-reg": 7},
        pm_sids={"sid-pm"},
    )
    # Seed stale state for the registry-mapped session: the pass must CLEAR
    # it (the session left scope), never accumulate it.
    stale = isolated_registry / "idle-unmapped-sid-reg.json"
    stale.write_text(json.dumps({"missed": 1, "alerted": False, "first_over_ts": 999_000.0}))
    t0 = 1_000_000.0
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert not stale.exists()
    assert not list(isolated_registry.glob("idle-unmapped-*.json"))


def test_idle_unmapped_pass_tty_session_never_touched(isolated_registry, monkeypatch):
    # A wrapper holding a controlling TTY (terminal-run session) clears any
    # accumulated state and is never stopped, however idle the transcript.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    children = [{"happySessionId": "sid-t", "pid": 11}]
    meta = {"sid-t": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch, children=children, meta=meta, idle_age=over, has_tty=True
    )
    state_path = isolated_registry / "idle-unmapped-sid-t.json"
    state_path.write_text(json.dumps({"missed": 1, "alerted": False, "first_over_ts": 999_000.0}))
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=1_000_000.0)
    assert not state_path.exists()
    assert stops == [] and records == []


def test_idle_unmapped_pass_missing_signal_fails_toward_keep(
    isolated_registry, monkeypatch, capsys
):
    # The resolver miss: never accumulates, never stops, logs loudly, and
    # leaves a pre-existing episode FROZEN (not erased) so a flapping
    # resolver can't reset a real episode.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    children = [{"happySessionId": "sid-m", "pid": 13}]
    meta = {"sid-m": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch, children=children, meta=meta, idle_age=None, signal_reason="no happy log"
    )
    state_path = isolated_registry / "idle-unmapped-sid-m.json"
    seeded = json.dumps({"missed": 1, "alerted": False, "first_over_ts": 999_000.0})
    state_path.write_text(seeded)
    t0 = 1_000_000.0
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert state_path.read_text() == seeded  # frozen, not erased or grown
    assert "failing toward KEEP" in capsys.readouterr().err


def test_idle_unmapped_pass_kill_switch_alert_only(isolated_registry, monkeypatch):
    # EPM_UNMAPPED_IDLE_REAP=0: one alert record per episode, never a stop.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP", "0")
    children = [{"happySessionId": "sid-k", "pid": 9}]
    meta = {"sid-k": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=over)
    t0 = 1_000_000.0
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == []
    assert len(records) == 1 and "NOT auto-stopped" in records[0]


def test_idle_unmapped_pass_stop_verification_retry_then_alert(
    isolated_registry, monkeypatch, capsys
):
    # ACK != kill: a session still live after its ACKed stop gets ONE retry,
    # then ONE loud record, then quiet — the state is reaped only when the
    # session actually leaves the live set.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    children = [{"happySessionId": "sid-v", "pid": 13}]
    meta = {"sid-v": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=over)
    state_path = isolated_registry / "idle-unmapped-sid-v.json"
    t0 = 1_000_000.0

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)  # miss 1
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)  # stop ACK
    assert stops == ["sid-v"] and len(records) == 1
    capsys.readouterr()

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 1200)  # retry
    assert stops == ["sid-v", "sid-v"]
    assert "IDLE-UNMAPPED STOP-VERIFY FAILED" in capsys.readouterr().err

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 1800)  # loud record
    assert stops == ["sid-v", "sid-v"]
    assert len(records) == 2  # stop + stop-failed records

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 2400)  # quiet
    assert stops == ["sid-v", "sid-v"] and len(records) == 2
    assert state_path.exists()

    # The session finally dies -> the live-session-keyed GC reaps the state.
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 3000)
    assert not state_path.exists()


def test_idle_unmapped_pass_dry_run_mutates_nothing(isolated_registry, monkeypatch):
    # Dry-run discipline: with an episode seeded AT the stop point, a dry-run
    # tick must not stop, must not record, and must leave the state file
    # byte-for-byte untouched.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    children = [{"happySessionId": "sid-d", "pid": 21}]
    meta = {"sid-d": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=over)
    # Mirror the REAL _stop_session contract (returns False without acting
    # when dry_run=True).
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: (not dry_run) and (stops.append(sid) or True)
    )
    t0 = 1_000_000.0
    state_path = isolated_registry / "idle-unmapped-sid-d.json"
    seeded = json.dumps({"missed": 1, "alerted": False, "first_over_ts": t0})
    state_path.write_text(seeded)

    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == [] and records == []
    assert state_path.read_text() == seeded  # untouched, not even rewritten


def test_idle_unmapped_pass_daemon_unreachable_skips(isolated_registry, monkeypatch):
    # Daemon-gated: liveness + the stop RPC both need the daemon.
    import autonomous_session_watch as asw

    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=[{"happySessionId": "sid-x", "pid": 1}],
        meta={"sid-x": {"path": _Z_ROOT}},
        idle_age=over,
    )
    asw.idle_unmapped_pass(False, 2, daemon_reachable=False, now=1_000_000.0)
    assert stops == [] and records == []
    assert not list(isolated_registry.glob("idle-unmapped-*.json"))


def test_idle_unmapped_transcript_signal_is_happy_log_only(tmp_path, monkeypatch):
    # The idleness signal uses ONLY session_resolver's per-pid happy-log path.
    # The shared-projects-dir filesystem fallback can attribute ANOTHER
    # session's OLDER transcript (a WRONG signal, not a missing one) and is
    # never consulted: a happy-log miss returns (None, reason) -> skip/keep.
    import os

    import autonomous_session_watch as asw
    import session_resolver

    def _boom(*_a, **_k):  # the fallback-bearing resolver must not be called
        raise AssertionError("resolve_transcript (with fs fallback) must not be used")

    monkeypatch.setattr(session_resolver, "resolve_transcript", _boom)

    monkeypatch.setattr(
        session_resolver,
        "_resolve_transcript_via_happy_log",
        lambda pid: (None, "no happy log file for this node pid"),
    )
    age, reason = asw._transcript_idle_age_s(123, now=1_000_000.0)
    assert age is None and "no happy log" in reason

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n")
    os.utime(transcript, (999_000.0, 999_000.0))
    monkeypatch.setattr(
        session_resolver,
        "_resolve_transcript_via_happy_log",
        lambda pid: (str(transcript), None),
    )
    age, reason = asw._transcript_idle_age_s(123, now=1_000_000.0)
    assert reason is None and age == 1_000.0


def test_gc_pass_never_touches_session_reaper_state_files(isolated_registry, monkeypatch):
    # The generic per-issue GC must not reap the per-SESSION state files of
    # the zombie-wrapper and idle-unmapped passes (sid stems are non-int and
    # their prefixes are not in _GC_TARGETS) — those are owned by each pass's
    # live-session-keyed GC. Reaping them here would reset miss counters
    # every tick and the thresholds could never be reached.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    payload = json.dumps({"missed": 1, "alerted": False, "first_over_ts": 0.0})
    zombie = isolated_registry / "zombie-wrapper-sid-abc.json"
    idle = isolated_registry / "idle-unmapped-sid-abc.json"
    zombie.write_text(payload)
    idle.write_text(payload)

    asw.gc_pass(False, now=10 * asw.MAX_ENTRY_AGE_S)

    assert zombie.exists() and idle.exists()


# ═══ infra-drain pass (execute the PM-adjudicated dispatch queue; #633) ══════
#
# Pure-decision tests run against decide_infra_drain / parse_infra_drain_queue
# with zero filesystem/subprocess; executor tests use isolated_registry +
# monkeypatch recorder stubs for _dispatch_infra_drain / _task_status_kind /
# _infra_drain_occupancy / _post_progress_marker / _live_session_ids.

_DRAIN_NOW = 1_800_000_000.0  # fixed epoch (2027-01); the live-file-shaped
# updated_ts below parses to mid-2026, safely in the past (never clamped).


def _decide_drain(
    ids,
    *,
    cap=3,
    holds=None,
    statuses=None,
    kinds=None,
    registered=None,
    occupied=0,
    pending=0,
    attempts=None,
    now=_DRAIN_NOW,
    updated_ts=None,
    backoff_s=INFRA_DRAIN_BACKOFF_S_DEFAULT,
    max_attempts=INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT,
):
    """decide_infra_drain with eligible-by-default fixtures: every ID is
    proposed/infra unless the test overrides the signal under test."""
    statuses = statuses if statuses is not None else {i: "proposed" for i in ids}
    kinds = kinds if kinds is not None else {i: "infra" for i in ids}
    return decide_infra_drain(
        ids,
        cap,
        holds or {},
        statuses,
        kinds,
        registered or set(),
        occupied,
        pending,
        attempts or {},
        now,
        updated_ts,
        backoff_s=backoff_s,
        max_attempts=max_attempts,
    )


def _write_drain_queue(reg_dir, ids, *, cap=3, holds=None, updated_ts="2026-06-12T22:40:00Z"):
    import json

    payload = {
        "ripe_oldest_first": ids,
        "cap": cap,
        "holds": holds or {},
        "updated_ts": updated_ts,
        "updated_by": "pm-session-drain-tick",
        "comment": "test fixture",
    }
    (reg_dir / "infra-drain-queue.json").write_text(json.dumps(payload))


def _stub_drain_executor(
    monkeypatch, *, status_kind=None, occupancy=None, live=None, real_dispatch=False
):
    """Stub every task.py/daemon-backed signal the executor consumes, and
    return the dispatch + marker recorders. ``live`` feeds the drain path's
    ``_live_session_ids_or_none`` (``None`` here means "daemon up, zero
    sessions" — pass-through tests that want the UNAVAILABLE shape patch the
    wrapper themselves). ``real_dispatch=True`` keeps the REAL
    ``_dispatch_infra_drain`` (incl. its pre-spawn registration re-check) and
    stubs only ``subprocess.run`` — pinning the dispatch path end-to-end
    (round-1 Critical: a stubbed dispatch masked the re-check aborting every
    stale-registration re-dispatch)."""
    import autonomous_session_watch as asw

    sk = status_kind or {}
    monkeypatch.setattr(asw, "_task_status_kind", lambda i: sk.get(i, (None, None)))
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: occupancy)
    monkeypatch.setattr(
        asw, "_live_session_ids_or_none", lambda: live if live is not None else set()
    )
    dispatched: list[int] = []
    if real_dispatch:
        from types import SimpleNamespace

        def _fake_run(cmd, **kw):
            assert cmd[3] == "scripts/spawn_session.py" and "spawn-issue" in cmd
            dispatched.append(int(cmd[cmd.index("--issue") + 1]))
            return SimpleNamespace(returncode=0, stdout="spawned sid-new\n", stderr="")

        monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    else:
        monkeypatch.setattr(
            asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: dispatched.append(i) or True
        )
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry, *, label: markers.append((issue, note, label)),
    )
    return dispatched, markers


# ── pure decision matrix ──────────────────────────────────────────────────────


@pytest.mark.parametrize("held", [True, False])
def test_infra_drain_held_ids_never_dispatch(held):
    # AC-2a: a held ID is skipped with "held" even with free slots and
    # proposed status; without the hold the same ID dispatches.
    holds = {7: "needs-thomas"} if held else {}
    dispatch, skipped = _decide_drain([7], holds=holds)
    if held:
        assert dispatch == [] and skipped == [(7, "held")]
    else:
        assert dispatch == [7] and skipped == []


@pytest.mark.parametrize("status", [*sorted(set(STATUSES) - {"proposed"}), None])
def test_infra_drain_non_proposed_never_dispatches(status):
    # AC-2b: every non-proposed enum member (and an unreadable status) skips
    # with the right reason; proposed dispatches.
    dispatch, skipped = _decide_drain([7], statuses={7: status})
    assert dispatch == []
    expected = "status-unreadable" if status is None else f"status-{status}"
    assert skipped == [(7, expected)]
    assert _decide_drain([7]) == ([7], [])


def test_infra_drain_registered_blocks_dispatch():
    # AC-2c: an existing (non-stale) registration blocks dispatch.
    dispatch, skipped = _decide_drain([7], registered={7})
    assert dispatch == [] and skipped == [(7, "already-registered")]


def test_infra_drain_cap_arithmetic():
    # AC-2d: free = max(0, cap - occupied - pending); oldest-first order
    # preserved (order is positional in ripe_oldest_first, not numeric).
    ids = [40, 30, 20, 10]
    dispatch, skipped = _decide_drain(ids, occupied=0)
    assert dispatch == [40, 30, 20] and skipped == [(10, "cap-full")]
    dispatch, skipped = _decide_drain(ids, occupied=2)
    assert dispatch == [40] and [r for _, r in skipped] == ["cap-full"] * 3
    assert _decide_drain(ids, occupied=3)[0] == []
    assert _decide_drain(ids, cap=0)[0] == []
    # occupied > cap must clamp at zero free, never go negative.
    dispatch, skipped = _decide_drain(ids, occupied=5)
    assert dispatch == [] and all(r == "cap-full" for _, r in skipped)


def test_infra_drain_pending_registration_counts_toward_cap():
    # A dispatched-but-still-proposed registration consumes a slot: occupied
    # 2 + 1 pending under cap 3 leaves zero free for the next eligible ID.
    dispatch, skipped = _decide_drain([7], occupied=2, pending=1)
    assert dispatch == [] and skipped == [(7, "cap-full")]


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",  # non-JSON
        "[1, 2]",  # JSON list at top level
        '{"cap": 3}',  # missing ripe_oldest_first (the file's entire point)
        '{"ripe_oldest_first": [1, "2"]}',  # non-int entry
        '{"ripe_oldest_first": [true]}',  # bool entry
        '{"ripe_oldest_first": [1], "cap": -1}',  # negative cap
        '{"ripe_oldest_first": [1], "cap": "3"}',  # non-int cap
        '{"ripe_oldest_first": [1], "cap": true}',  # bool cap
        '{"ripe_oldest_first": [1], "holds": [1]}',  # non-dict holds
        '{"ripe_oldest_first": [1], "holds": {"abc": "x"}}',  # unparseable hold key
    ],
)
def test_parse_infra_drain_queue_invalid_inputs(raw):
    # AC-2e (parse half): a garbled queue file must parse to None (the pass
    # then no-ops) — never to a silently-corrected dict that could DISPATCH
    # a held ID or invent a cap.
    assert parse_infra_drain_queue(raw) is None


def test_parse_infra_drain_queue_valid_inputs():
    import json

    # Missing cap -> default 3; missing holds -> empty; order-preserving
    # dedup (first occurrence wins); unparseable updated_ts -> None.
    q = parse_infra_drain_queue('{"ripe_oldest_first": [5, 3, 5]}')
    assert q == {"ids": [5, 3], "cap": 3, "holds": {}, "updated_ts": None}
    # Live-file-shaped input: string hold keys coerced to ints, ISO-8601 Z
    # updated_ts parsed to an epoch float, extra fields ignored.
    live = json.dumps(
        {
            "updated_ts": "2026-06-12T22:40:00Z",
            "updated_by": "pm-session-drain-tick",
            "comment": "c",
            "ripe_oldest_first": [630, 631],
            "cap": 3,
            "holds": {"609": "needs-thomas", "449": "spend"},
        }
    )
    q = parse_infra_drain_queue(live)
    assert q["ids"] == [630, 631]
    assert q["cap"] == 3
    assert q["holds"] == {609: "needs-thomas", 449: "spend"}
    assert isinstance(q["updated_ts"], float)


def test_infra_drain_backoff_and_attempt_cap():
    # The tight-loop guard (AC: a failed spawn is never retried every tick).
    now = _DRAIN_NOW
    # (a) within the window -> "backoff", ALWAYS — even when the PM epoch is
    # newer than the last attempt (the window is never bypassed: the PM
    # rewrites the file on EVERY STATUS pass).
    attempts = {7: {"attempts": 3, "last_attempt_ts": now - 60.0}}
    assert _decide_drain([7], attempts=attempts, updated_ts=now - 30.0) == (
        [],
        [(7, "backoff")],
    )
    # (b) past the window with the count exhausted and a STALE (or absent)
    # PM epoch -> parked.
    attempts = {7: {"attempts": 3, "last_attempt_ts": now - 7200.0}}
    assert _decide_drain([7], attempts=attempts, updated_ts=now - 10_000.0) == (
        [],
        [(7, "attempts-exhausted")],
    )
    assert _decide_drain([7], attempts=attempts, updated_ts=None) == (
        [],
        [(7, "attempts-exhausted")],
    )
    # (c) past the window with a FRESH PM epoch -> the COUNT resets ->
    # eligible again.
    assert _decide_drain([7], attempts=attempts, updated_ts=now - 60.0) == ([7], [])
    # (d) past the window, below the cap -> eligible.
    attempts = {7: {"attempts": 2, "last_attempt_ts": now - 7200.0}}
    assert _decide_drain([7], attempts=attempts, updated_ts=None) == ([7], [])


def test_infra_drain_occupied_statuses_subset_of_enum():
    # Kills phantom-status drift (mirrors
    # test_status_classes_subset_of_authoritative_enum).
    assert set(STATUSES) >= INFRA_DRAIN_OCCUPIED_STATUSES, (
        f"phantom statuses: {INFRA_DRAIN_OCCUPIED_STATUSES - set(STATUSES)}"
    )
    # proposed/blocked/terminal must NOT hold drain slots.
    assert {"proposed", "blocked", "completed", "archived", "awaiting_promotion"}.isdisjoint(
        INFRA_DRAIN_OCCUPIED_STATUSES
    )


def test_infra_drain_sentinel_registered():
    # A watcher-posted dispatch marker must never reset the orphan/stalled
    # staleness clocks for the session it just spawned.
    import autonomous_session_watch as asw

    assert asw._INFRA_DRAIN_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


@pytest.mark.parametrize(
    ("kind", "ok"),
    [("infra", True), ("batch", True), ("experiment", False), ("campaign", False), (None, False)],
)
def test_infra_drain_kind_guard(kind, ok):
    # A mis-queued experiment/campaign ID must never be spawned with --auto
    # (it would auto-approve GPU spend AND sit outside the cap arithmetic).
    dispatch, skipped = _decide_drain([7], kinds={7: kind})
    if ok:
        assert dispatch == [7] and skipped == []
    else:
        assert dispatch == []
        assert skipped == [(7, f"kind-{kind or 'unreadable'}")]


def test_infra_drain_future_updated_ts_clamped():
    # A future PM epoch (tz confusion / LLM timestamp bug) must not void the
    # attempt budget.
    now = _DRAIN_NOW
    future = now + 86_400.0
    # Plan fixture: within the window the backoff binds regardless.
    attempts = {7: {"attempts": 1, "last_attempt_ts": now - 60.0}}
    assert _decide_drain([7], attempts=attempts, updated_ts=future) == ([], [(7, "backoff")])
    # Discriminating fixture: past the window with the count exhausted, an
    # UNCLAMPED future ts would read as a perpetual fresh adjudication and
    # dispatch; the clamp parks it.
    attempts = {7: {"attempts": 3, "last_attempt_ts": now - 7200.0}}
    assert _decide_drain([7], attempts=attempts, updated_ts=future) == (
        [],
        [(7, "attempts-exhausted")],
    )
    # A ts within the tolerance window is NOT clamped (still a fresh epoch).
    assert _decide_drain([7], attempts=attempts, updated_ts=now + 100.0) == ([7], [])


def test_infra_drain_pending_conservative():
    # The widened pending count: unreadable status/kind counts toward the cap
    # (an unknown registered task might be a just-spawned infra task).
    import autonomous_session_watch as asw

    assert asw._infra_drain_pending({1: [{}]}, set(), {1: (None, None)}) == 1
    # ... and the slot math then parks the next eligible ID.
    dispatch, skipped = _decide_drain([2], occupied=2, pending=1)
    assert dispatch == [] and skipped == [(2, "cap-full")]
    # A NON-queue registration of a proposed infra task also counts (closes
    # the PM-prunes-a-dispatched-ID overshoot).
    assert asw._infra_drain_pending({99: [{}]}, set(), {99: ("proposed", "infra")}) == 1
    # A registration at an occupied status does NOT double-count (it is
    # already in occupied_active); terminal/blocked count zero.
    assert asw._infra_drain_pending({99: [{}]}, set(), {99: ("running", "infra")}) == 0
    assert asw._infra_drain_pending({99: [{}]}, set(), {99: ("blocked", "infra")}) == 0
    # A stale registration stops pinning a slot.
    assert asw._infra_drain_pending({99: [{}]}, {99}, {99: ("proposed", "infra")}) == 0
    # A proposed non-drain-kind registration counts zero.
    assert asw._infra_drain_pending({99: [{}]}, set(), {99: ("proposed", "experiment")}) == 0


def test_infra_drain_stale_registration(isolated_registry, monkeypatch, capsys):
    # Dead-at-boot handling: a stale registration stops pinning a pending
    # slot and stops blocking re-dispatch; ANY missing signal -> NOT stale.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    grace = 1800.0
    full = {"happy_session_id": "sid-x", "spawned_at": now - 3600.0}
    assert asw._infra_drain_stale(full, {"other"}, "proposed", now, grace) is True
    # Missing/uncertain signals all fail toward NOT stale (keep blocking):
    assert asw._infra_drain_stale(full, None, "proposed", now, grace) is False  # liveness n/a
    assert asw._infra_drain_stale(full, {"sid-x"}, "proposed", now, grace) is False  # live
    assert asw._infra_drain_stale(full, {"other"}, "running", now, grace) is False  # not proposed
    assert asw._infra_drain_stale(full, {"other"}, None, now, grace) is False  # status n/a
    no_spawned = {"happy_session_id": "sid-x"}
    assert asw._infra_drain_stale(no_spawned, {"other"}, "proposed", now, grace) is False
    no_sid = {"spawned_at": now - 3600.0}
    assert asw._infra_drain_stale(no_sid, {"other"}, "proposed", now, grace) is False
    young = {"happy_session_id": "sid-x", "spawned_at": now - 60.0}
    assert asw._infra_drain_stale(young, {"other"}, "proposed", now, grace) is False
    # Executor half — THROUGH THE REAL _dispatch_infra_drain (round-1
    # Critical: a stubbed dispatch masked the pre-spawn re-check aborting on
    # the stale registration's own file; only subprocess.run is stubbed, so
    # the decide -> re-check -> spawn conjunction is what's pinned): the
    # stale registration no longer pins pending and the ID is re-dispatched
    # — even when it is the ONLY queue ID (the raw pre-filter would
    # otherwise park it forever via the early exit).
    _write_drain_queue(isolated_registry, [483])
    (isolated_registry / "issue-483.json").write_text(
        json.dumps({"issue": 483, "happy_session_id": "sid-dead", "spawned_at": now - 3600.0})
    )
    dispatched, _markers = _stub_drain_executor(
        monkeypatch,
        status_kind={483: ("proposed", "infra")},
        occupancy=[],
        live={"sid-live"},
        real_dispatch=True,
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [483]  # the spawn subprocess actually ran
    out = capsys.readouterr().out
    assert "STALE registration for issue #483" in out
    assert "(+0 pending)" in out
    assert "INFRA-DRAIN DISPATCHED issue #483" in out
    assert "INFRA-DRAIN ABORT" not in out  # the known-stale file must not abort
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["last_result"] == "dispatched"


# ── executor (I/O) tests ──────────────────────────────────────────────────────


def test_infra_drain_pass_missing_or_invalid_file_is_noop(isolated_registry, monkeypatch, capsys):
    # AC-2e (executor half): missing or garbled queue file -> logged no-op —
    # zero dispatches, zero task.py reads, no state file created.
    import autonomous_session_watch as asw

    calls: list[int] = []
    monkeypatch.setattr(
        asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: calls.append(i) or True
    )
    monkeypatch.setattr(
        asw, "_task_status_kind", lambda i: pytest.fail("task.py read on a no-op tick")
    )
    monkeypatch.setattr(
        asw, "_infra_drain_occupancy", lambda: pytest.fail("occupancy read on a no-op tick")
    )
    asw.infra_drain_pass(dry_run=False, daemon_reachable=True)
    assert "no queue file" in capsys.readouterr().out
    (isolated_registry / "infra-drain-queue.json").write_text("{torn write")
    asw.infra_drain_pass(dry_run=False, daemon_reachable=True)
    assert "INVALID queue file" in capsys.readouterr().out
    assert calls == []
    assert not (isolated_registry / "infra-drain-state.json").exists()


def test_infra_drain_kill_switch(isolated_registry, monkeypatch, capsys):
    # AC-2f: EPM_DISABLE_INFRA_DRAIN=1 disables the pass before it even
    # reads the queue file.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_DISABLE_INFRA_DRAIN", "1")
    assert asw._infra_drain_enabled() is False
    monkeypatch.setattr(
        asw, "_infra_drain_queue_path", lambda: pytest.fail("queue read despite kill switch")
    )
    asw.infra_drain_pass(dry_run=False, daemon_reachable=True)
    assert "disabled via EPM_DISABLE_INFRA_DRAIN" in capsys.readouterr().out
    monkeypatch.setenv("EPM_DISABLE_INFRA_DRAIN", "0")
    assert asw._infra_drain_enabled() is True
    monkeypatch.delenv("EPM_DISABLE_INFRA_DRAIN")
    assert asw._infra_drain_enabled() is True


def test_infra_drain_pass_end_to_end_smoke(isolated_registry, monkeypatch, capsys):
    # Live-shaped queue + stubbed signals: exactly one dispatch for the
    # oldest eligible ID, attempt state written atomically, stale state entry
    # pruned, summary/skip lines printed.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483, 615, 630, 631], holds={"631": "needs-thomas"})
    (isolated_registry / "infra-drain-state.json").write_text(
        json.dumps(
            {
                "attempts": {
                    "999": {
                        "attempts": 2,
                        "last_attempt_ts": now - 50_000.0,
                        "last_result": "spawn-failed",
                        "exhausted_logged": False,
                    }
                }
            }
        )
    )
    dispatched, markers = _stub_drain_executor(
        monkeypatch,
        status_kind={
            483: ("proposed", "infra"),
            615: ("proposed", "infra"),
            630: ("planning", "infra"),
        },
        occupancy=[700, 701],
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    # occupied 2 + pending 0 under cap 3 -> exactly one free slot -> the
    # oldest eligible ID (483) and nothing else.
    assert dispatched == [483]
    assert len(markers) == 1 and markers[0][0] == 483
    assert asw._INFRA_DRAIN_NOTE_SENTINEL in markers[0][1]
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["attempts"] == 1
    assert state["attempts"]["483"]["last_attempt_ts"] == now
    assert state["attempts"]["483"]["last_result"] == "dispatched"
    assert "999" not in state["attempts"]  # pruned: the ID left the queue
    out = capsys.readouterr().out
    assert "INFRA-DRAIN SKIP issue #631 (held: needs-thomas)" in out
    assert "INFRA-DRAIN SKIP issue #630 (status-planning)" in out
    assert "INFRA-DRAIN SKIP issue #615 (cap-full)" in out
    assert "queue=4 occupied=2(+0 pending) cap=3 dispatched=1 skipped=3" in out
    assert "occupying=[700, 701]" in out  # slot-jam diagnosable from the log


def test_infra_drain_dry_run_no_mutation(isolated_registry, monkeypatch, capsys):
    # The live smoke's safety depends on this: dry-run decides + logs but
    # never spawns, never posts a marker, never writes state.
    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483])
    sk = {483: ("proposed", "infra")}
    monkeypatch.setattr(asw, "_task_status_kind", lambda i: sk.get(i, (None, None)))
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_live_session_ids_or_none", lambda: set())
    markers: list[tuple] = []
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: markers.append((a, k)))
    monkeypatch.setattr(
        asw.subprocess, "run", lambda *a, **k: pytest.fail("subprocess.run called in dry-run")
    )
    asw.infra_drain_pass(dry_run=True, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "[dry-run] would dispatch infra-drain" in out
    assert markers == []  # a dry-run dispatch returns False -> no marker
    assert not (isolated_registry / "infra-drain-state.json").exists()
    # The dispatch helper itself honours dry_run before any subprocess.
    assert asw._dispatch_infra_drain(483, "slot 1/3", dry_run=True) is False


def test_infra_drain_occupancy_none_skips_dispatch(isolated_registry, monkeypatch, capsys):
    # Fail-CLOSED: a partial occupancy read would UNDER-count and
    # over-dispatch past the cap, so None skips dispatching this tick
    # (state is still pruned + saved). Pins against the one-character
    # `or 0` inversion.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483])
    (isolated_registry / "infra-drain-state.json").write_text(
        json.dumps({"attempts": {"999": {"attempts": 1, "last_attempt_ts": now - 50_000.0}}})
    )
    dispatched, _markers = _stub_drain_executor(
        monkeypatch, status_kind={483: ("proposed", "infra")}, occupancy=None
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == []
    out = capsys.readouterr().out
    assert "occupancy read FAILED" in out and "fail-closed" in out
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state == {"attempts": {}}  # pruned (999 left the queue) + saved


def test_infra_drain_failed_spawn_records_attempt(isolated_registry, monkeypatch, capsys):
    # The tight-loop guard's executor half: a FAILED spawn still consumes an
    # attempt + arms the backoff window, so the next tick skips.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483])
    sk = {483: ("proposed", "infra")}
    monkeypatch.setattr(asw, "_task_status_kind", lambda i: sk.get(i, (None, None)))
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_live_session_ids_or_none", lambda: set())
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: None)
    calls: list[int] = []
    monkeypatch.setattr(
        asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: calls.append(i) or False
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert calls == [483]
    rec = json.loads((isolated_registry / "infra-drain-state.json").read_text())["attempts"]["483"]
    assert rec["attempts"] == 1
    assert rec["last_attempt_ts"] == now
    assert rec["last_result"] == "spawn-failed"
    # Second pass 60 s later (within the 1 h window): backoff skip, no
    # second dispatch call.
    asw.infra_drain_pass(dry_run=False, now=now + 60.0, daemon_reachable=True)
    assert calls == [483]
    assert "INFRA-DRAIN SKIP issue #483 (backoff)" in capsys.readouterr().out


def test_infra_drain_main_wiring(isolated_registry, monkeypatch):
    # --infra-drain-only alone cannot certify production wiring: pin that
    # main() calls infra_drain_pass exactly once, after orphan_sweep_pass,
    # before session_reconcile_pass, with the SAME reused daemon_reachable.
    import autonomous_session_watch as asw

    order: list[tuple[str, dict]] = []

    def rec(name):
        return lambda *a, **kw: order.append((name, kw))

    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    for pass_name in (
        "vm_disk_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "session_reconcile_pass",
        "gate_push_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
        "gc_pass",
    ):
        monkeypatch.setattr(asw, pass_name, rec(pass_name))

    rc = asw.main([])

    assert rc == 0
    names = [n for n, _ in order]
    assert names.count("infra_drain_pass") == 1
    assert (
        names.index("orphan_sweep_pass")
        < names.index("infra_drain_pass")
        < names.index("session_reconcile_pass")
    )
    infra_kw = next(kw for n, kw in order if n == "infra_drain_pass")
    assert infra_kw["daemon_reachable"] is True


def test_infra_drain_registered_blocks_dispatch_executor(isolated_registry, monkeypatch, capsys):
    # REAL registration files (autonomous + manual) under the registry dir
    # block dispatch AND count toward pending — a broken
    # _infra_drain_registrations could not be caught by the pure-function
    # test alone.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483, 615, 630])
    (isolated_registry / "issue-483.json").write_text(
        json.dumps(
            {"issue": 483, "happy_session_id": "sid-483", "spawned_at": now - 60.0, "missed": 0}
        )
    )
    (isolated_registry / "manual-issue-615.json").write_text(
        json.dumps(
            {
                "issue": 615,
                "happy_session_id": "sid-615",
                "spawned_at": now - 60.0,
                "mode": "manual",
            }
        )
    )
    dispatched, _markers = _stub_drain_executor(
        monkeypatch,
        status_kind={
            483: ("proposed", "infra"),
            615: ("proposed", "infra"),
            630: ("proposed", "infra"),
        },
        occupancy=[],
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [630]
    out = capsys.readouterr().out
    assert "INFRA-DRAIN SKIP issue #483 (already-registered)" in out
    assert "INFRA-DRAIN SKIP issue #615 (already-registered)" in out
    assert "occupied=0(+2 pending)" in out  # both registrations pin slots


def test_infra_drain_prefilter_agrees_with_decide(isolated_registry, monkeypatch):
    # Single-source-of-truth check: an exhausted record + a NEWER PM
    # updated_ts (past the backoff window) must survive the cheap pre-filter
    # and reach dispatch — a drifted re-implementation of the guards in the
    # pre-filter would park it forever.
    import json
    from datetime import UTC, datetime

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    iso = datetime.fromtimestamp(now - 60.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_drain_queue(isolated_registry, [483], updated_ts=iso)
    (isolated_registry / "infra-drain-state.json").write_text(
        json.dumps(
            {
                "attempts": {
                    "483": {
                        "attempts": 3,
                        "last_attempt_ts": now - 7200.0,
                        "last_result": "spawn-failed",
                        "exhausted_logged": True,
                    }
                }
            }
        )
    )
    dispatched, _markers = _stub_drain_executor(
        monkeypatch, status_kind={483: ("proposed", "infra")}, occupancy=[]
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [483]
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["attempts"] == 1  # fresh epoch reset the COUNT


def test_infra_drain_prespawn_recheck_aborts(isolated_registry, monkeypatch, capsys):
    # The double-spawn race shrinker, now snapshot-aware (round-2 fix #1):
    # abort ONLY when a registration APPEARED or CHANGED since the
    # decision-time snapshot; a byte-identical (known-stale) registration
    # proceeds to the spawn.
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    stale_bytes = b'{"issue": 42, "happy_session_id": "sid-dead"}'
    (isolated_registry / "issue-42.json").write_bytes(stale_bytes)
    monkeypatch.setattr(
        asw.subprocess, "run", lambda *a, **k: pytest.fail("spawned despite a lost race")
    )
    # (a) APPEARED: no snapshot context (direct call) -> any existing file
    # is a genuinely-new registration -> abort before the subprocess.
    ok = asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False)
    assert ok is False
    out = capsys.readouterr().out
    assert "lost race to concurrent dispatcher" in out and "appeared" in out
    # (a') APPEARED relative to an explicit snapshot that lacked the file.
    ok = asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={})
    assert ok is False
    assert "appeared" in capsys.readouterr().out
    # (b) CHANGED: the decision saw OTHER bytes (e.g. a concurrent PM spawn
    # overwrote the stale entry between decide and dispatch) -> abort.
    ok = asw._dispatch_infra_drain(
        42, "slot 1/3", dry_run=False, reg_snapshot={"issue-42.json": b'{"old": true}'}
    )
    assert ok is False
    assert "changed" in capsys.readouterr().out
    # (c) BYTE-IDENTICAL: the file IS the known-stale registration the
    # decision already classified -> proceed (spawn_session overwrites it on
    # success). The round-1 bug aborted here, wedging the ID forever.
    spawned: list[list] = []
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda cmd, **k: (
            spawned.append(cmd) or SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        ),
    )
    ok = asw._dispatch_infra_drain(
        42, "slot 1/3", dry_run=False, reg_snapshot={"issue-42.json": stale_bytes}
    )
    assert ok is True
    assert len(spawned) == 1 and "--issue" in spawned[0]
    assert "INFRA-DRAIN DISPATCHED issue #42" in capsys.readouterr().out


def test_infra_drain_liveness_unavailable_keeps_blocking(isolated_registry, monkeypatch, capsys):
    # Round-2 fix #2 (concern live-session-ids-empty-on-daemon-flap): a
    # daemon flap AFTER main()'s reachability probe must read as liveness
    # UNAVAILABLE (None -> nothing stale -> keep blocking), never as "zero
    # live sessions" (-> false-stale -> double-spawn). Exercises the REAL
    # _live_session_ids_or_none with the real unavailable shape: the /list
    # POST raises after daemon_reachable=True was already cached.
    import json
    import urllib.error

    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "daemon_port", lambda: 65535)

    def _refused(*a, **k):
        raise urllib.error.URLError("connection refused (flap after the probe)")

    monkeypatch.setattr("urllib.request.urlopen", _refused)
    assert asw._live_session_ids_or_none() is None  # unavailable, NOT set()

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483])
    # An old (grace-aged) registration for a still-proposed task — the
    # false-stale candidate the flap would have re-dispatched.
    (isolated_registry / "issue-483.json").write_text(
        json.dumps({"issue": 483, "happy_session_id": "sid-x", "spawned_at": now - 3600.0})
    )
    monkeypatch.setattr(
        asw, "_task_status_kind", lambda i: pytest.fail("task.py read despite liveness-unavailable")
    )
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: pytest.fail("double-spawned #483"))
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "INFRA-DRAIN SKIP issue #483 (already-registered)" in out
    assert "STALE registration" not in out
    assert "DISPATCHED" not in out


def test_infra_drain_garbled_attempts_state_fails_safe(isolated_registry, monkeypatch, capsys):
    # Round-2 fix #3 (Codex Major): a state record with numeric
    # last_attempt_ts but a garbled attempts count must not crash the pass
    # (int("bad")+1 / "bad" >= max_attempts both raised) and must fail toward
    # NOT dispatching — the count is normalized UP to the attempt cap, so the
    # record's budget reads exhausted until a fresh PM updated_ts resets it.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    (isolated_registry / "infra-drain-state.json").write_text(
        json.dumps({"attempts": {"483": {"attempts": "bad", "last_attempt_ts": 1}}})
    )
    # Stale PM epoch (no updated_ts): the unknown count must NOT bypass the
    # attempt budget into a dispatch.
    _write_drain_queue(isolated_registry, [483], updated_ts=None)
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: pytest.fail("budget bypassed"))
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)  # must not raise
    captured = capsys.readouterr()
    assert "garbled attempts count ('bad')" in captured.out
    assert "DISPATCHED" not in captured.out
    # The loud first-time exhausted skip goes to stderr.
    assert "attempts-exhausted" in captured.out + captured.err
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["attempts"] == INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT
    # Recovery path: a FRESH PM adjudication (updated_ts newer than the
    # record's last_attempt_ts) resets the count — the normalized record
    # parks the ID, it does not brick it.
    _write_drain_queue(isolated_registry, [483])  # default updated_ts (2026) > 1
    dispatched, _markers = _stub_drain_executor(
        monkeypatch, status_kind={483: ("proposed", "infra")}, occupancy=[], real_dispatch=True
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [483]
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["attempts"] == 1  # fresh epoch reset


def test_infra_drain_missing_attempts_key_normalizes_to_cap(isolated_registry, monkeypatch, capsys):
    # Round-4 fix (CONCERN: missing-attempts-normalizes-down): a state record
    # with a valid last_attempt_ts but NO ``attempts`` key would slip past
    # the type check via ``rec.get("attempts", 0)`` returning the 0 default
    # BEFORE the bool/non-int/negative branch fired — silently granting a
    # fresh budget on a half-written or hand-edited record. The fix
    # normalizes the missing-key shape UP to the attempt cap, same fail
    # direction the garbled-count sibling already uses.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    (isolated_registry / "infra-drain-state.json").write_text(
        # last_attempt_ts present and valid; attempts key entirely MISSING.
        json.dumps({"attempts": {"483": {"last_attempt_ts": 1}}})
    )
    # Stale PM epoch (no updated_ts): the missing count must NOT bypass the
    # attempt budget into a dispatch.
    _write_drain_queue(isolated_registry, [483], updated_ts=None)
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: pytest.fail("budget bypassed"))
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)  # must not raise
    captured = capsys.readouterr()
    assert "missing its attempts key" in captured.out
    assert "DISPATCHED" not in captured.out
    # The loud first-time exhausted skip goes to stderr.
    assert "attempts-exhausted" in captured.out + captured.err
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["attempts"] == INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT
    # Recovery path: a FRESH PM adjudication (updated_ts newer than the
    # record's last_attempt_ts) resets the count via the same fresh-epoch
    # branch the garbled-count sibling test pins — the normalized record
    # parks the ID, it does not brick it. This also pins that the fix did
    # NOT break the fresh-reset semantics for the missing-key shape.
    _write_drain_queue(isolated_registry, [483])  # default updated_ts (2026) > 1
    dispatched, _markers = _stub_drain_executor(
        monkeypatch, status_kind={483: ("proposed", "infra")}, occupancy=[], real_dispatch=True
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [483]
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"]["483"]["attempts"] == 1  # fresh epoch reset


def test_live_session_ids_or_none_shapes(monkeypatch):
    # Unit pin for the wrapper itself: a well-formed /list payload yields the
    # id set; a malformed payload or an unreachable daemon yields None
    # (UNAVAILABLE), never an empty set. A dict child carrying an invalid
    # ``happySessionId`` (missing/None/empty/non-str) is the same shape as a
    # missing ``children`` list — UNAVAILABLE, never ``{None}`` (round-3 fix
    # for the reconciler-flagged double-spawn class: a stray ``{None}`` set
    # would slip past the ``is None`` guard in :func:`_infra_drain_stale`
    # and make every real-string sid look NOT live).
    import io
    import urllib.error
    from contextlib import contextmanager

    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "daemon_port", lambda: 65535)

    def _responding(payload: bytes):
        @contextmanager
        def _fake_urlopen(req, timeout=10):
            yield io.BytesIO(payload)

        return _fake_urlopen

    monkeypatch.setattr(
        "urllib.request.urlopen",
        _responding(b'{"children": [{"happySessionId": "sid-1"}, "junk"]}'),
    )
    # Non-dict child ("junk") is skipped, NOT a contract violation — the
    # well-formed dict still contributes its sid.
    assert asw._live_session_ids_or_none() == {"sid-1"}
    monkeypatch.setattr("urllib.request.urlopen", _responding(b'{"children": []}'))
    assert asw._live_session_ids_or_none() == set()  # confirmed zero sessions
    monkeypatch.setattr("urllib.request.urlopen", _responding(b'{"no-children-key": 1}'))
    assert asw._live_session_ids_or_none() is None  # malformed -> unavailable

    # Sanity: a multi-sid happy path returns the full set.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _responding(b'{"children": [{"happySessionId": "real-1"}, {"happySessionId": "real-2"}]}'),
    )
    assert asw._live_session_ids_or_none() == {"real-1", "real-2"}

    # round-3: dict children with invalid happySessionId are daemon-contract
    # violations -> UNAVAILABLE (never ``{None}``).
    monkeypatch.setattr("urllib.request.urlopen", _responding(b'{"children": [{}]}'))
    assert asw._live_session_ids_or_none() is None  # missing sid key
    monkeypatch.setattr(
        "urllib.request.urlopen", _responding(b'{"children": [{"happySessionId": null}]}')
    )
    assert asw._live_session_ids_or_none() is None  # null sid
    monkeypatch.setattr(
        "urllib.request.urlopen", _responding(b'{"children": [{"happySessionId": ""}]}')
    )
    assert asw._live_session_ids_or_none() is None  # empty-string sid
    monkeypatch.setattr(
        "urllib.request.urlopen", _responding(b'{"children": [{"happySessionId": 42}]}')
    )
    assert asw._live_session_ids_or_none() is None  # non-str sid
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _responding(b'{"children": [{"happySessionId": "real-1"}, {}]}'),
    )
    # One bad child contaminates the whole reply — cannot tell whether the
    # others are real-but-incomplete or merely the well-formed survivors of
    # a partial write. Fail toward keep-blocking.
    assert asw._live_session_ids_or_none() is None

    def _down(*a, **k):
        raise urllib.error.URLError("daemon down")

    monkeypatch.setattr("urllib.request.urlopen", _down)
    assert asw._live_session_ids_or_none() is None


def test_live_session_ids_or_none_widens_exception_tuple(monkeypatch):
    # Round-4 fix (CONCERN: urlopen-catch-tuple-too-narrow): the previous
    # catch tuple (SystemExit, urllib.error.URLError, OSError,
    # json.JSONDecodeError) crashed the whole infra-drain pass on two
    # real daemon-flap shapes the original tuple missed:
    #   (a) http.client.HTTPException (incl. IncompleteRead) when the
    #       daemon hangs up mid-response-body — distinct from URLError,
    #       which fires at connection setup.
    #   (b) UnicodeDecodeError when the daemon emits invalid UTF-8 bytes —
    #       json.JSONDecodeError is a ValueError subclass, NOT a
    #       UnicodeDecodeError subclass, so the bytes read raises BEFORE
    #       json.loads ever sees a string.
    # Both must now return None (UNAVAILABLE — fail toward keep-blocking),
    # the same fail direction as a clean URLError.
    import http.client
    import io
    from contextlib import contextmanager

    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "daemon_port", lambda: 65535)

    # (a) http.client.IncompleteRead raised by urlopen itself: the response
    # connection drops while urlopen is still establishing the body stream.
    def _incomplete_read(*a, **k):
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr("urllib.request.urlopen", _incomplete_read)
    assert asw._live_session_ids_or_none() is None  # must not crash

    # (b) UnicodeDecodeError raised by json.load(s) on invalid UTF-8 bytes:
    # urlopen returns a body whose decode trips before any JSON parsing.
    class _BadBytes(io.BytesIO):
        def read(self, *a, **k):
            raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte")

    @contextmanager
    def _bad_utf8_urlopen(req, timeout=10):
        yield _BadBytes(b"\xff\xfe")

    monkeypatch.setattr("urllib.request.urlopen", _bad_utf8_urlopen)
    assert asw._live_session_ids_or_none() is None  # must not crash


def test_daemon_reachable_widens_exception_tuple(monkeypatch):
    # Round-4 sibling pin for _daemon_reachable: same widened catch tuple,
    # same fail direction (return False, NOT propagate). A daemon flap that
    # raises http.client.HTTPException or UnicodeDecodeError must not crash
    # the watcher's main() reachability probe (which would skip every pass
    # that consumes the result).
    import http.client
    import io
    from contextlib import contextmanager

    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "daemon_port", lambda: 65535)

    def _incomplete_read(*a, **k):
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr("urllib.request.urlopen", _incomplete_read)
    assert asw._daemon_reachable() is False  # must not crash

    class _BadBytes(io.BytesIO):
        def read(self, *a, **k):
            raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte")

    @contextmanager
    def _bad_utf8_urlopen(req, timeout=10):
        yield _BadBytes(b"\xff\xfe")

    monkeypatch.setattr("urllib.request.urlopen", _bad_utf8_urlopen)
    assert asw._daemon_reachable() is False  # must not crash


def test_infra_drain_malformed_child_keeps_blocking(isolated_registry, monkeypatch, capsys):
    # Round-3 production-trigger pin (reconciler verdict 2026-06-12): the
    # double-spawn class _live_session_ids_or_none used to reintroduce when
    # a /list child dict carried an invalid happySessionId — the bare set
    # comprehension returned ``{None}``, which slipped past
    # ``live_session_ids is None`` in _infra_drain_stale and made every
    # real-string sid look NOT live -> false-stale -> dispatch.
    #
    # Wires the malformed payload END-TO-END through infra_drain_pass with
    # a real urlopen stub: a stale registration that WOULD be re-dispatched
    # if liveness leaked ``{None}`` must stay blocked (the registration
    # pins a pending slot, no INFRA-DRAIN DISPATCHED line).
    import io
    import json
    from contextlib import contextmanager

    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "daemon_port", lambda: 65535)

    @contextmanager
    def _fake_urlopen(req, timeout=10):
        # The exact Codex fixture: a /list payload with an empty-dict
        # child. Pre-fix this returned ``{None}``; the round-3 fix returns
        # ``None`` (UNAVAILABLE).
        yield io.BytesIO(b'{"children": [{}]}')

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483])
    # A grace-aged registration that the false-stale read would otherwise
    # mark dead and re-dispatch.
    (isolated_registry / "issue-483.json").write_text(
        json.dumps({"issue": 483, "happy_session_id": "sid-x", "spawned_at": now - 3600.0})
    )
    monkeypatch.setattr(
        asw,
        "_task_status_kind",
        lambda i: pytest.fail("task.py read despite liveness-unavailable"),
    )
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: pytest.fail("double-spawned #483"))
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "INFRA-DRAIN SKIP issue #483 (already-registered)" in out
    assert "STALE registration" not in out
    assert "INFRA-DRAIN DISPATCHED" not in out


# ── predicate-hold auto-promotion (#633 follow-on) ────────────────────────────


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("predicate-535-slurm-attempt", 535),  # live example
        ("predicate-625-lands", 625),  # live example
        ("predicate-#535-x", 535),  # defensive: stray leading '#'
        ("predicate-7-a-b-c", 7),  # only the first token after the prefix matters
        ("needs-thomas", None),  # non-predicate PM-judgment hold
        ("credentials", None),
        ("spend", None),
        ("predicate-", None),  # bare prefix, no token
        ("predicate--x", None),  # empty token
        ("predicate-abc-x", None),  # non-digit token
        ("PREDICATE-5-x", None),  # case-sensitive prefix
    ],
)
def test_parse_predicate_hold(reason, expected):
    import autonomous_session_watch as asw

    assert asw._parse_predicate_hold(reason) == expected


def test_parse_predicate_hold_non_string():
    import autonomous_session_watch as asw

    # A garbled (non-string) reason must fail toward NOT touching the hold.
    assert asw._parse_predicate_hold(None) is None
    assert asw._parse_predicate_hold(123) is None


def test_satisfied_predicate_promotions_matrix():
    # The pure decision: only predicate holds whose blocking task is
    # completed/archived/awaiting_promotion are promoted; everything else
    # (non-predicate, malformed, unreadable, not-yet-terminal) is kept.
    import autonomous_session_watch as asw

    holds = {
        581: "predicate-535-slurm-attempt",  # 535 terminal -> promote
        700: "predicate-625-lands",  # 625 active -> keep
        609: "needs-thomas",  # non-predicate -> keep, never inspected
        710: "predicate-900-x",  # 900 unreadable -> keep
        720: "predicate-abc-x",  # malformed -> keep, never inspected
    }
    statuses = {535: "completed", 625: "running", 900: None}
    promote, remaining = asw._satisfied_predicate_promotions(holds, statuses)
    assert promote == [581]
    assert remaining == {
        700: "predicate-625-lands",
        609: "needs-thomas",
        710: "predicate-900-x",
        720: "predicate-abc-x",
    }


@pytest.mark.parametrize("status", sorted(INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES))
def test_satisfied_predicate_promotions_each_terminal_status(status):
    # All three landed/terminal statuses satisfy a predicate.
    import autonomous_session_watch as asw

    promote, remaining = asw._satisfied_predicate_promotions(
        {581: "predicate-535-x"}, {535: status}
    )
    assert promote == [581]
    assert remaining == {}


def test_satisfied_predicate_statuses_no_active_overlap():
    # The satisfaction set must be disjoint from the slot-occupying (active)
    # set — a still-running blocking task must never read as "finished".
    assert INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES.isdisjoint(
        {
            "planning",
            "plan_pending",
            "approved",
            "running",
            "verifying",
            "interpreting",
            "reviewing",
        }
    )


def test_satisfied_predicate_promotions_ascending_order():
    # promote_ids is deterministic ascending-id order regardless of dict order.
    import autonomous_session_watch as asw

    holds = {700: "predicate-9-x", 581: "predicate-9-x", 640: "predicate-9-x"}
    promote, remaining = asw._satisfied_predicate_promotions(holds, {9: "completed"})
    assert promote == [581, 640, 700]
    assert remaining == {}


def test_predicate_promote_rewrites_queue_and_dispatches(isolated_registry, monkeypatch, capsys):
    # End-to-end: a satisfied predicate hold is promoted, the queue file is
    # rewritten atomically (hold removed, id appended oldest-first, updated_by
    # stamped as the watcher), and the just-cleared id dispatches THIS tick.
    import json

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483], holds={"581": "predicate-535-slurm-attempt"})
    dispatched, _markers = _stub_drain_executor(
        monkeypatch,
        status_kind={
            483: ("proposed", "infra"),
            581: ("proposed", "infra"),
            535: ("completed", "experiment"),  # the blocking task is done
        },
        occupancy=[],
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    # Both the originally-ripe 483 and the just-promoted 581 dispatch (cap 3,
    # zero occupied) — promoted id flows through the normal cap/guard path.
    assert dispatched == [483, 581]
    out = capsys.readouterr().out
    assert "PREDICATE-PROMOTE 1 hold(s)" in out
    assert "#581 (cleared by predicate-535-slurm-attempt)" in out
    # Queue file rewritten: hold gone, 581 merged oldest-first, watcher-stamped.
    queue = json.loads((isolated_registry / "infra-drain-queue.json").read_text())
    assert queue["ripe_oldest_first"] == [483, 581]
    assert queue["holds"] == {}
    assert queue["updated_by"] == asw.INFRA_DRAIN_QUEUE_WRITER
    assert queue["updated_ts"].endswith("Z")


def test_promote_preserves_positional_order_not_ascending(monkeypatch):
    # `ripe_oldest_first` is POSITIONAL (oldest-first / urgency-first), NOT
    # ascending-id — a promoted id must be APPENDED to the END, never re-sorted.
    # Discriminating fixture: a non-ascending existing queue where sort != append.
    # Stub the status reader so the test never depends on live task state.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status_kind", lambda i: ("completed", "experiment"))
    new_ids, _ = asw._promote_satisfied_predicate_holds(
        ids=[640, 483, 615],  # PM urgency-first order (640 = urgent head)
        cap=3,
        holds={581: "predicate-535-x"},  # blocker #535 -> completed via stub
        now=_DRAIN_NOW,
        dry_run=True,  # decide only, no file write
    )
    # Wrong (re-sort): [483, 581, 615, 640] — pushes the urgent head to the back.
    # Right (append): [640, 483, 615, 581] — head preserved.
    assert new_ids == [640, 483, 615, 581]
    assert new_ids != sorted(new_ids)  # the bug would have made these equal


def test_promote_appends_multiple_in_ascending_subset_order(monkeypatch):
    # When several holds clear at once, the promoted SUBSET is appended in
    # ascending-id order (deterministic), AFTER the PM's positional head.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status_kind", lambda i: ("completed", "experiment"))
    new_ids, remaining = asw._promote_satisfied_predicate_holds(
        ids=[900, 100],  # non-ascending head
        cap=3,
        holds={700: "predicate-9-x", 581: "predicate-9-x", 640: "predicate-9-x"},
        now=_DRAIN_NOW,
        dry_run=True,
    )
    assert new_ids == [900, 100, 581, 640, 700]
    assert remaining == {}


def test_promote_dedupes_id_already_in_queue(monkeypatch):
    # Defensive: if a hold key is somehow already in ripe_oldest_first, the
    # append must not duplicate it (and must not reorder the existing head).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status_kind", lambda i: ("completed", "experiment"))
    new_ids, _ = asw._promote_satisfied_predicate_holds(
        ids=[640, 581, 483],  # 581 already present (shouldn't happen, but guard)
        cap=3,
        holds={581: "predicate-9-x"},
        now=_DRAIN_NOW,
        dry_run=True,
    )
    assert new_ids == [640, 581, 483]  # unchanged: no duplicate 581 appended


def test_predicate_promote_keeps_unsatisfied_and_nonpredicate(
    isolated_registry, monkeypatch, capsys
):
    # A non-predicate hold and an unsatisfied predicate hold are BOTH left
    # untouched; the queue file is NOT rewritten when nothing is promoted.

    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(
        isolated_registry,
        [483],
        holds={"609": "needs-thomas", "700": "predicate-625-lands"},
    )
    before = (isolated_registry / "infra-drain-queue.json").read_text()
    dispatched, _markers = _stub_drain_executor(
        monkeypatch,
        status_kind={
            483: ("proposed", "infra"),
            625: ("running", "experiment"),  # blocking task NOT finished
        },
        occupancy=[],
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [483]  # only the already-ripe id; neither hold promoted
    out = capsys.readouterr().out
    assert "PREDICATE-PROMOTE" not in out
    # Queue file byte-unchanged (no needless rewrite, no budget re-arm).
    assert (isolated_registry / "infra-drain-queue.json").read_text() == before


def test_predicate_promote_nonpredicate_holds_skip_status_read(
    isolated_registry, monkeypatch, capsys
):
    # Holds with NO predicate reason must cost ZERO task.py status reads — the
    # promotion step short-circuits before any subprocess.
    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483], holds={"609": "needs-thomas", "449": "spend"})
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_live_session_ids_or_none", lambda: set())
    monkeypatch.setattr(asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: None)

    # _task_status_kind is allowed for 483 (the dispatchable id) but must NEVER
    # be called for a non-predicate hold's nonexistent blocking issue.
    def _guarded_status_kind(i):
        assert i == 483, f"unexpected status read for #{i} (non-predicate holds inspected)"
        return ("proposed", "infra")

    monkeypatch.setattr(asw, "_task_status_kind", _guarded_status_kind)
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "PREDICATE-PROMOTE" not in out


def test_predicate_promote_dry_run_no_rewrite(isolated_registry, monkeypatch, capsys):
    # Dry-run decides + logs the promotion but never rewrites the queue file
    # and never spawns (mirrors the dispatch dry-run discipline).
    import autonomous_session_watch as asw

    now = _DRAIN_NOW
    _write_drain_queue(isolated_registry, [483], holds={"581": "predicate-535-x"})
    before = (isolated_registry / "infra-drain-queue.json").read_text()
    sk = {483: ("proposed", "infra"), 581: ("proposed", "infra"), 535: ("completed", "experiment")}
    monkeypatch.setattr(asw, "_task_status_kind", lambda i: sk.get(i, (None, None)))
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_live_session_ids_or_none", lambda: set())
    monkeypatch.setattr(
        asw.subprocess, "run", lambda *a, **k: pytest.fail("subprocess.run in dry-run")
    )
    asw.infra_drain_pass(dry_run=True, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "PREDICATE-PROMOTE 1 hold(s)" in out
    assert "would rewrite the queue file" in out
    # Queue file byte-unchanged under dry-run.
    assert (isolated_registry / "infra-drain-queue.json").read_text() == before
