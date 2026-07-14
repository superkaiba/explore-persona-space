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

from tests.conftest import _FLEET_MUTATING_PASS_NAMES, _stub_fleet_mutating_passes

# scripts/ holds autonomous_session_watch.py (and its spawn_session import).
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# src/ holds explore_persona_space.task_workflow (the watcher's follow-up
# label helpers lazy-import it; the tests import it directly too). Inserted
# ahead of any installed copy so THIS checkout's helpers win (#894).
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import spawn_session  # noqa: E402
from autonomous_session_watch import (  # noqa: E402
    ACTIVE,
    ALERT_STALE_HOURS,
    AUTO_STOP_DONE,
    AUTO_STOP_PAUSED,
    CAPACITY_RETRY_BACKOFF_S_DEFAULT,
    CAPACITY_RETRY_MAX_PER_DAY_DEFAULT,
    INFRA_DRAIN_BACKOFF_S_DEFAULT,
    INFRA_DRAIN_CAP_DEFAULT,
    INFRA_DRAIN_MAX_ATTEMPTS_DEFAULT,
    INFRA_DRAIN_OCCUPIED_STATUSES,
    INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES,
    ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
    ORPHAN_SPAWN_GRACE_S,
    ORPHAN_STALENESS_S_DEFAULT,
    PARK,
    POD_ACTIVE,
    POD_SAFETY_AUTO_STOP,
    PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT,
    PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT,
    RESPAWN_SPAWN_GRACE_S,
    STALLED_MARKER_WINDOW_S_DEFAULT,
    STALLED_MAX_RESPAWNS,
    STALLED_WINDOW_S,
    TERMINAL,
    TRANSIENT_CAPACITY_REASONS,
    decide,
    decide_capacity_retry,
    decide_infra_drain,
    decide_orphan,
    decide_pod_safety,
    decide_proposed_infra_sweep,
    parse_infra_drain_queue,
    program_orchestrator_pass,
)

from explore_persona_space.task_workflow import STATUSES  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_stagger_sleep(monkeypatch):
    """Hermeticity for the #1059 session-dispatch stagger: no watcher test may
    ever REALLY sleep (a real-spawn earlier in the same test records a fresh
    stamp into the isolated registry, which would put a ~60s ``time.sleep``
    inside the next dispatch). Records the requested delays so the stagger
    wiring tests can assert on them; every other test just never sleeps."""
    import autonomous_session_watch as asw

    sleeps: list[float] = []
    monkeypatch.setattr(asw, "_stagger_sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


# The #1247 hermeticity guards (`_forbid_real_marker_posts`,
# `_forbid_real_task_status_reads`) are now shared autouse fixtures in
# tests/conftest.py (task #1265) — they apply here automatically.


def _p(issue: int, pod_id: str, name: str):
    """A non-wedged 4-tuple for ``_running_managed_issue_pods`` stubs (#692).

    ``_running_managed_issue_pods`` now returns ``(issue, pod_id, name, info)``
    4-tuples carrying the live :class:`runpod_api.PodInfo`. These status-class
    pod-safety / session-reconcile tests are NOT about the wedge arm, so the
    ``info`` is HEALTHY (a public SSH port present) — the wedge predicate
    ``backend_poll._pod_is_runpod_runtime_wedged`` reads False and the wedge arm
    is a no-op, so existing status-class behavior is unchanged."""
    from runpod_api import PodInfo

    return (
        issue,
        pod_id,
        name,
        PodInfo(
            pod_id=pod_id,
            name=name,
            desired_status="RUNNING",
            ssh_host="1.2.3.4",
            ssh_port=22000,
        ),
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


# ─── #759: crash-recovery spawn-grace (bug class a) ──────────────────────────
# A just-(re)spawned ACTIVE session whose id has not yet propagated into the
# daemon's /list reply reads `alive=False` for the registration-latency window.
# Without a grace, the 2-miss respawn fires and spawns a DUPLICATE driver. The
# grace mirrors the orphan sweep's `ORPHAN_SPAWN_GRACE_S` (decide_orphan).


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_decide_grace_keeps_not_yet_live_entry_inside_window(status):
    # Criterion 1: inside the grace window, an ACTIVE not-alive entry does NOT
    # respawn even at the 2nd miss — its id may simply not be in /list yet.
    # Mirror of test_orphan_fresh_registration_keeps.
    assert decide(
        status,
        alive=False,
        missed=1,
        entry_age_s=RESPAWN_SPAWN_GRACE_S - 1,
    ) == ("keep", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_decide_grace_expired_respawns_as_before(status):
    # Criterion 2: past the grace window, the existing 2-miss respawn still
    # fires — no regression of the genuine-death path.
    assert decide(
        status,
        alive=False,
        missed=1,
        entry_age_s=RESPAWN_SPAWN_GRACE_S + 1,
    ) == ("respawn", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_decide_grace_none_age_preserves_today_behavior(status):
    # Criterion 3: a missing spawned_at (entry_age_s=None) preserves today's
    # behavior exactly — fail toward respawn, never silently suppress.
    assert decide(status, alive=False, missed=1, entry_age_s=None) == ("respawn", 0)


@pytest.mark.parametrize("status", sorted(ACTIVE))
def test_decide_alive_short_circuits_before_grace(status):
    # Criterion 4: a LIVE session is kept regardless of grace (alive checked
    # first, so a fresh-and-live entry resets misses without touching grace).
    assert decide(status, alive=True, missed=3, entry_age_s=1) == ("keep", 0)


def test_decide_grace_resets_accumulated_miss():
    # A miss accumulated BEFORE the (re)spawn must not straddle the grace into
    # an immediate respawn: inside the window the count resets to 0.
    assert decide(
        "running",
        alive=False,
        missed=5,
        entry_age_s=RESPAWN_SPAWN_GRACE_S - 1,
    ) == ("keep", 0)


def test_respawn_spawn_grace_env_override_and_fallback(monkeypatch):
    # Criterion 10 (a-half): EPM_RESPAWN_SPAWN_GRACE_MIN parses a valid int
    # (minutes) and falls back to the default on a garbled value. Mirrors the
    # _orphan_staleness_s env-parse coverage.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_RESPAWN_SPAWN_GRACE_MIN", raising=False)
    assert asw._respawn_spawn_grace_s() == float(RESPAWN_SPAWN_GRACE_S)

    monkeypatch.setenv("EPM_RESPAWN_SPAWN_GRACE_MIN", "20")
    assert asw._respawn_spawn_grace_s() == 20 * 60.0

    monkeypatch.setenv("EPM_RESPAWN_SPAWN_GRACE_MIN", "not-a-number")
    assert asw._respawn_spawn_grace_s() == float(RESPAWN_SPAWN_GRACE_S)


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


def test_followup_active_pause_shaped_timeline():
    # #980: the pause-shaped event timeline. `set-status <N> on_hold` posts
    # `epm:status-changed` (it is in _DONE_TRANSITION_KINDS), so at pause
    # time the park is the newest done-transition and any PRIOR run-launched
    # is stale -> NOT a live follow-up -> the auto-stop proceeds. A follow-up
    # signal posted AFTER the park correctly re-arms the skip.
    import autonomous_session_watch as asw

    # run-launched OLDER than the on_hold park -> False (stop proceeds).
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:run-launched", "ts": "2026-07-01T10:00:00Z", "note": ""},
                # the pause commit point (`set-status <N> on_hold`)
                {"kind": "epm:status-changed", "ts": "2026-07-01T12:00:00Z", "note": ""},
            ],
        )
        is False
    )
    # followup-scope NEWER than the park -> True (followup-skip applies).
    assert (
        asw._task_followup_active(
            0,
            events=[
                {"kind": "epm:status-changed", "ts": "2026-07-01T12:00:00Z", "note": ""},
                {"kind": "epm:followup-scope", "ts": "2026-07-01T13:00:00Z", "note": ""},
            ],
        )
        is True
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
    # #980: the widened pod-safety trigger set must stay disjoint from
    # POD_ACTIVE too, and the paused overlay must not overlap the DONE set
    # (a member in both would make the union's provenance ambiguous).
    assert POD_SAFETY_AUTO_STOP.isdisjoint(POD_ACTIVE)
    assert AUTO_STOP_PAUSED.isdisjoint(AUTO_STOP_DONE)
    # blocked is deliberately in NEITHER (kept, alert-only-if-stale).
    assert "blocked" not in AUTO_STOP_DONE
    assert "blocked" not in POD_ACTIVE


def test_status_classes_subset_of_authoritative_enum():
    # Every status named by AUTO_STOP_DONE / AUTO_STOP_PAUSED / POD_ACTIVE MUST
    # exist in the
    # authoritative runtime enum task_workflow.STATUSES — otherwise the member
    # is a phantom that can never match what `_task_status` returns (the prior
    # round shipped `cancelled` / `uploading` / `followups_running` as phantoms,
    # silently making the auto-stop / no-auto-stop guarantees vacuous;
    # `followups_running` was later un-phantomed on 2026-06-10 — it joined the
    # runtime enum and POD_ACTIVE for the same-issue follow-up loop). This
    # pin catches that whole class of bug.

    enum = set(STATUSES)
    assert enum >= AUTO_STOP_DONE, f"phantom AUTO_STOP_DONE members: {AUTO_STOP_DONE - enum}"
    assert enum >= AUTO_STOP_PAUSED, f"phantom AUTO_STOP_PAUSED members: {AUTO_STOP_PAUSED - enum}"
    assert enum >= POD_ACTIVE, f"phantom POD_ACTIVE members: {POD_ACTIVE - enum}"


# ─── _status_class classifier ────────────────────────────────────────────────


def test_status_class_done_statuses():
    import autonomous_session_watch as asw

    now = 1_000_000.0
    # #980: iterate the full pod-safety trigger set (DONE + on_hold).
    for s in sorted(POD_SAFETY_AUTO_STOP):
        assert asw._status_class(s, latest_progress_ts=now, now=now) == "auto-stop-done"


def test_status_class_on_hold_is_auto_stop_done():
    # #980: a user-paused task's RUNNING pod is an escaped pod — the #919
    # pause affordance stops the pod BEFORE parking, so on_hold + RUNNING
    # means the teardown leg failed. Progress freshness is irrelevant on the
    # auto-stop arm (both None and fresh classify the same).
    import autonomous_session_watch as asw

    now = 1_000_000.0
    assert asw._status_class("on_hold", latest_progress_ts=None, now=now) == "auto-stop-done"
    assert asw._status_class("on_hold", latest_progress_ts=now, now=now) == "auto-stop-done"
    assert "on_hold" in AUTO_STOP_PAUSED
    assert "on_hold" in POD_SAFETY_AUTO_STOP


def test_on_hold_not_in_auto_stop_done_or_session_reconcile():
    # #980 decoupling pin: `on_hold` widens ONLY the pod-safety trigger set.
    # Folding it into AUTO_STOP_DONE would silently widen the
    # SESSION_RECONCILE_DONE alias and start reaping paused tasks' sessions
    # (the user may be live-parked in them — same conservatism as `blocked`).
    from autonomous_session_watch import SESSION_RECONCILE_DONE

    assert "on_hold" not in AUTO_STOP_DONE
    assert "on_hold" not in SESSION_RECONCILE_DONE
    assert AUTO_STOP_DONE | {"on_hold"} == POD_SAFETY_AUTO_STOP


def test_pod_safety_on_hold_exemptions_still_apply():
    # #980: the keep-running tag and the inferred-follow-up predicate key on
    # status_class == "auto-stop-done", which `on_hold` now produces — both
    # exemptions extend to a paused task with zero decide-layer changes.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    status_class = asw._status_class("on_hold", None, now)
    assert decide_pod_safety(
        status_class=status_class,
        missed=5,
        stale=False,
        alerted=False,
        keep_running=True,
    ) == ("keep-running-skip", 0)
    assert decide_pod_safety(
        status_class=status_class,
        missed=5,
        stale=False,
        alerted=False,
        followup_active=True,
    ) == ("followup-skip", 0)


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


def test_latest_progress_ts_excludes_deliberate_stop_breadcrumb():
    # The exact spawn_session.py cmd_stop emitter shape: a deliberate session
    # stop is the death record of the task's driver, not progress — it must
    # NOT advance the clock (#990; precedent #949/#810 in
    # task_workflow.stage_dispatch_should_skip).
    import autonomous_session_watch as asw

    events = [
        # Real progress carries a benign ``by`` — it must still COUNT (pins
        # the exclusion to by == "spawn_session-stop", not by-presence).
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T10:00:00Z",
            "by": "unknown",
            "note": "step 100",
        },
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T12:00:00Z",
            "by": "spawn_session-stop",
            "note": (
                "deliberate-stop pid=n/a target=happy-session:abc123 "
                "reason=operator stop via spawn_session.py stop"
            ),
        },
    ]
    ts = asw._latest_progress_ts(events)
    # The 12:00 stop record is excluded; newest real progress stays 10:00.
    assert ts == asw._parse_event_ts("2026-07-01T10:00:00Z")


def test_latest_progress_ts_counts_real_progress_with_benign_by():
    # A lone real-progress event with a benign ``by`` returns ITS timestamp
    # (non-None) — kills a by-presence mutation (`if ev.get("by"): continue`).
    import autonomous_session_watch as asw

    events = [
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T11:00:00Z",
            "by": "unknown",
            "note": "step 200",
        },
    ]
    assert asw._latest_progress_ts(events) == asw._parse_event_ts("2026-07-01T11:00:00Z")


def test_latest_progress_ts_excludes_by_field_regardless_of_note():
    # The by-half in isolation: by == "spawn_session-stop" excludes the event
    # even when the note lacks the "deliberate-stop " prefix (note-text
    # drift), and even when the ``note`` key is ABSENT entirely (pins the
    # `note = ev.get("note") or ""` normalization against the predicate).
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-07-01T10:00:00Z", "note": "step 100"},
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T12:00:00Z",
            "by": "spawn_session-stop",
            "note": "stopping for operator replacement",
        },
        # No "note" key at all — the by-half alone must exclude it.
        {"kind": "epm:progress", "ts": "2026-07-01T13:00:00Z", "by": "spawn_session-stop"},
    ]
    assert asw._latest_progress_ts(events) == asw._parse_event_ts("2026-07-01T10:00:00Z")


def test_latest_progress_ts_excludes_deliberate_stop_prefix_without_by():
    # The prefix-half in isolation: a PM-posted stop record (by="pm-chat",
    # the research-pm.md shape) is excluded on the lstripped note prefix
    # alone; leading whitespace does not defeat the lstrip.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-07-01T10:00:00Z", "note": "step 100"},
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T12:00:00Z",
            "by": "pm-chat",
            "note": "deliberate-stop pid=12345 target=tick-loop reason=operator-replace",
        },
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T13:00:00Z",
            "by": "pm-chat",
            "note": "  deliberate-stop pid=999 target=tick-loop reason=leading-whitespace",
        },
    ]
    assert asw._latest_progress_ts(events) == asw._parse_event_ts("2026-07-01T10:00:00Z")


def test_latest_progress_ts_midnote_deliberate_stop_mention_still_counts():
    # Prefix boundary (mirror of the #949 pin in test_stage_dispatch_dedup):
    # a note merely MENTIONING deliberate-stop mid-text, with no special
    # ``by``, DOES advance the clock.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-07-01T10:00:00Z", "note": "step 100"},
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T12:00:00Z",
            "note": "noting the earlier deliberate-stop was expected; resuming phase 2",
        },
    ]
    assert asw._latest_progress_ts(events) == asw._parse_event_ts("2026-07-01T12:00:00Z")


def test_latest_progress_ts_only_deliberate_stop_returns_none():
    # A list containing only the stop record has no real progress at all.
    import autonomous_session_watch as asw

    events = [
        {
            "kind": "epm:progress",
            "ts": "2026-07-01T12:00:00Z",
            "by": "spawn_session-stop",
            "note": "deliberate-stop pid=n/a target=happy-session:abc reason=operator stop",
        },
    ]
    assert asw._latest_progress_ts(events) is None


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
    got = sorted(asw._running_managed_issue_pods(), key=lambda t: t[0])
    # pod-444, pod-489, and the legacy epm-issue-377 are recognized; the EXITED
    # and unmanaged ones are excluded. The third element is the pod NAME,
    # threaded out so callers (e.g. the #488 stale-port self-heal in
    # ``_handle_stalled_alert``) can address the pod by name without a
    # second ``list_team_pods`` round-trip; the FOURTH (#692) is the live
    # ``PodInfo`` itself, so the wedge backstop can read the raw no-port wedge
    # condition off it without a second ``list_team_pods`` round-trip.
    assert [(i, pid, name) for i, pid, name, _info in got] == [
        (377, "pold", "epm-issue-377"),
        (444, "p444", "pod-444"),
        (489, "p489", "pod-489"),
    ]
    # The 4th element is the live PodInfo for that pod (pod_id matches).
    assert [info.pod_id for _i, _pid, _name, info in got] == ["pold", "p444", "p489"]
    assert all(isinstance(info, PodInfo) for *_rest, info in got)


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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(7, "p7", "pod-7")]
    )
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.pod_safety_pass(dry_run=False, threshold=1, now=now)  # threshold=1 -> stop immediately
    assert stops == [7]


def test_process_pod_on_hold_plain_two_tick_stop(isolated_registry, monkeypatch):
    # #980 incident shape, PLAIN (non-wedged, port-present) pod: a task parked
    # `on_hold` whose #919 pause teardown leg failed leaves a healthy RUNNING
    # pod. _process_pod must treat it exactly like the DONE escaped-pod case:
    # tick 1 accumulates (missed=1, no stop), tick 2 hits the 2-miss guard and
    # stops ONCE with the `auto-stop` marker posted, then the state clears.
    import json

    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_status", lambda issue: "on_hold")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    info = _p(980, "p980", "pod-980")[3]  # healthy PodInfo (SSH port present)
    state_path = isolated_registry / "pod-safety-980.json"

    asw._process_pod(980, "p980", info, now, dry_run=False, threshold=2)
    assert stops == []
    assert json.loads(state_path.read_text())["missed"] == 1

    asw._process_pod(980, "p980", info, now, dry_run=False, threshold=2)
    assert stops == [980]
    assert posts == [(980, "auto-stop")]
    assert not state_path.exists()  # cleared after stop


@pytest.mark.parametrize("status", sorted(POD_SAFETY_AUTO_STOP))
def test_pod_safety_auto_stop_dry_run_no_mutation(isolated_registry, monkeypatch, capsys, status):
    # Dry-run coverage on the auto-stop arm across the full #980 trigger set
    # (DONE + on_hold): under dry_run=True a would-stop candidate produces the
    # "would stop pod" log line ONLY — no `pod.py stop` subprocess, no marker
    # post, no per-pod state save.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(980, "p980", "pod-980")]
    )
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    # Any subprocess spawn under dry-run is a mutation-path leak — fail loud.
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess.run under dry_run")),
    )

    # threshold=1 -> the decision layer picks "stop" on the first tick.
    asw.pod_safety_pass(dry_run=True, threshold=1, now=now)

    out = capsys.readouterr().out
    assert "would stop pod" in out
    assert posts == []  # _stop_pod returns False under dry-run -> no marker
    assert not (isolated_registry / "pod-safety-980.json").exists()  # no state save


def test_auto_stop_failure_posts_stop_failed_marker_once_and_stays_retryable(
    isolated_registry, monkeypatch
):
    # #1155: a real `pod.py stop` failure (rc!=0 -> _stop_pod False, NOT
    # dry-run) must (1) post ONE durable stop-failed marker per episode,
    # (2) preserve the pod-safety state so the stop retries next tick,
    # (3) still auto-stop + clear state when a later retry succeeds.
    import json

    import autonomous_session_watch as asw

    now = 1_000_000.0
    stops: list[int] = []
    posts: list[tuple[int, str, str]] = []
    stop_results = iter([False, False, True])  # ticks 2, 3, 4
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(
        asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or next(stop_results)
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label, note)),
    )

    info = _p(489, "p489", "pod-489")[3]
    state_path = isolated_registry / "pod-safety-489.json"

    # tick 1: first miss accumulates (threshold=2) — no stop attempt yet.
    asw._process_pod(489, "p489", info, now, dry_run=False, threshold=2)
    assert stops == [] and posts == []
    assert json.loads(state_path.read_text())["missed"] == 1

    # tick 2: stop attempted and FAILS -> ONE stop-failed marker; state
    # preserved with the miss count untouched (episode retryable).
    asw._process_pod(489, "p489", info, now, dry_run=False, threshold=2)
    assert stops == [489]
    assert [(i, lbl) for i, lbl, _ in posts] == [(489, "stop-failed")]
    assert asw._AUTOSTOP_FAILED_NOTE_SENTINEL in posts[0][2]
    state = json.loads(state_path.read_text())
    assert state["stop_failed_noted"] is True
    assert state["missed"] == 1  # unchanged -> next tick re-fires "stop"

    # Interleaved other-arm save (no stop_failed_noted param): the None-carry
    # in _save_pod_safety_state must preserve the flag on disk, or the dedup
    # would silently reset whenever another arm saves state mid-episode.
    asw._save_pod_safety_state(
        489, "p489", missed=1, alerted=False, last_progress_ts=None, prev=state
    )
    assert json.loads(state_path.read_text())["stop_failed_noted"] is True

    # tick 3: stop RETRIED, fails again -> NO second marker (dedup).
    asw._process_pod(489, "p489", info, now, dry_run=False, threshold=2)
    assert stops == [489, 489]
    assert [(i, lbl) for i, lbl, _ in posts] == [(489, "stop-failed")]
    assert state_path.exists()

    # tick 4: stop retried and SUCCEEDS -> auto-stop marker, state cleared.
    asw._process_pod(489, "p489", info, now, dry_run=False, threshold=2)
    assert stops == [489, 489, 489]
    assert [(i, lbl) for i, lbl, _ in posts] == [
        (489, "stop-failed"),
        (489, "auto-stop"),
    ]
    assert not state_path.exists()


def test_autostop_failed_sentinel_in_watcher_self_set():
    # House pattern: a watcher-posted marker's sentinel must be excluded from
    # the real-progress staleness clocks (the _WATCHER_NOTE_SENTINELS
    # convention comment: add a new watcher-posted marker -> add it here).
    import autonomous_session_watch as asw

    assert asw._AUTOSTOP_FAILED_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(530, "p530", "pod-530")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(530, "p530", "pod-530")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(477, "p477", "pod-477")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(477, "p477", "pod-477")]
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
    monkeypatch.setattr(
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(7, "p7", "pod-7")]
    )
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(489, "p489", "pod-489")]
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


def test_stub_fleet_mutating_passes_covers_main_pass_roster():
    """Drift guard for the shared #1247 stub helper (task #1278): every
    ``*_pass(`` invocation in main() must be classified — either stubbed by
    ``_stub_fleet_mutating_passes`` (tests/conftest.py) or named in the
    benign/per-test set below. #1267 added boot_death_pass to main() and
    only one of the two pre-consolidation helper copies picked it up; with
    a single shared copy, this is the one remaining silent-drift channel.
    A NEW pass added to main() fails here until its author classifies it."""
    import inspect
    import re

    import autonomous_session_watch as asw

    invoked = set(re.findall(r"\b([a-z_]+_pass)\(", inspect.getsource(asw.main)))
    # Passes main() runs that the full-main() tests handle per-test
    # (recorders / per-test stubs, e.g. triage_observer_pass #967,
    # stale_blocked_flag_pass #1021) or that are safe against the test
    # fixtures (isolated_registry + the patched daemon/live-ids seams).
    # Classification inherited from the pre-#1278 helper call sites.
    benign_or_per_test = {
        "vm_disk_pass",
        "triage_observer_pass",
        "auth_outage_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "stale_registration_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
    }
    unclassified = invoked - set(_FLEET_MUTATING_PASS_NAMES) - benign_or_per_test
    assert not unclassified, (
        f"main() invokes unclassified pass(es) {sorted(unclassified)} — add each to "
        "_FLEET_MUTATING_PASS_NAMES (tests/conftest.py) if it can mutate fleet / live "
        "VM state, else to benign_or_per_test here (the #1267 boot_death_pass gap)."
    )
    # Guard-hollowing direction: if a refactor moves the fleet-mutating passes
    # out of main()'s direct body (a decomposition into sub-dispatchers), the
    # regex-derived `invoked` set shrinks and `unclassified` stays empty — the
    # guard above would pass while covering nothing. Fail LOUD instead.
    missing_stubbed = set(_FLEET_MUTATING_PASS_NAMES) - invoked
    assert not missing_stubbed, (
        f"stubbed pass(es) {sorted(missing_stubbed)} are no longer invoked directly in "
        "main()'s source — a main() decomposition refactor hollows this drift guard. "
        "Point the regex at the new dispatch body (or prune retired names from "
        "_FLEET_MUTATING_PASS_NAMES in tests/conftest.py)."
    )
    # Symmetric stale-name check for the allowlist (same fail-loud direction).
    stale_allowlisted = benign_or_per_test - invoked
    assert not stale_allowlisted, (
        f"allowlisted pass(es) {sorted(stale_allowlisted)} are no longer invoked in "
        "main() — prune them from benign_or_per_test so the classification stays honest."
    )
    # Stale-name direction: every stubbed name must still exist on the module
    # (monkeypatch.setattr raising=True also enforces this at every call site).
    for name in _FLEET_MUTATING_PASS_NAMES:
        assert callable(getattr(asw, name)), name


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
    # Bypass the #845 retry wrapper's real 5s/10s backoff sleeps — the
    # single-probe stub above already encodes "daemon down" for this test.
    monkeypatch.setattr(asw, "_daemon_reachable_with_retry", lambda *a, **kw: False)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))
    monkeypatch.setattr(asw, "_process_entry", lambda *a, **kw: respawn_entry_calls.append((a, kw)))
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: vm_disk_calls.append((a, kw)))
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: orphan_calls.append((a, kw)))
    # #967: never sweep the LIVE registry/events tree from a unit test.
    monkeypatch.setattr(asw, "triage_observer_pass", lambda *a, **kw: None)
    # #1021: the stale-blocked flag pass shells task.py against the LIVE
    # blocked set (daemon-independent) — never run it from a unit test.
    monkeypatch.setattr(asw, "stale_blocked_flag_pass", lambda *a, **kw: None)
    _stub_fleet_mutating_passes(asw, monkeypatch)

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
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    monkeypatch.setattr(asw, "_live_children", lambda: snapshot)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: None)
    # #967: never sweep the LIVE registry/events tree from a unit test.
    monkeypatch.setattr(asw, "triage_observer_pass", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: orphan_calls.append((a, kw)))
    # Patched so the unit test never RPCs the real daemon / scans real /proc /
    # spawns task.py subprocesses for whatever sessions are live on the VM.
    monkeypatch.setattr(asw, "zombie_wrapper_pass", lambda *a, **kw: zombie_calls.append((a, kw)))
    monkeypatch.setattr(asw, "idle_unmapped_pass", lambda *a, **kw: idle_calls.append((a, kw)))
    # #1021: the stale-blocked flag pass shells task.py against the LIVE
    # blocked set (daemon-independent) — never run it from a unit test.
    monkeypatch.setattr(asw, "stale_blocked_flag_pass", lambda *a, **kw: None)
    # #1247: with _daemon_reachable forced True, the UNSTUBBED sweep passes
    # would dispatch REAL sessions (this exact test spawned one for #1227).
    _stub_fleet_mutating_passes(asw, monkeypatch)

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
        # #1155: the stop arm's failed-branch dedup flag is part of the schema
        # now; a save with no prior episode defaults it False.
        "stop_failed_noted": False,
        "first_seen": 1234.0,
        # #692 MF3: the wedge fields are part of the schema now; a status-class
        # save with no wedge state defaults them (no prior wedge to carry).
        "wedge_first_seen": None,
        "wedge_missed": 0,
        "wedge_alerted": False,
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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

    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
        lambda issue, cap, dry_run: spawns.append((issue, cap)) or "spawned",
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
    # Neutralize the #845 host-state probes the same way: the worktree-
    # activity hold walks the REAL .claude/worktrees/issue-<N> tree, and the
    # stop-failed / daemon-blocked escalations enqueue a REAL phone push.
    # Hold/wedge/push-specific tests re-patch these explicitly.
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    return stops, spawns, markers


def test_stalled_active_status_auto_respawns_after_two_misses(
    isolated_registry, monkeypatch, stalled_recorder
):
    # An ACTIVE-status stalled session auto-respawns instead of alerting
    # only. #845 (a-ii): the respawn is FENCED — the stop fires on the tick
    # the action trips, and the spawn only on the NEXT tick, after the sid
    # is verified absent from the live set (stop != kill; #763).
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 518, "sess-518", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1: increments to missed=1, no action.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []

    # Tick 2: threshold met, ACTIVE + daemon_reachable -> respawn action;
    # the fence issues the STOP only (no spawn, no marker yet).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-518"]
    assert spawns == [] and markers == []

    # Tick 3: pending sid verified dead (absent from the live set) -> spawn.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-518"]  # no second stop
    assert spawns == [(518, 24.0)]
    assert markers == [(518, "session-auto-respawn")]


# ─── #759 bug class b.1: live-session bounded K-escalation ───────────────────
# When the stalled detector wants to respawn an ACTIVE session whose Happy id
# is STILL in the daemon's live set, the first K-1 consecutive episodes
# downgrade respawn->alert (no duplicate driver on a transient busy stretch);
# the Kth escalates to the canonical respawn (#506 dead-bg-chain class). A
# genuinely-dead id (not in live_ids) respawns immediately, K not consulted.
# Criteria 6/7/11/12 pin all branches + the counter resets.


def _read_live_escalation_events(reg_dir):
    """Return the parsed rows of the #759 stalled-live-escalation sidecar
    (``[]`` when the file was never written)."""
    import json

    path = reg_dir / "stalled-live-escalation-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _read_live_consecutive(reg_dir, issue):
    """Return the persisted live_consecutive from stalled-<issue>.json
    (``None`` when the state file is absent)."""
    import json

    path = reg_dir / f"stalled-{issue}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("live_consecutive")


def test_stalled_live_k_minus_1_downgrades_to_alert(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Criterion 6 (default K=2 -> K-1 = 1 downgrade tick): a stale ACTIVE
    # session whose id is IN live_ids, with NO provision / followups exemption,
    # downgrades respawn->alert. Across the 2 ticks needed to reach the respawn
    # decision: ZERO respawns, exactly ONE session-stalled-alert, and
    # live_consecutive == 1 (== K-1) persisted afterward.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)  # default K=2
    _write_autonomous_entry(isolated_registry, 739, "sess-739", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1: first miss only (below threshold) -> keep, live_consecutive reset 0.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
    )
    assert stops == [] and spawns == [] and markers == []

    # Tick 2: threshold met -> decide() wants respawn, but the LIVE id triggers
    # the 1st live-stall episode (1 < K=2) -> downgrade to alert.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
    )
    assert spawns == []  # NO duplicate driver on the live session
    assert stops == []  # the alert arm never stops the session
    assert markers == [(739, "session-stalled-alert")]
    assert _read_live_consecutive(isolated_registry, 739) == 1  # == K-1

    rows = _read_live_escalation_events(isolated_registry)
    assert len(rows) == 1
    assert rows[0]["issue"] == 739
    assert rows[0]["event"] == "stalled-live-downgrade"
    assert rows[0]["live_consecutive"] == 1
    assert rows[0]["k"] == 2


def test_stalled_live_kth_episode_escalates_to_respawn(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Criterion 7 (default K=2): continue the criterion-6 scenario one more
    # tick (same live_ids, still stale). The Kth (2nd) consecutive live-stall
    # episode ESCALATES to the canonical respawn (stop+spawn), and resets
    # live_consecutive to 0. A persistently-stalled LIVE wrapper (#506 class)
    # is recovered, not alert-only forever.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)  # default K=2
    _write_autonomous_entry(isolated_registry, 739, "sess-739", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Ticks 1-2: reach the 1st live-stall episode (downgrade -> alert).
    for _ in range(2):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
        )
    assert spawns == [] and stops == []
    assert markers == [(739, "session-stalled-alert")]
    assert _read_live_consecutive(isolated_registry, 739) == 1

    # Tick 3: alerted=True short-circuits the miss guard -> decide() returns
    # respawn -> the 2nd (== K) consecutive live-stall episode ESCALATES.
    # #845 fence: the escalation tick issues the STOP only; the spawn waits
    # for the next tick's verified-dead read.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
    )
    assert stops == ["sess-739"]
    assert spawns == []
    assert markers == [(739, "session-stalled-alert")]
    # The escalation reset the counter (a fresh --auto session = a new episode).
    assert _read_live_consecutive(isolated_registry, 739) == 0

    # Tick 4: the stop landed (sid gone from the live set) -> verified dead
    # -> the fence spawns.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids=set()
    )
    assert spawns == [(739, 24.0)]
    assert markers == [(739, "session-stalled-alert"), (739, "session-auto-respawn")]

    rows = _read_live_escalation_events(isolated_registry)
    assert [r["event"] for r in rows] == ["stalled-live-downgrade", "stalled-live-escalation"]
    assert rows[1]["live_consecutive"] == 2  # the count that tripped the escalation
    assert rows[1]["k"] == 2


def test_stalled_live_k_equals_3_takes_one_alert_then_respawn(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Criteria 6+7 generalized for K=3: K-1 = 2 downgrade EPISODES (both in
    # the sidecar), then the 3rd escalates. Pins that the escalation tracks
    # the env-tuned K, not a hardcoded 2. #1137: only the FIRST downgrade
    # posts a session-stalled-alert marker — the second downgrade reaches
    # the alert handler with alerted=True and its marker is suppressed
    # (marker-only dedup; the counter/sidecar dynamics are unchanged).
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "3")
    _write_autonomous_entry(isolated_registry, 739, "sess-739", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1 (miss=1, keep) + tick 2 (1st live-stall episode -> alert).
    for _ in range(2):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
        )
    assert spawns == [] and markers == [(739, "session-stalled-alert")]
    assert _read_live_consecutive(isolated_registry, 739) == 1

    # Tick 3: alerted=True -> respawn-wanted -> 2nd episode (2 < 3) -> the
    # downgrade fires again but the repeat alert MARKER is suppressed (#1137
    # episode-total dedup); the incremented counter still persists (the dedup
    # is marker-only, never a dynamics change).
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
    )
    assert spawns == []
    assert markers == [(739, "session-stalled-alert")]
    assert _read_live_consecutive(isolated_registry, 739) == 2

    # Tick 4: 3rd (== K) episode -> ESCALATE to respawn, reset to 0. #845
    # fence: stop on this tick, spawn on the next verified-dead tick.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-739"}
    )
    assert stops == ["sess-739"]
    assert spawns == []
    assert _read_live_consecutive(isolated_registry, 739) == 0

    # Tick 5: the stop landed -> verified dead -> spawn.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids=set()
    )
    assert spawns == [(739, 24.0)]
    rows = _read_live_escalation_events(isolated_registry)
    assert [r["event"] for r in rows] == [
        "stalled-live-downgrade",
        "stalled-live-downgrade",
        "stalled-live-escalation",
    ]


def test_stalled_dead_id_respawns_immediately_k_not_consulted(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Criterion 11 (the regression guard): the SAME stale scenario but the
    # entry's id is NOT in live_ids (genuinely-dead wrapper). The K counter is
    # for LIVE ids only -> respawn fires on the 2nd miss exactly as before, no
    # downgrade, and live_consecutive stays 0 (reset on the dead path). No
    # escalation-sidecar row is written.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)
    _write_autonomous_entry(isolated_registry, 740, "sess-740", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # live_ids contains some OTHER session, so _session_alive(entry) is False.
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"other-sid"}
    )
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"other-sid"}
    )
    # #845 fence: even a genuinely-dead wrapper waits ONE tick between stop
    # and spawn (the fence predicate is evaluated once per tick) — that
    # +10 min is deliberate, it is what closes #763.
    assert stops == ["sess-740"]
    assert spawns == []
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"other-sid"}
    )
    assert spawns == [(740, 24.0)]
    assert markers == [(740, "session-auto-respawn")]
    assert _read_live_consecutive(isolated_registry, 740) == 0
    assert _read_live_escalation_events(isolated_registry) == []


def test_stalled_keep_resets_live_consecutive(isolated_registry, monkeypatch, stalled_recorder):
    # Criterion 12 (the "recovered then re-stalled" guard): drive one
    # live-stall tick (-> live_consecutive becomes 1 via the downgrade), then a
    # tick where decide() returns keep/clear (self-report is now FRESH) ->
    # live_consecutive resets to 0, so a LATER unrelated live stall starts its
    # K count fresh and the K-1 debounce is never short-cut by a stale counter.
    import autonomous_session_watch as asw

    # stalled_recorder's monkeypatching of the side-effects is what we need
    # (the assertions read the persisted state file, not the recorded calls).
    _ = stalled_recorder
    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)  # default K=2
    _write_autonomous_entry(isolated_registry, 741, "sess-741", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Ticks 1-2: reach the 1st live-stall episode (downgrade -> alert),
    # live_consecutive == 1.
    for _ in range(2):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-741"}
        )
    assert _read_live_consecutive(isolated_registry, 741) == 1

    # Tick 3: the self-report is now FRESH (5 min old) -> decide() returns keep
    # -> the corroboration's non-respawn branch resets live_consecutive to 0.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (5 * 60, "ts-fresh"))
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-741"}
    )
    assert _read_live_consecutive(isolated_registry, 741) == 0


# ─── #1137: episode-total alert dedup across BOTH alert producers ─────────────
# decide_session_stalled's alerted=True dedup covers only its own keep branch;
# the #759 downgrade lane rewrites respawn->alert AFTER decide() and used to
# post a fresh session-stalled-alert marker every eligible tick. Incident
# #1092 (2026-07-07): the escalate -> wt-hold-defer -> downgrade 2-tick cycle
# posted one marker every 20 min, indefinitely, on a healthy session.


def _read_stalled_alerted(reg_dir, issue):
    """Return the persisted ``alerted`` flag from stalled-<issue>.json
    (``None`` when the state file is absent)."""
    import json

    path = reg_dir / f"stalled-{issue}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("alerted")


def test_stalled_live_downgrade_repeat_alert_suppressed_across_wt_held_cycle(
    isolated_registry, monkeypatch, stalled_recorder
):
    # The #1092 2026-07-07 replay (fix A acceptance 1): K=2, LIVE sid, FRESH
    # worktree activity. Steady-state 2-tick cycle — escalate (criterion-7
    # reset) -> #845 (b) wt-hold defer -> downgrade 1/2 -> alert — reached
    # _handle_stalled_alert with alerted=True every other tick. Post-fix:
    # exactly ONE session-stalled-alert marker for the whole 6-tick episode
    # (pre-#1137 baseline: 3), zero stops, zero spawns; the sidecar
    # downgrade/escalation observability and the alerted persist survive.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)  # default K=2
    _write_autonomous_entry(isolated_registry, 1092, "sess-1092", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    # Fresh worktree activity (#1092: P0's file writes kept the worktree
    # warm) -> every escalated respawn is DEFERRED by the #845 (b) hold,
    # which persists the already-reset counter -> the next tick is a fresh
    # 1/2 downgrade. Re-patch AFTER the fixture's default False.
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    now = 1_000_000.0

    for _ in range(6):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-1092"}
        )

    assert markers == [(1092, "session-stalled-alert")]  # ONE per episode (was 3)
    assert stops == []
    assert spawns == []
    assert _read_stalled_alerted(isolated_registry, 1092) is True
    rows = _read_live_escalation_events(isolated_registry)
    assert [r["event"] for r in rows] == [
        "stalled-live-downgrade",
        "stalled-live-escalation",
        "stalled-live-downgrade",
        "stalled-live-escalation",
        "stalled-live-downgrade",
    ]


def test_stalled_repeat_alert_dry_run_no_marker_no_persist(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Fix A dry-run discipline: drive the #1092 replay to the repeat-downgrade
    # tick with real writes, then run the SUPPRESSED tick under dry_run=True —
    # the new dedup path posts no marker AND persists nothing (the state file
    # is unchanged; _persist_stalled_ctx no-ops on dry_run). The recorder's
    # _post_progress_marker patch records regardless of dry_run, so a marker
    # post on this tick would be visible.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)  # default K=2
    _write_autonomous_entry(isolated_registry, 1092, "sess-1092", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    now = 1_000_000.0

    # Ticks 1-3 (real): miss, downgrade -> first alert, escalate -> wt-hold.
    for _ in range(3):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-1092"}
        )
    assert markers == [(1092, "session-stalled-alert")]
    state_before = (isolated_registry / "stalled-1092.json").read_text()

    # Tick 4 (dry run): the repeat downgrade hits the #1137 dedup branch.
    asw.stalled_session_pass(
        dry_run=True, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-1092"}
    )
    assert markers == [(1092, "session-stalled-alert")]  # no new marker
    assert (isolated_registry / "stalled-1092.json").read_text() == state_before
    assert stops == [] and spawns == []


def test_stalled_repeat_alert_refires_on_new_episode(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Fix A is EPISODE-scoped, not permanent silence (acceptance 4): after the
    # self-report ts ADVANCES (episode over -> alerted cleared) and a new
    # staleness episode forms, a fresh alert fires again. K is pinned high so
    # every respawn downgrades to alert (no stop/fence noise in the read).
    # Episode advancement is a LEXICOGRAPHIC string compare on the raw
    # self-report ts, so the ts sequence must be lexicographically increasing:
    # "ts-old" -> "ts-old-2" -> "ts-old-3".
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "99")
    _write_autonomous_entry(isolated_registry, 1093, "sess-1093", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0
    signal = {"age": STALLED_WINDOW_S + 60, "ts": "ts-old"}
    monkeypatch.setattr(
        asw, "_self_report_age_seconds", lambda issue, now: (signal["age"], signal["ts"])
    )

    # Episode 1: miss -> downgrade -> ONE alert; the tick-3 repeat downgrade
    # is suppressed (pure downgrade-lane shape, no wt hold needed).
    for _ in range(3):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-1093"}
        )
    assert markers == [(1093, "session-stalled-alert")]

    # Self-report advances + goes fresh: the episode is over, alerted clears.
    signal["age"] = 60.0
    signal["ts"] = "ts-old-2"
    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-1093"}
    )
    assert _read_stalled_alerted(isolated_registry, 1093) is False

    # New staleness episode (a third, newer ts, stale again): re-alerts once.
    signal["age"] = STALLED_WINDOW_S + 60
    signal["ts"] = "ts-old-3"
    for _ in range(3):
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids={"sess-1093"}
        )
    assert markers == [(1093, "session-stalled-alert"), (1093, "session-stalled-alert")]
    assert stops == [] and spawns == []


# ─── #1137 fix (B): deliberate-blocked-park alert suppression ─────────────────
# The CLAUDE.md halt contract posts epm:failure then sets status blocked (1s
# apart on #1092); a stalled-session alert on that shape duplicates a by-design
# park the gate-push pass already phone-pushed. A blocked task with NO failure
# trail (hand-moved / unexplained) keeps the one-time alert (fail-open).


def _blocked_park_events(*, with_failure_trail: bool):
    """Events for the #1092 blocked-park shape. The ts are 1970-era so their
    age under the fake ``now=1_000_000.0`` (~1970-01-12) is ~11.5 days — far
    past the 2h STALLED_MARKER_WINDOW_S_DEFAULT (a 2026 ISO ts would read as
    NEGATIVE-age fresh and the detector would vacuously keep). Both blocked
    tests share this helper so the no-trail positive control provably runs
    under the SAME ts scheme as the suppressed trail case."""
    events = [{"kind": "epm:status-changed", "ts": "1970-01-01T00:00:01Z"}]
    if with_failure_trail:
        # 1s BEFORE the status change — the canonical halt-contract pair
        # (#1092: 13:21:34Z epm:failure, 13:21:35Z status-changed -> blocked).
        events.insert(0, {"kind": "epm:failure", "ts": "1970-01-01T00:00:00Z"})
    return events


def test_stalled_blocked_deliberate_park_suppresses_alert(
    isolated_registry, monkeypatch, stalled_recorder
):
    # The #1092 15:33Z replay (acceptance 2): status=blocked with the
    # halt-contract trail, both staleness signals stale -> the deliberate-
    # blocked-park exemption suppresses the alert entirely (print-only, no
    # marker — the epm:failure marker itself carries the user ask, and the
    # gate-push pass already phone-pushed the blocked transition). Pre-#1137
    # baseline: 1 alert marker on the 2nd miss.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1092, "sess-1092", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="blocked")
    monkeypatch.setattr(
        asw, "_task_events", lambda issue: _blocked_park_events(with_failure_trail=True)
    )
    now = 1_000_000.0

    for _ in range(3):
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert markers == []
    assert stops == [] and spawns == []


def test_stalled_blocked_without_failure_trail_still_alerts(
    isolated_registry, monkeypatch, stalled_recorder
):
    # Fail-open positive control (acceptance 3): status=blocked with NO
    # epm:failure trail (hand-moved / unexplained block) keeps the one-time
    # alert — exactly ONE marker after 2 misses, then decide()'s own
    # alerted dedup holds. Same events helper / ts scheme as the suppressed
    # trail case above.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1094, "sess-1094", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="blocked")
    monkeypatch.setattr(
        asw, "_task_events", lambda issue: _blocked_park_events(with_failure_trail=False)
    )
    now = 1_000_000.0

    for _ in range(3):
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert markers == [(1094, "session-stalled-alert")]
    assert stops == [] and spawns == []


def test_deliberate_blocked_park_reason_trail_within_window():
    # The canonical 1s-apart halt-contract pair -> reason fires.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:failure", "ts": "1970-01-02T00:00:00Z"},
        {"kind": "epm:status-changed", "ts": "1970-01-02T00:00:01Z"},
    ]
    assert asw._deliberate_blocked_park_reason(events) is not None


def test_deliberate_blocked_park_reason_failure_older_than_window():
    # Failure > 3600s BEFORE the newest status change -> no trail -> None.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:failure", "ts": "1970-01-01T00:00:00Z"},
        {"kind": "epm:status-changed", "ts": "1970-01-01T02:00:00Z"},
    ]
    assert asw._deliberate_blocked_park_reason(events) is None


def test_deliberate_blocked_park_reason_failure_after_slack():
    # Failure > 300s AFTER the newest status change (order inversion beyond
    # the slack) -> None.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:status-changed", "ts": "1970-01-01T00:00:00Z"},
        {"kind": "epm:failure", "ts": "1970-01-01T00:05:01Z"},
    ]
    assert asw._deliberate_blocked_park_reason(events) is None


def test_deliberate_blocked_park_reason_unparseable_ts_fails_open():
    # Unparseable ts on both rows -> None (fail-open: the alert still fires).
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:failure", "ts": "not-a-timestamp"},
        {"kind": "epm:status-changed", "ts": "also-garbage"},
    ]
    assert asw._deliberate_blocked_park_reason(events) is None


def test_deliberate_blocked_park_reason_empty_events():
    import autonomous_session_watch as asw

    assert asw._deliberate_blocked_park_reason([]) is None


def test_deliberate_blocked_park_reason_newest_status_change_wins():
    # The failure trails an OLD status change; the NEWEST status change
    # (a day later, e.g. a manual re-block) has no failure inside its
    # window -> None (the trail must corroborate the CURRENT block).
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:failure", "ts": "1970-01-01T00:00:00Z"},
        {"kind": "epm:status-changed", "ts": "1970-01-01T00:00:01Z"},
        {"kind": "epm:status-changed", "ts": "1970-01-02T00:00:00Z"},
    ]
    assert asw._deliberate_blocked_park_reason(events) is None


# ─── #759 bug class b.2: STALLED_WINDOW_S raised 45 -> 60 min, env-tunable ────


def test_stalled_window_default_is_sixty_minutes():
    # Criterion 8 floor + the b.2 raise: the module default is now 60 min, and
    # a short (5-min) no-marker subagent stretch is FAR under it.
    import autonomous_session_watch as asw

    assert asw.STALLED_WINDOW_S == 60 * 60
    # A 5-min-old self-report + 5-min-old marker -> keep (well inside window),
    # regardless of the new vs old window (documents the regression floor).
    assert asw.decide_session_stalled(
        self_report_age_s=5 * 60,
        marker_progress_age_s=5 * 60,
        has_pod=False,
        missed=1,
        alerted=False,
        respawn_eligible=True,
    ) == ("keep", 0)


def test_stalled_window_50min_keeps_under_new_window():
    # Criterion 9 (the test that DISTINGUISHES the new 60-min window from the
    # old 45-min one): a self-report + marker 50 min old sits BETWEEN the old
    # and new windows. With the new 60-min window the window is NOT yet open ->
    # keep. (Under the old 45-min window this same input would have respawned.)
    import autonomous_session_watch as asw

    age_50min = 50 * 60
    # First miss path: 50 min < 60 min window -> not stale -> keep, missed reset.
    assert asw.decide_session_stalled(
        self_report_age_s=age_50min,
        marker_progress_age_s=age_50min,
        has_pod=False,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        window_s=asw.STALLED_WINDOW_S,
    ) == ("keep", 0)
    # Control: under the OLD 45-min window this same input DOES open the window
    # (proving the input is genuinely between the two windows, not trivially
    # under both). marker_window_s is pinned alongside window_s here — the
    # control is about the SELF-REPORT window, and the #845 2h marker default
    # would otherwise mask it (a 50-min marker is fresh under 2h).
    assert asw.decide_session_stalled(
        self_report_age_s=age_50min,
        marker_progress_age_s=age_50min,
        has_pod=False,
        missed=1,
        alerted=False,
        respawn_eligible=True,
        window_s=45 * 60,
        marker_window_s=45 * 60,
    ) == ("respawn", 0)


def test_stalled_window_env_override_and_fallback(monkeypatch):
    # Criterion 10 (b-half): EPM_STALLED_WINDOW_MIN parses a valid int (minutes)
    # and falls back to the default on a garbled value.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_STALLED_WINDOW_MIN", raising=False)
    assert asw._stalled_window_s() == float(asw.STALLED_WINDOW_S_DEFAULT)

    monkeypatch.setenv("EPM_STALLED_WINDOW_MIN", "90")
    assert asw._stalled_window_s() == 90 * 60.0

    monkeypatch.setenv("EPM_STALLED_WINDOW_MIN", "garbled")
    assert asw._stalled_window_s() == float(asw.STALLED_WINDOW_S_DEFAULT)


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


# ─── long-phase-heartbeat exemption (incident #761) ─────────────────────────


def _hb_event(asw, *, ts_iso: str, note: str, kind: str = "epm:progress") -> dict:
    """Build a single task event dict for the heartbeat-reason tests."""
    return {"kind": kind, "ts": ts_iso, "note": note}


def test_long_phase_heartbeat_freshness_env_override_and_fallback(monkeypatch):
    # Mirror test_stalled_window_env_override_and_fallback: the env knob parses
    # a valid int (minutes) and falls back to the default on a garbled value.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN", raising=False)
    assert (
        asw._long_phase_heartbeat_freshness_s()
        == float(asw.LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT)
        == 90 * 60.0
    )

    monkeypatch.setenv("EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN", "120")
    assert asw._long_phase_heartbeat_freshness_s() == 120 * 60.0

    monkeypatch.setenv("EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN", "not-a-number")
    assert asw._long_phase_heartbeat_freshness_s() == float(
        asw.LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT
    )


def test_long_phase_heartbeat_sentinel_not_in_watcher_self_set():
    """[long-phase-heartbeat] is REAL progress (emitter-stamped, not watcher-posted),
    so adding it to _WATCHER_NOTE_SENTINELS would break the design: heartbeats
    would be excluded from _latest_progress_ts and stop counting as progress
    (the exemption would still fire but the underlying staleness signal would
    be broken). Guard the inverse-of-watcher-self property."""
    import autonomous_session_watch as asw

    for sentinel in asw._WATCHER_NOTE_SENTINELS:
        assert asw._LONG_PHASE_HEARTBEAT_PREFIX not in sentinel, (
            f"_LONG_PHASE_HEARTBEAT_PREFIX leaked into _WATCHER_NOTE_SENTINELS via {sentinel!r}"
        )


def test_long_phase_heartbeat_reason_fresh_returns_reason():
    # A fresh epm:progress note carrying the sentinel -> non-None reason string.
    import autonomous_session_watch as asw

    ts_iso = "2026-06-30T12:00:00Z"
    ts = asw._parse_event_ts(ts_iso)
    events = [
        _hb_event(asw, ts_iso=ts_iso, note=f"{asw._LONG_PHASE_HEARTBEAT_PREFIX} verifying r2")
    ]
    reason = asw._long_phase_heartbeat_reason(events, now=ts + 30 * 60.0)  # 30m < 90m window
    assert reason is not None and "heartbeat" in reason


def test_long_phase_heartbeat_reason_sentinel_absent_returns_none():
    # An ordinary fresh epm:progress note WITHOUT the sentinel -> no exemption.
    import autonomous_session_watch as asw

    ts_iso = "2026-06-30T12:00:00Z"
    ts = asw._parse_event_ts(ts_iso)
    events = [_hb_event(asw, ts_iso=ts_iso, note="verifying r2 (no opt-in)")]
    assert asw._long_phase_heartbeat_reason(events, now=ts + 30 * 60.0) is None


def test_long_phase_heartbeat_reason_non_progress_kind_with_sentinel_returns_none():
    """Pin the kind=='epm:progress' filter (asw.py:4335): a FRESH non-progress
    event (e.g. epm:status-changed) whose note carries the heartbeat sentinel
    MUST NOT exempt the session — only epm:progress kinds count as heartbeats.
    Guards against a future refactor that broadens the predicate; without the
    kind filter this event's age (30m < 90m window) would return a reason.
    """
    import autonomous_session_watch as asw

    ts_iso = "2026-06-30T12:00:00Z"
    ts = asw._parse_event_ts(ts_iso)
    events = [
        _hb_event(
            asw,
            ts_iso=ts_iso,
            kind="epm:status-changed",
            note=f"some status change {asw._LONG_PHASE_HEARTBEAT_PREFIX} carrying the sentinel",
        )
    ]
    # 30m < 90m freshness window: would be fresh-enough to exempt IF the kind
    # filter were absent. The kind filter is what makes this return None.
    assert asw._long_phase_heartbeat_reason(events, now=ts + 30 * 60.0) is None


def test_long_phase_heartbeat_reason_stale_returns_none():
    # Sentinel present but older than the freshness window -> no exemption.
    import autonomous_session_watch as asw

    ts_iso = "2026-06-30T12:00:00Z"
    ts = asw._parse_event_ts(ts_iso)
    events = [
        _hb_event(asw, ts_iso=ts_iso, note=f"{asw._LONG_PHASE_HEARTBEAT_PREFIX} verifying r2")
    ]
    # well past LONG_PHASE_HEARTBEAT_FRESH_S (default 90 min)
    now = ts + asw.LONG_PHASE_HEARTBEAT_FRESH_S + 60.0
    assert asw._long_phase_heartbeat_reason(events, now=now) is None


def test_long_phase_heartbeat_reason_watcher_self_note_excluded():
    # A fresh note carrying BOTH the heartbeat sentinel AND a watcher-self
    # sentinel is filtered out (the watcher-self filter wins) -> no exemption.
    import autonomous_session_watch as asw

    ts_iso = "2026-06-30T12:00:00Z"
    ts = asw._parse_event_ts(ts_iso)
    watcher_sentinel = next(iter(asw._WATCHER_NOTE_SENTINELS))
    note = f"{watcher_sentinel} {asw._LONG_PHASE_HEARTBEAT_PREFIX} spurious"
    events = [_hb_event(asw, ts_iso=ts_iso, note=note)]
    assert asw._long_phase_heartbeat_reason(events, now=ts + 30 * 60.0) is None


def test_long_phase_heartbeat_reason_empty_events_returns_none():
    # Empty events list -> no exemption (no crash on best_ts is None).
    import autonomous_session_watch as asw

    assert asw._long_phase_heartbeat_reason([], now=1_000_000.0) is None


def test_long_phase_heartbeat_reason_future_ts_returns_none():
    # A future ts (negative age, clock skew / fake clock) is NOT fresh -> None,
    # mirroring the `0 <= age` guard in _provision_in_flight_reason.
    import autonomous_session_watch as asw

    ts_iso = "2026-06-30T12:00:00Z"
    ts = asw._parse_event_ts(ts_iso)
    events = [_hb_event(asw, ts_iso=ts_iso, note=f"{asw._LONG_PHASE_HEARTBEAT_PREFIX} verifying")]
    assert asw._long_phase_heartbeat_reason(events, now=ts - 60.0) is None


def test_long_phase_heartbeat_reason_newest_wins_older_fresh_newer_stale():
    # Multiple sentinel events: the NEWEST is chosen. Older one fresh but the
    # newer one stale -> None (the newest decides).
    import autonomous_session_watch as asw

    older_iso = "2026-06-30T12:00:00Z"
    newer_iso = "2026-06-30T20:00:00Z"  # 8h later -> stale vs a `now` just past it
    older_ts = asw._parse_event_ts(older_iso)
    newer_ts = asw._parse_event_ts(newer_iso)
    sentinel = asw._LONG_PHASE_HEARTBEAT_PREFIX
    events = [
        _hb_event(asw, ts_iso=older_iso, note=f"{sentinel} older"),
        _hb_event(asw, ts_iso=newer_iso, note=f"{sentinel} newer"),
    ]
    # now is 2h past the NEWER event (stale, > 90m) but the older event would be
    # ~10h old (also stale) — so this case confirms newest-wins-and-is-stale.
    now = newer_ts + 2 * 3600.0
    assert now - older_ts > asw.LONG_PHASE_HEARTBEAT_FRESH_S  # older also stale
    assert asw._long_phase_heartbeat_reason(events, now=now) is None


def test_long_phase_heartbeat_reason_newest_wins_older_stale_newer_fresh():
    # Older one stale, newer one fresh -> non-None (the newest, fresh, decides).
    import autonomous_session_watch as asw

    older_iso = "2026-06-30T12:00:00Z"
    newer_iso = "2026-06-30T20:00:00Z"
    newer_ts = asw._parse_event_ts(newer_iso)
    sentinel = asw._LONG_PHASE_HEARTBEAT_PREFIX
    events = [
        _hb_event(asw, ts_iso=older_iso, note=f"{sentinel} older"),
        _hb_event(asw, ts_iso=newer_iso, note=f"{sentinel} newer"),
    ]
    reason = asw._long_phase_heartbeat_reason(events, now=newer_ts + 30 * 60.0)  # 30m < 90m
    assert reason is not None and "heartbeat" in reason


def test_stalled_exemption_fresh_heartbeat_blocks_respawn(
    isolated_registry, monkeypatch, stalled_recorder
):
    # incident #761: a session whose only recent activity is a fresh
    # long-phase heartbeat is NOT stalled. When the heartbeat-reason probe
    # returns a reason, the stalled detector must neither accumulate misses
    # nor stop/spawn/post markers across ticks past the threshold. Mirrors
    # test_stalled_exemption_live_provision_blocks_respawn.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 761, "sess-761", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(
        asw,
        "_long_phase_heartbeat_reason",
        lambda events, now: "fresh long-phase heartbeat (12.3m old)",
    )
    now = 1_000_000.0

    for _ in range(4):  # well past the 2-miss threshold
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == [] and markers == []


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
    # #845 fence: tick 2 stops; tick 3 spawns after the verified-dead read.
    assert stops == ["sess-518"] and spawns == []
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

    # Drive the episode forward: each "respawn" needs THREE stale ticks
    # under the #845 fence (1st increments to missed=1, 2nd fires the
    # action and issues the STOP, 3rd verifies the sid is dead and SPAWNS).
    # After each spawn the state is persisted with respawn_count++. The cap
    # is hit when respawn_count reaches STALLED_MAX_RESPAWNS, then the next
    # cycle posts the exhausted marker.
    for _ in range(STALLED_MAX_RESPAWNS):
        # tick A: missed -> 1
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
        # tick B: respawn action fires; the fence issues the stop
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
        # tick C: pending sid verified dead; spawn fires, respawn_count++
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

    # Episode 1: drive one full respawn (3 ticks under the #845 fence:
    # miss -> stop -> verified-dead spawn).
    _patch_stale_signals(monkeypatch, asw, status="running")
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
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
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)

    # Two episodes -> two respawns; cap was NOT reached.
    assert len(spawns) == 2
    assert (950, "session-auto-respawn-exhausted") not in markers


def test_stalled_stop_failure_skips_spawn(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (a-ii) fence, retry-then-stop-failed: a stop that does NOT
    # actually kill the session (the sid stays in the daemon's live set
    # across the verify ticks — the daemon ACK is not a kill) gets exactly
    # ONE retry, then the one-time loud stop-failed alert (marker + push) —
    # and NEVER a spawn next to the live session (two drivers racing on the
    # same issue = the #763 4h overlap). respawn_count is never bumped
    # (nothing spawned), so the cap is unaffected.
    import json

    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or False)
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append(msg) or True)
    _write_autonomous_entry(isolated_registry, 960, "sess-960")
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0
    live = {"sess-960"}

    def tick():
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids=live
        )

    # Ticks 1-2: misses accumulate; tick 2's respawn is K-downgraded to the
    # one-time alert (live id, 1st live-stall episode; default K=2).
    tick()
    tick()
    assert markers == [(960, "session-stalled-alert")]
    # Tick 3: Kth live episode escalates -> the fence issues the stop.
    tick()
    assert stops == ["sess-960"] and spawns == []
    # Tick 4: sid STILL live on the verify tick -> exactly one retry.
    tick()
    assert stops == ["sess-960", "sess-960"]
    # Tick 5: STILL live after the retry -> one-time stop-failed alert +
    # push; never a spawn.
    tick()
    # Tick 6: already alerted -> no repeat marker/push, still no spawn.
    tick()

    assert spawns == []  # never spawned next to a live session
    labels = [label for _i, label in markers]
    assert labels.count("session-stop-failed") == 1
    assert "session-auto-respawn" not in labels
    assert len(pushes) == 1
    assert stops == ["sess-960", "sess-960"]  # one stop + one retry, no more

    state = json.loads((isolated_registry / "stalled-960.json").read_text())
    assert state["respawn_count"] == 0
    assert state["exhausted"] is False
    assert state["stop_failed_alerted"] is True
    assert state["stop_pending_sid"] == "sess-960"


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
    # Bypass the #845 retry wrapper's real backoff sleeps.
    monkeypatch.setattr(asw, "_daemon_reachable_with_retry", lambda *a, **kw: False)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: None)
    # #967: never sweep the LIVE registry/events tree from a unit test.
    monkeypatch.setattr(asw, "triage_observer_pass", lambda *a, **kw: None)

    def _record_stalled(*a, **kw):
        captured_kwargs.update(kw)

    monkeypatch.setattr(asw, "stalled_session_pass", _record_stalled)
    # #1021: the stale-blocked flag pass shells task.py against the LIVE
    # blocked set (daemon-independent) — never run it from a unit test.
    monkeypatch.setattr(asw, "stale_blocked_flag_pass", lambda *a, **kw: None)
    _stub_fleet_mutating_passes(asw, monkeypatch)

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
    # ``_followups_awaiting_child_reason``. ``live_consecutive`` (default 0)
    # is the #759 bug-class-b.1 consecutive-live-stall counter; see
    # ``_process_stalled_session``'s live-session corroboration block.
    # Schema-shape coverage stays exhaustive.
    # The stop_pending_* / stop_retried / stop_failed_alerted / wt_hold_count
    # / daemon_blocked_* / wedge_hits fields are the #845 hardening state
    # (stop-verify fence, worktree hold, daemon-blocked escalation,
    # prompt-wedge observability); all default-absent-safe.
    # dead_silence_respawn_day / dead_silence_respawns_today are the #1209
    # day-keyed dead-silence fence-episode cap (advancement-clear-EXEMPT;
    # bumped once per episode at stop-initiation); default-absent-safe.
    # wedge_respawn_day / wedge_respawns_today are the #1241 twin fields —
    # the SHARED day-keyed cap for the four pre-#1209 wedge triggers, same
    # advancement-clear-EXEMPT / stop-initiation-bump contract, an
    # INDEPENDENT budget; default-absent-safe.
    assert payload == {
        "happy_session_id": "sess-7",
        "missed": 1,
        "alerted": False,
        "respawn_count": 2,
        "exhausted": False,
        "refresh_attempted": False,
        "followups_child_alerted": False,
        "live_consecutive": 0,
        "stop_pending_sid": None,
        "stop_pending_ts": None,
        "stop_retried": False,
        "stop_failed_alerted": False,
        "wt_hold_count": 0,
        "daemon_blocked_ticks": 0,
        "daemon_blocked_pushed": False,
        "wedge_hits": 0,
        "dead_silence_respawn_day": None,
        "dead_silence_respawns_today": 0,
        "wedge_respawn_day": None,
        "wedge_respawns_today": 0,
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
        live_consecutive=1,
        prev=payload,
    )
    payload2 = json.loads((isolated_registry / "stalled-7.json").read_text())
    assert payload2["first_seen"] == 1234.0  # carried forward
    assert payload2["respawn_count"] == 3
    assert payload2["exhausted"] is True
    assert payload2["alerted"] is True
    assert payload2["live_consecutive"] == 1  # persisted across saves


def test_load_stalled_state_defaults_live_consecutive_zero_on_legacy_file(isolated_registry):
    # #759 b.1 backward-compat: an on-disk stalled-<N>.json written BEFORE the
    # live_consecutive field existed has no such key. _load_stalled_state
    # returns the raw dict (no live_consecutive), and the read site in
    # _process_stalled_session must default it to 0 via .get(..., 0) — same
    # guard shape as prev_missed. We assert both the raw load (key absent) and
    # the canonical .get default so a future refactor can't silently start
    # treating a missing key as anything but 0.
    import json

    import autonomous_session_watch as asw

    (isolated_registry / "stalled-9.json").write_text(
        json.dumps(
            {
                "happy_session_id": "sess-9",
                "missed": 1,
                "alerted": False,
                "respawn_count": 0,
                "exhausted": False,
                "refresh_attempted": False,
                "followups_child_alerted": False,
                "last_self_report_ts": "ts-legacy",
                "first_seen": 1.0,
            }
        )
    )
    state = asw._load_stalled_state(9)
    assert "live_consecutive" not in state  # legacy file predates the field
    assert state.get("live_consecutive", 0) == 0  # the read-site default


def test_stalled_live_escalation_k_env_override_and_fallback(monkeypatch):
    # #759 b.1: EPM_STALLED_LIVE_ESCALATION_K parses a valid positive int COUNT
    # (not minutes) and falls back to the default on a garbled OR non-positive
    # value — a typo must neither disable the escalation (K too large) nor
    # disable the debounce (K <= 0 = immediate escalation, the duplicate-driver
    # bug). Mirrors the _respawn_spawn_grace_s / _orphan_staleness_s env-parse
    # coverage.
    import autonomous_session_watch as asw
    from autonomous_session_watch import STALLED_LIVE_ESCALATION_K

    monkeypatch.delenv("EPM_STALLED_LIVE_ESCALATION_K", raising=False)
    assert asw._stalled_live_escalation_k() == STALLED_LIVE_ESCALATION_K

    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "4")
    assert asw._stalled_live_escalation_k() == 4

    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "garbled")
    assert asw._stalled_live_escalation_k() == STALLED_LIVE_ESCALATION_K

    # Non-positive falls back (never 0/negative — that would disable debounce).
    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "0")
    assert asw._stalled_live_escalation_k() == STALLED_LIVE_ESCALATION_K
    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "-3")
    assert asw._stalled_live_escalation_k() == STALLED_LIVE_ESCALATION_K


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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(488, "p488", "pod-488")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(488, "p488", "pod-488")]
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
        asw, "_running_managed_issue_pods", lambda *_a, **_k: [_p(488, "p488", "pod-488")]
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

    # Tick 2: the autonomous respawn action fires once (fence: stop only);
    # still no stalled-alert from the manual sibling.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-510"]
    assert spawns == [] and markers == []

    # Tick 3: verified dead -> the fence spawns; the manual sibling still
    # contributes nothing.
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


def test_vm_disk_pass_dry_run_no_subprocess_even_with_cache_candidate(
    isolated_registry, tmp_path, monkeypatch
):
    """Regression for the #681 r3 BLOCKER: the dry-run sub-floor attribution
    must invoke ZERO ``subprocess.run`` even when a per-issue cache candidate
    EXISTS (so the pre-fix `_top_issue_cache_paths` would reach the per-dir
    ``du`` shell-out).

    The pre-fix bug was environment-dependent: `test_vm_disk_pass_dry_run_
    mutates_nothing` only tripped the crash on a machine whose glob roots
    happened to contain an ``hf_dl``/``g*_dl`` cache dir; in a worktree with no
    such candidate the loop never ran and the call silently no-op'd. This test
    FORCES a candidate present (the glob root has a real ``issue999/hf_dl``
    dir), so pre-fix the dry-run pass calls ``subprocess.run`` on the synthetic
    ``S`` subprocess type whose ``run`` returns None — crashing in the
    ``except (subprocess.SubprocessError, OSError)`` clause (``S`` has no
    ``SubprocessError`` attr). Post-fix the attribution short-circuits to ``[]``
    under ``dry_run`` and shells out to nothing.
    """
    import autonomous_session_watch as asw

    # A glob root WITH a real cache candidate -> pre-fix code reaches du.
    data_root = tmp_path / "data"
    (data_root / "issue999" / "hf_dl").mkdir(parents=True)
    monkeypatch.setattr(asw, "_issue_cache_glob_roots", lambda: [data_root])
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 4 * 2**30)  # critical, below sub-floor
    monkeypatch.setattr(asw, "_sweep_stale_claude_tmp", lambda now, dry_run: 0)
    # The same hostile synthetic subprocess type as the sibling test: only `run`,
    # NO `SubprocessError` attr -> any real shell-out crashes, recording the call.
    ran: list[bool] = []
    monkeypatch.setattr(
        asw,
        "subprocess",
        type("S", (), {"run": staticmethod(lambda *a, **kw: ran.append(True))}),
    )

    # Must NOT raise (the pre-fix AttributeError) and must NOT shell out.
    asw.vm_disk_pass(dry_run=True, now=1_000_000.0)

    assert ran == []  # zero subprocess.run despite a cache candidate present
    assert not (isolated_registry / "vm-disk.json").exists()
    assert not (isolated_registry / "vm-disk-events.jsonl").exists()


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


# ─── orphan sweep signal counts ANY non-watcher marker (#661/#658 sibling) ────
#
# decide_orphan above is pinned on a directly-supplied marker_age_s; these tests
# pin the marker-KIND SEMANTICS of the call site (_process_orphan_task), which
# the pure-gate tests cannot reach. The site read _latest_progress_ts (narrow
# _PROGRESS_KINDS run/upload/interpret allowlist), so a pre-pod lifecycle marker
# (epm:experiment-implementation, epm:plan, ...) was invisible -> the
# alive-but-unregistered session in a long pre-pod phase read as zero-progress
# and was falsely respawned. It now reads _latest_nonwatcher_event_ts.


def _run_orphan_task(
    asw,
    monkeypatch,
    *,
    issue,
    events,
    now,
    missed=1,
    task_status="running",
    dry_run=False,
    respawns_today=0,
    status_calls=None,
):
    """Drive _process_orphan_task end-to-end through the actual marker-age call
    site (rec=None -> fully-unregistered #472 class, so it is an orphan
    candidate). Pre-seeds orphan state with ``missed`` so a stale read fires a
    respawn on the SECOND consecutive miss (threshold default 2).

    #1247 hermeticity: the driver previously left ``_post_progress_marker``
    LIVE with ``dry_run=False``, so every suite run shelled the real
    ``task.py post-marker`` and committed a junk marker on the REAL task named
    by ``issue`` — the two-week #662/#663/#867 marker loop. It now records
    marker posts, seams ``_task_status`` (the #1247 act-time guard's live
    re-read; ``task_status`` sets the returned status, ``status_calls``
    optionally records the issue argument), and drives SYNTHETIC issue ids
    only. Returns ``(respawns, markers)``."""
    respawns: list[int] = []
    markers: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_task_events", lambda _i: events)
    monkeypatch.setattr(asw, "_stalled_cap_gpu_hours", lambda _i: 24.0)
    monkeypatch.setattr(
        asw, "_respawn_orphan", lambda i, cap, dry_run: respawns.append(i) or "spawned"
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda i, note, dry_run, label: markers.append((i, note)),
    )
    monkeypatch.setattr(
        asw,
        "_task_status",
        lambda i: (status_calls.append(i) if status_calls is not None else None) or task_status,
    )
    asw._save_orphan_state(
        issue,
        missed=missed,
        alerted=False,
        respawn_day="2026-06-25" if respawns_today else None,
        respawns_today=respawns_today,
        prev=None,
    )
    _process_orphan_task = asw._process_orphan_task
    _process_orphan_task(
        issue,
        "running",
        None,  # rec=None: fully-unregistered orphan candidate
        set(),  # no live session ids
        now,
        dry_run,
        2,  # threshold
        staleness_s=asw.ORPHAN_STALENESS_S_DEFAULT,
        max_per_day=asw.ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
        day_key="2026-06-25",
    )
    return respawns, markers


def test_orphan_pre_run_lifecycle_marker_keeps_session(isolated_registry, monkeypatch):
    # Load-bearing #661/#658 regression: an active+unregistered task whose
    # NEWEST non-watcher marker is a pre-pod lifecycle kind (excluded from
    # _PROGRESS_KINDS) and is RECENT must NOT be respawned on freshness grounds.
    # On the pre-fix _latest_progress_ts line this respawns (the lifecycle
    # marker is invisible -> marker_age_s=None -> stale); on the fixed
    # _latest_nonwatcher_event_ts line it is kept.
    import autonomous_session_watch as asw

    # The only _PROGRESS_KINDS marker (epm:status-changed) is OLD (>90-min
    # staleness window), so the narrow helper reads stale/respawn. The RECENT
    # sign of life is the lifecycle marker, which ONLY the broad helper sees.
    now = asw._parse_event_ts("2026-06-25T03:40:00Z")
    events = [
        {"kind": "epm:status-changed", "ts": "2026-06-25T00:39:00Z", "note": "running"},
        {
            "kind": "epm:experiment-implementation",
            "ts": "2026-06-25T03:31:00Z",  # 9 min before now — well inside staleness window
            "note": "implemented dispatch script",
        },
    ]
    respawns, _markers = _run_orphan_task(asw, monkeypatch, issue=90661, events=events, now=now)
    assert respawns == []  # kept — the recent lifecycle marker is a sign of life


def test_orphan_watcher_sentinel_marker_still_ignored(isolated_registry, monkeypatch):
    # The broadened signal must STILL ignore the sweep's own respawn/alert posts
    # (they land on the very task whose inactivity they measure). A recent
    # watcher-sentinel'd note with an OLD real lifecycle marker behind it must
    # read as stale -> respawn (the sentinel does not reset the clock).
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T05:00:00Z")
    events = [
        {
            "kind": "epm:experiment-implementation",
            "ts": "2026-06-25T01:00:00Z",  # 4h ago — past the 90-min staleness window
            "note": "real progress, long ago",
        },
        {
            "kind": "epm:progress",
            "ts": "2026-06-25T04:55:00Z",  # recent, but a watcher post
            "note": f"{asw._ORPHAN_RESPAWN_NOTE_SENTINEL} auto-respawn attempt",
        },
    ]
    respawns, _markers = _run_orphan_task(asw, monkeypatch, issue=90662, events=events, now=now)
    assert respawns == [90662]  # the watcher sentinel is filtered; real marker is stale


def test_orphan_old_lifecycle_marker_still_stale(isolated_registry, monkeypatch):
    # Sanity floor: a genuinely OLD newest non-watcher marker still reads stale
    # and respawns — the fix counts more kinds, it does not disable the clock.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    events = [
        {
            "kind": "epm:plan",
            "ts": "2026-06-25T02:00:00Z",  # 4h ago — past the staleness window
            "note": "planning done long ago",
        },
    ]
    respawns, _markers = _run_orphan_task(asw, monkeypatch, issue=90663, events=events, now=now)
    assert respawns == [90663]


# ─── #1247: act-time terminal-status guard (orphan pass) ─────────────────────
#
# The acting branches of _process_orphan_task (respawn / alert / exemption)
# previously trusted the pass-start _active_status_tasks() snapshot for the
# whole tick; a stale snapshot produced respawns + junk markers + git commits
# against TERMINAL tasks, unboundedly (the two-week #662/#663/#867 loop). The
# guard re-reads the LIVE status via _task_status immediately before acting
# and requires a POSITIVE ACTIVE confirmation. All tests drive the REAL
# _process_orphan_task body via _run_orphan_task; only the subprocess-boundary
# seams (_task_status, _task_events, _respawn_orphan, _post_progress_marker)
# are monkeypatched.

_STALE_EVENTS_90XXX = [
    {"kind": "epm:plan", "ts": "2026-06-25T02:00:00Z", "note": "planning done long ago"}
]


def test_orphan_act_guard_aborts_on_terminal_live_status(isolated_registry, monkeypatch, capsys):
    # Durability pin for #1247: snapshot says "running", the live re-read says
    # "archived" — NO spawn, NO marker, orphan episode state CLEARED, one loud
    # stderr line naming snapshot vs live status.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    respawns, markers = _run_orphan_task(
        asw,
        monkeypatch,
        issue=90701,
        events=_STALE_EVENTS_90XXX,
        now=now,
        task_status="archived",
    )
    assert respawns == []
    assert markers == []
    assert asw._load_orphan_state(90701) == {}  # positively non-ACTIVE: episode over
    err = capsys.readouterr().err
    assert "ORPHAN-ACT-GUARD" in err
    assert "'archived'" in err and "status=running" in err and "action=respawn" in err


def test_orphan_act_guard_aborts_on_parked_live_status(isolated_registry, monkeypatch, capsys):
    # The PARK half of the terminal/parked list: on_hold is not ACTIVE either.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    respawns, markers = _run_orphan_task(
        asw,
        monkeypatch,
        issue=90702,
        events=_STALE_EVENTS_90XXX,
        now=now,
        task_status="on_hold",
    )
    assert respawns == []
    assert markers == []
    assert asw._load_orphan_state(90702) == {}
    assert "ORPHAN-ACT-GUARD" in capsys.readouterr().err


def test_orphan_act_guard_defers_on_unreadable_live_status(isolated_registry, monkeypatch, capsys):
    # _task_status -> None (transient task.py read failure): abort the act but
    # KEEP the episode state (missed intact) — retried next tick. Fail toward
    # not-acting, never toward erasing episode state on a task.py glitch.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    respawns, markers = _run_orphan_task(
        asw,
        monkeypatch,
        issue=90703,
        events=_STALE_EVENTS_90XXX,
        now=now,
        task_status=None,
    )
    assert respawns == []
    assert markers == []
    assert asw._load_orphan_state(90703).get("missed") == 1  # state PRESERVED
    assert "ORPHAN-ACT-GUARD" in capsys.readouterr().err


def test_orphan_act_guard_allows_live_active(isolated_registry, monkeypatch):
    # Regression floor: a positively-ACTIVE live read lets the respawn + its
    # marker proceed. The recording seam also closes the wrong-argument hole:
    # the guard must query the SAME issue it is about to act on.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    status_calls: list[int] = []
    respawns, markers = _run_orphan_task(
        asw,
        monkeypatch,
        issue=90704,
        events=_STALE_EVENTS_90XXX,
        now=now,
        task_status="running",
        status_calls=status_calls,
    )
    assert respawns == [90704]
    assert len(markers) == 1 and markers[0][0] == 90704
    assert status_calls == [90704]  # the live re-read queried the acted-on issue


@pytest.mark.parametrize("branch", ["alert", "exemption"])
def test_orphan_alert_and_exemption_branches_also_guarded(
    isolated_registry, monkeypatch, capsys, branch
):
    # The guard sits at ONE site covering ALL acting branches: the daily-cap
    # ALERT branch and the exemption-action dispatch are aborted the same way
    # the respawn branch is (no marker, episode state cleared).
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    if branch == "alert":
        # Cap exhausted -> decide_orphan returns "alert".
        kwargs = dict(respawns_today=asw.ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT)
        events = _STALE_EVENTS_90XXX
        expect_action = "action=alert"
    else:
        # An old prose USER-PAUSE note as the latest word rewrites the respawn
        # to the "user-pause-hold-skip" exemption action (#816 arm).
        kwargs = {}
        events = [
            {
                "kind": "epm:progress",
                "ts": "2026-06-25T02:00:00Z",
                "note": "USER PAUSE (verbatim: hold this); paused_from=running",
            }
        ]
        expect_action = "action=user-pause-hold-skip"
    respawns, markers = _run_orphan_task(
        asw,
        monkeypatch,
        issue=90705,
        events=events,
        now=now,
        task_status="completed",
        **kwargs,
    )
    assert respawns == []
    assert markers == []  # neither the alert nor the exemption marker posted
    assert asw._load_orphan_state(90705) == {}
    err = capsys.readouterr().err
    assert "ORPHAN-ACT-GUARD" in err and expect_action in err


def test_orphan_guard_marker_note_uses_live_status(isolated_registry, monkeypatch):
    # Active->active drift (running -> verifying): the act proceeds and the
    # marker note names the TRUE live status, not the stale snapshot.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    respawns, markers = _run_orphan_task(
        asw,
        monkeypatch,
        issue=90706,
        events=_STALE_EVENTS_90XXX,
        now=now,
        task_status="verifying",
    )
    assert respawns == [90706]
    assert len(markers) == 1
    assert "(status=verifying)" in markers[0][1]


def test_orphan_respawn_note_carries_source_stamp(isolated_registry, monkeypatch):
    # #1247 source stamp: the respawn note self-identifies the posting
    # process (host/user/pid/sha/root) as its trailing token.
    import re

    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    _respawns, markers = _run_orphan_task(
        asw, monkeypatch, issue=90707, events=_STALE_EVENTS_90XXX, now=now
    )
    assert len(markers) == 1
    assert re.search(r"\[src: host=\S+ user=\S+ pid=\d+ sha=\S+ root=/\S+\]$", markers[0][1])


def test_source_stamp_format_and_stability():
    # The stamp matches the documented shape, names THIS process's pid, and is
    # stable within a process (lru_cache -> two calls compare equal).
    import os
    import re

    import autonomous_session_watch as asw

    stamp = asw._source_stamp()
    assert re.fullmatch(r"\[src: host=\S+ user=\S+ pid=\d+ sha=\S+ root=/\S+\]", stamp)
    assert f"pid={os.getpid()}" in stamp
    assert asw._source_stamp() == stamp  # process-stable (cached)


def test_source_stamp_never_raises_when_getuser_fails(monkeypatch):
    # #1247 round-2 fail-soft pin: getpass.getuser() raises KeyError/OSError
    # in an environment with no USER/LOGNAME env vars and no pw entry for the
    # uid; the docstring claims "never raises", so the stamp must degrade to
    # user=unknown instead of killing the acting pass at note-format time.
    # Pre-fix this test raises OSError out of _source_stamp().
    import re

    import autonomous_session_watch as asw

    def _no_user():
        raise OSError("no USER/LOGNAME env and no pwd entry for uid")

    monkeypatch.setattr(asw.getpass, "getuser", _no_user)
    # lru_cache: clear so the stubbed getuser is actually consulted, and clear
    # again afterwards so no later test reads the poisoned cached stamp.
    asw._source_stamp.cache_clear()
    try:
        stamp = asw._source_stamp()
    finally:
        asw._source_stamp.cache_clear()
    assert "user=unknown" in stamp
    assert re.fullmatch(r"\[src: host=\S+ user=unknown pid=\d+ sha=\S+ root=/\S+\]", stamp)


def test_orphan_act_guard_dry_run_never_mutates_state(isolated_registry, monkeypatch, capsys):
    # --dry-run exercises the guard's read but must mutate NOTHING: no spawn,
    # no marker, and the pre-seeded orphan state file survives unmodified
    # (the guard's state-clear is dry_run-gated). Without this pin the §12
    # `--dry-run` live smoke would be the first place a missing gate mutates
    # real ~/.eps-autonomous state.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T06:00:00Z")
    respawns: list[int] = []
    markers: list[tuple] = []
    monkeypatch.setattr(asw, "_task_events", lambda _i: _STALE_EVENTS_90XXX)
    monkeypatch.setattr(asw, "_stalled_cap_gpu_hours", lambda _i: 24.0)
    monkeypatch.setattr(
        asw, "_respawn_orphan", lambda i, cap, dry_run: respawns.append(i) or "spawned"
    )
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: markers.append(a))
    monkeypatch.setattr(asw, "_task_status", lambda _i: "archived")
    asw._save_orphan_state(
        90708, missed=1, alerted=False, respawn_day=None, respawns_today=0, prev=None
    )
    state_path = asw._orphan_state_path(90708)
    before = state_path.read_bytes()
    asw._process_orphan_task(
        90708,
        "running",
        None,
        set(),
        now,
        True,  # dry_run
        2,
        staleness_s=asw.ORPHAN_STALENESS_S_DEFAULT,
        max_per_day=asw.ORPHAN_MAX_RESPAWNS_PER_DAY_DEFAULT,
        day_key="2026-06-25",
    )
    assert respawns == []
    assert markers == []
    assert state_path.read_bytes() == before  # state file untouched under dry-run
    assert "ORPHAN-ACT-GUARD" in capsys.readouterr().err


# ─── #866/#903: deliberate-takeover sentinel skips the orphan sweep ──────────
#
# A deliberate session takeover renames `issue-<N>.json` ->
# `issue-<N>.json.paused-takeover-<suffix>`, making the registration invisible
# to the registry-keyed passes — which turned the handoff into a guaranteed
# orphan-respawn (#866: a duplicate `/issue 866 --auto` orchestrator 21 min
# into the takeover). The orphan sweep must SKIP the issue while the sentinel
# is FRESH (mtime < EPS_TAKEOVER_TTL_H, default 6h) and behave EXACTLY as
# today when it is stale/missing (FAIL OPEN — the existing orphan suite above
# is the no-sentinel control).


@pytest.fixture
def clear_takeover_ttl_env(monkeypatch):
    """The sentinel tests below pin the DEFAULT 6h TTL (and the 7d GC floor's
    max(7d, TTL) contract); an operator shell exporting the fleet knob
    ``EPS_TAKEOVER_TTL_H`` must not flip them."""
    monkeypatch.delenv("EPS_TAKEOVER_TTL_H", raising=False)


def test_takeover_sentinel_fresh_skips_orphan_respawn(
    isolated_registry, monkeypatch, clear_takeover_ttl_env
):
    import time

    import autonomous_session_watch as asw

    (isolated_registry / "issue-90866.json.paused-takeover-20260702").write_text("{}")
    now = time.time()
    # Marker is comfortably stale — without the sentinel this respawns (the
    # control is test_takeover_sentinel_stale_respawns below).
    events = [{"kind": "epm:plan", "ts": "2026-06-25T02:00:00Z", "note": "stale marker"}]
    respawns, _markers = _run_orphan_task(asw, monkeypatch, issue=90866, events=events, now=now)
    assert respawns == []
    # Frozen episode: the skip returns BEFORE any state read/write, so the
    # pre-seeded missed count is untouched and expiry resumes where it left off.
    assert asw._load_orphan_state(90866).get("missed") == 1


def test_takeover_sentinel_stale_respawns(isolated_registry, monkeypatch, clear_takeover_ttl_env):
    import os
    import time

    import autonomous_session_watch as asw

    sentinel = isolated_registry / "issue-90867.json.paused-takeover-20260702"
    sentinel.write_text("{}")
    now = time.time()
    stale = now - 7 * 3600  # past the 6h default TTL
    os.utime(sentinel, (stale, stale))
    events = [{"kind": "epm:plan", "ts": "2026-06-25T02:00:00Z", "note": "stale marker"}]
    respawns, _markers = _run_orphan_task(asw, monkeypatch, issue=90867, events=events, now=now)
    assert respawns == [90867]  # FAIL OPEN: today's behavior once the sentinel ages out


def test_takeover_sentinel_manual_prefix_also_honored(
    isolated_registry, monkeypatch, clear_takeover_ttl_env
):
    import time

    import autonomous_session_watch as asw

    (isolated_registry / "manual-issue-90868.json.paused-takeover-x").write_text("{}")
    events = [{"kind": "epm:plan", "ts": "2026-06-25T02:00:00Z", "note": "stale marker"}]
    respawns, _markers = _run_orphan_task(
        asw, monkeypatch, issue=90868, events=events, now=time.time()
    )
    assert respawns == []


def test_takeover_ttl_env_malformed_falls_back(monkeypatch):
    # A typo'd env var must not disable crash recovery (mirror of the
    # _orphan_staleness_s malformed-env fallback pattern).
    monkeypatch.setenv("EPS_TAKEOVER_TTL_H", "banana")
    assert spawn_session._takeover_ttl_s() == pytest.approx(6.0 * 3600.0)
    monkeypatch.setenv("EPS_TAKEOVER_TTL_H", "2")
    assert spawn_session._takeover_ttl_s() == pytest.approx(2.0 * 3600.0)
    monkeypatch.delenv("EPS_TAKEOVER_TTL_H")
    assert spawn_session._takeover_ttl_s() == pytest.approx(6.0 * 3600.0)


def test_takeover_sentinel_future_mtime_not_fresh(isolated_registry, clear_takeover_ttl_env):
    # A future-dated mtime (clock skew / `touch -d` typo) would be PERMANENTLY
    # fresh — an indefinite crash-recovery suppression inverting the fail-open
    # guarantee. Beyond the clock-jitter slack it is treated as NOT fresh.
    import os
    import time

    sentinel = isolated_registry / "issue-869.json.paused-takeover-x"
    sentinel.write_text("{}")
    now = time.time()
    future = now + 3600.0  # well beyond FUTURE_MTIME_SLACK_S (300s)
    os.utime(sentinel, (future, future))
    assert (
        spawn_session.takeover_sentinel_fresh(869, now=now, registry_dir=isolated_registry) is None
    )


def test_gc_reaps_stale_takeover_sentinels(isolated_registry, clear_takeover_ttl_env):
    # Sentinels never match the `*.json` GC globs, so without the dedicated
    # reap they linger forever; the 7-day floor keeps the forensics record
    # well past the (inert-after-6h) TTL.
    import os
    import time

    import autonomous_session_watch as asw

    now = time.time()
    old = isolated_registry / "issue-901.json.paused-takeover-old"
    old.write_text("{}")
    os.utime(old, (now - 8 * 86400, now - 8 * 86400))  # past the 7d floor
    fresh = isolated_registry / "issue-902.json.paused-takeover-new"
    fresh.write_text("{}")
    os.utime(fresh, (now - 3600, now - 3600))
    asw.gc_pass(dry_run=False, now=now)
    assert not old.exists()
    assert fresh.exists()


def test_gc_keeps_sentinel_under_extended_ttl(isolated_registry, monkeypatch):
    # The max(7d, TTL) contract: with EPS_TAKEOVER_TTL_H above 168h a fixed
    # 7-day reap would delete a sentinel STILL protecting a live takeover.
    import os
    import time

    import autonomous_session_watch as asw

    monkeypatch.setenv("EPS_TAKEOVER_TTL_H", "240")  # 10 days > the 7d GC floor
    now = time.time()
    sentinel = isolated_registry / "issue-904.json.paused-takeover-live"
    sentinel.write_text("{}")
    os.utime(sentinel, (now - 8 * 86400, now - 8 * 86400))  # 8d old: past 7d, inside 240h
    assert asw._gc_stale_takeover_sentinels(now, dry_run=False) == 0
    assert sentinel.exists()


def test_campaign_child_pre_run_lifecycle_marker_reads_fresh(monkeypatch):
    # Campaign watchdog parity (#661/#658 sibling): a child in a long pre-pod
    # planning/implementation phase posts only excluded lifecycle markers; the
    # watchdog must still read it as a FRESH child (no over-alert). On the
    # pre-fix _latest_progress_ts line this returned False (lifecycle markers
    # invisible); on the fixed line it returns True.
    import autonomous_session_watch as asw

    now = asw._parse_event_ts("2026-06-25T01:40:00Z")
    monkeypatch.setattr(asw, "_campaign_children", lambda _i: [{"id": 700, "status": "running"}])
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda _i: [
            {
                "kind": "epm:experiment-implementation",
                "ts": "2026-06-25T01:31:00Z",  # 9 min ago — inside the window
                "note": "child implementing",
            }
        ],
    )
    assert asw._campaign_child_marker_fresh(590, window_s=90 * 60, now=now) is True


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


def _make_progress_event(note: str, ts: str) -> dict:
    """Minimal epm:progress row (stage-dispatch breadcrumbs, watcher-sentinel
    alerts) for the #837 witness/freshness-gate fixtures."""
    return {"ts": ts, "kind": "epm:progress", "version": 1, "note": note}


def _make_generic_event(kind: str, ts: str, note: str = "") -> dict:
    """Minimal event row of an arbitrary kind (witness-kind / parent-tail
    fixtures for the #837 gates)."""
    return {"ts": ts, "kind": kind, "version": 1, "note": note}


def _issue_778_condensed_events() -> list[dict]:
    """Condensed, time-faithful replay of the verified #778 incident record
    (tasks/*/778/events.jsonl) up to the FIRST premature watcher repark
    (2026-07-02T04:43:26Z): 9b scope v2 → the PARENT pass's own tail parks
    (step 10, then 9a-bis — the mis-attributed round-end signal) → the round
    demonstrably running (loop entry, stage-dispatch breadcrumbs,
    run-launched, results, upload-verification PASS)."""
    return [
        _make_followup_scope_event("2026-07-01T21:35:20Z"),  # scope v2 (authoritative)
        _make_step_completed_event(step="10", ts="2026-07-01T21:46:17Z"),  # parent tail
        _make_generic_event("epm:merged", "2026-07-01T21:49:01Z", note="worktree merged"),
        _make_step_completed_event(step="9a-bis", ts="2026-07-01T21:49:35Z"),  # parent tail
        _make_generic_event(
            "epm:status-changed",
            "2026-07-01T21:50:27Z",
            note="awaiting_promotion -> followups_running (follow-up loop entry)",
        ),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt-778",
            "2026-07-01T22:28:53Z",
        ),
        _make_generic_event("epm:run-launched", "2026-07-02T00:40:51Z", note="pod-778"),
        _make_generic_event("epm:results", "2026-07-02T03:28:41Z", note="pod phases complete"),
        _make_progress_event(
            "stage-dispatch stage=followup-null-battery round=1 "
            "subagent=orchestrator-bg-poll worktree=/tmp/wt-778",
            "2026-07-02T03:29:51Z",
        ),
        _make_generic_event("epm:upload-verification", "2026-07-02T03:36:37Z", note="PASS"),
    ]


def _issue_778_second_firing_extra_events() -> list[dict]:
    """The rows between #778's first (04:43Z) and second (06:13Z) premature
    reparks — including the 04:43 WATCHER rows themselves (the repark's
    status-changed + its sentinel-noted progress marker), so the state-2
    fixture exercises the watcher-marker handling exactly where it bit."""
    import autonomous_session_watch as asw

    return [
        _make_generic_event(
            "epm:status-changed",
            "2026-07-02T04:43:26Z",
            note="followups_running -> awaiting_promotion",
        ),
        _make_progress_event(
            f"{asw._FOLLOWUP_ROUND_REPARK_NOTE_SENTINEL} same-issue follow-up "
            f"round complete (premature — the #837 incident's first firing)",
            "2026-07-02T04:43:28Z",
        ),
        _make_generic_event(
            "epm:status-changed",
            "2026-07-02T05:01:45Z",
            note="Replacement session (watcher-registered) resuming the round",
        ),
        _make_progress_event(
            "stage-dispatch stage=followup-null-battery round=1 "
            "subagent=orchestrator-bg-poll worktree=/tmp/wt-778 -- takeover",
            "2026-07-02T05:01:48Z",
        ),
    ]


def test_followup_round_complete_reason_fires_on_533_round_end_shape():
    # The #533 freeze shape: round started (followup-scope), round-end
    # step-completed (9a-bis, parked) NEWER than the scope — the designed
    # re-park never ran. The predicate MUST fire for 9a-bis AND for the
    # step-10 parks the respawned sessions posted. Fixture carries a
    # stage-dispatch breadcrumb between scope and park: the #837 round-start
    # witness gate requires proof the round actually STARTED (a genuinely
    # completed round always has one — SKILL.md's breadcrumb contract).
    import autonomous_session_watch as asw

    for step in ("9a-bis", "10"):
        reason = asw._followup_round_complete_reason(
            [
                _make_followup_scope_event("2026-06-11T09:00:00Z"),
                _make_progress_event(
                    "stage-dispatch stage=followup-implementing round=1 "
                    "subagent=experiment-implementer worktree=/tmp/wt",
                    "2026-06-11T09:30:00Z",
                ),
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


# ─── #837: the two repark gates + awaiting-child stand-down (#778 incident) ──


@pytest.mark.parametrize("state", ["first-firing-0443Z", "second-firing-0613Z"])
def test_followup_round_complete_reason_inert_on_778_mid_round_shape(state):
    # #837 acceptance 1: replaying the verified #778 event history, the
    # predicate MUST return None at BOTH historical firing states — the
    # matched 9a-bis park (21:49:35Z) is the PARENT pass's own tail, and
    # round activity is HOURS newer than it (freshness gate). The second
    # state includes the 04:43 watcher rows (the premature repark's
    # status-changed + sentinel-noted progress marker).
    import autonomous_session_watch as asw

    events = _issue_778_condensed_events()
    if state == "second-firing-0613Z":
        events += _issue_778_second_firing_extra_events()
    assert asw._followup_round_complete_reason(events, issue=778) is None


def test_followup_round_complete_reason_inert_in_pre_activity_race_window():
    # #837 acceptance 2 + 8 (gate 1, round-start witness): #778's
    # 21:49:35Z -> 21:50:27Z window — the mis-attributed parent-tail park is
    # the NEWEST event and no round marker exists yet. The fixture carries
    # realistic PRE-SCOPE parent-history witness-KIND events (ts <= scope_ts):
    # an implementation that checks kind membership WITHOUT the timestamp
    # comparison must fail this test. A watcher-sentinel alert after the
    # park changes nothing.
    import autonomous_session_watch as asw

    events = [
        _make_generic_event("epm:plan-approved", "2026-07-01T08:00:00Z"),
        _make_generic_event("epm:run-launched", "2026-07-01T10:00:00Z", note="parent run"),
        _make_followup_scope_event("2026-07-01T21:35:20Z"),
        _make_step_completed_event(step="9a-bis", ts="2026-07-01T21:49:35Z"),
        _make_progress_event(
            f"{asw._STALLED_ALERT_NOTE_SENTINEL} session stalled alert",
            "2026-07-01T23:00:00Z",
        ),
    ]
    assert asw._followup_round_complete_reason(events, issue=778) is None


@pytest.mark.parametrize("witness_kind", ["epm:plan", "epm:run-launched", "epm:plan-verify"])
def test_followup_round_complete_reason_fires_on_kind_only_witness(witness_kind):
    # #837 acceptance 9: a round whose only round-start proof is a
    # witness-KIND marker (NO stage-dispatch breadcrumb anywhere — the
    # documented missed-breadcrumb limitation) must still repark. An emptied
    # or typo'd _FOLLOWUP_ROUND_WITNESS_KINDS must fail this test.
    # epm:plan-verify is the load-bearing member for the sparsest rounds
    # (#537 class).
    import autonomous_session_watch as asw

    reason = asw._followup_round_complete_reason(
        [
            _make_followup_scope_event("2026-06-11T09:00:00Z"),
            _make_generic_event(witness_kind, "2026-06-11T09:30:00Z"),
            _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
        ]
    )
    assert reason is not None, witness_kind
    assert "designed re-park" in reason


def test_followup_round_complete_reason_ignores_watcher_markers_in_freshness():
    # #837 acceptance 3 (gate 2's watcher exclusion): watcher-sentinel-noted
    # markers routinely post AFTER a genuine round-end park (stalled alerts,
    # awaiting-child alerts) — they must NOT veto the repark.
    import autonomous_session_watch as asw

    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt",
            "2026-06-11T09:30:00Z",
        ),
        _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
        _make_progress_event(
            f"{asw._STALLED_ALERT_NOTE_SENTINEL} session stalled alert",
            "2026-06-11T12:00:00Z",
        ),
        _make_progress_event(
            f"{asw._FOLLOWUPS_AWAITING_CHILD_NOTE_SENTINEL} waiting on child",
            "2026-06-11T13:00:00Z",
        ),
    ]
    reason = asw._followup_round_complete_reason(events)
    assert reason is not None
    assert "designed re-park" in reason


def test_followup_round_complete_reason_ignores_deliberate_stop_in_freshness():
    # #1053 round-2 pin: a FRESH deliberate-stop breadcrumb (the Step 0
    # stale-wake YIELD death record, by="issue-session-guard") posted AFTER a
    # valid round-end park is a driver DEATH record, never round activity —
    # it must NOT veto the repark via the #837 gate-2 freshness read
    # (_has_nonwatcher_event_after carries the same two-leg exclusion as
    # _latest_nonwatcher_event_ts / _latest_progress_ts).
    import autonomous_session_watch as asw

    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt",
            "2026-06-11T09:30:00Z",
        ),
        _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
        {
            **_make_progress_event(
                "deliberate-stop pid=n/a target=self reason=stale-wake-yield "
                "replacement=happy-session:def456 — stale /issue 1053 session "
                "yielding on wake; the replacement owns the task; no state mutated",
                "2026-06-11T12:00:00Z",
            ),
            "by": "issue-session-guard",
        },
    ]
    reason = asw._followup_round_complete_reason(events)
    assert reason is not None
    assert "designed re-park" in reason


def test_followup_round_complete_reason_converges_after_respawn_park():
    # #837 acceptance 4 (the §4b convergence argument, pinned): a stray
    # NON-watcher cross-post after a genuine round end blocks the repark
    # this tick (freshness gate) — then the respawned session re-derives
    # state and posts a FRESH round-end park; nothing is newer than it, the
    # witness still exists, and the repark fires on the next tick.
    import autonomous_session_watch as asw

    base = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt",
            "2026-06-11T09:30:00Z",
        ),
        _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
        _make_generic_event(
            "epm:workflow-fix-applied", "2026-06-11T12:00:00Z", note="applied_task: #900"
        ),
    ]
    assert asw._followup_round_complete_reason(base) is None
    respawned = [*base, _make_step_completed_event(step="10", ts="2026-06-11T13:00:00Z")]
    reason = asw._followup_round_complete_reason(respawned)
    assert reason is not None
    assert "designed re-park" in reason


def test_followup_round_complete_reason_inert_on_scope_repost_after_completion():
    # OPTIONAL documentation fixture (plan §Test plan, Alt-Claude concern): a
    # content-identical scope RE-POST after round completion resets scope_ts
    # newer than every witness AND the round-end park -> the predicate stays
    # inert (in-flight early return); recovery is respawn-convergence, which
    # the §4d stand-down keeps reachable.
    import autonomous_session_watch as asw

    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt",
            "2026-06-11T09:30:00Z",
        ),
        _make_step_completed_event(step="9a-bis", ts="2026-06-11T10:54:12Z"),
        _make_followup_scope_event("2026-06-11T11:00:00Z"),  # the re-post
    ]
    assert asw._followup_round_complete_reason(events) is None


def test_followups_awaiting_child_reason_stands_down_on_unrun_scope(monkeypatch):
    # #837 acceptance 10 (§4d): an UNRUN epm:followup-scope (no newer run
    # marker) means a same-issue round is pending or executing — the
    # children-wait suppression must stand down (return None) EVEN WITH an
    # open child, so the never-started / blocked-repark shapes fall through
    # to RESPAWN instead of latching alert-only (#778 has open child #816).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_children", lambda issue: [{"id": 816, "status": "running"}])
    events = [
        _make_followup_scope_event("2026-07-01T21:35:20Z"),
        _make_step_completed_event(step="10", ts="2026-07-01T21:46:17Z"),
    ]
    reason = asw._followups_awaiting_child_reason(
        778, status="followups_running", has_pod=False, events=events
    )
    assert reason is None


def test_check_orphan_followups_exemption_respawns_on_unrun_scope_no_witness(monkeypatch):
    # #837 acceptance 10 (orphan-pass action assertion): the no-witness
    # unrun-scope shape — the witness gate vetoes the repark AND the §4d
    # stand-down vetoes the awaiting-child latch — must keep action
    # "respawn" (never "followups-awaiting-child", never a repark).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_children", lambda issue: [{"id": 816, "status": "running"}])
    action, reason = asw._check_orphan_followups_exemption(
        issue=778,
        status="followups_running",
        has_pod=False,
        events=[
            _make_followup_scope_event("2026-07-01T21:35:20Z"),
            _make_step_completed_event(step="10", ts="2026-07-01T21:46:17Z"),
        ],
        action="respawn",
    )
    assert action == "respawn"
    assert reason is None


def test_followups_awaiting_child_reason_fires_on_recorded_scope(monkeypatch):
    # #837 §4d companion: a RECORDED scope (run marker strictly newer than
    # the scope) keeps the legacy children-in-flight suppression semantics
    # untouched — the exemption still fires.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_task_children", lambda issue: [{"id": 546, "status": "awaiting_promotion"}]
    )
    events = [
        _make_followup_scope_event("2026-06-11T09:00:00Z"),
        _make_followup_run_event("2026-06-11T10:55:00Z"),
        _make_step_completed_event(step="10", ts="2026-06-12T08:00:00Z"),
    ]
    reason = asw._followups_awaiting_child_reason(
        533, status="followups_running", has_pod=False, events=events
    )
    assert reason is not None
    assert "#546" in reason


# ─── #894: label-grouped unrun predicate in the watcher passes ───────────────


def _make_labeled_scope_event(label: str, source: str, ts: str, version: int = 1) -> dict:
    """epm:followup-scope row with an explicit label (multi-label fixtures)."""
    return {
        "ts": ts,
        "kind": "epm:followup-scope",
        "version": version,
        "note": f"followup_label: {label}\nsource: {source}",
    }


def _make_labeled_run_event(label: str, source: str, ts: str, version: int = 1) -> dict:
    """epm:same-issue-followup-run row with an explicit label."""
    return {
        "ts": ts,
        "kind": "epm:same-issue-followup-run",
        "version": version,
        "note": f"followup_label: {label}\nsource: {source}\nround: 1",
    }


def test_followup_round_complete_reason_fires_for_older_label_round():
    # #894 test 9: an OLDER queued label's round executes AFTER a newer
    # label's run marker — scope A (t1), scope B (t2), run B (t3),
    # round-start witness (t4), 9a-bis park (t5, newest). Label A is unrun,
    # its round demonstrably started and parked → the repark MUST fire.
    # Pre-fix the `run_ts > scope_ts` early exit returned None (blind to A).
    import autonomous_session_watch as asw

    events = [
        _make_labeled_scope_event("label-a", "user-chat", "2026-07-01T09:00:00Z", version=1),
        _make_labeled_scope_event(
            "label-b", "proposer-9b-cheap", "2026-07-01T10:00:00Z", version=2
        ),
        _make_labeled_run_event("label-b", "proposer-9b-cheap", "2026-07-01T12:00:00Z"),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt label=label-a",
            "2026-07-01T13:00:00Z",
        ),
        _make_step_completed_event(step="9a-bis", ts="2026-07-01T15:00:00Z"),
    ]
    reason = asw._followup_round_complete_reason(events, issue=763)
    assert reason is not None
    assert "designed re-park" in reason


def test_followup_round_complete_reason_anchor_rejects_stale_witness():
    # #894 test 9b (Stat-Codex MF1 — the max-anchor MUTANT pin): scope A
    # (t1), scope B (t2), B's round-start witness (t2.5), run B (t3),
    # round-end park (t4 > t3). Label A is unrun but has NO witness of its
    # own newer than max(scope_ts, run_ts) = t3 → the predicate MUST stay
    # inert. A scope_ts-only-anchor mutant counts B's stale t2.5 witness for
    # A and false-fires (reparking + wrong-closing the head-of-queue label).
    import autonomous_session_watch as asw

    events = [
        _make_labeled_scope_event("label-a", "user-chat", "2026-07-01T09:00:00Z", version=1),
        _make_labeled_scope_event(
            "label-b", "proposer-9b-cheap", "2026-07-01T10:00:00Z", version=2
        ),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt label=label-b",
            "2026-07-01T10:30:00Z",
        ),
        _make_labeled_run_event("label-b", "proposer-9b-cheap", "2026-07-01T12:00:00Z"),
        _make_step_completed_event(step="9a-bis", ts="2026-07-01T15:00:00Z"),
    ]
    assert asw._followup_round_complete_reason(events, issue=763) is None


def test_followups_awaiting_child_reason_stands_down_on_763_multilabel_shape(monkeypatch):
    # #894 test 10: the #763 shape — an UNRUN older label behind a NEWER
    # label's run marker, plus a step-10 park and open children. The
    # label-keyed stand-down MUST return None (respawn reaches the queued
    # label); the pre-fix ts-keyed read saw run_ts > scope_ts → no
    # stand-down → the awaiting-child latch suppressed the recovery.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_children", lambda issue: [{"id": 816, "status": "running"}])
    events = [
        _make_labeled_scope_event(
            "neutral-contrast-and-cofit", "user-chat", "2026-06-30T22:10:33Z", version=1
        ),
        _make_labeled_scope_event(
            "deception-rubric-reanchor", "proposer-9b-cheap", "2026-07-02T10:46:52Z", version=2
        ),
        _make_labeled_run_event(
            "deception-rubric-reanchor", "proposer-9b-cheap", "2026-07-02T21:45:13Z"
        ),
        _make_step_completed_event(step="10", ts="2026-07-02T22:00:00Z"),
    ]
    reason = asw._followups_awaiting_child_reason(
        763, status="followups_running", has_pod=False, events=events
    )
    assert reason is None


def test_parse_followup_note_field_parses_bold_markdown_fields():
    # #837 acceptance 5 (§4c), migrated to the shared parser (#894 — the
    # watcher's `_scope_note_field` is superseded by
    # `task_workflow.parse_followup_note_field`): the modern scope note
    # writes bold-markdown fields (`**followup_label:** …`) — the verified
    # cause of #778's second premature firing (the 04:43 disarm run marker
    # never posted: "no followup_label parseable"). Fixture = the VERBATIM
    # #778 scope-v2 note (2026-07-01T21:35:20Z) from the incident record.
    from explore_persona_space.task_workflow import parse_followup_note_field

    note = (Path(__file__).resolve().parent / "fixtures" / "issue778_scope_v2_note.md").read_text()
    # fixture-integrity guards: the verbatim note really carries the bold form
    assert "**followup_label:** corrected-monitoring-8prompt-ladder" in note
    assert "**source:** user-chat" in note
    assert parse_followup_note_field(note, "followup_label") == (
        "corrected-monitoring-8prompt-ladder"
    )
    assert parse_followup_note_field(note, "source") == "user-chat"


def test_post_followup_run_marker_parses_bold_scope(monkeypatch):
    # #837 §4c end-to-end (now via task_workflow.executing_followup_label,
    # #894): the disarm run marker posts with the label AND the real source
    # parsed from a bold-markdown scope note (source no longer degrades to
    # "unknown", so a proposer-9b-cheap round counts toward the 2-round
    # cheap cap).
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    bold_scope = {
        "ts": "2026-07-01T21:35:20Z",
        "kind": "epm:followup-scope",
        "version": 2,
        "note": (
            "**source:** proposer-9b-cheap\n"
            "**followup_label:** corrected-monitoring-8prompt-ladder\n"
        ),
    }
    assert asw._post_followup_run_marker(778, [bold_scope], dry_run=False) is True
    assert len(calls) == 1
    note = calls[0][-1]
    assert "followup_label: corrected-monitoring-8prompt-ladder" in note
    assert "source: proposer-9b-cheap" in note


def test_parse_followup_note_field_plain_format_still_parses():
    # #837 §4c regression companion (migrated to the shared parser, #894):
    # the plain `<field>: <value>` form the legacy fixtures use must keep
    # parsing bit-identically.
    from explore_persona_space.task_workflow import parse_followup_note_field

    note = _make_followup_scope_event("2026-06-11T09:00:00Z")["note"]
    assert parse_followup_note_field(note, "followup_label") == "bare-word-install-step-grid"
    assert parse_followup_note_field(note, "source") == "user-chat"
    assert parse_followup_note_field(note, "gpu_hours_estimate") is None
    assert parse_followup_note_field("", "followup_label") is None


def test_executing_followup_label_resolves_queue_head():
    # Migration of the old "reads the LATEST scope" test (#894): with two
    # unrun labels the resolver returns the QUEUE HEAD (user-initiated
    # first, then oldest armed ts) rather than the bare latest scope; no
    # scopes -> None.
    from explore_persona_space.task_workflow import executing_followup_label

    events = [
        {
            "ts": "2026-06-10T09:00:00Z",
            "kind": "epm:followup-scope",
            "version": 1,
            "note": "followup_label: old-round\nsource: proposer-9b",
        },
        _make_followup_scope_event("2026-06-11T09:00:00Z"),  # user-chat
    ]
    group = executing_followup_label(events)
    assert group is not None
    assert group["followup_label"] == "bare-word-install-step-grid"  # user-initiated wins
    assert group["source"] == "user-chat"
    assert executing_followup_label([]) is None


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


def test_post_followup_run_marker_fails_soft_without_label(monkeypatch, capsys):
    # No parseable followup_label -> the scope becomes a NON-dispatchable
    # pseudo-label group (#894), so the resolver returns nothing to close:
    # no marker posted, returns False (fail-soft: the re-park already
    # happened), and the stderr log LOUDLY names the repair (re-post with a
    # proper label / retro-close).
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
    err = capsys.readouterr().err
    assert "REPAIR" in err
    assert "followup_label" in err


def test_post_followup_run_marker_closes_executing_label_not_latest_scope(monkeypatch):
    # #894 test 11: the #763 shape mid-round on the OLDER queued label — the
    # LATEST scope is a different (already-run) label, so parsing the latest
    # scope would close the WRONG label (stranding the executed round unrun
    # and closing a never-run one). The resolver must post label A.
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    events = [
        _make_labeled_scope_event(
            "neutral-contrast-and-cofit", "user-chat", "2026-06-30T22:10:33Z", version=1
        ),
        _make_labeled_scope_event(
            "deception-rubric-reanchor", "proposer-9b-cheap", "2026-07-02T10:46:52Z", version=2
        ),
        _make_labeled_run_event(
            "deception-rubric-reanchor", "proposer-9b-cheap", "2026-07-02T21:45:13Z"
        ),
    ]
    assert asw._post_followup_run_marker(763, events, dry_run=False) is True
    assert len(calls) == 1
    note = calls[0][-1]
    assert "followup_label: neutral-contrast-and-cofit" in note
    assert "source: user-chat" in note
    assert "round: 2" in note  # 1 + the existing run marker


def test_post_followup_run_marker_breadcrumb_beats_queue_head(monkeypatch):
    # #894 test 11b (Stat-Codex MF2 — watcher-level breadcrumb-first pin):
    # label B is executing via a `stage-dispatch … label=B` breadcrumb NEWER
    # than the latest run marker, and a LATER user-chat label A now heads
    # the queue → the posted note must carry label B (a head-of-queue-only
    # mutant posts A and fails).
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    events = [
        _make_labeled_scope_event("label-b", "proposer-9b-cheap", "2026-07-01T09:00:00Z"),
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt label=label-b",
            "2026-07-01T10:00:00Z",
        ),
        # A user-chat label posted MID-ROUND — heads the queue but is NOT
        # the executing round.
        _make_labeled_scope_event("label-a", "user-chat", "2026-07-01T11:00:00Z", version=2),
    ]
    assert asw._post_followup_run_marker(763, events, dry_run=False) is True
    assert len(calls) == 1
    note = calls[0][-1]
    assert "followup_label: label-b" in note
    assert "source: proposer-9b-cheap" in note


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
        # witness event between scope and park — required by the #837
        # round-start witness gate (assertions unchanged).
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt",
            "2026-06-11T09:30:00Z",
        ),
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
    # A FAILED re-park must fall through to the pre-existing handling.
    # Pre-#837 that meant the awaiting-child suppression; post-#837 the
    # repark only ever fires on an UNRUN scope, and the §4d stand-down makes
    # the awaiting-child suppression decline exactly there (even with an
    # open child), so the fall-through now reaches RESPAWN — the designed
    # recovery for a pending/executing round (#778's 05:01Z replacement
    # session demonstrated it live). Witness event added for the #837
    # round-start witness gate so the repark probe still fires.
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
        _make_progress_event(
            "stage-dispatch stage=followup-implementing round=1 "
            "subagent=experiment-implementer worktree=/tmp/wt",
            "2026-06-11T09:30:00Z",
        ),
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
    # Fall-through reaches RESPAWN (action passes through unchanged); the
    # awaiting-child alert must NOT post (unrun scope → §4d stand-down).
    assert (action, new_missed, child_alerted) == ("respawn", 2, False)
    assert posted == []


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
            # witness event between scope and park — required by the #837
            # round-start witness gate (assertions unchanged).
            _make_progress_event(
                "stage-dispatch stage=followup-implementing round=1 "
                "subagent=experiment-implementer worktree=/tmp/wt",
                "2026-06-11T09:30:00Z",
            ),
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


# ─── over-cap spend-approval park exemption (incident #653, 2026-06-18) ───────
# Status-hold variant (SKILL.md Step 9b): an over-cap plan estimate parks the
# task IN PLACE at the ACTIVE status `followups_running` (the status does NOT
# move to plan_pending). decide() therefore sees an ACTIVE task and the
# missing-self-report drove 5 respawn-and-park cycles in ~4h, each re-posting
# the same epm:step-completed step=2c exit_kind=parked. This is a user-only
# gate; the exemption diverts the would-be respawn to a budget-free alert and
# self-disarms when the user approves / re-plans (a real progress marker newer
# than the park).


def _make_spend_approval_event(ts: str = "2026-06-18T00:34:11Z") -> dict:
    """Minimal epm:awaiting-spend-approval row — the over-cap autonomous
    plan-gate park (task.py --auto-approve-if-autonomous, parked_over_cap)."""
    return {
        "ts": ts,
        "kind": "epm:awaiting-spend-approval",
        "version": 1,
        "by": "autonomous-gate",
        "note": (
            "Autonomous plan-gate parked IN PLACE at followups_running: est 132.0 "
            "GPU-h exceeds 100.0h auto-approve cap; awaiting user approval "
            "(status-hold rule, SKILL.md Step 9b)."
        ),
    }


def test_spend_approval_park_reason_fires_on_canonical_653_shape():
    # The #653 shape: latest non-watcher event is epm:awaiting-spend-approval,
    # followed only by parked epm:step-completed re-posts (NOT in
    # _PROGRESS_KINDS) and watcher respawn markers (sentinel-filtered). The
    # exemption MUST fire.
    import autonomous_session_watch as asw

    events = [
        _make_spend_approval_event("2026-06-18T00:34:11Z"),
        _make_step_completed_event(step="2c", exit_kind="parked", ts="2026-06-18T00:34:24Z"),
        # watcher respawn note — sentinel-filtered out of _latest_progress_ts
        {
            "ts": "2026-06-18T01:33:07Z",
            "kind": "epm:progress",
            "note": f"{asw._STALLED_RESPAWN_NOTE_SENTINEL} ALIVE-BUT-STALLED auto-respawn",
        },
        _make_step_completed_event(step="2c", exit_kind="parked", ts="2026-06-18T02:14:56Z"),
    ]
    reason = asw._spend_approval_park_reason(events)
    assert reason is not None
    assert "over-cap autonomous plan-gate" in reason
    assert "user-only gate" in reason


def test_spend_approval_park_reason_self_disarms_on_real_progress():
    # When the user approves / re-plans, a REAL progress marker
    # (epm:status-changed, in _PROGRESS_KINDS) newer than the park resolves
    # the gate — the exemption MUST stop applying so respawn coverage resumes.
    import autonomous_session_watch as asw

    events = [
        _make_spend_approval_event("2026-06-18T00:34:11Z"),
        {
            "ts": "2026-06-18T05:00:00Z",
            "kind": "epm:status-changed",
            "from": "followups_running",
            "to": "approved",
        },
    ]
    assert asw._spend_approval_park_reason(events) is None


def test_spend_approval_park_reason_inert_without_spend_marker():
    # No epm:awaiting-spend-approval on record = not the over-cap park shape.
    import autonomous_session_watch as asw

    events = [_make_step_completed_event(step="2c", exit_kind="parked")]
    assert asw._spend_approval_park_reason(events) is None


def test_spend_approval_skip_already_noted_dedup():
    # Self-contained events-log dedup: a skip marker NEWER than the gating
    # spend-approval marker means this episode's alert already fired; an OLDER
    # one (from a prior episode) does NOT count, so a fresh park re-arms.
    import autonomous_session_watch as asw

    spend = _make_spend_approval_event("2026-06-18T00:34:11Z")
    newer_skip = {
        "ts": "2026-06-18T00:35:00Z",
        "kind": "epm:progress",
        "note": f"{asw._SPEND_APPROVAL_SKIP_NOTE_SENTINEL} parked at the over-cap gate.",
    }
    older_skip = {
        "ts": "2026-06-18T00:00:00Z",
        "kind": "epm:progress",
        "note": f"{asw._SPEND_APPROVAL_SKIP_NOTE_SENTINEL} parked at the over-cap gate.",
    }
    assert asw._spend_approval_skip_already_noted([spend, newer_skip]) is True
    assert asw._spend_approval_skip_already_noted([spend, older_skip]) is False
    assert asw._spend_approval_skip_already_noted([newer_skip]) is False  # no spend marker


def test_apply_stalled_followups_exemption_rewrites_spend_approval_respawn_to_keep(monkeypatch):
    # The stalled-detector helper: an `action="respawn"` on a spend-approval
    # park MUST become `action="keep"` with `new_missed=0`, and the one-time
    # skip alert MUST be posted (events-log dedup, not a state flag).
    import autonomous_session_watch as asw

    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    events = [
        _make_spend_approval_event("2026-06-18T00:34:11Z"),
        _make_step_completed_event(step="2c", exit_kind="parked", ts="2026-06-18T00:34:24Z"),
    ]
    action, new_missed, child_alerted = asw._apply_stalled_followups_exemption(
        issue=653,
        status="followups_running",
        has_pod=False,
        events=events,
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action, new_missed, child_alerted) == ("keep", 0, False)
    assert len(posted) == 1
    assert posted[0][0] == 653
    assert posted[0][2] == "spend-approval-skip"
    assert "Respawn suppressed" in posted[0][1]

    # Second call within the same episode: a skip marker now exists newer than
    # the spend-approval marker in `events` -> the alert MUST NOT re-post.
    posted.clear()
    events_with_skip = [
        *events,
        {
            "ts": "2026-06-18T00:35:00Z",
            "kind": "epm:progress",
            "note": f"{asw._SPEND_APPROVAL_SKIP_NOTE_SENTINEL} parked at the over-cap gate.",
        },
    ]
    action2, new_missed2, _ = asw._apply_stalled_followups_exemption(
        issue=653,
        status="followups_running",
        has_pod=False,
        events=events_with_skip,
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action2, new_missed2) == ("keep", 0)
    assert posted == []  # dedup'd via the events log


def test_check_orphan_followups_exemption_returns_spend_approval_skip(monkeypatch):
    # Orphan pass: a spend-approval-parked task with no live registered session
    # rewrites respawn -> "spend-approval-skip" WITHOUT consulting children
    # (the spend-approval probe is checked first and is events-only).
    import autonomous_session_watch as asw

    def _boom(issue):
        raise AssertionError("_task_children must not be consulted on the spend-approval path")

    monkeypatch.setattr(asw, "_task_children", _boom)
    action, reason = asw._check_orphan_followups_exemption(
        issue=653,
        status="followups_running",
        has_pod=False,
        events=[
            _make_spend_approval_event("2026-06-18T00:34:11Z"),
            _make_step_completed_event(step="2c", exit_kind="parked", ts="2026-06-18T00:34:24Z"),
        ],
        action="respawn",
    )
    assert action == "spend-approval-skip"
    assert reason is not None
    assert "over-cap autonomous plan-gate" in reason


def test_handle_orphan_spend_approval_skip_posts_once_and_skips_budget(
    isolated_registry, monkeypatch
):
    # The orphan handler MUST (a) post the one-time alert dedup'd via the
    # events log; (b) persist state WITHOUT incrementing respawns_today.
    import autonomous_session_watch as asw

    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    spend_events = [_make_spend_approval_event("2026-06-18T00:34:11Z")]
    asw._handle_orphan_spend_approval_skip(
        issue=653,
        reason="parked at the over-cap autonomous plan-gate",
        new_missed=2,
        alerted=False,
        respawn_day="2026-06-18",
        respawns_today=0,
        followups_child_alerted=False,
        events=spend_events,
        state={},
        dry_run=False,
    )
    assert len(posted) == 1
    assert posted[0][2] == "spend-approval-skip"
    state = asw._load_orphan_state(653)
    assert state["respawns_today"] == 0  # NOT incremented

    # Second call within the same episode: a skip marker now exists newer than
    # the spend-approval marker -> dedup'd, alert MUST NOT re-post.
    posted.clear()
    spend_events_with_skip = [
        *spend_events,
        {
            "ts": "2026-06-18T00:35:00Z",
            "kind": "epm:progress",
            "note": f"{asw._SPEND_APPROVAL_SKIP_NOTE_SENTINEL} parked at the over-cap gate.",
        },
    ]
    asw._handle_orphan_spend_approval_skip(
        issue=653,
        reason="parked at the over-cap autonomous plan-gate",
        new_missed=3,
        alerted=False,
        respawn_day="2026-06-18",
        respawns_today=0,
        followups_child_alerted=False,
        events=spend_events_with_skip,
        state=state,
        dry_run=False,
    )
    assert posted == []
    assert asw._load_orphan_state(653)["respawns_today"] == 0


def test_spend_approval_skip_sentinel_in_watcher_filter():
    # The skip alert marker must NEVER reset the staleness clock it is measured
    # against — pin the sentinel into the shared exclusion set.
    from autonomous_session_watch import (
        _SPEND_APPROVAL_SKIP_NOTE_SENTINEL,
        _WATCHER_NOTE_SENTINELS,
    )

    assert _SPEND_APPROVAL_SKIP_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS


# ─── prose USER-PAUSE hold exemption (incident #816, 2026-07-02) ──────────────
# A session posted a prose-only `USER PAUSE ...` hold note (epm:progress) and
# left the task at the ACTIVE status `running`; the orphan-respawn pass cannot
# parse prose and respawned against the hold (attempt 1/2). The durable
# affordance is `set-status on_hold` (#919); this exemption is defense-in-depth
# for sessions that still post a prose hold. Test-fixture discipline: no
# helper/test note below may begin with the bare `USER PAUSE` literal outside
# the deliberate pause fixtures built by _make_user_pause_event.


def _make_user_pause_event(
    ts: str = "2026-07-03T06:44:44Z", note: str | None = None, kind: str = "epm:progress"
) -> dict:
    """Minimal prose USER-PAUSE hold row. Default note = the verbatim #816
    incident prefix; callers pass ``note=`` for the other observed variants
    (canonical SKILL.md durable-park format, the older #919 sketch). This
    helper is the deliberate pause fixture — the only sanctioned source of
    notes beginning with the bare literal."""
    if note is None:
        note = (
            "USER PAUSE (2026-07-02, verbatim: 'pause 816'): session stopped at "
            "user request. DO NOT resume, respawn, or auto-dispatch this task."
        )
    return {"ts": ts, "kind": kind, "version": 1, "by": "user", "note": note}


def test_user_pause_hold_reason_fires_on_816_incident_shape():
    # Incident-SHAPE replay of #816: the prose pause note, a NEWER
    # watcher-sentinel respawn marker (sentinel-filtered out of
    # _latest_progress_ts), and a NEWER parked epm:step-completed. Neither
    # newer row disarms the hold — the exemption MUST fire. NOTE: the parked
    # step-completed element is imported from the #653 spend-approval fixture
    # shape — the raw #816 log has no post-pause step-completed; it is added
    # here to pin that a parked re-post cannot disarm the hold either.
    import autonomous_session_watch as asw

    events = [
        _make_user_pause_event("2026-07-03T06:44:44Z"),
        {
            "ts": "2026-07-03T08:33:00Z",
            "kind": "epm:progress",
            "note": f"{asw._ORPHAN_RESPAWN_NOTE_SENTINEL} active task auto-respawned (1/2)",
        },
        _make_step_completed_event(step="2c", exit_kind="parked", ts="2026-07-03T08:35:00Z"),
    ]
    reason = asw._user_pause_hold_reason(events)
    assert reason is not None
    assert "USER PAUSE" in reason
    assert "#816" in reason


def test_user_pause_hold_reason_fires_on_canonical_skill_format():
    # The canonical SKILL.md § User pause affordance durable-park note rides
    # the `set-status <N> on_hold` epm:status-changed row — the SECOND
    # self-inclusion kind (epm:status-changed is in _PROGRESS_KINDS, so
    # pause_ts == progress_ts and only the strict `>` keeps the hold armed).
    # Arming here is harmless in production: on_hold is in the watcher PARK
    # set, so a durably-parked task is never orphan-evaluated.
    import autonomous_session_watch as asw

    events = [
        _make_user_pause_event(
            ts="2026-07-03T06:44:44Z",
            note=(
                "USER PAUSE (verbatim: 'pause 42'); paused_from=running; "
                "resume: user-greenlight only."
            ),
            kind="epm:status-changed",
        )
    ]
    assert asw._user_pause_hold_reason(events) is not None


def test_user_pause_hold_reason_fires_on_919_older_variant():
    # The older #919 sketch variant (no parenthetical, no 'verbatim' literal)
    # — the anchor is variant-agnostic by construction.
    import autonomous_session_watch as asw

    events = [
        _make_user_pause_event(
            note="USER PAUSE pause 42; resume: set-status 42 running + spawn-issue --auto."
        )
    ]
    assert asw._user_pause_hold_reason(events) is not None


def test_user_pause_hold_reason_ignores_mid_note_quote():
    # The real quote carriers (the #979 clarify / #919 completion-audit /
    # #920 triage-note shapes) carry the literal MID-note only — the anchored
    # prefix must NOT match any of them.
    import autonomous_session_watch as asw

    events = [
        {
            "ts": "2026-07-03T10:00:00Z",
            "kind": "epm:clarify",
            "note": "Goal: grep the task's recent markers for a 'USER PAUSE'-shaped note.",
        },
        {
            "ts": "2026-07-03T11:00:00Z",
            "kind": "epm:completion-audit",
            "note": "Ask 3: the watcher should recognize the USER PAUSE format — ADDRESSED.",
        },
        {
            "ts": "2026-07-03T12:00:00Z",
            "kind": "epm:progress",
            "note": "external-markers-triaged: 1 applied (the USER PAUSE hold format note).",
        },
    ]
    assert asw._user_pause_hold_reason(events) is None


def test_user_pause_hold_reason_self_disarms_on_real_progress():
    # A real _PROGRESS_KINDS marker STRICTLY newer than the pause note (the
    # canonical resume path posts epm:status-changed) disarms the hold so
    # respawn coverage resumes.
    import autonomous_session_watch as asw

    events = [
        _make_user_pause_event("2026-07-03T06:44:44Z"),
        {
            "ts": "2026-07-03T09:00:00Z",
            "kind": "epm:status-changed",
            "note": "status running -> running (resumed by user)",
        },
    ]
    assert asw._user_pause_hold_reason(events) is None


def test_user_pause_hold_reason_inert_without_pause_note():
    # No anchored pause note anywhere = not the prose-hold shape.
    import autonomous_session_watch as asw

    events = [
        _make_step_completed_event(step="10", exit_kind="parked"),
        {
            "ts": "2026-07-03T09:00:00Z",
            "kind": "epm:progress",
            "note": "round 2 implementing",
        },
    ]
    assert asw._user_pause_hold_reason(events) is None


def test_user_pause_skip_already_noted_dedup():
    # Self-contained events-log dedup: a skip marker at/after the gating pause
    # note means this episode's alert already fired; an OLDER one (prior
    # episode) does NOT count, so a fresh pause note re-arms the alert.
    import autonomous_session_watch as asw

    pause = _make_user_pause_event("2026-07-03T06:44:44Z")
    newer_skip = {
        "ts": "2026-07-03T06:50:00Z",
        "kind": "epm:progress",
        "note": f"{asw._USER_PAUSE_SKIP_NOTE_SENTINEL} prose hold — respawn suppressed.",
    }
    older_skip = {
        "ts": "2026-07-03T00:00:00Z",
        "kind": "epm:progress",
        "note": f"{asw._USER_PAUSE_SKIP_NOTE_SENTINEL} prose hold — respawn suppressed.",
    }
    assert asw._user_pause_skip_already_noted([pause, newer_skip]) is True
    assert asw._user_pause_skip_already_noted([pause, older_skip]) is False
    assert asw._user_pause_skip_already_noted([newer_skip]) is False  # no pause note


def test_check_orphan_followups_exemption_returns_user_pause_skip(monkeypatch):
    # Orphan pass: a prose-paused task with no live registered session
    # rewrites respawn -> "user-pause-hold-skip" WITHOUT consulting children
    # (the pause probe is checked first and is events-only); any other action
    # passes through unchanged.
    import autonomous_session_watch as asw

    def _boom(issue):
        raise AssertionError("_task_children must not be consulted on the user-pause path")

    monkeypatch.setattr(asw, "_task_children", _boom)
    pause_events = [_make_user_pause_event("2026-07-03T06:44:44Z")]
    action, reason = asw._check_orphan_followups_exemption(
        issue=816,
        status="running",
        has_pod=False,
        events=pause_events,
        action="respawn",
    )
    assert action == "user-pause-hold-skip"
    assert reason is not None
    assert "USER PAUSE" in reason

    # Non-respawn actions pass through unchanged (the early return).
    action2, reason2 = asw._check_orphan_followups_exemption(
        issue=816,
        status="running",
        has_pod=False,
        events=pause_events,
        action="keep",
    )
    assert (action2, reason2) == ("keep", None)


def test_user_pause_checked_before_spend_approval():
    # Ordering pin: events carrying BOTH a prose pause note and a
    # spend-approval park route to the pause action (checked FIRST — the most
    # specific gate signal; its alert carries the actionable durable-fix
    # recipe).
    import autonomous_session_watch as asw

    events = [
        _make_spend_approval_event("2026-07-03T06:00:00Z"),
        _make_user_pause_event("2026-07-03T06:44:44Z"),
    ]
    action, reason = asw._check_orphan_followups_exemption(
        issue=816,
        status="followups_running",
        has_pod=False,
        events=events,
        action="respawn",
    )
    assert action == "user-pause-hold-skip"
    assert reason is not None


def test_handle_orphan_user_pause_skip_posts_once_and_skips_budget(isolated_registry, monkeypatch):
    # The orphan handler MUST (a) post the one-time alert dedup'd via the
    # events log, naming the durable-park recipe (stable substring `on_hold`);
    # (b) persist state WITHOUT incrementing respawns_today.
    import autonomous_session_watch as asw

    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    pause_events = [_make_user_pause_event("2026-07-03T06:44:44Z")]
    asw._handle_orphan_user_pause_skip(
        issue=816,
        reason="prose USER PAUSE hold",
        new_missed=2,
        alerted=False,
        respawn_day="2026-07-03",
        respawns_today=0,
        followups_child_alerted=False,
        events=pause_events,
        state={},
        dry_run=False,
    )
    assert len(posted) == 1
    assert posted[0][0] == 816
    assert posted[0][2] == "user-pause-hold-skip"
    assert "on_hold" in posted[0][1]  # the durable-park recipe (AC3)
    assert "does NOT consume the daily respawn budget" in posted[0][1]
    state = asw._load_orphan_state(816)
    assert state["respawns_today"] == 0  # NOT incremented

    # Second call within the same episode: a skip marker now exists at/after
    # the pause note -> dedup'd, alert MUST NOT re-post.
    posted.clear()
    pause_events_with_skip = [
        *pause_events,
        {
            "ts": "2026-07-03T06:50:00Z",
            "kind": "epm:progress",
            "note": f"{asw._USER_PAUSE_SKIP_NOTE_SENTINEL} prose hold — respawn suppressed.",
        },
    ]
    asw._handle_orphan_user_pause_skip(
        issue=816,
        reason="prose USER PAUSE hold",
        new_missed=3,
        alerted=False,
        respawn_day="2026-07-03",
        respawns_today=0,
        followups_child_alerted=False,
        events=pause_events_with_skip,
        state=state,
        dry_run=False,
    )
    assert posted == []
    assert asw._load_orphan_state(816)["respawns_today"] == 0


def test_apply_stalled_followups_exemption_rewrites_user_pause_respawn_to_keep(monkeypatch):
    # The stalled-detector helper: an `action="respawn"` on a prose pause hold
    # MUST become `action="keep"` with `new_missed=0` and the third element
    # passed through unchanged (fixture passes followups_child_alerted=False
    # IN, so the assert pins pass-through, not an unrelated flag), plus the
    # one-time alert (events-log dedup).
    import autonomous_session_watch as asw

    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, label)),
    )
    events = [
        _make_user_pause_event("2026-07-03T06:44:44Z"),
        _make_step_completed_event(step="2c", exit_kind="parked", ts="2026-07-03T06:45:00Z"),
    ]
    action, new_missed, child_alerted = asw._apply_stalled_followups_exemption(
        issue=816,
        status="running",
        has_pod=False,
        events=events,
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action, new_missed, child_alerted) == ("keep", 0, False)
    assert len(posted) == 1
    assert posted[0][2] == "user-pause-hold-skip"
    assert "on_hold" in posted[0][1]  # the durable-park recipe (AC3)

    # Second call within the same episode: skip marker in events -> no re-post.
    posted.clear()
    events_with_skip = [
        *events,
        {
            "ts": "2026-07-03T06:50:00Z",
            "kind": "epm:progress",
            "note": f"{asw._USER_PAUSE_SKIP_NOTE_SENTINEL} prose hold — respawn suppressed.",
        },
    ]
    action2, new_missed2, _ = asw._apply_stalled_followups_exemption(
        issue=816,
        status="running",
        has_pod=False,
        events=events_with_skip,
        action="respawn",
        new_missed=2,
        followups_child_alerted=False,
        dry_run=False,
    )
    assert (action2, new_missed2) == ("keep", 0)
    assert posted == []  # dedup'd via the events log


def test_user_pause_skip_sentinel_in_watcher_filter():
    # The skip alert marker must NEVER reset the staleness clock it is
    # measured against — pin the sentinel into the shared exclusion set.
    from autonomous_session_watch import (
        _USER_PAUSE_SKIP_NOTE_SENTINEL,
        _WATCHER_NOTE_SENTINELS,
    )

    assert _USER_PAUSE_SKIP_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS


def test_user_pause_skip_in_orphan_exemption_actions_and_dispatch(monkeypatch):
    # Routing pin: the action string is a member of the exemption set, and
    # the dispatch routes it to the new handler with the reason forwarded.
    import autonomous_session_watch as asw

    assert "user-pause-hold-skip" in asw._ORPHAN_EXEMPTION_ACTIONS

    calls: list[dict] = []
    monkeypatch.setattr(
        asw,
        "_handle_orphan_user_pause_skip",
        lambda **kwargs: calls.append(kwargs),
    )
    asw._dispatch_orphan_exemption_action(
        action="user-pause-hold-skip",
        issue=816,
        followups_reason="prose USER PAUSE hold",
        events=[],
        new_missed=1,
        alerted=False,
        day_key="2026-07-03",
        respawns_today=0,
        followups_child_alerted=False,
        state={},
        dry_run=True,
    )
    assert len(calls) == 1
    assert calls[0]["issue"] == 816
    assert calls[0]["reason"] == "prose USER PAUSE hold"
    assert calls[0]["respawns_today"] == 0


def test_user_pause_hold_reason_is_case_sensitive():
    # [Must-Fix, statistics reconciler] A note beginning LOWERCASE
    # `user pause ...` (ordinary discussion prose) must NOT arm the probe —
    # kills the case-insensitive mutant the rest of the battery cannot catch.
    import autonomous_session_watch as asw

    events = [
        {
            "ts": "2026-07-03T06:44:44Z",
            "kind": "epm:progress",
            "note": "user pause requested? no — continuing with the planned round.",
        }
    ]
    assert asw._user_pause_hold_reason(events) is None


def test_user_pause_hold_reason_tie_with_distinct_marker_suppresses():
    # Tie-pin: a DISTINCT real progress marker (epm:results) at ts exactly ==
    # pause_ts must NOT disarm (strict `>` only) — kills the
    # `>=`-with-pause-filtered mutant that the self-inclusion tests (1-3)
    # cannot catch.
    import autonomous_session_watch as asw

    events = [
        _make_user_pause_event("2026-07-03T06:44:44Z"),
        {
            "ts": "2026-07-03T06:44:44Z",
            "kind": "epm:results",
            "note": "eval numbers for round 1 landed",
        },
    ]
    assert asw._user_pause_hold_reason(events) is not None


def test_user_pause_hold_reason_inert_on_malformed_ts():
    # A pause note with an absent / garbage ts leaves the probe INERT (fail
    # direction: respawn proceeds — the incident direction, documented in the
    # probe docstring), rather than crashing or arming with a bogus anchor.
    import autonomous_session_watch as asw

    events = [
        _make_user_pause_event(ts=None),  # type: ignore[arg-type]
        _make_user_pause_event(ts="not-a-timestamp"),
    ]
    assert asw._user_pause_hold_reason(events) is None


def test_handle_orphan_user_pause_skip_dry_run_skips_state(isolated_registry, monkeypatch):
    # dry_run=True: no _save_orphan_state write (state file absent), and the
    # dry_run flag is forwarded to the alert post (_post_progress_marker owns
    # the dry-run print semantics).
    import autonomous_session_watch as asw

    posted: list[tuple[int, str, bool, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, note, dry_run, label)),
    )
    asw._handle_orphan_user_pause_skip(
        issue=816,
        reason="prose USER PAUSE hold",
        new_missed=2,
        alerted=False,
        respawn_day="2026-07-03",
        respawns_today=0,
        followups_child_alerted=False,
        events=[_make_user_pause_event("2026-07-03T06:44:44Z")],
        state={},
        dry_run=True,
    )
    assert len(posted) == 1
    assert posted[0][2] is True  # dry_run forwarded to the poster
    assert asw._load_orphan_state(816) == {}  # no state write under dry_run


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
    # The DONE set shares the pod-safety DONE set AUTO_STOP_DONE — NOT the
    # wider pod-safety trigger set POD_SAFETY_AUTO_STOP (#980 widened the pod
    # pass to on_hold; sessions of paused tasks are deliberately kept):
    # awaiting_promotion /
    # completed / archived (2026-06-10 user request: "stop the happy sessions
    # once they reach awaiting promotion"). followups_running (a same-issue
    # follow-up round is executing) and blocked (under investigation) are
    # excluded — the session may be legitimately live there.
    from autonomous_session_watch import AUTO_STOP_DONE, SESSION_RECONCILE_DONE

    assert SESSION_RECONCILE_DONE == AUTO_STOP_DONE
    assert {"completed", "awaiting_promotion", "archived"} == SESSION_RECONCILE_DONE
    assert "followups_running" not in SESSION_RECONCILE_DONE
    assert "blocked" not in SESSION_RECONCILE_DONE
    assert "on_hold" not in SESSION_RECONCILE_DONE


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
        "on_hold",
    ],
)
@pytest.mark.parametrize("idle", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_session_reconcile_non_done_always_clears(status, idle, missed):
    # Any non-parked status (including the follow-up-executing
    # followups_running, the user-parked blocked, the user-paused on_hold —
    # #980 widened the POD pass to on_hold but sessions of paused tasks are
    # deliberately kept — and an unreadable None)
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
        monkeypatch, status="awaiting_promotion", pods=[_p(42, "pod-id-x", "pod-42")]
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
    detached_tmux_ttys=frozenset(),
    tmux_activity=None,
    registry=None,
    pm_sids=frozenset(),
    orphaned_predicate=None,
    controlling_tty_path=None,
):
    """Common monkeypatching for the idle-unmapped I/O tests: daemon children
    + session metadata + the TTY probe + the transcript-idle signal, leaving
    state files and decisions real. Pins asw.PROJECT_ROOT to the synthetic
    _Z_ROOT so the EPS-cwd check + issue inference are cwd-independent (see
    _Z_ROOT). Returns the (stops, records) recorders.

    ``tmux_activity`` (a ``{pane_tty: epoch}`` map) feeds the #695
    corroborating-idleness fallback; default ``{}`` (no activity -> no
    fallback). Both the detached set AND the activity map are served through
    the SINGLE combined helper ``_detached_tmux_panes_with_activity`` that the
    pass actually calls (and the legacy ``_detached_tmux_pane_ttys`` is patched
    too for the `_process_idle_unmapped`-default / `_is_live_user_tty`
    paths).

    ``orphaned_predicate`` (default ``lambda pid: False``) pins
    ``_wrapper_on_orphaned_tmux_server`` so the orphaned-tmux widening branch is
    INERT by default — every current caller keeps its pre-change behavior — and
    an orphaned-session test can flip it to ``lambda pid: True``.
    ``controlling_tty_path`` (default ``None``) pins
    ``_wrapper_controlling_tty_path`` — a pin the pre-#818 helper did NOT apply,
    added so the fold-1 observability read is deterministic without shelling to
    /proc."""
    import autonomous_session_watch as asw

    stops: list[str] = []
    records: list[str] = []
    activity = dict(tmux_activity or {})
    detached = set(detached_tmux_ttys)
    _orphaned = orphaned_predicate or (lambda pid: False)
    _tty_path = controlling_tty_path
    monkeypatch.setattr(asw, "PROJECT_ROOT", Path(_Z_ROOT))
    monkeypatch.setattr(asw, "_live_children", lambda: list(children))
    monkeypatch.setattr(asw, "_load_session_meta", lambda: dict(meta))
    monkeypatch.setattr(asw, "_load_session_issue_map", lambda: dict(registry or {}))
    monkeypatch.setattr(asw, "_load_pm_session_ids", lambda: set(pm_sids))
    monkeypatch.setattr(asw, "_wrapper_has_controlling_tty", lambda pid: has_tty)
    monkeypatch.setattr(asw, "_wrapper_on_orphaned_tmux_server", _orphaned)
    monkeypatch.setattr(asw, "_wrapper_controlling_tty_path", lambda pid: _tty_path)
    # Pin BOTH tmux probes so the I/O tests never shell out to a live tmux
    # server (deterministic; default = no detached panes, no activity). The
    # pass calls the combined helper; the legacy single-return wrapper is
    # patched too for callers that use it directly.
    monkeypatch.setattr(
        asw, "_detached_tmux_panes_with_activity", lambda: (set(detached), dict(activity))
    )
    monkeypatch.setattr(asw, "_detached_tmux_pane_ttys", lambda: set(detached))
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


def test_is_live_user_tty_detached_tmux_pane_is_not_live(monkeypatch):
    # The 2026-06-24 fix: a wrapper whose controlling tty is a DETACHED tmux
    # pane (in detached_tmux_ttys) is NOT a live-user tty, so it falls through
    # to the transcript-idle check. An ATTACHED pane / raw login pts / an
    # unresolvable tty stays live (keep-leaning).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_wrapper_has_controlling_tty", lambda pid: True)
    monkeypatch.setattr(asw, "_wrapper_controlling_tty_path", lambda pid: "/dev/pts/24")
    # /dev/pts/24 is a detached pane -> not live.
    assert asw._is_live_user_tty(99, {"/dev/pts/24"}) is False
    # /dev/pts/24 is NOT in the detached set (it is attached) -> live, keep.
    assert asw._is_live_user_tty(99, {"/dev/pts/99"}) is True
    # Tty path unresolvable -> cannot confirm detached -> live, keep.
    monkeypatch.setattr(asw, "_wrapper_controlling_tty_path", lambda pid: None)
    assert asw._is_live_user_tty(99, {"/dev/pts/24"}) is True
    # No controlling tty at all -> not a tty session (the headless case).
    monkeypatch.setattr(asw, "_wrapper_has_controlling_tty", lambda pid: False)
    assert asw._is_live_user_tty(99, {"/dev/pts/24"}) is False


# ── #818 orphaned-tmux-server widening tests ──────────────────────────────────


def _build_proc_tree(tmp_path, *, procs):
    """Build a ``/proc``-shaped tmp tree driving the REAL parentage walk (no
    helper is stubbed out). ``procs`` maps ``pid -> {"comm": str, "ppid": int,
    "pts_fds": [str, ...]}``: ``comm`` writes ``<proc>/<pid>/comm``, ``ppid``
    writes a minimal ``<proc>/<pid>/stat`` whose post-``)`` fields place the
    ppid at index 1 (``state ppid pgrp ...``), and ``pts_fds`` (optional)
    writes ``<proc>/<pid>/fd/<i>`` symlinks pointing at each ``/dev/pts`` target
    (the target need not resolve — ``_tmux_server_client_ttys`` reads the
    readlink STRING). Returns the proc-root Path."""
    import os

    proc_root = tmp_path / "proc"
    for pid, spec in procs.items():
        d = proc_root / str(pid)
        d.mkdir(parents=True)
        (d / "comm").write_text(spec["comm"] + "\n")
        ppid = spec.get("ppid", 1)
        # comm field is parenthesised (may contain spaces); the parser splits
        # after the LAST ')' so fields become: state(0) ppid(1) pgrp(2) ...
        (d / "stat").write_text(f"{pid} ({spec['comm']}) S {ppid} {ppid} 0 0 -1\n")
        pts_fds = spec.get("pts_fds")
        if pts_fds is not None:
            fdd = d / "fd"
            fdd.mkdir()
            for i, target in enumerate(pts_fds, start=3):
                os.symlink(target, fdd / str(i))
    return proc_root


def _mk_unix_socket(path):
    """Create a real AF_UNIX socket FILE at ``path`` (so ``Path.is_socket()``
    reports True) — the "a tmux server is reattachable" signal. The socket is
    bound then closed; the filesystem entry persists as a socket file."""
    import socket

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.close()


def test_orphaned_tmux_reap_env_toggle(monkeypatch):
    # EPM_ORPHANED_TMUX_REAP: unset / "1" -> enabled; explicit falsy disables
    # (mirrors test_idle_unmapped_env_helpers for _unmapped_idle_reap_enabled).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_ORPHANED_TMUX_REAP", raising=False)
    assert asw._orphaned_tmux_reap_enabled() is True
    monkeypatch.setenv("EPM_ORPHANED_TMUX_REAP", "1")
    assert asw._orphaned_tmux_reap_enabled() is True
    for falsy in ("0", "false", "no", " FALSE "):
        monkeypatch.setenv("EPM_ORPHANED_TMUX_REAP", falsy)
        assert asw._orphaned_tmux_reap_enabled() is False, falsy


def test_is_live_user_tty_orphaned_server_wrapper_is_not_live(monkeypatch):
    # The #818 branch, at the guard level. A wrapper with a controlling tty
    # that is NOT a detached pane, whose owning tmux server is orphaned, is
    # not-live ONLY when check_orphaned is on. The kill-switch (check_orphaned
    # omitted / False) keeps it. No controlling tty -> not-live regardless.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_wrapper_has_controlling_tty", lambda pid: True)
    monkeypatch.setattr(asw, "_wrapper_controlling_tty_path", lambda pid: "/dev/pts/7")
    monkeypatch.setattr(asw, "_wrapper_on_orphaned_tmux_server", lambda pid: True)
    # Orphaned + enabled -> not-live.
    assert asw._is_live_user_tty(99, set(), check_orphaned=True) is False
    # Orphaned but the widening is DISABLED -> live, keep (kill-switch at guard).
    assert asw._is_live_user_tty(99, set(), check_orphaned=False) is True
    # Default (check_orphaned omitted) never runs the branch -> live, keep
    # (the back-compat shape the existing 2-arg callers rely on).
    assert asw._is_live_user_tty(99, set()) is True
    # Enabled but the parentage predicate says NOT orphaned -> live, keep.
    monkeypatch.setattr(asw, "_wrapper_on_orphaned_tmux_server", lambda pid: False)
    assert asw._is_live_user_tty(99, set(), check_orphaned=True) is True
    # No controlling tty at all -> not-live regardless of the orphaned branch.
    monkeypatch.setattr(asw, "_wrapper_has_controlling_tty", lambda pid: False)
    monkeypatch.setattr(asw, "_wrapper_on_orphaned_tmux_server", lambda pid: True)
    assert asw._is_live_user_tty(99, set(), check_orphaned=True) is False


def test_wrapper_on_orphaned_tmux_server_fixture_proc_tree(tmp_path, monkeypatch):
    # Drive the REAL ppid walk + REAL client-fd read against a /proc-shaped tmp
    # tree (no helper stubbed): 700 -> 600 -> 500 (tmux: server, zero pts fds).
    # Socket dir empty -> orphaned (True). A socket file present -> keep (False).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")
    proc_root = _build_proc_tree(
        tmp_path,
        procs={
            500: {"comm": "tmux: server", "ppid": 1, "pts_fds": []},  # zero clients
            600: {"comm": "node", "ppid": 500},
            700: {"comm": "node", "ppid": 600},
        },
    )
    empty_sock_dir = tmp_path / "tmux-sock"
    empty_sock_dir.mkdir()
    monkeypatch.setattr(asw, "_tmux_socket_dir", lambda: empty_sock_dir)
    # Socket absent + zero clients -> the only reap-widening case.
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_root) is True

    # Add a socket file -> signal 1 says reattachable -> keep.
    sock = empty_sock_dir / "default"
    _mk_unix_socket(sock)
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_root) is False


def test_wrapper_on_orphaned_tmux_server_no_tmux_ancestor_keeps(tmp_path, monkeypatch):
    # No tmux: server anywhere in the chain (700 -> 600 -> pid 1) -> not our
    # class -> False regardless of the (empty) socket dir.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")
    proc_root = _build_proc_tree(
        tmp_path,
        procs={
            600: {"comm": "bash", "ppid": 1},
            700: {"comm": "node", "ppid": 600},
        },
    )
    empty_sock_dir = tmp_path / "tmux-sock"
    empty_sock_dir.mkdir()
    monkeypatch.setattr(asw, "_tmux_socket_dir", lambda: empty_sock_dir)
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_root) is False


def test_wrapper_on_orphaned_tmux_server_failsoft(tmp_path, monkeypatch):
    # Every uncertain probe -> False (KEEP). Parametrized cases (a)-(g).
    import autonomous_session_watch as asw

    # (a) tmux binary absent -> False (nothing to reap).
    monkeypatch.setattr(asw.shutil, "which", lambda name: None)
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=tmp_path / "nope") is False

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    # (b) <proc>/700/stat unreadable (_proc_ppid None) at a non-tmux hop -> walk
    # stops -> False. 700 has a comm ("node", not tmux) but no stat file.
    proc_b = tmp_path / "proc_b"
    (proc_b / "700").mkdir(parents=True)
    (proc_b / "700" / "comm").write_text("node\n")  # no stat -> _proc_ppid None
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_b) is False

    # (c) <proc>/700/comm missing at a walked hop (_proc_comm None): the walk
    # STOPS immediately on the unreadable comm (round-2 fail-toward-keep
    # guard; it never reaches _proc_ppid) -> no raise, False.
    proc_c = tmp_path / "proc_c"
    (proc_c / "700").mkdir(parents=True)  # neither comm nor stat
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_c) is False

    # (d) tmux: server ancestor found + socket-dir .iterdir() raises ->
    # _live_tmux_socket_present True -> False (keep).
    proc_d = _build_proc_tree(
        tmp_path / "d",
        procs={
            500: {"comm": "tmux: server", "ppid": 1, "pts_fds": []},
            700: {"comm": "node", "ppid": 500},
        },
    )
    monkeypatch.setattr(asw, "_tmux_socket_dir", lambda: tmp_path / "d" / "no-such-sock-dir")
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_d) is False

    # (e) a ppid cycle (600 <-> 700) -> seen-set guard -> False.
    proc_e = _build_proc_tree(
        tmp_path / "e",
        procs={
            600: {"comm": "node", "ppid": 700},
            700: {"comm": "node", "ppid": 600},
        },
    )
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_e) is False

    # (f) max_depth exhausted before a tmux: server -> False. A long non-tmux
    # chain, walked with a small max_depth.
    long_chain = {pid: {"comm": "node", "ppid": pid - 1} for pid in range(700, 690, -1)}
    proc_f = _build_proc_tree(tmp_path / "f", procs=long_chain)
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_f, max_depth=3) is False

    # (g) tmux: server ancestor found + socket dir EMPTY + server /fd UNREADABLE
    # (_tmux_server_client_ttys None) -> cannot prove zero clients -> False.
    proc_g = _build_proc_tree(
        tmp_path / "g",
        procs={
            # No pts_fds key -> no fd/ dir at all -> iterdir raises -> None.
            500: {"comm": "tmux: server", "ppid": 1},
            700: {"comm": "node", "ppid": 500},
        },
    )
    empty_g = tmp_path / "g" / "sock"
    empty_g.mkdir()
    monkeypatch.setattr(asw, "_tmux_socket_dir", lambda: empty_g)
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_g) is False


def test_wrapper_on_orphaned_tmux_server_comm_unreadable_intermediate_hop_keeps(
    tmp_path, monkeypatch
):
    # #818 round-2 regression (concern comm-unreadable-intermediate-hop-can-reap):
    # an INTERMEDIATE hop whose /proc/<pid>/comm is UNREADABLE but whose stat IS
    # readable must STOP the walk and KEEP — a socketless clientless tmux: server
    # reachable BEYOND the unclassifiable hop must NOT reap the wrapper. Before
    # the fix the walk continued past the None-comm hop, reached 500, and
    # returned True (reap-eligible). This is the fail-toward-keep contract for an
    # unreadable comm (plan v4 3.6; background-automation.md).
    import os

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")
    proc_root = tmp_path / "proc"
    # 500: a genuinely orphaned tmux: server (empty fd dir -> zero clients),
    # reachable ONLY by walking PAST the comm-unreadable hop 700.
    d500 = proc_root / "500"
    (d500 / "fd").mkdir(parents=True)  # empty fd dir -> zero /dev/pts clients
    (d500 / "comm").write_text("tmux: server\n")
    (d500 / "stat").write_text("500 (tmux: server) S 1 1 0 0 -1\n")
    # 700: the wrapper. stat IS readable (ppid -> 500) but comm is ABSENT
    # (the intermediate-hop comm-unreadable case).
    d700 = proc_root / "700"
    d700.mkdir(parents=True)
    (d700 / "stat").write_text("700 (node) S 500 500 0 0 -1\n")  # no comm file
    empty_sock_dir = tmp_path / "tmux-sock"
    empty_sock_dir.mkdir()  # socketless
    monkeypatch.setattr(asw, "_tmux_socket_dir", lambda: empty_sock_dir)
    # Sanity: the ancestor 500 IS a socketless zero-client orphaned server, so
    # were the walk to reach it the pre-fix code would return True. The fix must
    # stop at 700 (unreadable comm) -> False, with no raise.
    assert os.path.exists(d500 / "comm")  # ancestor is classifiable if reached
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_root) is False


def test_wrapper_on_orphaned_tmux_server_socketless_but_attached_keeps(tmp_path, monkeypatch):
    # The Must-Fix-1 regression: a tmux: server whose socket is deleted BUT that
    # still holds one /dev/pts client fd (an attached SSH session survives the
    # unlink) -> signal 2 proves a client is attached -> KEEP (False). This is
    # the systemd-tmpfiles-swept-a-live-SSH-session case the client proof blocks.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")
    proc_root = _build_proc_tree(
        tmp_path,
        procs={
            500: {"comm": "tmux: server", "ppid": 1, "pts_fds": ["/dev/pts/9"]},  # 1 client
            700: {"comm": "node", "ppid": 500},
        },
    )
    empty_sock_dir = tmp_path / "tmux-sock"
    empty_sock_dir.mkdir()  # socketless
    monkeypatch.setattr(asw, "_tmux_socket_dir", lambda: empty_sock_dir)
    assert asw._wrapper_on_orphaned_tmux_server(700, proc_root=proc_root) is False


def test_wrapper_on_orphaned_tmux_server_live_child_integration():
    # NON-monkeypatched real-/proc walk: spawn a throwaway child whose parent is
    # the pytest process (not a tmux: server). The parentage walk reads real
    # /proc/<pid>/stat ppid + comm and terminates with no tmux ancestor -> False
    # (keep). Proves the walk works against real /proc semantics, not a stub.
    import subprocess

    import autonomous_session_watch as asw

    child = subprocess.Popen(["sleep", "30"])
    try:
        assert asw._wrapper_on_orphaned_tmux_server(child.pid, proc_root=Path("/proc")) is False
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_tmux_server_client_ttys_live_integration():
    # NON-monkeypatched: read the pytest process's own /proc/<pid>/fd. It holds
    # no /dev/pts client fds via SCM_RIGHTS the way a tmux server would, so the
    # returned set is empty-or-small and the fd-read + readlink path does not
    # raise against real /proc.
    import os

    import autonomous_session_watch as asw

    result = asw._tmux_server_client_ttys(os.getpid(), Path("/proc"))
    assert result is None or isinstance(result, set)


def test_idle_unmapped_pass_orphaned_tmux_session_reaped_after_threshold(
    isolated_registry, monkeypatch
):
    # I/O test: an unmapped repo-root session with a controlling tty (not a
    # detached pane) whose owning tmux server is orphaned reads not-live ->
    # enters the idle branch -> idle >=12h accumulates a miss tick 1, stopped
    # tick 2. Mirrors test_idle_unmapped_pass_stop_fires_after_threshold.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    monkeypatch.delenv("EPM_ORPHANED_TMUX_REAP", raising=False)  # default ON
    children = [{"happySessionId": "sid-orph", "pid": 8181}]
    meta = {"sid-orph": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=over,
        has_tty=True,  # real _wrapper_has_controlling_tty pinned True
        controlling_tty_path="/dev/pts/7",  # not in the (empty) detached set
        orphaned_predicate=lambda pid: True,  # owning server orphaned
    )
    state_path = isolated_registry / "idle-unmapped-sid-orph.json"
    t0 = 1_000_000.0

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)
    state = json.loads(state_path.read_text())
    assert state["missed"] == 1 and stops == []

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == ["sid-orph"]
    assert len(records) == 1 and "auto-stopped idle unmapped" in records[0]


def test_idle_unmapped_pass_orphaned_reap_disabled_keeps(isolated_registry, monkeypatch):
    # Kill-switch end-to-end: EPM_ORPHANED_TMUX_REAP=0 -> check_orphaned False
    # -> the orphaned wrapper reads live (its tty is not a detached pane and the
    # widening is off) -> KEEP, state cleared, never stopped.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.setenv("EPM_ORPHANED_TMUX_REAP", "0")
    children = [{"happySessionId": "sid-off", "pid": 8282}]
    meta = {"sid-off": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=over,
        has_tty=True,
        controlling_tty_path="/dev/pts/7",
        orphaned_predicate=lambda pid: True,
    )
    state_path = isolated_registry / "idle-unmapped-sid-off.json"
    for now in (1_000_000.0, 1_000_600.0, 1_001_200.0):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert not state_path.exists()


def test_idle_unmapped_pass_orphaned_dry_run_mutates_nothing(isolated_registry, monkeypatch):
    # Dry-run discipline for the orphaned path (mirrors
    # test_idle_unmapped_pass_dry_run_mutates_nothing): an orphaned-tmux episode
    # seeded AT the stop point + the real _stop_session dry-run contract -> a
    # dry-run tick must not stop, not record, and leave the state byte-untouched.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_ORPHANED_TMUX_REAP", raising=False)
    children = [{"happySessionId": "sid-od", "pid": 8383}]
    meta = {"sid-od": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=over,
        has_tty=True,
        controlling_tty_path="/dev/pts/7",
        orphaned_predicate=lambda pid: True,
    )
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: (not dry_run) and (stops.append(sid) or True)
    )
    t0 = 1_000_000.0
    state_path = isolated_registry / "idle-unmapped-sid-od.json"
    seeded = json.dumps({"missed": 1, "alerted": False, "first_over_ts": t0})
    state_path.write_text(seeded)

    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == [] and records == []
    assert state_path.read_text() == seeded  # untouched, not even rewritten


def test_idle_unmapped_pass_live_socket_pane_never_reaped(isolated_registry, monkeypatch):
    # Regression guard for the healthy fleet: a wrapper on a LIVE-socket tmux
    # server (orphaned predicate False) with a controlling tty -> reads live ->
    # KEEP across 3 ticks, never stopped, state never accumulated. Complements
    # the live-VM verification (one live-socket server -> zero behavior change).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_ORPHANED_TMUX_REAP", raising=False)
    children = [{"happySessionId": "sid-live", "pid": 8484}]
    meta = {"sid-live": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=over,
        has_tty=True,
        controlling_tty_path="/dev/pts/7",
        orphaned_predicate=lambda pid: False,  # socket present -> not orphaned
    )
    state_path = isolated_registry / "idle-unmapped-sid-live.json"
    for now in (1_000_000.0, 1_000_600.0, 1_001_200.0):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert not state_path.exists()


def test_detached_tmux_pane_ttys_failsoft_when_tmux_absent(monkeypatch):
    # Fail-soft contract: tmux missing -> empty set -> every tty-bearing
    # wrapper stays "live" -> keep-all preserved (never an accidental reap).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: None)
    assert asw._detached_tmux_pane_ttys() == set()


def test_detached_tmux_pane_ttys_parses_attached_count(monkeypatch):
    # Only panes whose tmux session has zero attached clients are reported as
    # detached; attached panes and unparseable rows are excluded.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    class _Out:
        returncode = 0
        stdout = (
            "/dev/pts/24\t0\n"  # detached -> included
            "/dev/pts/39\t0\n"  # detached -> included
            "/dev/pts/47\t1\n"  # attached -> excluded
            "/dev/pts/50\t2\n"  # attached (2 clients) -> excluded
            "\t0\n"  # empty pane_tty -> skipped
            "/dev/pts/9\tnope\n"  # unparseable count -> skipped
        )

    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: _Out())
    assert asw._detached_tmux_pane_ttys() == {"/dev/pts/24", "/dev/pts/39"}


def test_detached_tmux_pane_ttys_failsoft_on_nonzero_rc(monkeypatch):
    # No tmux server running -> non-zero rc -> empty set (keep-all preserved).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    class _Out:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: _Out())
    assert asw._detached_tmux_pane_ttys() == set()


def test_idle_unmapped_pass_detached_tmux_reaps_attached_kept(isolated_registry, monkeypatch):
    # End-to-end: two unmapped EPS sessions, both tty-bearing and both idle
    # past the window. One sits in a DETACHED tmux pane (reapable), the other
    # in an ATTACHED pane (Thomas is live -> never touched). After the 2-miss
    # guard only the detached one is stopped.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    children = [
        {"happySessionId": "sid-detached", "pid": 100},
        {"happySessionId": "sid-attached", "pid": 200},
    ]
    meta = {"sid-detached": {"path": _Z_ROOT}, "sid-attached": {"path": _Z_ROOT}}
    over = asw.UNMAPPED_IDLE_REAP_S + 3600
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=over,
        has_tty=True,  # both wrappers hold a controlling tty
        detached_tmux_ttys={"/dev/pts/24"},
    )
    # pid 100 -> detached pane; pid 200 -> attached pane (not in the set).
    monkeypatch.setattr(
        asw,
        "_wrapper_controlling_tty_path",
        lambda pid: "/dev/pts/24" if pid == 100 else "/dev/pts/47",
    )
    t0 = 1_000_000.0
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)  # accumulate
    assert stops == []
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)  # stop
    assert stops == ["sid-detached"]
    assert len(records) == 1 and "auto-stopped idle unmapped" in records[0]
    # The attached session never accumulated state and was never stopped.
    assert not (isolated_registry / "idle-unmapped-sid-attached.json").exists()


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


# ── #695 corroborating-idleness fallback tests ────────────────────────────────


def _patch_fallback_gates(
    monkeypatch,
    *,
    pane_tty="/dev/pts/24",
    has_work_descendant=False,
    running_pods=None,
    pending_input=False,
):
    """Stub the four real dependencies of the #695 fallback gate evaluation
    that `_patch_idle_io` does NOT cover (so a fallback test never shells out
    to a live tmux / RunPod API / /proc): the wrapper's controlling-tty path
    (gate 1), the work-descendant probe (gate 3), the running-pod snapshot
    (gate 4), and the pending-pane-input probe (gate 5). ``running_pods``
    default ``[]`` (genuinely no pods -> gate passes); pass ``None`` to
    simulate a failed snapshot, or a non-empty list to simulate a live pod."""
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_wrapper_controlling_tty_path", lambda pid: pane_tty)
    monkeypatch.setattr(
        asw, "_has_running_work_descendant", lambda pid, cmap=None: has_work_descendant
    )
    pods = [] if running_pods is None else running_pods
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *a, **k: pods)
    monkeypatch.setattr(asw, "_pane_has_pending_input", lambda pane: pending_input)


def test_idle_unmapped_fallback_reaps_when_all_gates_pass(isolated_registry, monkeypatch):
    # Test 1 (load-bearing REAP): detached + unmapped + no work + no pod + over
    # the fallback threshold + no pending input -> the fallback supplies a
    # substitute idle age and the session is stopped after the 2-miss guard.
    # The pre-stop audit row is written BEFORE the stop, and the post-stop note
    # is the fallback-DISTINCT narrative.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_TMUX_IDLE_FALLBACK_S", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_TMUX_IDLE_FALLBACK_ENABLED", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-fb", "pid": 4242}]
    meta = {"sid-fb": {"path": _Z_ROOT}}
    # has_tty True + the pane in the detached set => not a live-user tty =>
    # falls through to the idle branch; primary transcript signal None =>
    # fallback eligible.
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch)
    state_path = isolated_registry / "idle-unmapped-sid-fb.json"
    events_path = isolated_registry / "idle-unmapped-events.jsonl"

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)  # miss 1
    assert stops == [] and records == []
    assert json.loads(state_path.read_text())["missed"] == 1

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)  # stop
    assert stops == ["sid-fb"]
    assert len(records) == 1
    assert asw._IDLE_UNMAPPED_STOP_FALLBACK_NOTE_SENTINEL in records[0]
    # A pre-stop would_stop_fallback audit row landed in the events file.
    rows = [json.loads(ln) for ln in events_path.read_text().splitlines() if ln.strip()]
    audits = [r for r in rows if r.get("kind") == "would_stop_fallback"]
    assert len(audits) == 1
    assert audits[0]["fallback_source"] == "tmux_session_activity"


def test_idle_unmapped_fallback_keeps_when_work_descendant_present(isolated_registry, monkeypatch):
    # Test 2 (work-descendant KEEP): a running codex / experimenter / train.py
    # descendant blocks the fallback reap entirely (the experimenter incident:
    # 1/6 sessions). Never stops, never accumulates a fallback episode.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-w", "pid": 100}]
    meta = {"sid-w": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch, has_work_descendant=True)
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert not list(isolated_registry.glob("idle-unmapped-*.json"))


def test_idle_unmapped_fallback_keeps_when_running_pod_present(isolated_registry, monkeypatch):
    # Test 3 (running-pod KEEP): a non-empty managed-RUNNING-pod snapshot defers
    # the fallback reap (the conservative no-issue-key floor for unmapped
    # sessions).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-p", "pid": 100}]
    meta = {"sid-p": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch, running_pods=[_p(489, "p489", "pod-489")])
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []


def test_idle_unmapped_fallback_keeps_when_pod_snapshot_failed(isolated_registry, monkeypatch):
    # Test 3b (uncertain-pod KEEP): a None snapshot (API error) is uncertain ->
    # KEEP (no_running_pods is False unless the snapshot is a real empty list).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-pn", "pid": 100}]
    meta = {"sid-pn": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch, running_pods=None)  # None == failed snapshot
    # Patch the helper to actually return None (the helper, not the default []).
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *a, **k: None)
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []


def test_idle_unmapped_fallback_keeps_when_under_threshold(isolated_registry, monkeypatch):
    # Test 4 (under-threshold KEEP): session_activity age under the fallback
    # window -> no substitute idle age over the floor -> ("skip", missed), never
    # stops, no fallback episode accumulated.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    under = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S - 3600  # fresh-ish: under the floor
    children = [{"happySessionId": "sid-u", "pid": 100}]
    meta = {"sid-u": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - under},
    )
    _patch_fallback_gates(monkeypatch)
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert not list(isolated_registry.glob("idle-unmapped-*.json"))


def test_idle_unmapped_fallback_keeps_when_activity_unavailable(isolated_registry, monkeypatch):
    # Test 5 (unavailable-signal KEEP): the pane has no session_activity entry
    # (empty activity map) -> the fallback finds no substitute age -> KEEP.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    children = [{"happySessionId": "sid-na", "pid": 100}]
    meta = {"sid-na": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={},  # pane absent from the activity map
    )
    _patch_fallback_gates(monkeypatch)
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []


def test_idle_unmapped_fallback_keeps_attached_pane(isolated_registry, monkeypatch):
    # Test 6 (attached-pane KEEP): a session whose pane is NOT in the detached
    # set is a live-user tty -> has_tty stays True -> ("clear", 0), the fallback
    # is never reached. Preserves the existing detached-vs-attached behavior.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-att", "pid": 100}]
    meta = {"sid-att": {"path": _Z_ROOT}}
    # has_tty True but the controlling pane is /dev/pts/47, NOT in the detached
    # set -> _is_live_user_tty True -> clear.
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/47": t0 - over},
    )
    monkeypatch.setattr(asw, "_wrapper_controlling_tty_path", lambda pid: "/dev/pts/47")
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []


def test_idle_unmapped_fallback_not_consulted_when_primary_signal_present(
    isolated_registry, monkeypatch
):
    # Test 7: when the PRIMARY transcript signal resolves, the fallback gate
    # evaluation is NEVER reached — assert the fallback evaluator (and its
    # pending-input probe) are not called. The primary path drives the decision.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    primary_over = asw.UNMAPPED_IDLE_REAP_S + 3600
    children = [{"happySessionId": "sid-pri", "pid": 100}]
    meta = {"sid-pri": {"path": _Z_ROOT}}
    stops, _records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=primary_over,  # PRIMARY signal available
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - 999_999},
    )
    called = {"fallback": 0, "pending": 0}
    monkeypatch.setattr(
        asw,
        "_evaluate_idle_unmapped_fallback",
        lambda *a, **k: called.__setitem__("fallback", called["fallback"] + 1) or (None, None),
    )
    monkeypatch.setattr(
        asw,
        "_pane_has_pending_input",
        lambda pane: called.__setitem__("pending", called["pending"] + 1) or False,
    )
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)
    # The primary path stopped it (transcript over the primary window).
    assert stops == ["sid-pri"]
    assert called["fallback"] == 0 and called["pending"] == 0


def test_idle_unmapped_empty_detached_set_beacon(isolated_registry, monkeypatch, capsys):
    # Test 8: tmux present but the detached set is EMPTY -> the once-per-pass
    # loud WARNING beacon fires (the silent-regression guard). Fail-soft set
    # stays empty; nothing is reaped.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")
    children = [{"happySessionId": "sid-b", "pid": 100}]
    meta = {"sid-b": {"path": _Z_ROOT}}
    stops, _records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        detached_tmux_ttys=set(),  # tmux present but no detached panes
        tmux_activity={},
    )
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=1_000_000.0)
    assert "tmux present but detached set empty" in capsys.readouterr().err
    assert stops == []


def test_idle_unmapped_fallback_dry_run_mutates_nothing(isolated_registry, monkeypatch):
    # Test 9 (extended dry-run): a dry-run tick at the fallback stop point
    # neither stops, writes the pre-stop audit row, nor rewrites state.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-dr", "pid": 100}]
    meta = {"sid-dr": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch)
    # Mirror the REAL _stop_session dry-run contract (returns False, no action).
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: (not dry_run) and (stops.append(sid) or True)
    )
    state_path = isolated_registry / "idle-unmapped-sid-dr.json"
    seeded = json.dumps({"missed": 1, "alerted": False, "first_over_ts": t0})
    state_path.write_text(seeded)
    events_path = isolated_registry / "idle-unmapped-events.jsonl"

    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == [] and records == []
    assert state_path.read_text() == seeded  # untouched
    assert not events_path.exists()  # no audit row written under dry-run


def test_idle_unmapped_fallback_keeps_when_pending_input(isolated_registry, monkeypatch):
    # Test 10 (MF1 typed-but-unsent KEEP): all five other gates pass but the
    # pane shows pending un-submitted input -> KEEP. No stop, no audit row, no
    # accumulated episode. This is the dominant-class (4/6) protection.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-pi", "pid": 100}]
    meta = {"sid-pi": {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch, pending_input=True)  # buffered input present
    events_path = isolated_registry / "idle-unmapped-events.jsonl"
    for now in (t0, t0 + 600, t0 + 1200):
        asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=now)
    assert stops == [] and records == []
    assert not events_path.exists()
    assert not list(isolated_registry.glob("idle-unmapped-sid-pi.json"))


# A real-shape Claude TUI render: a ✻ status line, a /clear hint, a top
# box-rule (carrying the ↯ token-count glyph), the caret input row (U+276F caret +
# a U+00A0 non-breaking space separator), a bottom box-rule, and the
# ⏵⏵ permissions footer. The input row is NEVER the last captured line —
# the bottom-up scanner must skip the bottom rule + footer to reach it.
# (#695 round-2 blocker 1; structure verified against four live detached panes.)
_REAL_CLAUDE_TOP_RULE = "───────────────────────────────── ↯ ─"
_REAL_CLAUDE_BOTTOM_RULE = "─────────────────────────────────────"
_REAL_CLAUDE_FOOTER = "  ⏵⏵ bypass permissions on  · ← for…"


def _real_claude_render(input_row: str) -> str:
    """A capture-pane stdout in the live Claude TUI shape, with ``input_row``
    placed above the bottom rule + permissions footer."""
    return (
        "✻ Cogitated for 28s\n"
        "  new task? /clear to save 281.2k t…\n"
        f"{_REAL_CLAUDE_TOP_RULE}\n"
        f"{input_row}\n"
        f"{_REAL_CLAUDE_BOTTOM_RULE}\n"
        f"{_REAL_CLAUDE_FOOTER}\n"
    )


def test_pane_has_pending_input_heuristic(monkeypatch):
    # Test 11 (MF1 heuristic unit tests, #695 round-2 bottom-up scanner): the
    # KEEP-leaning text heuristic over REAL-shape capture-pane output — the
    # input row sits ABOVE a bottom rule + ⏵⏵ footer, so the scanner walks
    # bottom-up skipping border/footer lines. Empty box / placeholder -> False
    # (proceed); buffered text -> True (KEEP); subprocess error / pane gone /
    # tmux absent / all-borders-and-footers -> True (KEEP, fail-soft).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    def _set_capture(stdout, returncode=0):
        class _Out:
            pass

        o = _Out()
        o.stdout = stdout
        o.returncode = returncode
        monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: o)

    # Real-shape EMPTY input box (caret + non-breaking space only) -> proceed.
    _set_capture(_real_claude_render("\u276f\xa0"))
    assert asw._pane_has_pending_input("/dev/pts/24") is False
    # Real-shape BUFFERED input (caret + NBSP + typed text) -> KEEP (True).
    _set_capture(_real_claude_render("\u276f\xa0promote it useful"))
    assert asw._pane_has_pending_input("/dev/pts/24") is True
    # Whitespace-only capture -> no input row -> KEEP (True).
    _set_capture("   \n  \n")
    assert asw._pane_has_pending_input("/dev/pts/24") is True
    # Empty-prompt placeholder hint inside the real shape -> proceed (False).
    _set_capture(_real_claude_render('\u276f\xa0Try "fix the bug"'))
    assert asw._pane_has_pending_input("/dev/pts/24") is False
    _set_capture(_real_claude_render("\u276f\xa0/for shortcuts"))  # 'for shortcuts' hint
    assert asw._pane_has_pending_input("/dev/pts/24") is False
    # Older / idealized ASCII box render (no ⏵⏵ footer): the bottom line is the
    # ╰──╯ rule, skipped; the input row above it is judged. Empty -> proceed.
    _set_capture("╭──────────────╮\n│ >            │\n╰──────────────╯\n")
    assert asw._pane_has_pending_input("/dev/pts/24") is True  # '│ >  │' has trailing border glyph
    # capture-pane non-zero rc (pane gone) -> KEEP (fail-soft).
    _set_capture("", returncode=1)
    assert asw._pane_has_pending_input("/dev/pts/24") is True

    # subprocess raises -> KEEP (fail-soft).
    def _boom(*a, **k):
        raise asw.subprocess.SubprocessError("boom")

    monkeypatch.setattr(asw.subprocess, "run", _boom)
    assert asw._pane_has_pending_input("/dev/pts/24") is True
    # tmux absent -> KEEP (fail-soft).
    monkeypatch.setattr(asw.shutil, "which", lambda name: None)
    assert asw._pane_has_pending_input("/dev/pts/24") is True


def test_pane_has_pending_input_real_render_buffered_keeps(monkeypatch):
    # Test 11b (#695 round-2 blocker 1, load-bearing): the REAL Claude TUI
    # render (top rule -> caret input row -> bottom rule -> ⏵⏵ footer) with
    # BUFFERED input. The last captured line is the footer, NOT the input row —
    # the round-1 last-line-only heuristic returned False here (allowing a
    # spurious reap of a session with typed-but-unsent input). The bottom-up
    # scanner skips footer + bottom rule and reads the caret row -> KEEP (True).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    class _Out:
        stdout = _real_claude_render("\u276f\xa0check progress")
        returncode = 0

    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: _Out())
    # Sanity: the captured LAST line really is the footer, not the input row.
    assert _Out.stdout.splitlines()[-1].lstrip().startswith("⏵")
    assert asw._pane_has_pending_input("/dev/pts/24") is True


def test_pane_has_pending_input_real_render_empty_proceeds(monkeypatch):
    # Test 11c (#695 round-2 blocker 1): the REAL Claude render with a genuinely
    # EMPTY input box (the caret caret + a lone U+00A0 separator, nothing typed).
    # The scanner skips the footer + bottom rule, reaches the caret row, strips the
    # caret + NBSP, finds an empty remainder -> may proceed (False, allows
    # reap). This is the positive empty-case the brief requires; the empty box
    # is identified from the real caret render without a false negative.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    class _Out:
        stdout = _real_claude_render("\u276f\xa0")
        returncode = 0

    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: _Out())
    assert asw._pane_has_pending_input("/dev/pts/24") is False


def test_pane_has_pending_input_all_borders_and_footers_keeps(monkeypatch):
    # Test 11d (#695 round-2 blocker 1): a capture consisting ONLY of border /
    # rule lines and footer lines (no recognizable input row at all) -> the
    # bottom-up scanner finds no input line -> cannot confirm empty -> KEEP.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw.shutil, "which", lambda name: "/usr/bin/tmux")

    capture = (
        f"{_REAL_CLAUDE_TOP_RULE}\n"
        f"{_REAL_CLAUDE_BOTTOM_RULE}\n"
        "╭──────────────╮\n"
        "╰──────────────╯\n"
        f"{_REAL_CLAUDE_FOOTER}\n"
        "  ? for shortcuts\n"
    )

    class _Out:
        stdout = capture
        returncode = 0

    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: _Out())
    assert asw._pane_has_pending_input("/dev/pts/24") is True


def test_pane_line_classifiers(monkeypatch):
    # Test 11e (#695 round-2 blocker 1): the bottom-up scanner's two line
    # classifiers. Border lines: pure box-drawing / rule glyphs (incl. the ↯
    # token-count glyph on the top rule). Footer lines: ⏵ / ? / nav-arrow
    # leading glyph. The caret input row is NEITHER (so the scanner stops on it).
    import autonomous_session_watch as asw

    # Borders.
    assert asw._pane_line_is_border("─────────────────────────────────────")
    assert asw._pane_line_is_border("╭──────────────╮")
    assert asw._pane_line_is_border("╰──────────────╯")
    assert asw._pane_line_is_border(_REAL_CLAUDE_TOP_RULE)  # carries ↯
    assert asw._pane_line_is_border("   ") is False  # all-whitespace is not a border
    assert asw._pane_line_is_border("\u276f\xa0promote it useful") is False
    # Footers.
    assert asw._pane_line_is_footer(_REAL_CLAUDE_FOOTER)  # ⏵⏵ permissions
    assert asw._pane_line_is_footer("  ? for shortcuts")
    assert asw._pane_line_is_footer("↑/↓ to navigate")
    assert asw._pane_line_is_footer("\u276f\xa0promote it useful") is False
    assert asw._pane_line_is_footer("   ") is False  # all-whitespace is not a footer


def test_work_descendant_denylist_import_pin():
    # Test 12 (MF2 import-pin): ORPHAN_HOLDER_PATTERNS resolves + compiles and
    # the union matches a known codex cmdline; each LOCAL workload marker is in
    # the gate's denylist. A future rename of either source trips this test.
    import re

    import autonomous_session_watch as asw
    from worktree_audit import ORPHAN_HOLDER_PATTERNS

    assert isinstance(ORPHAN_HOLDER_PATTERNS, tuple) and ORPHAN_HOLDER_PATTERNS
    assert all(isinstance(p, re.Pattern) for p in ORPHAN_HOLDER_PATTERNS)
    # The union matches a real codex companion cmdline.
    codex_cmd = "node /home/x/.claude/plugins/cache/openai-codex/dist/index.js app-server"
    assert any(p.search(codex_cmd) for p in ORPHAN_HOLDER_PATTERNS)
    # asw imported the SAME tuple object.
    assert asw.ORPHAN_HOLDER_PATTERNS is ORPHAN_HOLDER_PATTERNS
    # Every named LOCAL workload marker is in the gate's denylist.
    for marker in (
        "scripts/train.py",
        "scripts/eval.py",
        "scripts/run_sweep.py",
        "scripts/dispatch_issue.py",
        "backend_poll.py",
        "experiment-implementer",
    ):
        assert marker in asw._IDLE_UNMAPPED_WORK_CMDLINE_MARKERS, marker


def test_work_descendant_unreadable_child_keeps(monkeypatch):
    # Test 12b (#695 round-2 blocker 2): a wrapper subtree with a child whose
    # /proc/<pid>/cmdline read raises OSError. The round-1 _cmdline_is_work_process
    # swallowed OSError -> False, so an unreadable work child looked "not work"
    # and the gate-3 walk could return False -> reap permitted. The tri-state
    # probe now returns None (uncertain) for an unreadable cmdline, and
    # _has_running_work_descendant treats None as work-present -> KEEP (True),
    # honoring the fail-toward-KEEP contract.
    import autonomous_session_watch as asw

    # Topology: wrapper 100 -> child 200 -> grandchild 300.
    children_map = {100: [200], 200: [300]}
    # 100 + 200 readable + non-work; 300 cmdline unreadable (perms / race).
    readable_nonwork = {100: b"node /happy/index.mjs claude\x00", 200: b"node mcp\x00"}

    def _fake_read_bytes(self):
        # self is a Path("/proc/<pid>/cmdline")
        s = str(self)
        pid = int(s.split("/proc/")[1].split("/")[0])
        if pid in readable_nonwork:
            return readable_nonwork[pid]
        raise OSError("EACCES")  # 300 -> unreadable

    monkeypatch.setattr(asw.Path, "read_bytes", _fake_read_bytes)

    # Tri-state probe directly: readable-nonwork -> False, unreadable -> None.
    assert asw._cmdline_is_work_process(100) is False
    assert asw._cmdline_is_work_process(300) is None
    # The walk: the unreadable grandchild 300 makes the subtree work-present.
    assert asw._has_running_work_descendant(100, children_map) is True


def test_work_descendant_all_readable_nonwork_allows_reap(monkeypatch):
    # Test 12c (#695 round-2 blocker 2, complement): when EVERY child cmdline is
    # readable and NONE match the work-process denylist, the walk returns False
    # (no work descendant -> gate 3 allows reap) — the pre-existing behavior is
    # preserved by the tri-state change (False is still positively not-work).
    import autonomous_session_watch as asw

    children_map = {100: [200], 200: [300]}
    readable_nonwork = {
        100: b"node /happy/index.mjs claude\x00",
        200: b"node /mcp/runpod\x00",
        300: b"node /mcp/arxiv\x00",
    }

    def _fake_read_bytes(self):
        pid = int(str(self).split("/proc/")[1].split("/")[0])
        return readable_nonwork[pid]  # all readable

    monkeypatch.setattr(asw.Path, "read_bytes", _fake_read_bytes)

    for pid in (100, 200, 300):
        assert asw._cmdline_is_work_process(pid) is False
    assert asw._has_running_work_descendant(100, children_map) is False
    # And a positive work marker anywhere in the subtree still trips it.
    readable_nonwork[300] = b"python scripts/train.py condition=c1\x00"
    assert asw._cmdline_is_work_process(300) is True
    assert asw._has_running_work_descendant(100, children_map) is True


def test_idle_unmapped_fallback_audit_before_stop_and_payload(isolated_registry, monkeypatch):
    # Test 13 (MF3 ordering + payload): the pre-stop audit write happens BEFORE
    # the _stop_session call (audit_ts < stop_ts), and the audit payload
    # carries all nine named fields.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0
    over = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    children = [{"happySessionId": "sid-ord", "pid": 100}]
    meta = {"sid-ord": {"path": _Z_ROOT}}
    _stops, _records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over},
    )
    _patch_fallback_gates(monkeypatch)
    order: list[tuple[str, float]] = []
    seq = {"n": 0}

    def _next():
        seq["n"] += 1
        return float(seq["n"])

    captured_payload: list[dict] = []

    def _audit(payload, dry_run):
        captured_payload.append(payload)
        order.append(("audit", _next()))

    monkeypatch.setattr(asw, "_append_idle_unmapped_audit", _audit)
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: order.append(("stop", _next())) or True
    )

    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)  # miss 1
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)  # stop

    audit_ts = next(ts for name, ts in order if name == "audit")
    stop_ts = next(ts for name, ts in order if name == "stop")
    assert audit_ts < stop_ts
    assert len(captured_payload) == 1
    payload = captured_payload[0]
    for field in (
        "sid",
        "pid",
        "fallback_source",
        "idle_age_s",
        "threshold_env_value",
        "detached_verdict",
        "work_descendant",
        "running_pods",
        "pending_input",
    ):
        assert field in payload, field
    assert payload["work_descendant"] is False
    assert payload["running_pods"] == []
    assert payload["pending_input"] is False


def test_idle_unmapped_fallback_post_stop_note_is_distinct(isolated_registry, monkeypatch):
    # Test 14 (MF3 fallback-distinct): the fallback reap's post-stop note does
    # NOT contain the primary-transcript narrative ("its resolved Claude
    # transcript has been idle") and DOES carry fallback-source language;
    # the PRIMARY reap's note is the existing transcript narrative unchanged.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    t0 = 1_000_000.0

    # ── fallback reap ──
    over_fb = asw.UNMAPPED_TMUX_IDLE_FALLBACK_S + 3600
    _stops, records = _patch_idle_io(
        monkeypatch,
        children=[{"happySessionId": "sid-fbn", "pid": 100}],
        meta={"sid-fbn": {"path": _Z_ROOT}},
        idle_age=None,
        has_tty=True,
        detached_tmux_ttys={"/dev/pts/24"},
        tmux_activity={"/dev/pts/24": t0 - over_fb},
    )
    _patch_fallback_gates(monkeypatch)
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)
    assert len(records) == 1
    fb_note = records[0]
    assert "its resolved Claude transcript has been idle" not in fb_note
    assert asw._IDLE_UNMAPPED_STOP_FALLBACK_NOTE_SENTINEL in fb_note
    assert "session_activity" in fb_note

    # ── primary reap (separate session) ──
    over_pri = asw.UNMAPPED_IDLE_REAP_S + 3600
    _stops2, records2 = _patch_idle_io(
        monkeypatch,
        children=[{"happySessionId": "sid-prn", "pid": 200}],
        meta={"sid-prn": {"path": _Z_ROOT}},
        idle_age=over_pri,
    )
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)
    assert len(records2) == 1
    pri_note = records2[0]
    assert "its resolved Claude transcript has been idle" in pri_note
    assert asw._IDLE_UNMAPPED_STOP_NOTE_SENTINEL in pri_note


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


def test_watcher_helpers_share_transcript_memo_within_scope(tmp_path, monkeypatch):
    # The #1182 Goal, mechanically: within ONE tick-scope, the wedge-probe
    # helper (_transcript_tail_rows) and the idle-age helper
    # (_transcript_idle_age_s) resolve the SAME pid with exactly ONE
    # underlying happy-log resolution (cross-call-site dedup). The fake
    # returns a REAL on-disk transcript with mtime <= now — both helpers
    # stat/open the file AFTER resolution.
    import os

    import autonomous_session_watch as asw
    import session_resolver

    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"type":"user","sessionId":"s"}\n')
    now = 1_000_000.0
    os.utime(transcript, (now - 500.0, now - 500.0))

    calls: list[int] = []

    def fake_uncached(pid):
        calls.append(pid)
        return (str(transcript), None)

    monkeypatch.setattr(
        session_resolver, "_resolve_transcript_via_happy_log_uncached", fake_uncached
    )
    with session_resolver.transcript_resolution_scope():
        rows = asw._transcript_tail_rows(123)
        age, reason = asw._transcript_idle_age_s(123, now=now)
    assert rows == [{"type": "user", "sessionId": "s"}]
    assert reason is None and age == 500.0
    assert calls == [123]


def test_main_runs_tick_under_transcript_memo_scope(isolated_registry, monkeypatch):
    # #1182 wiring pin: the whole watcher tick (main()) executes inside ONE
    # transcript_resolution_scope() — an ACTIVE memo while the passes run,
    # and None again after main() returns (never-across-ticks, acceptance
    # criterion 2). Stub harness copied from
    # test_main_order_stale_registration_after_gate_push.
    import autonomous_session_watch as asw
    import session_resolver

    seen: list[bool] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    # #1247 r3: shared hermeticity helper FIRST — covers the fleet-mutating +
    # live-observer set incl. vm_ledger_reap_pass (absent from the loop below,
    # which would otherwise mutate the live ~/.task-workflow/vm-ledger.json);
    # the loop + the stale_registration_pass recorder are patched AFTER, so
    # they win (later monkeypatch wins).
    _stub_fleet_mutating_passes(asw, monkeypatch)
    for name in (
        "vm_disk_pass",
        "data_disk_pass",
        "happy_patch_pass",
        "cpu_guard_pass",
        "triage_observer_pass",
        "verdict_disagree_pass",
        "program_orchestrator_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "proposed_infra_sweep_pass",
        "capacity_retry_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "gate_push_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
        "gc_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(
        asw,
        "stale_registration_pass",
        lambda *a, **kw: seen.append(session_resolver._TRANSCRIPT_MEMO is not None),
    )
    assert session_resolver._TRANSCRIPT_MEMO is None
    rc = asw.main([])
    assert rc == 0
    assert seen == [True]
    assert session_resolver._TRANSCRIPT_MEMO is None


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
            asw,
            "_dispatch_infra_drain",
            lambda i, slot, dry, **kw: dispatched.append(i) or "spawned",
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

    # Missing cap -> default 5; missing holds -> empty; order-preserving
    # dedup (first occurrence wins); unparseable updated_ts -> None.
    q = parse_infra_drain_queue('{"ripe_oldest_first": [5, 3, 5]}')
    assert q == {"ids": [5, 3], "cap": 5, "holds": {}, "updated_ts": None}
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
        asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: calls.append(i) or "spawned"
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
    assert markers == []  # a dry-run dispatch returns "failed" -> no marker
    assert not (isolated_registry / "infra-drain-state.json").exists()
    # The dispatch helper itself honours dry_run before any subprocess
    # (tri-state as of #843: dry-run returns "failed" — nothing spawned).
    assert asw._dispatch_infra_drain(483, "slot 1/3", dry_run=True) == "failed"


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
        asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: calls.append(i) or "failed"
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
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    # #1247: with _daemon_reachable forced True, the unstubbed sweep passes
    # would dispatch REAL sessions from this unit test. Called BEFORE the
    # recorder loop so the loop's own recorders (gate_push_pass / gc_pass
    # overlap the helper's stub set) win and keep recording.
    _stub_fleet_mutating_passes(asw, monkeypatch)
    for pass_name in (
        "vm_disk_pass",
        "triage_observer_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "stale_blocked_flag_pass",
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
    assert ok == "failed"
    out = capsys.readouterr().out
    assert "lost race to concurrent dispatcher" in out and "appeared" in out
    # (a') APPEARED relative to an explicit snapshot that lacked the file.
    ok = asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={})
    assert ok == "failed"
    assert "appeared" in capsys.readouterr().out
    # (b) CHANGED: the decision saw OTHER bytes (e.g. a concurrent PM spawn
    # overwrote the stale entry between decide and dispatch) -> abort.
    ok = asw._dispatch_infra_drain(
        42, "slot 1/3", dry_run=False, reg_snapshot={"issue-42.json": b'{"old": true}'}
    )
    assert ok == "failed"
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
    assert ok == "spawned"
    assert len(spawned) == 1 and "--issue" in spawned[0]
    assert "INFRA-DRAIN DISPATCHED issue #42" in capsys.readouterr().out


# ── #1059 session-dispatch stagger (chokepoint wiring) ────────────────────────
# The stagger paces REAL spawns >= EPM_SESSION_DISPATCH_STAGGER_S (60s) apart
# via spawn_session's last-session-dispatch stamp; the module-level autouse
# _no_real_stagger_sleep fixture (top of file) replaces the _stagger_sleep
# seam with a recorder, so these tests assert the requested delay without
# ever sleeping.


def _stagger_fake_spawn(monkeypatch, *, stdout="spawned sid-new\n", rc=0):
    """subprocess.run stub for direct ``_dispatch_infra_drain`` calls; records
    each argv and returns a canned result (rc / stdout configurable so tests
    can produce the suppressed-sentinel and failed shapes)."""
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr="boom" if rc else "")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    return calls


def test_dispatch_infra_drain_staggers_second_spawn(
    isolated_registry, monkeypatch, _no_real_stagger_sleep, capsys
):
    # A stamp 5s old under the default 60s window -> one sleep of ~55s, then
    # a normal "spawned" result that UPDATES the stamp to this dispatch.
    import json
    import time as _time

    import autonomous_session_watch as asw

    calls = _stagger_fake_spawn(monkeypatch)
    spawn_session.record_session_dispatch(99, "test-prior", now=_time.time() - 5.0)
    result = asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={})
    assert result == "spawned"
    assert len(calls) == 1  # the spawn still ran (sleep-then-proceed, never skip)
    assert len(_no_real_stagger_sleep) == 1
    assert 54.0 <= _no_real_stagger_sleep[0] <= 56.0
    assert "INFRA-DRAIN STAGGER issue #42" in capsys.readouterr().out
    entry = json.loads((isolated_registry / "last-session-dispatch.json").read_text())
    assert entry["issue"] == 42 and entry["holder"] == "watcher-infra-dispatch"


def test_dispatch_infra_drain_two_consecutive_spawns_full_window(
    isolated_registry, monkeypatch, _no_real_stagger_sleep
):
    # End-to-end through the REAL helpers (record -> age -> delay): with no
    # prior stamp the first spawn never sleeps and records; the second,
    # issued immediately after, sleeps out (approximately) the FULL window.
    import autonomous_session_watch as asw

    _stagger_fake_spawn(monkeypatch)
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "spawned"
    assert _no_real_stagger_sleep == []  # no prior dispatch -> no sleep
    assert asw._dispatch_infra_drain(43, "slot 2/3", dry_run=False, reg_snapshot={}) == "spawned"
    assert len(_no_real_stagger_sleep) == 1
    assert 59.0 <= _no_real_stagger_sleep[0] <= 60.0


def test_dispatch_infra_drain_no_stagger_when_disabled(
    isolated_registry, monkeypatch, _no_real_stagger_sleep
):
    # EPM_SESSION_DISPATCH_STAGGER_S=0 is the kill switch: a fresh stamp
    # produces zero sleep and the spawn proceeds.
    import time as _time

    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "0")
    calls = _stagger_fake_spawn(monkeypatch)
    spawn_session.record_session_dispatch(99, "test-prior", now=_time.time())
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "spawned"
    assert _no_real_stagger_sleep == []
    assert len(calls) == 1


def test_dispatch_infra_drain_dry_run_never_sleeps_never_records(
    isolated_registry, monkeypatch, _no_real_stagger_sleep
):
    # Dry-run returns BEFORE the stagger block: zero sleeps, zero subprocess
    # calls, and the stamp is byte-untouched (dry-run never records either).
    import time as _time

    import autonomous_session_watch as asw

    spawn_session.record_session_dispatch(99, "test-prior", now=_time.time())
    stamp = isolated_registry / "last-session-dispatch.json"
    before = stamp.read_bytes()
    monkeypatch.setattr(
        asw.subprocess, "run", lambda *a, **k: pytest.fail("subprocess ran on dry-run")
    )
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=True, reg_snapshot={}) == "failed"
    assert _no_real_stagger_sleep == []
    assert stamp.read_bytes() == before


def test_dispatch_infra_drain_suppressed_or_failed_does_not_record(
    isolated_registry, monkeypatch, _no_real_stagger_sleep
):
    # Only a REAL "spawned" outcome records the stamp: a rc-0 suppressed
    # no-op (lease held) and a rc!=0 failed spawn both leave no stamp, so a
    # no-op can never defer real work at the next dispatcher.
    import autonomous_session_watch as asw

    stamp = isolated_registry / "last-session-dispatch.json"
    _stagger_fake_spawn(monkeypatch, stdout="DISPATCH-LEASE HELD issue #42: in flight\n")
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "suppressed"
    assert not stamp.exists()
    _stagger_fake_spawn(monkeypatch, rc=1)
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "failed"
    assert not stamp.exists()


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
    monkeypatch.setattr(asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: "spawned")
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


# ─── capacity-retry pass (incident #642) ─────────────────────────────────────
#
# A wrong RE-DRIVE re-animates a session that should stay parked (worst case: a
# deliberate `failure_class: code` halt gets relaunched), so the scope is pinned
# exhaustively: ONLY a `blocked` task whose LATEST `epm:failure` is
# `failure_class: infra` + a TRANSIENT_CAPACITY_REASONS reason is ever touched.

# A realistic 2026 epoch (2026-06-20T00:00:00Z) — must be AFTER the 2026-06-16
# marker timestamps in the real-shape fixtures below so the backoff window is
# satisfied in the I/O-wrapper tests. The pure-decide matrix tests pass relative
# offsets, so the absolute value is immaterial there.
_CR_NOW = 1781913600.0


@pytest.mark.parametrize(
    "status",
    sorted(ACTIVE | PARK | TERMINAL | {None}, key=lambda s: (s is None, s)),
)
def test_capacity_retry_only_blocked_status_redrives(status):
    # Every NON-blocked status -> skip, even when retriable + out of backoff +
    # budget available. A deliberate halt (or any active/parked/terminal task)
    # is never re-driven by this pass.
    if status == "blocked":
        return  # covered by the matrix tests below
    assert (
        decide_capacity_retry(
            status,
            True,
            _CR_NOW - 99999,
            None,
            0,
            _CR_NOW,
            backoff_s=CAPACITY_RETRY_BACKOFF_S_DEFAULT,
            max_per_day=CAPACITY_RETRY_MAX_PER_DAY_DEFAULT,
        )
        == "skip"
    )


def test_capacity_retry_non_retriable_block_is_skipped():
    # A `blocked` task that is NOT a transient-capacity block stays parked.
    assert decide_capacity_retry("blocked", False, _CR_NOW - 99999, None, 0, _CR_NOW) == "skip"


def test_capacity_retry_redrive_when_clear():
    # blocked + retriable + block old + no prior attempt + budget -> redrive.
    assert decide_capacity_retry("blocked", True, _CR_NOW - 7200, None, 0, _CR_NOW) == "redrive"


def test_capacity_retry_backoff_from_block_ts():
    # Inside the backoff window measured from the block timestamp -> skip.
    assert decide_capacity_retry("blocked", True, _CR_NOW - 100, None, 0, _CR_NOW) == "skip"


def test_capacity_retry_backoff_from_last_attempt():
    # Block is old, but a recent attempt holds the backoff -> skip (no tight loop).
    assert (
        decide_capacity_retry("blocked", True, _CR_NOW - 99999, _CR_NOW - 100, 1, _CR_NOW) == "skip"
    )


def test_capacity_retry_daily_cap_exhausted():
    cap = CAPACITY_RETRY_MAX_PER_DAY_DEFAULT
    # At/over the cap, out of backoff -> exhausted (alert, no respawn).
    assert (
        decide_capacity_retry("blocked", True, _CR_NOW - 99999, _CR_NOW - 99999, cap, _CR_NOW)
        == "exhausted"
    )
    # One under the cap -> still redrives (boundary).
    assert (
        decide_capacity_retry("blocked", True, _CR_NOW - 99999, _CR_NOW - 99999, cap - 1, _CR_NOW)
        == "redrive"
    )


def test_capacity_retry_garbled_block_ts_does_not_freeze():
    # Unparseable block ts (None) must not permanently block recovery: the
    # last-attempt backoff still binds, and a first-ever (both None) redrives.
    assert decide_capacity_retry("blocked", True, None, _CR_NOW - 99999, 0, _CR_NOW) == "redrive"
    assert decide_capacity_retry("blocked", True, None, None, 0, _CR_NOW) == "redrive"


# ── _parse_failure_fields / _is_transient_capacity_block against REAL shapes ──

# Verbatim-shaped notes from #642's events.jsonl (the originating incident).
_NOTE_CAPACITY = (
    "failure_class: infra\nreason: no_compute_available\n"
    "detail: every auto lane failed or was unavailable (order: gcp -> nibi ...)"
)
_NOTE_CODEX_INFRA = (
    "Codex CR-critic R1 no-show: codex_task.py exhausted 1 transient retry ... "
    "failure_class: infra reason: codex-companion-probe-error. Proceeding "
    "Claude-only this round per skill fallback."
)
_NOTE_CODE = "failure_class: code\n\nvLLM ZeroDivisionError on every stage-A cmft worker ..."


def test_parse_failure_fields_real_shapes():
    import autonomous_session_watch as asw

    assert asw._parse_failure_fields(_NOTE_CAPACITY) == ("infra", "no_compute_available")
    # Inline (one-line) form, with a trailing sentence after the reason value.
    assert asw._parse_failure_fields(_NOTE_CODEX_INFRA) == (
        "infra",
        "codex-companion-probe-error",
    )
    assert asw._parse_failure_fields(_NOTE_CODE) == ("code", None)
    assert asw._parse_failure_fields(None) == (None, None)
    assert asw._parse_failure_fields("") == (None, None)


def _fail_ev(ts, note):
    return {"kind": "epm:failure", "ts": ts, "note": note}


def test_is_transient_capacity_block_latest_wins_and_allowlist():
    import autonomous_session_watch as asw

    cap = _fail_ev("2026-06-16T11:37:00Z", _NOTE_CAPACITY)
    codex = _fail_ev("2026-06-16T10:00:00Z", _NOTE_CODEX_INFRA)
    code = _fail_ev("2026-06-15T09:57:42Z", _NOTE_CODE)

    # Capacity failure is the LATEST -> retriable.
    retriable, reason, ts = asw._is_transient_capacity_block([code, codex, cap])
    assert retriable is True and reason == "no_compute_available" and ts is not None

    # A NON-capacity infra failure (codex probe) as the latest -> NOT retriable.
    retriable, reason, _ = asw._is_transient_capacity_block([cap, codex])
    assert retriable is False and reason == "codex-companion-probe-error"

    # A code failure as the latest -> NOT retriable.
    retriable, _, _ = asw._is_transient_capacity_block([cap, code])
    assert retriable is False

    # No failure marker at all -> NOT retriable.
    assert asw._is_transient_capacity_block(
        [{"kind": "epm:progress", "ts": "2026-06-16T00:00:00Z"}]
    ) == (False, None, None)


def test_no_compute_available_is_in_the_allowlist():
    # The one demonstrated transient-capacity reason; the allowlist stays tight.
    assert "no_compute_available" in TRANSIENT_CAPACITY_REASONS
    assert "codex-companion-probe-error" not in TRANSIENT_CAPACITY_REASONS


def test_capacity_retry_does_not_redrive_cpu_exhausted_reason():
    """#677: a `cpu_exhausted_no_runpod_lane` block is NOT a transient-capacity
    block — the watcher's capacity-retry pass must NOT hot-retry a structurally
    CPU-unservable run (no lane will ever free up to make RunPod accept a CPU
    intent).

    GREEN today purely because `cpu_exhausted_no_runpod_lane` is NOT in
    TRANSIENT_CAPACITY_REASONS — a REGRESSION GUARD pinning the contract (a
    future careless widening of the allowlist to include the CPU reason turns it
    RED).
    """
    import autonomous_session_watch as asw

    note = "failure_class: infra\nreason: cpu_exhausted_no_runpod_lane"
    ev = _fail_ev("2026-06-26T00:00:00Z", note)
    retriable, reason, _block_ts = asw._is_transient_capacity_block([ev])
    assert retriable is False
    assert reason == "cpu_exhausted_no_runpod_lane"
    # And the downstream decision is "skip" (never "redrive") even out of backoff
    # with the day-cap unspent — a non-retriable block always parks.
    assert decide_capacity_retry("blocked", retriable, _CR_NOW - 99999, None, 0, _CR_NOW) == "skip"


def test_capacity_retry_DOES_redrive_no_compute_available():
    """#677 control: the genuine transient-capacity reason IS still retriable,
    so the new CPU exclusion did not over-narrow the allowlist."""
    import autonomous_session_watch as asw

    ev = _fail_ev("2026-06-26T00:00:00Z", _NOTE_CAPACITY)
    retriable, reason, _block_ts = asw._is_transient_capacity_block([ev])
    assert retriable is True
    assert reason == "no_compute_available"


def test_cpu_fallback_infeasible_block_is_not_transient_capacity():
    """#1010: a `cpu_fallback_infeasible_for_plan` block (the RunPod
    CPU-fallback footprint-feasibility refusal, incident #958) is NOT a
    transient-capacity block — the RunPod instance can never grow to fit the
    plan, so the watcher's capacity-retry pass must never hot-retry it. The
    #677 mirror: GREEN purely because the reason is NOT in
    TRANSIENT_CAPACITY_REASONS (a future careless widening turns it RED)."""
    import autonomous_session_watch as asw

    note = "failure_class: infra\nreason: cpu_fallback_infeasible_for_plan"
    ev = _fail_ev("2026-07-04T00:00:00Z", note)
    retriable, reason, _block_ts = asw._is_transient_capacity_block([ev])
    assert retriable is False
    assert reason == "cpu_fallback_infeasible_for_plan"
    # Downstream: always "skip", even out of backoff with the day-cap unspent.
    assert decide_capacity_retry("blocked", retriable, _CR_NOW - 99999, None, 0, _CR_NOW) == "skip"


# ── I/O-wrapper scoping: only transient-infra blocks re-driven, halts untouched ──


def _patch_pass(monkeypatch, asw, blocked_ids, events_by_issue):
    monkeypatch.setattr(asw, "_blocked_issue_ids", lambda: list(blocked_ids))
    monkeypatch.setattr(asw, "_task_events", lambda i: events_by_issue.get(i, []))
    spawned = []

    def fake_run(cmd, *a, **k):
        spawned.append(cmd)

        class R:
            returncode = 0
            stdout = "session ok"
            stderr = ""

        return R()

    monkeypatch.setattr(asw.subprocess, "run", fake_run)
    posted = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, label)),
    )
    return spawned, posted


def test_capacity_retry_pass_redrives_only_transient_infra(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    cap = _fail_ev("2026-06-16T11:37:00Z", _NOTE_CAPACITY)  # #642 -> retriable
    code = _fail_ev("2026-06-16T11:00:00Z", _NOTE_CODE)  # deliberate code halt
    codex = _fail_ev("2026-06-16T11:00:00Z", _NOTE_CODEX_INFRA)  # non-capacity infra
    events = {642: [cap], 700: [code], 701: [codex]}
    spawned, posted = _patch_pass(monkeypatch, asw, [642, 700, 701], events)

    # now is far past the block ts so backoff is satisfied for #642.
    asw.capacity_retry_pass(dry_run=False, now=_CR_NOW, daemon_reachable=True)

    # Exactly ONE spawn — the transient-infra block. The code + non-capacity-
    # infra halts are untouched.
    assert len(spawned) == 1, spawned
    assert "642" in spawned[0]
    assert [iss for iss, _label in posted] == [642]


def test_capacity_retry_pass_daemon_gated(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    cap = _fail_ev("2026-06-16T11:37:00Z", _NOTE_CAPACITY)
    spawned, _posted = _patch_pass(monkeypatch, asw, [642], {642: [cap]})
    # Daemon down -> no spawn (spawn POSTs to the daemon RPC).
    asw.capacity_retry_pass(dry_run=False, now=_CR_NOW, daemon_reachable=False)
    assert spawned == []


def test_capacity_retry_pass_dry_run_no_mutation(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    cap = _fail_ev("2026-06-16T11:37:00Z", _NOTE_CAPACITY)
    monkeypatch.setattr(asw, "_blocked_issue_ids", lambda: [642])
    monkeypatch.setattr(asw, "_task_events", lambda i: [cap] if i == 642 else [])
    monkeypatch.setattr(
        asw.subprocess, "run", lambda *a, **k: pytest.fail("subprocess.run in dry-run")
    )
    asw.capacity_retry_pass(dry_run=True, now=_CR_NOW, daemon_reachable=True)
    # No state file written under dry-run.
    assert not (isolated_registry / "capacity-retry-642.json").exists()


def test_capacity_retry_pass_kill_switch(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_DISABLE_CAPACITY_RETRY", "1")
    monkeypatch.setattr(
        asw, "_blocked_issue_ids", lambda: pytest.fail("scanned despite kill switch")
    )
    asw.capacity_retry_pass(dry_run=False, now=_CR_NOW, daemon_reachable=True)


def test_capacity_retry_pass_daily_cap_then_exhausted_alert(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    cap = _fail_ev("2026-06-16T11:37:00Z", _NOTE_CAPACITY)
    spawned, posted = _patch_pass(monkeypatch, asw, [642], {642: [cap]})
    day_key = __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime(_CR_NOW))
    # Pre-seed state at the daily cap, last attempt long ago (backoff clear).
    (isolated_registry / "capacity-retry-642.json").write_text(
        __import__("json").dumps(
            {
                "retry_day": day_key,
                "retries_today": CAPACITY_RETRY_MAX_PER_DAY_DEFAULT,
                "last_attempt_ts": _CR_NOW - 99999,
                "alerted_day": None,
            }
        )
    )
    asw.capacity_retry_pass(dry_run=False, now=_CR_NOW, daemon_reachable=True)
    # Cap spent -> no respawn, one exhausted alert.
    assert spawned == []
    assert posted == [(642, "capacity-retry-exhausted")]


# ─── stale-blocked flag pass (task #1021, incident #742) ─────────────────────
#
# FLAG-ONLY: a wrong FLAG costs one digest line; a wrong FLIP would race the
# orchestrator's own-relaunch reconcile rule, so the pass NEVER mutates status
# (pinned two-pronged below, per the triage-observer non-gating precedent).

_SB_T0 = 1782000000.0  # arbitrary 2026 epoch anchor for the pure-predicate tests


def test_decide_stale_blocked_flag_fires_on_fresh_run_after_block():
    # The #742 shape: block < launch <= progress, progress 10 min old -> True.
    from autonomous_session_watch import decide_stale_blocked_flag

    assert (
        decide_stale_blocked_flag(
            "blocked",
            run_launched_ts=_SB_T0 + 100,
            blocked_since_ts=_SB_T0,
            progress_ts=_SB_T0 + 200,
            now=_SB_T0 + 200 + 600,
        )
        is True
    )


@pytest.mark.parametrize(
    "status", ["running", "followups_running", "on_hold", "awaiting_promotion", None]
)
def test_decide_stale_blocked_flag_skips_non_blocked_status(status):
    # Constraint interaction pinned: `followups_running` (the same-issue
    # follow-up loop's holding status) never flags; nor does any other status.
    from autonomous_session_watch import decide_stale_blocked_flag

    assert (
        decide_stale_blocked_flag(
            status,
            run_launched_ts=_SB_T0 + 100,
            blocked_since_ts=_SB_T0,
            progress_ts=_SB_T0 + 200,
            now=_SB_T0 + 800,
        )
        is False
    )


def test_decide_stale_blocked_flag_skips_launch_before_block():
    # The normal fail-then-block order (launch OLDER than the block) -> False;
    # a deliberately-blocked task with an old launch stays quiet.
    from autonomous_session_watch import decide_stale_blocked_flag

    assert (
        decide_stale_blocked_flag(
            "blocked",
            run_launched_ts=_SB_T0,
            blocked_since_ts=_SB_T0 + 100,
            progress_ts=_SB_T0 + 200,
            now=_SB_T0 + 800,
        )
        is False
    )


@pytest.mark.parametrize("missing", ["run_launched_ts", "blocked_since_ts", "progress_ts"])
def test_decide_stale_blocked_flag_skips_missing_signals(missing):
    # EVERY missing signal fails toward silence (e.g. a hand `git mv` block
    # that skipped `epm:status-changed` yields blocked_since_ts=None -> skip).
    from autonomous_session_watch import decide_stale_blocked_flag

    kwargs = {
        "run_launched_ts": _SB_T0 + 100,
        "blocked_since_ts": _SB_T0,
        "progress_ts": _SB_T0 + 200,
    }
    kwargs[missing] = None
    assert decide_stale_blocked_flag("blocked", now=_SB_T0 + 800, **kwargs) is False


def test_decide_stale_blocked_flag_skips_stale_progress():
    # Post-launch progress OLDER than the freshness window -> False (the run
    # may have died since; under-flagging is the safe failure direction).
    from autonomous_session_watch import decide_stale_blocked_flag

    assert (
        decide_stale_blocked_flag(
            "blocked",
            run_launched_ts=_SB_T0 + 100,
            blocked_since_ts=_SB_T0,
            progress_ts=_SB_T0 + 200,
            now=_SB_T0 + 200 + 7201,
            fresh_window_s=7200,
        )
        is False
    )


def test_decide_stale_blocked_flag_skips_pre_launch_progress():
    # v2 conjunct (progress_ts >= run_launched_ts): the newest "progress" is
    # the block-transition epm:status-changed marker itself (a _PROGRESS_KINDS
    # member) and the launch has no post-launch tick yet -> False. Pins the
    # vacuity fix: pre-launch progress never satisfies the liveness leg.
    from autonomous_session_watch import decide_stale_blocked_flag

    assert (
        decide_stale_blocked_flag(
            "blocked",
            run_launched_ts=_SB_T0 + 300,
            blocked_since_ts=_SB_T0,
            progress_ts=_SB_T0,  # == the block-transition marker's ts
            now=_SB_T0 + 400,
        )
        is False
    )


def _sb_events(*, block_iso, launch_iso, progress_iso=None, reblock_iso=None):
    """Minimal event-sequence fixture for the pass-level replays."""
    events = [
        {"kind": "epm:status-changed", "ts": block_iso, "note": "running -> blocked"},
        {"kind": "epm:run-launched", "ts": launch_iso, "note": "pid=123 log_abs=/w/x.log"},
    ]
    if progress_iso:
        events.append(
            {"kind": "epm:progress", "ts": progress_iso, "note": "[poll-tick:bg] launch alive"}
        )
    if reblock_iso:
        events.append(
            {"kind": "epm:status-changed", "ts": reblock_iso, "note": "running -> blocked (again)"}
        )
    return events


def _patch_sb_pass(monkeypatch, asw, blocked_ids, events_by_issue):
    """Isolate the pass's I/O seams: blocked-id scan, events fetch, marker
    post, Telegram push. State + sidecar writes go to isolated_registry."""
    monkeypatch.setattr(asw, "_blocked_issue_ids", lambda: list(blocked_ids))
    monkeypatch.setattr(asw, "_task_events", lambda i: events_by_issue.get(i, []))
    posted = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, *, label: posted.append((issue, label, note)),
    )
    pushed = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushed.append(msg) or True)
    return posted, pushed


def test_stale_blocked_pass_flags_once_per_launch_ts(isolated_registry, monkeypatch):
    # First tick: marker + push + sidecar row + state file. Second tick with
    # the same launch ts: nothing. A NEWER launch ts (fresh episode): re-alerts.
    import json

    import autonomous_session_watch as asw

    events = _sb_events(
        block_iso="2026-07-01T00:00:00Z",
        launch_iso="2026-07-01T08:00:00Z",
        progress_iso="2026-07-01T09:00:00Z",
    )
    posted, pushed = _patch_sb_pass(monkeypatch, asw, [742], {742: events})
    now = asw._parse_event_ts("2026-07-01T09:30:00Z")

    asw.stale_blocked_flag_pass(dry_run=False, now=now)
    assert [(i, label) for i, label, _n in posted] == [(742, "stale-blocked-flag")]
    assert len(pushed) == 1
    state_path = isolated_registry / "stale-blocked-742.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["flagged_run_launched_ts"] == asw._parse_event_ts("2026-07-01T08:00:00Z")
    sidecar = isolated_registry / "stale-blocked-events.jsonl"
    assert sidecar.exists() and len(sidecar.read_text().splitlines()) == 1
    # The marker note carries the sentinel + the reconcile command, flag-only.
    note = posted[0][2]
    assert asw._STALE_BLOCKED_FLAG_NOTE_SENTINEL in note
    assert "set-status 742 running" in note
    assert "FLAG-ONLY" in note

    # Same launch episode on the next tick: deduped, nothing new.
    asw.stale_blocked_flag_pass(dry_run=False, now=now + 600)
    assert len(posted) == 1 and len(pushed) == 1

    # A NEWER launch (a fresh episode) re-alerts.
    events2 = _sb_events(
        block_iso="2026-07-01T00:00:00Z",
        launch_iso="2026-07-02T08:00:00Z",
        progress_iso="2026-07-02T09:00:00Z",
    )
    monkeypatch.setattr(asw, "_task_events", lambda i: events2)
    asw.stale_blocked_flag_pass(dry_run=False, now=asw._parse_event_ts("2026-07-02T09:30:00Z"))
    assert len(posted) == 2 and len(pushed) == 2


def test_stale_blocked_pass_reblock_after_launch_unflags(isolated_registry, monkeypatch):
    # Pass-level replay: block -> launch -> post-launch progress -> RE-BLOCK
    # (newer epm:status-changed). The wiring extracts the LATEST
    # status-changed as "the transition into blocked", so launch < re-block ->
    # no flag (pins the latest-status-changed extraction, not just the
    # predicate).
    import autonomous_session_watch as asw

    events = _sb_events(
        block_iso="2026-07-01T00:00:00Z",
        launch_iso="2026-07-01T08:00:00Z",
        progress_iso="2026-07-01T09:00:00Z",
        reblock_iso="2026-07-01T10:00:00Z",
    )
    posted, pushed = _patch_sb_pass(monkeypatch, asw, [742], {742: events})
    asw.stale_blocked_flag_pass(dry_run=False, now=asw._parse_event_ts("2026-07-01T10:30:00Z"))
    assert posted == [] and pushed == []
    assert not (isolated_registry / "stale-blocked-742.json").exists()


def test_stale_blocked_pass_never_mutates_status(isolated_registry, tmp_path, monkeypatch):
    # The flag-only HARD invariant, TWO-PRONGED (per
    # test_triage_observer_never_calls_in_process_mutators): (i) no recorded
    # subprocess argv contains `set-status`; (ii) the in-process mutators
    # task_workflow.set_status / post_event raise if touched — the pass must
    # complete without tripping them. Helper-mediated or in-process status
    # mutation must fail this test, not just direct subprocess calls.
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    events = _sb_events(
        block_iso="2026-07-01T00:00:00Z",
        launch_iso="2026-07-01T08:00:00Z",
        progress_iso="2026-07-01T09:00:00Z",
    )
    monkeypatch.setattr(asw, "_blocked_issue_ids", lambda: [742])
    monkeypatch.setattr(asw, "_task_events", lambda i: events)

    def _forbidden(*a, **kw):
        raise AssertionError("stale_blocked_flag_pass must never mutate task state in-process")

    monkeypatch.setattr(task_workflow, "set_status", _forbidden)
    monkeypatch.setattr(task_workflow, "post_event", _forbidden)

    # Route the Telegram push through the recorded subprocess seam too.
    push_script = tmp_path / "push.sh"
    push_script.write_text("#!/usr/bin/env bash\n")
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(push_script))

    argvs: list[list[str]] = []

    def _record_run(cmd, *a, **kw):
        argvs.append(list(cmd))
        return _subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(asw.subprocess, "run", _record_run)
    asw.stale_blocked_flag_pass(dry_run=False, now=asw._parse_event_ts("2026-07-01T09:30:00Z"))
    # The flag DID fire (post-marker argv present) but nothing mutates status.
    assert any("post-marker" in cmd for cmd in argvs)
    assert not any("set-status" in cmd for cmd in argvs)


def test_stale_blocked_pass_kill_switch(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_DISABLE_STALE_BLOCKED_FLAG", "1")
    monkeypatch.setattr(
        asw, "_blocked_issue_ids", lambda: pytest.fail("scanned despite kill switch")
    )
    asw.stale_blocked_flag_pass(dry_run=False, now=_SB_T0)


def test_stale_blocked_pass_dry_run_no_writes(isolated_registry, monkeypatch):
    # dry_run -> zero state/sidecar writes, zero marker/push subprocesses.
    import autonomous_session_watch as asw

    events = _sb_events(
        block_iso="2026-07-01T00:00:00Z",
        launch_iso="2026-07-01T08:00:00Z",
        progress_iso="2026-07-01T09:00:00Z",
    )
    monkeypatch.setattr(asw, "_blocked_issue_ids", lambda: [742])
    monkeypatch.setattr(asw, "_task_events", lambda i: events)
    monkeypatch.setattr(
        asw.subprocess, "run", lambda *a, **k: pytest.fail("subprocess.run in dry-run")
    )
    asw.stale_blocked_flag_pass(dry_run=True, now=asw._parse_event_ts("2026-07-01T09:30:00Z"))
    assert not (isolated_registry / "stale-blocked-742.json").exists()
    assert not (isolated_registry / "stale-blocked-events.jsonl").exists()


def test_stale_blocked_sentinel_in_watcher_note_sentinels():
    # The flag note rides epm:progress; membership keeps it from ever
    # resetting the _latest_progress_ts staleness clocks (the set's own
    # comment mandates this for every new watcher-posted marker). Inverse of
    # the long-phase-heartbeat NON-membership test.
    import autonomous_session_watch as asw

    assert asw._STALE_BLOCKED_FLAG_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_main_wires_stale_blocked_pass_order(isolated_registry, monkeypatch):
    # Must-Fix (methodology reconciler): main() runs capacity_retry_pass ->
    # stale_blocked_flag_pass -> session_reconcile_pass in that order — closes
    # the silently-inert-backstop hole (the #681
    # test_main_wires_data_disk_pass_call_site class) where every isolation
    # test passes but the normal cadence never runs the pass.
    import autonomous_session_watch as asw

    order: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    # #1247 r3: shared hermeticity helper FIRST — adds verdict_disagree_pass
    # (scans the LIVE tasks tree) + vm_ledger_reap_pass (mutates the live
    # ~/.task-workflow/vm-ledger.json), both absent from the loop below; the
    # capacity_retry / stale_blocked_flag / session_reconcile recorders are
    # patched AFTER the helper, so they win.
    _stub_fleet_mutating_passes(asw, monkeypatch)
    for name in (
        "vm_disk_pass",
        "data_disk_pass",
        "happy_patch_pass",
        "cpu_guard_pass",
        "program_orchestrator_pass",
        "triage_observer_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "proposed_infra_sweep_pass",
        "gate_push_pass",
        "stale_registration_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
        "gc_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(asw, "capacity_retry_pass", lambda *a, **kw: order.append("capacity_retry"))
    monkeypatch.setattr(
        asw, "stale_blocked_flag_pass", lambda *a, **kw: order.append("stale_blocked_flag")
    )
    monkeypatch.setattr(
        asw, "session_reconcile_pass", lambda *a, **kw: order.append("session_reconcile")
    )
    rc = asw.main([])
    assert rc == 0
    assert (
        order.index("capacity_retry")
        < order.index("stale_blocked_flag")
        < order.index("session_reconcile")
    )


def test_main_stale_blocked_only_flag(isolated_registry, monkeypatch):
    # --stale-blocked-only runs JUST the new pass and exits (mirrors
    # test_main_proposed_infra_sweep_only_flag).
    import autonomous_session_watch as asw

    calls: list[str] = []
    monkeypatch.setattr(asw, "stale_blocked_flag_pass", lambda *a, **kw: calls.append("flag"))
    monkeypatch.setattr(
        asw, "vm_disk_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    monkeypatch.setattr(
        asw, "capacity_retry_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    rc = asw.main(["--stale-blocked-only", "--dry-run"])
    assert rc == 0
    assert calls == ["flag"]


# --- program-orchestrator crash-recovery pass (#660 bash daemon) ---
#
# The recovery path only fires on a real daemon crash, so it must be unit-tested
# (not just reasoned about). The pass takes injectable paths + a fake runner so
# every branch is exercised WITHOUT touching the live daemon, its STOP sentinel,
# or its log.


class _FakeProc:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr


def _po_runner(*, pgrep_rc: int, newsession_rc: int = 0, newsession_stderr: str = ""):
    """Fake subprocess runner: records every cmd; canned returns for pgrep / tmux."""
    calls: list[list[str]] = []

    def runner(cmd, **_kwargs):
        calls.append(list(cmd))
        if cmd[:1] == ["pgrep"]:
            return _FakeProc(returncode=pgrep_rc)
        if cmd[:2] == ["tmux", "new-session"]:
            return _FakeProc(returncode=newsession_rc, stderr=newsession_stderr)
        return _FakeProc(returncode=0)  # tmux kill-session

    runner.calls = calls
    return runner


def _po_relaunched(runner) -> bool:
    return any(c[:2] == ["tmux", "new-session"] for c in runner.calls)


def _po_paths(tmp_path, *, script_exists=True, stop_exists=False, log_text=""):
    script = tmp_path / "run_program_orchestrator.sh"
    if script_exists:
        script.write_text("#!/usr/bin/env bash\n")
    stop = tmp_path / "program_orchestrator.STOP"
    if stop_exists:
        stop.write_text("")
    log = tmp_path / "program_orchestrator.log"
    log.write_text(log_text)
    return script, stop, log


_PO_INFLIGHT = "[2026-06-26T10:00:00Z]   [Phase 2 (#664) #664] status=running\n"


def test_program_orchestrator_alive_no_relaunch(tmp_path):
    script, stop, log = _po_paths(tmp_path, log_text=_PO_INFLIGHT)
    runner = _po_runner(pgrep_rc=0)  # alive
    program_orchestrator_pass(False, script=script, stop=stop, log=log, runner=runner, env={})
    assert not _po_relaunched(runner)


def test_program_orchestrator_down_stop_present_no_relaunch(tmp_path):
    script, stop, log = _po_paths(tmp_path, stop_exists=True, log_text=_PO_INFLIGHT)
    runner = _po_runner(pgrep_rc=1)  # down, but a STOP sentinel = deliberate halt
    program_orchestrator_pass(False, script=script, stop=stop, log=log, runner=runner, env={})
    assert not _po_relaunched(runner)


def test_program_orchestrator_down_complete_no_relaunch(tmp_path):
    script, stop, log = _po_paths(
        tmp_path, log_text="ALL PHASES reached awaiting_promotion. Program complete\n"
    )
    runner = _po_runner(pgrep_rc=1)  # down, but normal completion -> leave down
    program_orchestrator_pass(False, script=script, stop=stop, log=log, runner=runner, env={})
    assert not _po_relaunched(runner)


def test_program_orchestrator_down_halts_no_relaunch(tmp_path):
    script, stop, log = _po_paths(
        tmp_path, log_text="Program finished WITH HALTS: Phase3 rc=1 Phase4 rc=0\n"
    )
    runner = _po_runner(pgrep_rc=1)  # down, but surfaced halt -> leave down
    program_orchestrator_pass(False, script=script, stop=stop, log=log, runner=runner, env={})
    assert not _po_relaunched(runner)


def test_program_orchestrator_down_inflight_dry_run_no_relaunch(tmp_path):
    script, stop, log = _po_paths(tmp_path, log_text=_PO_INFLIGHT)
    runner = _po_runner(pgrep_rc=1)
    program_orchestrator_pass(True, script=script, stop=stop, log=log, runner=runner, env={})
    assert not _po_relaunched(runner)  # dry-run: would-relaunch only, never acts


def test_program_orchestrator_down_inflight_relaunches(tmp_path):
    script, stop, log = _po_paths(tmp_path, log_text=_PO_INFLIGHT)
    runner = _po_runner(pgrep_rc=1)  # down, no STOP, in flight -> the crash case
    program_orchestrator_pass(False, script=script, stop=stop, log=log, runner=runner, env={})
    assert _po_relaunched(runner)


def test_program_orchestrator_kill_switch_noop(tmp_path):
    script, stop, log = _po_paths(tmp_path, log_text=_PO_INFLIGHT)
    runner = _po_runner(pgrep_rc=1)
    program_orchestrator_pass(
        False,
        script=script,
        stop=stop,
        log=log,
        runner=runner,
        env={"EPM_DISABLE_PROGRAM_ORCHESTRATOR_RECOVERY": "1"},
    )
    assert runner.calls == []  # kill switch: never even probes


def test_program_orchestrator_missing_script_noop(tmp_path):
    script, stop, log = _po_paths(tmp_path, script_exists=False, log_text=_PO_INFLIGHT)
    runner = _po_runner(pgrep_rc=1)
    program_orchestrator_pass(False, script=script, stop=stop, log=log, runner=runner, env={})
    assert runner.calls == []  # no script -> bail before probing


# ═══ proposed-infra sweep (always-on backstop for orphaned ripe infra; #690) ══
#
# Pure-decision tests run against decide_proposed_infra_sweep with zero
# filesystem/subprocess; executor tests use isolated_registry + monkeypatch
# recorder stubs (mirroring the infra-drain test group's _stub_drain_executor),
# with the candidate set fed via _proposed_infra_candidates and the holds map
# via a real infra-drain-queue.json (the SAME file the drain reads).

_SWEEP_NOW = 1_800_000_000.0  # fixed epoch, well clear of any backoff window


def _decide_sweep(
    candidates,
    *,
    holds=None,
    predicate_statuses=None,
    statuses=None,
    kinds=None,
    registered=None,
    occupied=0,
    pending=0,
    attempts=None,
    now=_SWEEP_NOW,
    cap=INFRA_DRAIN_CAP_DEFAULT,
    backoff_s=PROPOSED_INFRA_SWEEP_BACKOFF_S_DEFAULT,
    max_attempts=PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS_DEFAULT,
):
    """decide_proposed_infra_sweep with eligible-by-default fixtures: every
    candidate is proposed/infra and un-held unless the test overrides the
    signal under test. Mirrors _decide_drain."""
    statuses = statuses if statuses is not None else {i: "proposed" for i in candidates}
    kinds = kinds if kinds is not None else {i: "infra" for i in candidates}
    return decide_proposed_infra_sweep(
        candidates,
        holds or {},
        predicate_statuses or {},
        statuses,
        kinds,
        registered or set(),
        occupied,
        pending,
        attempts or {},
        now,
        cap,
        backoff_s=backoff_s,
        max_attempts=max_attempts,
    )


def _stub_sweep_executor(monkeypatch, *, candidates, status_kind=None, occupancy=None, live=None):
    """Stub every task.py/daemon signal the sweep consumes EXCEPT the holds
    read (which the test seeds as a real infra-drain-queue.json) and return the
    (dispatched, markers) recorders. ``candidates`` feeds
    _proposed_infra_candidates; ``status_kind`` feeds _task_status_kind (for
    both the candidate signals AND any predicate-blocker status read);
    ``occupancy`` feeds _infra_drain_occupancy; ``live`` feeds
    _live_session_ids_or_none."""
    import autonomous_session_watch as asw

    sk = status_kind or {}
    monkeypatch.setattr(asw, "_proposed_infra_candidates", lambda: candidates)
    monkeypatch.setattr(asw, "_task_status_kind", lambda i: sk.get(i, (None, None)))
    # #843 M3: the sweep loop reads each dispatch-list candidate's events for
    # the marker-freshness guard; stub it hermetic (no real task.py subprocess,
    # no skip) — the M3-specific tests override this with seeded events.
    monkeypatch.setattr(asw, "_task_events", lambda i: [])
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: occupancy)
    monkeypatch.setattr(
        asw, "_live_session_ids_or_none", lambda: live if live is not None else set()
    )
    dispatched: list[int] = []
    monkeypatch.setattr(
        asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: dispatched.append(i) or "spawned"
    )
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry, *, label: markers.append((issue, note, label)),
    )
    return dispatched, markers


# ── pure decision matrix ──────────────────────────────────────────────────────


def test_sweep_orphan_dispatches():
    # An orphan (not in holds) on a healthy system dispatches.
    assert _decide_sweep([7]) == ([7], [])


def test_sweep_non_predicate_hold_skips_regardless():
    # ANY non-predicate hold reason -> SKIP regardless (the PM said hold for a
    # user-park reason; the sweep never overrides it).
    for reason in ("spend", "credentials", "outward-facing", "cap", "predicate"):
        dispatch, skipped = _decide_sweep([7], holds={7: reason})
        assert dispatch == []
        assert skipped == [(7, f"held: {reason}")]


def test_sweep_predicate_hold_gated_on_blocker_status():
    # A predicate hold is eligible iff its blocking issue is satisfied.
    holds = {7: "predicate-999-foo"}
    dispatch, skipped = _decide_sweep([7], holds=holds, predicate_statuses={999: "running"})
    assert dispatch == [] and skipped == [(7, "held: predicate-999")]
    # Blocker satisfied -> eligible.
    for sat in sorted(INFRA_DRAIN_PREDICATE_SATISFIED_STATUSES):
        assert _decide_sweep([7], holds=holds, predicate_statuses={999: sat}) == ([7], [])
    # Blocker status unreadable -> still held (fail toward keep-blocking).
    dispatch, skipped = _decide_sweep([7], holds=holds, predicate_statuses={999: None})
    assert dispatch == [] and skipped == [(7, "held: predicate-999")]


def test_sweep_registered_blocks_dispatch():
    dispatch, skipped = _decide_sweep([7], registered={7})
    assert dispatch == [] and skipped == [(7, "already-registered")]


@pytest.mark.parametrize("status", [*sorted(set(STATUSES) - {"proposed"}), None])
def test_sweep_non_proposed_skips(status):
    # A candidate that changed status between the query and the re-confirming
    # read is skipped (defense in depth — the query already filtered).
    dispatch, skipped = _decide_sweep([7], statuses={7: status})
    expected = "status-unreadable" if status is None else f"status-{status}"
    assert dispatch == [] and skipped == [(7, expected)]


@pytest.mark.parametrize(
    ("kind", "ok"),
    [("infra", True), ("batch", True), ("experiment", False), ("campaign", False), (None, False)],
)
def test_sweep_kind_guard(kind, ok):
    dispatch, skipped = _decide_sweep([7], kinds={7: kind})
    if ok:
        assert dispatch == [7] and skipped == []
    else:
        assert dispatch == [] and skipped == [(7, f"kind-{kind or 'unreadable'}")]


def test_sweep_cap_arithmetic():
    # free = max(0, cap - occupied - pending); oldest-first preserved. cap is
    # pinned to 3 here to exercise the clamp at a fixed value independent of the
    # production default (INFRA_DRAIN_CAP_DEFAULT).
    ids = [10, 20, 30, 40]
    dispatch, skipped = _decide_sweep(ids, occupied=0, cap=3)
    assert dispatch == [10, 20, 30] and skipped == [(40, "cap-full")]
    dispatch, skipped = _decide_sweep(ids, occupied=2, cap=3)
    assert dispatch == [10] and [r for _, r in skipped] == ["cap-full"] * 3
    assert _decide_sweep(ids, occupied=3, cap=3)[0] == []
    # occupied > cap clamps at zero free.
    assert _decide_sweep(ids, occupied=5, cap=3)[0] == []
    # pending consumes a slot too.
    dispatch, skipped = _decide_sweep([7], occupied=2, pending=1, cap=3)
    assert dispatch == [] and skipped == [(7, "cap-full")]


def test_sweep_backoff_and_attempt_cap():
    # The tight-loop guard: a repeatedly-failing spawn backs off, then parks at
    # the attempt cap (the sweep has no PM epoch, so the count simply binds).
    now = _SWEEP_NOW
    attempts = {7: {"attempts": 1, "last_attempt_ts": now - 60.0}}
    assert _decide_sweep([7], attempts=attempts) == ([], [(7, "backoff")])
    attempts = {7: {"attempts": 3, "last_attempt_ts": now - 7200.0}}
    assert _decide_sweep([7], attempts=attempts) == ([], [(7, "attempts-exhausted")])
    attempts = {7: {"attempts": 2, "last_attempt_ts": now - 7200.0}}
    assert _decide_sweep([7], attempts=attempts) == ([7], [])


def test_sweep_sentinel_registered():
    # A watcher-posted sweep dispatch marker must never reset the
    # orphan/stalled staleness clocks for the session it just spawned.
    import autonomous_session_watch as asw

    assert asw._PROPOSED_INFRA_SWEEP_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


# ── (b) sweep dispatches an orphaned proposed infra task ───────────────────────


def test_sweep_dispatches_orphaned_proposed_infra(isolated_registry, monkeypatch, capsys):
    # AC (b): a ripe orphan (proposed infra, no registration, no queue entry)
    # is dispatched exactly once with the sweep marker.
    import autonomous_session_watch as asw

    dispatched, markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra")},
        occupancy=[],
    )
    # No infra-drain-queue.json on disk -> empty holds -> orphan eligible.
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched == [684]
    assert len(markers) == 1 and markers[0][0] == 684
    assert markers[0][2] == "proposed-infra-sweep"
    assert asw._PROPOSED_INFRA_SWEEP_NOTE_SENTINEL in markers[0][1]
    out = capsys.readouterr().out
    assert "candidates=1 occupied=0(+0 pending) cap=5 dispatched=1 skipped=0" in out


# ── (c-watcher) no double-dispatch when a live session exists ──────────────────


def test_sweep_skips_task_with_live_session(isolated_registry, monkeypatch, capsys):
    # AC (c): a candidate with a live (non-stale) issue-<N>.json is filtered
    # out before dispatch (registered_nonstale).
    import json

    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    (isolated_registry / "issue-684.json").write_text(
        json.dumps({"issue": 684, "happy_session_id": "sid-live", "spawned_at": now - 60.0})
    )
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra")},
        occupancy=[],
        live={"sid-live"},  # the recorded session IS live -> non-stale -> blocks
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == []
    out = capsys.readouterr().out
    assert "already-registered" in out


# ── (d) shared concurrency cap respected — via the REAL pending mechanism (R5) ─


def test_sweep_cap_full_via_occupancy(isolated_registry, monkeypatch, capsys):
    # (d.i) Cap full from occupancy alone: 5 infra tasks at occupied statuses
    # -> 0 free -> 0 dispatched (cap = INFRA_DRAIN_CAP_DEFAULT = 5).
    import autonomous_session_watch as asw

    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra")},
        occupancy=[700, 701, 702, 703, 704],
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched == []
    out = capsys.readouterr().out
    assert "cap-full" in out
    assert "occupying=[700, 701, 702, 703, 704]" in out


def test_sweep_cap_full_counts_real_pending_registration(isolated_registry, monkeypatch, capsys):
    # (d.ii, R5) "1 pending" produced through the REAL registration path: write
    # an actual issue-<X>.json for a still-proposed drain-kind task so the real
    # _infra_drain_pending counts it; occupancy=4 -> free = 5 - 4 - 1 = 0 -> 0
    # dispatched (cap = INFRA_DRAIN_CAP_DEFAULT = 5). Exercises the real
    # pending-counting layer end-to-end rather than stubbing the count wholesale.
    import json

    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    # A NON-candidate proposed infra task #900 already has a (non-stale,
    # dead-session-but-young) registration -> counts as 1 pending.
    (isolated_registry / "issue-900.json").write_text(
        json.dumps({"issue": 900, "happy_session_id": "sid-pending", "spawned_at": now - 60.0})
    )
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        # 684 is the candidate; 900 is the pending registration the real
        # _infra_drain_signals/_infra_drain_pending must read + count.
        status_kind={684: ("proposed", "infra"), 900: ("proposed", "infra")},
        occupancy=[700, 701, 702, 703],
        live={"sid-pending"},  # young + live -> non-stale -> pins a pending slot
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == []  # free = 5 - 4 occupied - 1 real pending = 0
    out = capsys.readouterr().out
    assert "occupied=4(+1 pending)" in out
    assert "cap-full" in out


# ── (M2) on_hold excluded via the --status proposed argv assertion ─────────────


def test_sweep_candidate_query_is_exactly_status_proposed(isolated_registry, monkeypatch):
    # #690 M2: the candidate-construction query MUST be exactly
    # `task.py list-by-status --status proposed --json`. on_hold is a different
    # status FOLDER, so a query restricted to --status proposed can never
    # enumerate an on_hold task — the STRUCTURAL exclusion. A regression that
    # broadens the query (drops --status, scans tasks/, asks for a different
    # status) trips this exact-argv assertion. Companion: a real ripe row flows
    # through and dispatches, so the test fails on the boundary it guards, not
    # merely on an emptied candidate set.
    import json
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    sk = {684: ("proposed", "infra")}
    monkeypatch.setattr(asw, "_task_status_kind", lambda i: sk.get(i, (None, None)))
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_live_session_ids_or_none", lambda: set())
    dispatched: list[int] = []
    monkeypatch.setattr(
        asw, "_dispatch_infra_drain", lambda i, slot, dry, **kw: dispatched.append(i) or "spawned"
    )
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: None)
    # #843 M3: the dispatch loop's marker-freshness read is a task.py signal
    # like the others — stubbed so the ONLY real subprocess stays the
    # candidate query this test pins.
    monkeypatch.setattr(asw, "_task_events", lambda i: [])

    seen_argv: list[list[str]] = []

    def _fake_run(cmd, **kw):
        # Only the candidate-construction list-by-status call should reach a
        # real subprocess (every other task.py/daemon signal is stubbed above).
        seen_argv.append(list(cmd))
        assert cmd == [
            "uv",
            "run",
            "python",
            "scripts/task.py",
            "list-by-status",
            "--status",
            "proposed",
            "--json",
        ], f"candidate query argv drifted: {cmd}"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"id": 684, "kind": "infra", "status": "proposed"}]),
            stderr="",
        )

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    asw.proposed_infra_sweep_pass(dry_run=False, now=now, daemon_reachable=True)
    assert len(seen_argv) == 1  # exactly one list-by-status query
    assert dispatched == [684]  # the real ripe row flows through


# ── non-infra excluded at the candidate-query layer ────────────────────────────


def test_sweep_candidate_query_filters_non_infra_kinds(isolated_registry, monkeypatch):
    # A kind: experiment row in the proposed list is filtered out by
    # _proposed_infra_candidates (kind not in INFRA_DRAIN_KINDS) -> never a
    # candidate, zero dispatches.
    import json
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    def _fake_run(cmd, **kw):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {"id": 684, "kind": "experiment", "status": "proposed"},
                    {"id": 685, "kind": "infra", "status": "proposed"},
                    {"id": 686, "kind": "campaign", "status": "proposed"},
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    cands = asw._proposed_infra_candidates()
    assert cands == [685]  # only the infra row, experiment/campaign filtered


# ── needs-human excluded at the candidate-query layer (#706) ───────────────────


def test_sweep_candidate_query_skips_needs_human(isolated_registry, monkeypatch):
    # A proposed infra row tagged `needs-human` (a /daily route-3 held judgment
    # call, task #706) MUST NOT be an auto-dispatch candidate — it surfaces in
    # the PM `Needs you` block instead. The always-on watcher sweep would
    # otherwise auto-dispatch it the moment a slot frees, defeating the entire
    # purpose of routing /daily-held items to a human. The kind-only filter in
    # `_proposed_infra_candidates` currently lets it through; the tag-skip is
    # the single load-bearing invariant this test pins.
    import json
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    def _fake_run(cmd, **kw):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "id": 700,
                        "kind": "infra",
                        "status": "proposed",
                        "tags": ["needs-human", "daily-held"],
                    },
                    {
                        "id": 701,
                        "kind": "infra",
                        "status": "proposed",
                        "tags": ["daily-auto-filed"],
                    },
                    {"id": 702, "kind": "infra", "status": "proposed", "tags": []},
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    cands = asw._proposed_infra_candidates()
    assert cands == [701, 702]  # needs-human row #700 filtered out


def test_sweep_candidate_query_admits_row_without_tags_key(isolated_registry, monkeypatch):
    # Backward-compat (#706, Statistics Claude critic concern #2): a LEGACY
    # proposed infra row that predates the `tags` field — no `tags` key at all
    # — must STILL be admitted as a candidate. The tag-skip MUST use safe
    # access (`row.get("tags") or []`); a `row["tags"]` lookup would KeyError on
    # this row and crash the whole sweep. This row has NO needs-human tag (it
    # has no tags), so it is a normal candidate.
    import json
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    def _fake_run(cmd, **kw):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "id": 710,
                        "kind": "infra",
                        "status": "proposed",
                        "tags": ["needs-human"],
                    },
                    {
                        "id": 711,
                        "kind": "infra",
                        "status": "proposed",
                        "tags": ["daily-auto-filed"],
                    },
                    {"id": 712, "kind": "infra", "status": "proposed"},  # legacy: no tags key
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    cands = asw._proposed_infra_candidates()
    assert cands == [711, 712]  # needs-human #710 skipped; no-tags-key #712 admitted


# ── unmet predicate held / satisfied (queue-file gate, executor half) ──────────


def test_sweep_predicate_queue_gate_executor(isolated_registry, monkeypatch, capsys):
    # Seed infra-drain-queue.json with holds = {684: predicate-999-foo}; #999
    # running -> held; flip #999 to completed -> dispatched. Exercises the
    # _parse_predicate_hold reuse on the PM queue file's PARSED int-keyed holds
    # (NOT an on-task tag, NOT raw JSON string keys).
    import autonomous_session_watch as asw

    _write_drain_queue(isolated_registry, [], holds={"684": "predicate-999-foo"})
    # #999 running -> not satisfied -> held.
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra"), 999: ("running", None)},
        occupancy=[],
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched == []
    assert "held: predicate-999" in capsys.readouterr().out

    # #999 completed -> satisfied -> dispatched.
    dispatched2, _m2 = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra"), 999: ("completed", None)},
        occupancy=[],
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched2 == [684]


def test_sweep_non_predicate_user_hold_honored_executor(isolated_registry, monkeypatch, capsys):
    # A non-predicate user-park (spend) in the queue holds the candidate even
    # though it is otherwise ripe — the sweep never overrides a user-park.
    import autonomous_session_watch as asw

    _write_drain_queue(isolated_registry, [], holds={"684": "spend"})
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra")},
        occupancy=[],
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched == []
    assert "held: spend" in capsys.readouterr().out


# ── (R1) corrupt-or-None-parsing queue does NOT silently un-hold ───────────────


def test_sweep_missing_queue_dispatches_orphan(isolated_registry, monkeypatch, capsys):
    # (R1.i) Genuinely missing queue file with an otherwise-ripe ORPHAN ->
    # treated as no holds -> dispatched (the orphan path is never blocked by an
    # unreadable queue).
    import autonomous_session_watch as asw

    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra")},
        occupancy=[],
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched == [684]


def test_sweep_corrupt_queue_does_not_unhold_predicate_candidate(
    isolated_registry, monkeypatch, capsys
):
    # (R1.ii) A present-but-None-parsing (corrupt) queue file must NOT silently
    # un-hold a candidate a valid map had held. Concretely: the SAME candidate
    # #684 that test_sweep_predicate_queue_gate_executor holds under
    # predicate-999-foo (#999 still running) must NOT flip skipped->dispatched
    # just because the queue read produced no usable holds map THIS tick. A
    # corrupt read is treated as "no holds", so #684 becomes an un-held orphan
    # and WOULD dispatch — that is the documented fail-soft behavior; what R1
    # guards is that the corrupt read does not DROP a held entry from an
    # otherwise-valid map. We pin the predicate-held behavior under a VALID map
    # (above) and here pin that a corrupt map parses to None (so the gate has
    # no held entry to drop — it never sees one).
    import autonomous_session_watch as asw

    # `holds` is a LIST -> parse_infra_drain_queue returns None (corrupt).
    (isolated_registry / "infra-drain-queue.json").write_text(
        '{"ripe_oldest_first": [], "holds": [684]}'
    )
    assert asw._infra_drain_read_queue() is None  # corrupt -> None -> "no usable holds map"
    # The gate then sees holds={}; #684 is an un-held orphan and dispatches.
    # The KEY invariant: a corrupt read never carries a stale held entry that
    # could mask a NEW hold, and never drops a held entry from a valid map (the
    # valid-map predicate hold is pinned by the executor test above).
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch,
        candidates=[684],
        status_kind={684: ("proposed", "infra")},
        occupancy=[],
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert dispatched == [684]


# ── kill switch / daemon-down no-ops ───────────────────────────────────────────


def test_sweep_kill_switch(isolated_registry, monkeypatch, capsys):
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_DISABLE_PROPOSED_INFRA_SWEEP", "1")
    assert asw._proposed_infra_sweep_enabled() is False
    monkeypatch.setattr(
        asw,
        "_proposed_infra_candidates",
        lambda: pytest.fail("candidate scan despite kill switch"),
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=True)
    assert "disabled via EPM_DISABLE_PROPOSED_INFRA_SWEEP" in capsys.readouterr().out
    monkeypatch.setenv("EPM_DISABLE_PROPOSED_INFRA_SWEEP", "0")
    assert asw._proposed_infra_sweep_enabled() is True
    monkeypatch.delenv("EPM_DISABLE_PROPOSED_INFRA_SWEEP")
    assert asw._proposed_infra_sweep_enabled() is True


def test_sweep_daemon_down_noop(isolated_registry, monkeypatch, capsys):
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw,
        "_proposed_infra_candidates",
        lambda: pytest.fail("candidate scan despite daemon down"),
    )
    asw.proposed_infra_sweep_pass(dry_run=False, now=_SWEEP_NOW, daemon_reachable=False)
    assert "Happy daemon unreachable" in capsys.readouterr().out


# ── (R6) main() pass-order assertion ───────────────────────────────────────────


def test_main_runs_sweep_after_infra_drain(isolated_registry, monkeypatch):
    # #690 R6: the sweep MUST run AFTER infra_drain in main() — so the sweep's
    # pending count sees any ID the drain dispatched THIS tick (its fresh
    # registration), and the shared cap holds across both. A reorder would not
    # be caught by the in-isolation pass tests; this cheap order check pins it.
    import autonomous_session_watch as asw

    order: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    # #1247 r3: shared hermeticity helper FIRST — adds data_disk / happy_patch
    # / cpu_guard / verdict_disagree / vm_ledger_reap (all absent from the
    # loop below; live observer scans + a live ledger mutate); the infra_drain
    # + proposed_infra_sweep recorders are patched AFTER, so they win.
    _stub_fleet_mutating_passes(asw, monkeypatch)
    # Neutralize every other pass so main() runs cheaply and deterministically.
    for name in (
        "vm_disk_pass",
        "program_orchestrator_pass",
        "triage_observer_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "capacity_retry_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "gate_push_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
        "gc_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(asw, "infra_drain_pass", lambda *a, **kw: order.append("infra_drain"))
    monkeypatch.setattr(
        asw, "proposed_infra_sweep_pass", lambda *a, **kw: order.append("proposed_infra_sweep")
    )
    rc = asw.main([])
    assert rc == 0
    assert "infra_drain" in order and "proposed_infra_sweep" in order
    assert order.index("infra_drain") < order.index("proposed_infra_sweep")


def test_main_proposed_infra_sweep_only_flag(isolated_registry, monkeypatch):
    # --proposed-infra-sweep-only runs JUST the sweep pass and exits.
    import autonomous_session_watch as asw

    calls: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "proposed_infra_sweep_pass", lambda *a, **kw: calls.append("sweep"))
    monkeypatch.setattr(
        asw, "infra_drain_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    monkeypatch.setattr(
        asw, "vm_disk_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    rc = asw.main(["--proposed-infra-sweep-only"])
    assert rc == 0
    assert calls == ["sweep"]


# ─── #681 data-disk PERCENT thresholds (Must-Fix #2 — size-invariant) ────────
#
# These drive the PRODUCTION decision functions against a large total, NOT a
# percent monkeypatch seam — the v1→v2 coverage fix. The byte-floor /-path
# (decide_subfloor / decide_vm_disk) is UNCHANGED and stays correct for the
# 485 GB boot disk; the data disk uses percent thresholds so a future resize
# cannot push the fire point past the wedge.

_GIB = 2**30
_TIB = 2**40


def _used_pct(total_bytes: int, free_bytes: int) -> float:
    """Percent USED for a (total, free) — the statvfs-derived input the data-disk
    pass computes from `total` and `free`, exactly as production does."""
    return 100.0 * (total_bytes - free_bytes) / total_bytes


def test_data_disk_subfloor_fires_at_intended_fullness():
    # 512 GiB total: at 85-90% full the sub-floor FIRES; at 80% it does NOT.
    # Proves escalation PRECEDES the wedge (the v1 byte-floor bug would have
    # stayed quiescent until ~88-94% on this disk).
    from autonomous_session_watch import decide_subfloor_pct

    total = 512 * _GIB
    free_88 = int(total * 0.12)  # 88% used
    free_80 = int(total * 0.20)  # 80% used
    assert decide_subfloor_pct(_used_pct(total, free_88), None) is True  # fires
    assert decide_subfloor_pct(_used_pct(total, free_80), None) is False  # quiet


def test_data_disk_subfloor_realerts_only_on_climb():
    # An already-sub-floor episode re-fires only when usage CLIMBS by > the
    # re-alert fraction; a stable footprint does not re-fire every tick.
    from autonomous_session_watch import VM_DISK_SUBFLOOR_GROWTH_REALERT, decide_subfloor_pct

    total = 512 * _GIB
    used_86 = _used_pct(total, int(total * 0.14))  # 86%
    used_87 = _used_pct(total, int(total * 0.13))  # ~87%, ~1.2% relative climb
    used_99 = _used_pct(total, int(total * 0.01))  # 99%, ~15% relative climb
    # The re-alert is a > VM_DISK_SUBFLOOR_GROWTH_REALERT (default 0.10) RELATIVE
    # climb of used_pct since the last row. Bracket the threshold explicitly.
    assert (used_87 - used_86) / used_86 < VM_DISK_SUBFLOOR_GROWTH_REALERT
    assert (used_99 - used_86) / used_86 > VM_DISK_SUBFLOOR_GROWTH_REALERT
    # Stable / tiny climb since last row → no re-alert.
    assert decide_subfloor_pct(used_87, used_86) is False
    # A large climb → re-alert.
    assert decide_subfloor_pct(used_99, used_86) is True


def test_data_disk_alert_and_reclaim_fire_before_wedge():
    # The data-disk ALERT arm fires at 90% and the (escalate-only) CRITICAL arm
    # at 95% of a 512 GiB total; NO reclaim-tier action is even RETURNED (the
    # function returns only (level, do_alert) — there is no do_reclaim/do_audit).
    from autonomous_session_watch import decide_vm_disk_pct

    total = 512 * _GIB
    used_60 = _used_pct(total, int(total * 0.40))
    used_91 = _used_pct(total, int(total * 0.09))
    used_96 = _used_pct(total, int(total * 0.04))

    assert decide_vm_disk_pct(used_60, alerted=False) == ("ok", False)
    assert decide_vm_disk_pct(used_91, alerted=False) == ("low", True)
    assert decide_vm_disk_pct(used_96, alerted=False) == ("critical", True)
    # Already-alerted episode does not re-alert.
    assert decide_vm_disk_pct(used_96, alerted=True) == ("critical", False)
    # The return is a 2-tuple — there is structurally no reclaim/audit action on
    # the data disk (escalate-only).
    assert len(decide_vm_disk_pct(used_96, alerted=False)) == 2


def test_data_disk_thresholds_size_invariant():
    # The CANARY: repeat the 85-90%-fires / 80%-quiet assertions with total=2 TiB
    # (a future resize). The PERCENT basis must fire at the SAME fullness — the
    # mirrored-byte-floor bug would regress here (a 20 GiB free floor on a 2 TiB
    # disk is ~99% full, firing AFTER the wedge).
    from autonomous_session_watch import decide_subfloor_pct, decide_vm_disk_pct

    total = 2 * _TIB
    free_88 = int(total * 0.12)  # 88% used
    free_80 = int(total * 0.20)  # 80% used
    assert decide_subfloor_pct(_used_pct(total, free_88), None) is True
    assert decide_subfloor_pct(_used_pct(total, free_80), None) is False
    # And the alert arm fires at 90% / 95% identically on the 2 TiB disk.
    assert decide_vm_disk_pct(_used_pct(total, int(total * 0.09)), alerted=False) == ("low", True)
    assert decide_vm_disk_pct(_used_pct(total, int(total * 0.04)), alerted=False) == (
        "critical",
        True,
    )
    # Sanity vs the byte-floor /-path: 20 GiB free on a 2 TiB disk is ~99% used —
    # the mirrored byte floor (decide_subfloor at <60 GiB free) would only have
    # fired at the very brink. The percent floor already fired at 88%.
    twenty_gib_free_used = _used_pct(total, 20 * _GIB)
    assert twenty_gib_free_used > 98.0


def test_subfloor_attributes_worktree_data(tmp_path, monkeypatch):
    # The sub-floor attribution must name the WORKTREE-INTERNAL caches
    # (.claude/worktrees/issue-<N>/data/issue_<N>/{hf_dl,g*_dl}), not just
    # repo-root data/ — the per-issue caches the data disk actually holds live
    # in the worktree (#681 / #658 evidence).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)
    # A worktree-internal cache for issue 658 (where the live run writes).
    wt_cache = tmp_path / ".claude" / "worktrees" / "issue-658" / "data" / "issue_658" / "hf_dl"
    wt_cache.mkdir(parents=True)
    (wt_cache / "blob.bin").write_bytes(b"x" * 8192)
    # A repo-root data/ cache too, to prove BOTH roots are globbed.
    repo_cache = tmp_path / "data" / "issue_700" / "g1_dl"
    repo_cache.mkdir(parents=True)
    (repo_cache / "blob.bin").write_bytes(b"y" * 4096)

    roots = asw._issue_cache_glob_roots()
    # Both the repo-root data/ AND the worktree data/ are glob roots.
    assert (tmp_path / "data") in roots
    assert (tmp_path / ".claude" / "worktrees" / "issue-658" / "data") in roots

    top = asw._top_issue_cache_paths(top_n=5)
    named = {rel for rel, _ in top}
    # The worktree-internal cache is attributed (the bug was it was missed).
    assert any("worktrees/issue-658/data/issue_658/hf_dl" in rel for rel in named)
    # The repo-root cache is still attributed too.
    assert any("data/issue_700/g1_dl" in rel for rel in named)


def test_repquota_attribution_parses_per_project_rows(monkeypatch):
    # The data-disk PRIMARY attribution reads per-PROJECT usage via repquota -P
    # in one cheap call (project id == issue number); a non-zero rc / unparseable
    # output returns None so the caller falls back to the du-based path.
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    # repquota -Ocsv emits: Project,BlockStatus,FileStatus,BlockUsed(KiB),...
    csv = (
        "#0,ok,ok,512,0,0,0,ok,1,0,0\n"
        "#658,ok,ok,104857600,0,134217728,0,ok,10,0,0\n"  # 100 GiB used
        "#700,ok,ok,52428800,0,134217728,0,ok,5,0,0\n"  # 50 GiB used
    )

    def fake_run(cmd, *a, **k):
        assert cmd[:3] == ["repquota", "-Ocsv", "-P"]
        return _subprocess.CompletedProcess(cmd, 0, stdout=csv, stderr="")

    monkeypatch.setattr(asw.subprocess, "run", fake_run)
    rows = asw._top_issue_caches_by_project_quota("/mnt/eps-data", top_n=5)
    assert rows is not None
    # project 0 (the unbounded default) is excluded; sorted by usage desc.
    assert rows[0][0].startswith("issue-658")
    assert rows[0][1] == 104857600 * 1024
    assert rows[1][0].startswith("issue-700")
    assert all("issue-0" not in r[0] for r in rows)

    # A non-zero rc (repquota missing / no prjquota) → None (du fallback).
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda cmd, *a, **k: _subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no prjquota"),
    )
    assert asw._top_issue_caches_by_project_quota("/mnt/eps-data") is None


# ── Data-disk pass — PRODUCTION call site (#681 round-2 BLOCKER #1) ───────────
# The round-1 diff DEFINED the percent helpers but never DROVE them from a live
# watcher pass — plan §4 "Add a parallel data-disk check ... that the data-disk
# path drives" requires a production call site, not just unit-pinned helpers.
# These tests pin the wrapper data_disk_pass (driven from main(), the sibling
# of vm_disk_pass) AND the source-level fact that main() wires it.


def _stub_data_disk_io(asw, monkeypatch, *, mounted, used_pct, top=None):
    """Make data_disk_pass deterministic: control the mount probe, the
    statvfs-derived used_pct, and the attribution. Returns the list the
    sidecar-append closure records into.

    The mount gate is ``_is_mounted`` (st_dev vs parent), NOT ``Path.is_dir()``
    (#681 round-2 Major) — a plain directory passes ``is_dir()`` but is not a
    mount, so the gate had to become a real mount check. Stub ``_is_mounted``
    accordingly; ``test_data_disk_pass_production_gate_*`` below exercise the
    REAL ``_is_mounted`` against real plain/mount dirs (no stub)."""
    monkeypatch.setattr(asw, "_is_mounted", lambda dd_path: mounted)
    monkeypatch.setattr(asw, "_data_disk_used_pct", lambda dd_path: used_pct)
    # The production `_data_disk_top_caches` takes a `dry_run` kwarg (#681 r3 —
    # dry-run short-circuits the `du`/`repquota` attribution so a dry-run pass
    # shells out to nothing); the stub must accept it to stay signature-faithful.
    monkeypatch.setattr(asw, "_data_disk_top_caches", lambda dd_path, *, dry_run=False: top or [])
    recorded: list[dict] = []
    monkeypatch.setattr(
        asw, "_append_disk_guard_sidecar", lambda event, dry_run: recorded.append(event)
    )
    return recorded


def test_data_disk_pass_fires_subfloor_when_mounted_and_full(tmp_path, monkeypatch):
    # The PRODUCTION wrapper (the one main() calls) writes the data-disk sub-floor
    # sidecar row at 96% used when the mount is present. Drives the REAL pass, not
    # only the pure decide_* helpers (Codex's explicit round-1 miss).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)  # isolate dedup state
    recorded = _stub_data_disk_io(
        asw,
        monkeypatch,
        mounted=True,
        used_pct=96.0,
        top=[("issue-658 (project quota, /mnt/eps-data)", 100 * 2**30)],
    )

    wrote = asw.data_disk_pass(dry_run=False)

    assert wrote is True
    kinds = {r["kind"] for r in recorded}
    # Both the alert/critical arm (decide_vm_disk_pct -> critical at 96%) AND the
    # sub-floor arm (decide_subfloor_pct -> True at 96%) escalate.
    assert "vm-disk-data-critical" in kinds
    assert "vm-disk-data-subfloor" in kinds
    # Every data-disk row is tagged disk=data with the WORKTREE-internal cache
    # attribution carried through.
    assert all(r.get("disk") == "data" for r in recorded)
    sub = next(r for r in recorded if r["kind"] == "vm-disk-data-subfloor")
    assert sub["band"] == "sub-floor"
    assert any("issue-658" in c["path"] for c in sub["top_cache_paths"])


def test_data_disk_pass_is_clean_noop_pre_cutover(tmp_path, monkeypatch):
    # Pre-cutover (the mount does not exist) the data-disk pass is a CLEAN no-op:
    # no sidecar row, no state write, even when used_pct would otherwise escalate.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    recorded = _stub_data_disk_io(asw, monkeypatch, mounted=False, used_pct=99.0)

    wrote = asw.data_disk_pass(dry_run=False)

    assert wrote is False
    assert recorded == []  # nothing escalated
    assert not (tmp_path / "vm-disk-data.json").exists()  # no dedup state touched


def test_data_disk_pass_dry_run_writes_no_state(tmp_path, monkeypatch):
    # --dry-run decides + logs but mutates nothing (no dedup-state file written).
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    _stub_data_disk_io(asw, monkeypatch, mounted=True, used_pct=96.0)

    asw.data_disk_pass(dry_run=True)

    assert not (tmp_path / "vm-disk-data.json").exists()


def test_data_disk_pass_production_gate_noops_on_plain_dir(tmp_path, monkeypatch):
    # PRODUCTION-PROBE (#681 round-2 Major): point the data-disk path at a plain
    # (non-mount) dir on the root fs — an existing-but-unmounted /mnt/eps-data
    # (Phase-1 `mkdir -p` / `nofail`-boot state) — and drive the REAL _is_mounted
    # gate (no stub). data_disk_pass MUST no-op (return False, write NO sidecar
    # row), NEVER misreading /'s statvfs as the data disk's. The old
    # Path(dd_path).is_dir() gate (True for any dir) would have run the pass and
    # emitted a row mirroring /'s percent.
    import autonomous_session_watch as asw

    plain = tmp_path / "mnt-eps-data-not-mounted"
    plain.mkdir()  # exists, but NOT a mount (shares tmp_path's st_dev)
    monkeypatch.setenv("EPS_VM_DATA_DISK_PATH", str(plain))
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)

    recorded: list[dict] = []
    monkeypatch.setattr(
        asw, "_append_disk_guard_sidecar", lambda event, dry_run: recorded.append(event)
    )
    # Force a 99%-full read so ONLY the mount gate can suppress escalation: if the
    # plain dir were wrongly treated as mounted, this would escalate loudly.
    monkeypatch.setattr(asw, "_data_disk_used_pct", lambda dd_path: 99.0)

    wrote = asw.data_disk_pass(dry_run=False)

    assert wrote is False, "a plain (unmounted) data-disk dir must make the pass no-op"
    assert recorded == [], "no sidecar row may be written for an unmounted data disk"
    assert not (tmp_path / "vm-disk-data.json").exists(), "no dedup state on a no-op pass"


def test_data_disk_pass_production_gate_fires_on_real_mount(tmp_path, monkeypatch):
    # Counterpart: when _is_mounted reports a live mount, the REAL gate lets the
    # pass run (it escalates at 96%). Stub only _is_mounted -> True (we cannot
    # create a real mount without privilege); everything else is the production
    # path, proving the gate is not stuck always-off after the fix.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    recorded = _stub_data_disk_io(asw, monkeypatch, mounted=True, used_pct=96.0)

    wrote = asw.data_disk_pass(dry_run=False)

    assert wrote is True
    assert any(r["kind"] == "vm-disk-data-subfloor" for r in recorded)


def test_is_mounted_false_on_plain_dir(tmp_path):
    # Direct unit test of the shared helper: a plain dir on the root fs shares its
    # parent's st_dev → not a mount. Mirrors vm_disk_guard._is_mounted.
    import autonomous_session_watch as asw

    plain = tmp_path / "fake-data-disk"
    plain.mkdir()
    assert asw._is_mounted(str(plain)) is False
    assert asw._is_mounted(str(tmp_path / "missing")) is False  # fail-soft on missing


def test_main_wires_data_disk_pass_call_site():
    # The mechanizable round-2 check: main() must DRIVE data_disk_pass, not just
    # define the helpers. Codex: "fail if main() only calls vm_disk_pass". Pin the
    # production call site at the source level so a future refactor that drops the
    # call (regressing to helpers-without-callsite) fails loudly.
    import inspect

    import autonomous_session_watch as asw

    src = inspect.getsource(asw.main)
    assert "data_disk_pass(args.dry_run)" in src, (
        "main() must call data_disk_pass(args.dry_run) next to vm_disk_pass — "
        "the percent helpers must be DRIVEN by a production call site (#681 BLOCKER #1)"
    )


# ─── #720 short reap window for last-mapped-terminal sessions ────────────────
# An autonomous /issue --auto session goes UNMAPPED the instant its task hits a
# terminal status (the respawn pass deletes issue-<N>.json), dropping it into
# the generic 12h idle-unmapped bucket. The #720 fix drops a breadcrumb at that
# unmapping moment and applies a SHORT (30-min) reap window to the now-unmapped
# session — but ONLY after two lazy protected-class guards (no running managed
# pod for the issue; no live same-issue follow-up) both clear.


def _PROJECT_ROOT_for_720(monkeypatch):
    """Pin asw.PROJECT_ROOT to the synthetic root so the EPS-cwd check is
    cwd-independent (mirrors _patch_idle_io / _patch_zombie_io)."""
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "PROJECT_ROOT", Path(_Z_ROOT))


def _write_terminal_breadcrumb(reg_dir, sid, issue, terminal_status, recorded_at=0.0):
    import json

    (reg_dir / f"last-mapped-terminal-{sid}.json").write_text(
        json.dumps(
            {
                "happy_session_id": sid,
                "issue": issue,
                "terminal_status": terminal_status,
                "recorded_at": recorded_at,
            }
        )
    )


def test_short_window_worst_case_under_acceptance():
    # MF-2: pure-arithmetic bound, no clock, no fixtures. The body's "within
    # ~1h" claim must hold STRICTLY: with the 10-min cron + the 2-miss guard the
    # worst-case reap latency is LAST_MAPPED_TERMINAL_REAP_S + 2*600. 30 min +
    # 20 min = 50 min <= 60 min (10-min margin). A future change to the constant
    # or cron cadence that breaks the bound trips this test.
    import autonomous_session_watch as asw

    cadence_s = 600  # cron 3-59/10 (every 10 min)
    threshold = 2
    idle_reap_s = asw.LAST_MAPPED_TERMINAL_REAP_S
    worst_case_s = idle_reap_s + threshold * cadence_s
    acceptance_s = 3600  # strict "within ~1h"
    assert worst_case_s <= acceptance_s, (
        f"worst_case={worst_case_s}s exceeds acceptance={acceptance_s}s"
    )
    assert asw.LAST_MAPPED_TERMINAL_REAP_S == 30 * 60


def test_last_mapped_terminal_env_helper(monkeypatch):
    # EPM_LAST_MAPPED_TERMINAL_REAP_S: a positive override wins; garbled /
    # non-positive falls back to the 30-min default.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    assert asw._last_mapped_terminal_reap_s() == asw.LAST_MAPPED_TERMINAL_REAP_S
    monkeypatch.setenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", "120")
    assert asw._last_mapped_terminal_reap_s() == 120.0
    monkeypatch.setenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", "garbled")
    assert asw._last_mapped_terminal_reap_s() == asw.LAST_MAPPED_TERMINAL_REAP_S
    monkeypatch.setenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", "-5")
    assert asw._last_mapped_terminal_reap_s() == asw.LAST_MAPPED_TERMINAL_REAP_S
    monkeypatch.setenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", "0")
    assert asw._last_mapped_terminal_reap_s() == asw.LAST_MAPPED_TERMINAL_REAP_S


def test_record_last_mapped_terminal_writes_for_terminal_only(isolated_registry, monkeypatch):
    # _record_last_mapped_terminal writes the breadcrumb ONLY for a status in
    # TERMINAL; a PARK/ACTIVE status no-ops (it would widen scope on read).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    for status in sorted(asw.TERMINAL):
        sid = f"sid-{status}"
        asw._record_last_mapped_terminal(sid, 720, status, dry_run=False, now=123.0)
        assert (isolated_registry / f"last-mapped-terminal-{sid}.json").is_file()
        assert asw._last_mapped_terminal(sid) == (status, 720)
    # A non-terminal status writes nothing.
    for status in ("blocked", "running", "proposed"):
        sid = f"sid-{status}"
        asw._record_last_mapped_terminal(sid, 99, status, dry_run=False)
        assert not (isolated_registry / f"last-mapped-terminal-{sid}.json").exists()
        assert asw._last_mapped_terminal(sid) is None
    # dry_run never writes.
    asw._record_last_mapped_terminal("sid-dry", 5, "completed", dry_run=True)
    assert not (isolated_registry / "last-mapped-terminal-sid-dry.json").exists()


def test_last_mapped_terminal_read_roundtrip_and_rejections(isolated_registry):
    # Read back (status, issue); a blocked / garbled / missing / non-int-issue
    # breadcrumb reads None (the scope guard against a stale/widened value).
    import json

    import autonomous_session_watch as asw

    _write_terminal_breadcrumb(isolated_registry, "sid-ok", 720, "completed")
    assert asw._last_mapped_terminal("sid-ok") == ("completed", 720)
    # Missing breadcrumb.
    assert asw._last_mapped_terminal("sid-absent") is None
    # Non-terminal status (PARK) -> None.
    _write_terminal_breadcrumb(isolated_registry, "sid-park", 720, "blocked")
    assert asw._last_mapped_terminal("sid-park") is None
    # Non-int issue -> None.
    (isolated_registry / "last-mapped-terminal-sid-strissue.json").write_text(
        json.dumps({"terminal_status": "completed", "issue": "720"})
    )
    assert asw._last_mapped_terminal("sid-strissue") is None
    # Garbled JSON -> None.
    (isolated_registry / "last-mapped-terminal-sid-garbled.json").write_text("{not json")
    assert asw._last_mapped_terminal("sid-garbled") is None


def test_process_entry_writes_breadcrumb_on_terminal_delete(isolated_registry, monkeypatch):
    # The respawn pass: _process_entry on a TERMINAL task whose sid is live
    # writes last-mapped-terminal-<sid>.json (carrying issue) AND unlinks
    # issue-<N>.json.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    # Pin the clock so spawned_at stays inside MAX_ENTRY_AGE_S (else the
    # backstop-age branch fires before the delete branch we are testing).
    monkeypatch.setattr(asw.time, "time", lambda: 2_000_000.0)
    reg = isolated_registry / "issue-720.json"
    reg.write_text(
        json.dumps({"issue": 720, "happy_session_id": "sid-live", "spawned_at": 2_000_000.0})
    )
    asw._process_entry(reg, live_ids={"sid-live"}, dry_run=False, threshold=2)
    assert not reg.exists()
    crumb = isolated_registry / "last-mapped-terminal-sid-live.json"
    assert crumb.is_file()
    assert asw._last_mapped_terminal("sid-live") == ("completed", 720)


def test_process_entry_no_breadcrumb_when_session_dead(isolated_registry, monkeypatch):
    # A terminal task whose sid is NOT in live_ids: registry deleted, NO
    # breadcrumb (a dead session has nothing to reap on the short window).
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status", lambda issue: "archived")
    monkeypatch.setattr(asw.time, "time", lambda: 2_000_000.0)
    reg = isolated_registry / "issue-721.json"
    reg.write_text(
        json.dumps({"issue": 721, "happy_session_id": "sid-dead", "spawned_at": 2_000_000.0})
    )
    asw._process_entry(reg, live_ids={"some-other-sid"}, dry_run=False, threshold=2)
    assert not reg.exists()
    assert not (isolated_registry / "last-mapped-terminal-sid-dead.json").exists()


def test_process_entry_no_breadcrumb_on_park(isolated_registry, monkeypatch):
    # A blocked (PARK) task: registry KEPT (never deleted), NO breadcrumb.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status", lambda issue: "blocked")
    monkeypatch.setattr(asw.time, "time", lambda: 2_000_000.0)
    reg = isolated_registry / "issue-722.json"
    reg.write_text(
        json.dumps({"issue": 722, "happy_session_id": "sid-park", "spawned_at": 2_000_000.0})
    )
    asw._process_entry(reg, live_ids={"sid-park"}, dry_run=False, threshold=2)
    assert reg.exists()  # PARK keeps the registration
    assert not (isolated_registry / "last-mapped-terminal-sid-park.json").exists()


def test_process_entry_dry_run_writes_no_breadcrumb(isolated_registry, monkeypatch):
    # dry-run on a terminal delete logs but writes neither the breadcrumb nor
    # removes the registry.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw.time, "time", lambda: 2_000_000.0)
    reg = isolated_registry / "issue-723.json"
    reg.write_text(
        json.dumps({"issue": 723, "happy_session_id": "sid-dr", "spawned_at": 2_000_000.0})
    )
    asw._process_entry(reg, live_ids={"sid-dr"}, dry_run=True, threshold=2)
    assert reg.exists()
    assert not (isolated_registry / "last-mapped-terminal-sid-dr.json").exists()


def test_process_entry_spawn_grace_suppresses_then_respawns(isolated_registry, monkeypatch):
    # Criterion 5 (#759, bug class a): an ACTIVE entry with a DEAD recorded id.
    # When spawned_at is INSIDE the grace window, _process_entry must NOT
    # respawn (the id may not have propagated to /list yet) across 2 ticks;
    # when spawned_at is well OUTSIDE the window, the existing 2-miss respawn
    # still fires on the 2nd tick. Mirror of
    # test_active_dead_needs_two_misses_before_respawn at the I/O layer.
    import json

    import autonomous_session_watch as asw

    now = 2_000_000.0
    monkeypatch.setattr(asw.time, "time", lambda: now)
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    respawns: list[int] = []
    monkeypatch.setattr(asw, "_respawn", lambda entry, dry_run: respawns.append(entry.get("issue")))

    # ── Inside grace: spawned 60s ago, dead id, ACTIVE → no respawn over 2 ticks
    reg = isolated_registry / "issue-901.json"
    reg.write_text(
        json.dumps(
            {"issue": 901, "happy_session_id": "sid-dead", "spawned_at": now - 60, "missed": 0}
        )
    )
    asw._process_entry(reg, live_ids={"some-other-sid"}, dry_run=False, threshold=2)
    asw._process_entry(reg, live_ids={"some-other-sid"}, dry_run=False, threshold=2)
    assert respawns == []  # grace held across both ticks
    # The miss count stays reset (grace returns ("keep", 0)), so no stale miss
    # can carry past the grace boundary.
    assert json.loads(reg.read_text()).get("missed", 0) == 0

    # ── Outside grace: spawned 1h ago, dead id, ACTIVE → respawn on 2nd tick
    reg2 = isolated_registry / "issue-902.json"
    reg2.write_text(
        json.dumps(
            {"issue": 902, "happy_session_id": "sid-dead2", "spawned_at": now - 3600, "missed": 0}
        )
    )
    asw._process_entry(reg2, live_ids={"some-other-sid"}, dry_run=False, threshold=2)
    assert respawns == []  # first miss only increments
    asw._process_entry(reg2, live_ids={"some-other-sid"}, dry_run=False, threshold=2)
    assert respawns == [902]  # 2nd miss past grace → respawn


def _patch_short_window_guards(monkeypatch, *, running_pods, followup_active):
    """Patch the two lazy MF-1 guards the short-window branch consults.
    ``running_pods`` is the _running_managed_issue_pods return ([], None, or a
    list of (issue, pod_id, pod_name, info) tuples); ``followup_active`` is the
    _task_followup_active bool."""
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda caller="": running_pods)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None: followup_active)


def _run_short_window_case(
    isolated_registry,
    monkeypatch,
    *,
    terminal_status="completed",
    running_pods,
    followup_active,
    idle_age_s=35 * 60,
    has_tty=False,
    write_breadcrumb=True,
    registry=None,
):
    """Drive idle_unmapped_pass end-to-end for ONE unmapped EPS session over two
    consecutive ticks under the short (30-min) reap window, with the breadcrumb
    present and the two MF-1 guards patched. Returns (stops, records)."""
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    sid = "sid-720"
    children = [{"happySessionId": sid, "pid": 7720}]
    meta = {sid: {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch,
        children=children,
        meta=meta,
        idle_age=idle_age_s,
        has_tty=has_tty,
        registry=registry,
    )
    _patch_short_window_guards(
        monkeypatch, running_pods=running_pods, followup_active=followup_active
    )
    if write_breadcrumb:
        _write_terminal_breadcrumb(isolated_registry, sid, 720, terminal_status)
    t0 = 1_000_000.0
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)  # miss 1
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)  # decide
    return stops, records


def test_idle_unmapped_pass_short_window_reaps_completed(isolated_registry, monkeypatch):
    # Happy path: unmapped + `completed` breadcrumb + idle 35 min (> 30-min short
    # window, < 12h default) + no TTY + no pod + no follow-up + 2 consecutive
    # ticks -> reap fires on tick 2.
    stops, records = _run_short_window_case(
        isolated_registry, monkeypatch, running_pods=[], followup_active=False
    )
    assert stops == ["sid-720"]
    assert len(records) == 1 and "auto-stopped idle unmapped" in records[0]


def test_idle_unmapped_pass_short_window_reaps_archived(isolated_registry, monkeypatch):
    stops, _ = _run_short_window_case(
        isolated_registry,
        monkeypatch,
        terminal_status="archived",
        running_pods=[],
        followup_active=False,
    )
    assert stops == ["sid-720"]


def test_idle_unmapped_pass_short_window_reaps_awaiting_promotion(isolated_registry, monkeypatch):
    stops, _ = _run_short_window_case(
        isolated_registry,
        monkeypatch,
        terminal_status="awaiting_promotion",
        running_pods=[],
        followup_active=False,
    )
    assert stops == ["sid-720"]


def test_idle_unmapped_pass_no_short_window_for_blocked(isolated_registry, monkeypatch):
    # A `blocked` breadcrumb (PARK) is NOT in TERMINAL -> _last_mapped_terminal
    # returns None -> the short window is never applied -> the 12h default holds
    # and a 35-min-idle session is NOT reaped across 2 ticks. (In production no
    # breadcrumb is even written for blocked; here we write one and confirm the
    # read-side scope guard also keeps it on the long window.)
    stops, records = _run_short_window_case(
        isolated_registry,
        monkeypatch,
        terminal_status="blocked",
        running_pods=[],
        followup_active=False,
    )
    assert stops == [] and records == []


def test_idle_unmapped_pass_short_window_skipped_when_running_pod(isolated_registry, monkeypatch):
    # MF-1 Guard 1: a RUNNING managed pod for the breadcrumb's issue keeps the
    # session on the 12h window -> NOT reaped across 2 ticks, even though the
    # 30-min short-window idle threshold is crossed. The guard reads only t[0]
    # (the issue) from each 4-tuple, so the PodInfo slot is a placeholder.
    running = [(720, "p720", "pod-720", None)]
    stops, records = _run_short_window_case(
        isolated_registry, monkeypatch, running_pods=running, followup_active=False
    )
    assert stops == [] and records == []


def test_idle_unmapped_pass_short_window_keeps_when_pod_snapshot_none(
    isolated_registry, monkeypatch
):
    # MF-1 Guard 1 fail-toward-keep: a None pod snapshot (API transport error)
    # is "uncertain -> keep the long window" -> NOT reaped across 2 ticks.
    stops, records = _run_short_window_case(
        isolated_registry, monkeypatch, running_pods=None, followup_active=False
    )
    assert stops == [] and records == []


def test_idle_unmapped_pass_short_window_running_pod_other_issue_reaps(
    isolated_registry, monkeypatch
):
    # Guard 1 is issue-scoped: a RUNNING pod for a DIFFERENT issue does NOT
    # protect this session -> the short window still applies and it reaps.
    running = [(999, "p999", "pod-999", None)]
    stops, _ = _run_short_window_case(
        isolated_registry, monkeypatch, running_pods=running, followup_active=False
    )
    assert stops == ["sid-720"]


def test_idle_unmapped_pass_short_window_skipped_when_followup_active(
    isolated_registry, monkeypatch
):
    # MF-1 Guard 2: a live same-issue follow-up (fresh epm:run-launched newer
    # than the latest done-transition) keeps the session on the 12h window ->
    # NOT reaped across 2 ticks.
    stops, records = _run_short_window_case(
        isolated_registry, monkeypatch, running_pods=[], followup_active=True
    )
    assert stops == [] and records == []


def test_idle_unmapped_pass_short_window_excludes_tty(isolated_registry, monkeypatch):
    # A TTY-attached session is never reaped (the breadcrumb branch is guarded by
    # `not has_tty`), so even with a terminal breadcrumb it stays untouched.
    stops, records = _run_short_window_case(
        isolated_registry,
        monkeypatch,
        running_pods=[],
        followup_active=False,
        has_tty=True,
    )
    assert stops == [] and records == []


def test_idle_unmapped_pass_short_window_excludes_mapped(isolated_registry, monkeypatch):
    # A RE-MAPPED session (issue-<N>.json present again, e.g. a follow-up loop's
    # register-current) is excluded from the breadcrumb branch entirely (mapped
    # wins) -> the 12h window holds even with a stale breadcrumb -> a 35-min
    # idle session is NOT reaped.
    stops, records = _run_short_window_case(
        isolated_registry,
        monkeypatch,
        running_pods=[],
        followup_active=False,
        registry={"sid-720": 720},
    )
    assert stops == [] and records == []


def test_idle_unmapped_pass_short_window_two_miss_guard(isolated_registry, monkeypatch):
    # The 2-miss guard still binds under the short window: a single qualifying
    # tick accumulates a miss but does NOT stop; the stop fires only on the
    # second consecutive qualifying tick.
    import json

    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    sid = "sid-720"
    children = [{"happySessionId": sid, "pid": 7720}]
    meta = {sid: {"path": _Z_ROOT}}
    stops, _records = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=35 * 60)
    _patch_short_window_guards(monkeypatch, running_pods=[], followup_active=False)
    _write_terminal_breadcrumb(isolated_registry, sid, 720, "completed")
    state_path = isolated_registry / f"idle-unmapped-{sid}.json"
    t0 = 1_000_000.0
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)  # miss 1
    assert stops == []
    assert json.loads(state_path.read_text())["missed"] == 1
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)  # stop
    assert stops == [sid]


def test_idle_unmapped_pass_short_window_dry_run(isolated_registry, monkeypatch):
    # --dry-run: no breadcrumb is written by _record_last_mapped_terminal (tested
    # above) and no stop call fires even when the session would otherwise reap.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    sid = "sid-720"
    children = [{"happySessionId": sid, "pid": 7720}]
    meta = {sid: {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=35 * 60)
    _patch_short_window_guards(monkeypatch, running_pods=[], followup_active=False)
    _write_terminal_breadcrumb(isolated_registry, sid, 720, "completed")
    t0 = 1_000_000.0
    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0)
    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == [] and records == []


def test_idle_unmapped_short_window_min_never_lengthens(isolated_registry, monkeypatch):
    # Defensive: idle_reap_s = min(_unmapped_idle_reap_s(), short) can only ever
    # SHORTEN. If an operator set the 12h window BELOW 30 min, the short window
    # does not lengthen it. Here EPM_UNMAPPED_IDLE_REAP_S=300s (5 min) < 30 min:
    # a session idle 6 min is reaped on the (5-min) window even with the
    # breadcrumb present, not held to 30 min.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    monkeypatch.setenv("EPM_UNMAPPED_IDLE_REAP_S", "300")  # 5 min
    sid = "sid-720"
    children = [{"happySessionId": sid, "pid": 7720}]
    meta = {sid: {"path": _Z_ROOT}}
    stops, _ = _patch_idle_io(monkeypatch, children=children, meta=meta, idle_age=360)  # 6 min
    _patch_short_window_guards(monkeypatch, running_pods=[], followup_active=False)
    _write_terminal_breadcrumb(isolated_registry, sid, 720, "completed")
    t0 = 1_000_000.0
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0)
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=t0 + 600)
    assert stops == [sid]


def test_gc_orphan_last_mapped_terminal_removes_stale(isolated_registry):
    # A breadcrumb whose sid is NOT in the live set is unlinked; a live-sid
    # breadcrumb is kept.
    import autonomous_session_watch as asw

    _write_terminal_breadcrumb(isolated_registry, "sid-live", 720, "completed")
    _write_terminal_breadcrumb(isolated_registry, "sid-dead", 721, "completed")
    asw._gc_orphan_last_mapped_terminal({"sid-live"}, dry_run=False)
    assert (isolated_registry / "last-mapped-terminal-sid-live.json").exists()
    assert not (isolated_registry / "last-mapped-terminal-sid-dead.json").exists()
    # dry-run never unlinks.
    _write_terminal_breadcrumb(isolated_registry, "sid-dead", 721, "completed")
    asw._gc_orphan_last_mapped_terminal({"sid-live"}, dry_run=True)
    assert (isolated_registry / "last-mapped-terminal-sid-dead.json").exists()


def test_idle_unmapped_pass_calls_breadcrumb_gc(isolated_registry, monkeypatch):
    # The pass GCs orphan breadcrumbs once per daemon-reachable tick (even with
    # zero candidates): a breadcrumb for a sid not in the live set is reaped.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    # No live children -> live_sids empty -> the stale breadcrumb is GC'd.
    _patch_idle_io(monkeypatch, children=[], meta={})
    _write_terminal_breadcrumb(isolated_registry, "sid-gone", 720, "completed")
    asw.idle_unmapped_pass(False, 2, daemon_reachable=True, now=1_000_000.0)
    assert not (isolated_registry / "last-mapped-terminal-sid-gone.json").exists()


# ── #795 diagnosis (NO-CHANGE verification) ──────────────────────────────────
# #795 was filed to "verify/tighten the reconcile-pass auto-stop predicate for
# completed-task sessions, or add a completed-task fast-path". The diagnosis
# (plan v2, Route A) found the class is ALREADY reaped by the #720 short-window
# idle-unmapped path (worst case ~50 min). These tests pin that NO-CHANGE
# conclusion: they assert the diagnosis-smoke's decision core, so a future
# regression that removes / lengthens the #720 fast lane trips here.
def test_issue795_diagnosis_smoke_verdict_holds(isolated_registry, monkeypatch):
    # The offline diagnosis (synthetic ghost class + arithmetic bound) must
    # return the NO-CHANGE verdict on the current tree. Runs the smoke's two
    # deterministic checks (A + B) directly — the same asserts the smoke's
    # exit code depends on.
    import importlib.util

    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    spec = importlib.util.spec_from_file_location(
        "issue795_diagnosis_smoke",
        str(SCRIPTS / "issue795_diagnosis_smoke.py"),
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    assert smoke.check_synthetic_short_window() is True
    assert smoke.check_worst_case_bound() is True


def test_issue795_completed_ghost_class_reaped_via_720(isolated_registry, monkeypatch):
    # The exact #795 ghost class (698/705/706): a `completed`-task session,
    # unmapped (registry entry deleted by the respawn pass), repo-root cwd,
    # non-TTY, no pod, no follow-up -> reaped by the idle-unmapped pass on the
    # 30-min short window within 2 ticks. This is the end-to-end proof that no
    # reconcile-pass change is needed; it reuses the #720 short-window harness.
    stops, records = _run_short_window_case(
        isolated_registry,
        monkeypatch,
        terminal_status="completed",
        running_pods=[],
        followup_active=False,
    )
    assert stops == ["sid-720"]
    assert len(records) == 1 and "auto-stopped idle unmapped" in records[0]


def test_issue795_dry_run_does_not_stop(isolated_registry, monkeypatch):
    # c11 (dry-run coverage): the same ghost class under dry_run=True is
    # DECIDED but never actually stopped — no daemon stop call, no reap record.
    # Pins that the diagnosis path is side-effect-free in dry-run mode.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP", raising=False)
    monkeypatch.delenv("EPM_UNMAPPED_IDLE_REAP_S", raising=False)
    monkeypatch.delenv("EPM_LAST_MAPPED_TERMINAL_REAP_S", raising=False)
    sid = "sid-795-dry"
    children = [{"happySessionId": sid, "pid": 7795}]
    meta = {sid: {"path": _Z_ROOT}}
    stops, records = _patch_idle_io(
        monkeypatch, children=children, meta=meta, idle_age=35 * 60, has_tty=False
    )
    _patch_short_window_guards(monkeypatch, running_pods=[], followup_active=False)
    _write_terminal_breadcrumb(isolated_registry, sid, 795, "completed")
    t0 = 2_000_000.0
    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0)  # dry_run miss 1
    asw.idle_unmapped_pass(True, 2, daemon_reachable=True, now=t0 + 600)  # dry_run decide
    assert stops == [] and records == []


# ─── #843: dispatch-lease loop pre-checks, M1b suppressed-output handling, ────
# ─── M3 marker-freshness guard, GC of terminal leases ─────────────────────────
#
# The lease PRIMITIVE (acquire/release/takeover/race) is pinned in
# tests/test_dispatch_lease.py; these tests pin the WATCHER-side consumers:
# the caller-loop advisory pre-checks record NO attempt (a lease-held skip
# must not consume the 1 h backoff — the crashed-winner recovery bound),
# the tri-state `_dispatch_infra_drain` never books a suppressed rc-0 no-op,
# the respawn family never logs/marks a suppressed no-op as RESPAWNED, and
# the sweep skips candidates with a < 600 s dispatch-sentinel marker.


def _iso_ts(epoch: float) -> str:
    """Task-event `ts` string (UTC, `%Y-%m-%dT%H:%M:%SZ`) for a fixed epoch."""
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch, tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_drain_loop_skips_on_fresh_lease_records_no_attempt(isolated_registry, monkeypatch, capsys):
    # Test 15: a fresh per-issue dispatch lease -> the DRAIN caller loop
    # `continue`s with the loud skip line, never reaches the spawn
    # subprocess, and records NO attempt (no backoff consumed — the round-1
    # crashed-winner recovery-bound fix).
    import json

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
        asw.subprocess, "run", lambda *a, **k: pytest.fail("spawned despite a fresh lease")
    )
    assert spawn_session.acquire_dispatch_lease(483, "other-dispatcher", now=now - 30) is not None
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "INFRA-DRAIN SKIP issue #483 (dispatch-lease held" in out
    assert "holder=other-dispatcher" in out
    assert markers == []
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"] == {}  # NO attempt recorded -> no backoff armed


def test_sweep_loop_skips_on_fresh_lease_records_no_attempt(isolated_registry, monkeypatch, capsys):
    # Test 15 sibling for the SWEEP caller loop (same contract).
    import json

    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    dispatched, markers = _stub_sweep_executor(
        monkeypatch, candidates=[771], status_kind={771: ("proposed", "infra")}, occupancy=[]
    )
    assert spawn_session.acquire_dispatch_lease(771, "other-dispatcher", now=now - 30) is not None
    asw.proposed_infra_sweep_pass(dry_run=False, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "PROPOSED-INFRA-SWEEP SKIP issue #771 (dispatch-lease held" in out
    assert dispatched == [] and markers == []
    state = json.loads((isolated_registry / "proposed-infra-sweep-state.json").read_text())
    assert state["attempts"] == {}  # NO attempt recorded


@pytest.mark.parametrize("dry_run", [False, True])
def test_sweep_skips_on_recent_dispatch_marker(isolated_registry, monkeypatch, capsys, dry_run):
    # Test 16: a dispatch-sentinel marker younger than 600 s -> skip (no
    # dispatch, no attempt). Parametrized over dry_run (the #596/#607/#633
    # broken-dry-run-thread pattern): the skip decision must survive dry-run.
    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch, candidates=[771], status_kind={771: ("proposed", "infra")}, occupancy=[]
    )
    events = [
        {
            "kind": "epm:progress",
            "note": f"{asw._PROPOSED_INFRA_SWEEP_NOTE_SENTINEL} watcher auto-dispatched",
            "ts": _iso_ts(now - 300),
        }
    ]
    monkeypatch.setattr(asw, "_task_events", lambda i: events)
    asw.proposed_infra_sweep_pass(dry_run=dry_run, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "PROPOSED-INFRA-SWEEP SKIP issue #771 (recent-dispatch-marker 300s < 600s)" in out
    assert dispatched == []


def test_sweep_dispatches_on_old_marker(isolated_registry, monkeypatch, capsys):
    # Test 16 companion: a 20-min-old dispatch marker does NOT skip.
    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    dispatched, _markers = _stub_sweep_executor(
        monkeypatch, candidates=[771], status_kind={771: ("proposed", "infra")}, occupancy=[]
    )
    events = [
        {
            "kind": "epm:progress",
            "note": f"{asw._INFRA_DRAIN_NOTE_SENTINEL} watcher dispatched",
            "ts": _iso_ts(now - 1200),
        }
    ]
    monkeypatch.setattr(asw, "_task_events", lambda i: events)
    asw.proposed_infra_sweep_pass(dry_run=False, now=now, daemon_reachable=True)
    assert dispatched == [771]
    assert "recent-dispatch-marker" not in capsys.readouterr().out


def test_recent_dispatch_marker_age_matches_both_sentinels():
    # BOTH sentinels disqualify (a drain dispatch is exactly as disqualifying
    # as a sweep one); an unrelated epm:progress note never matches.
    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    for sentinel in (asw._PROPOSED_INFRA_SWEEP_NOTE_SENTINEL, asw._INFRA_DRAIN_NOTE_SENTINEL):
        events = [{"kind": "epm:progress", "note": f"{sentinel} x", "ts": _iso_ts(now - 120)}]
        assert asw._recent_dispatch_marker_age_s(events, now) == pytest.approx(120, abs=2)
    plain = [{"kind": "epm:progress", "note": "ordinary progress", "ts": _iso_ts(now - 120)}]
    assert asw._recent_dispatch_marker_age_s(plain, now) is None
    non_progress = [
        {
            "kind": "epm:results",
            "note": f"{asw._INFRA_DRAIN_NOTE_SENTINEL} x",
            "ts": _iso_ts(now - 120),
        }
    ]
    assert asw._recent_dispatch_marker_age_s(non_progress, now) is None


def test_recent_dispatch_marker_age_unparseable_ts_returns_none():
    # Fail-soft: an unparseable ts row is skipped -> None -> no skip (the M1
    # lease still protects).
    import autonomous_session_watch as asw

    events = [
        {
            "kind": "epm:progress",
            "note": f"{asw._INFRA_DRAIN_NOTE_SENTINEL} x",
            "ts": "not-a-timestamp",
        }
    ]
    assert asw._recent_dispatch_marker_age_s(events, _SWEEP_NOW) is None


def test_sweep_marker_fresh_s_env_override(monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S", "60")
    assert asw._proposed_infra_sweep_marker_fresh_s() == 60.0
    monkeypatch.setenv("EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S", "garbage")
    assert (
        asw._proposed_infra_sweep_marker_fresh_s()
        == asw.PROPOSED_INFRA_SWEEP_MARKER_FRESH_S_DEFAULT
    )
    monkeypatch.delenv("EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S")
    assert (
        asw._proposed_infra_sweep_marker_fresh_s()
        == asw.PROPOSED_INFRA_SWEEP_MARKER_FRESH_S_DEFAULT
    )


def test_lease_ttl_default_equals_respawn_spawn_grace():
    # Test 17: pins the M2/#759 coupling — neither default may drift alone
    # (the lease TTL must never postpone a recovery the watcher would run).
    import autonomous_session_watch as asw

    assert spawn_session.DISPATCH_LEASE_TTL_S == asw.RESPAWN_SPAWN_GRACE_S


def test_gc_reaps_terminal_dispatch_lease(isolated_registry, monkeypatch, capsys):
    # Test 18: a TERMINAL task's leftover lease is reaped; an ACTIVE task's
    # fresh lease is NEVER touched (pins the conservative keep branch against
    # future _GC_TARGETS edits); the PERMANENT .lock sidecar is not swept
    # (the glob is dispatch-lease-*.json).
    import autonomous_session_watch as asw

    (isolated_registry / "dispatch-lease-900.json").write_text("{}")
    (isolated_registry / "dispatch-lease-900.lock").write_text("")
    (isolated_registry / "dispatch-lease-901.json").write_text("{}")
    statuses = {900: "completed", 901: "running"}
    monkeypatch.setattr(asw, "_task_status", lambda i: statuses.get(i))
    asw.gc_pass(dry_run=False, now=_SWEEP_NOW)
    assert not (isolated_registry / "dispatch-lease-900.json").exists()  # reaped
    assert (isolated_registry / "dispatch-lease-900.lock").exists()  # sidecar kept
    assert (isolated_registry / "dispatch-lease-901.json").exists()  # active kept
    assert "dispatch-lease-900.json" in capsys.readouterr().out


def test_dispatch_infra_drain_tristate_suppressed(isolated_registry, monkeypatch, capsys):
    # Test 20 (helper level): rc-0 stdout carrying a suppression sentinel ->
    # "suppressed"; plain rc-0 -> "spawned"; rc!=0 -> "failed".
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    outputs = {"rc": 0, "stdout": "DISPATCH-LEASE HELD issue #42: a dispatch is in flight\n"}

    def _fake_run(cmd, **kw):
        return SimpleNamespace(returncode=outputs["rc"], stdout=outputs["stdout"], stderr="boom")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "suppressed"
    assert "INFRA-DRAIN SUPPRESSED issue #42" in capsys.readouterr().out
    outputs["stdout"] = "REGISTRATION-COLLISION issue #42: kept sid-a, stopped sid-b\n"
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "suppressed"
    outputs["stdout"] = "Issue #42 session spawned: sid-new\n"
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "spawned"
    outputs["rc"] = 1
    assert asw._dispatch_infra_drain(42, "slot 1/3", dry_run=False, reg_snapshot={}) == "failed"


def test_drain_pass_suppressed_records_no_attempt_no_marker(isolated_registry, monkeypatch, capsys):
    # Test 20 (caller level, through the REAL _dispatch_infra_drain): a
    # suppressed rc-0 no-op records NO attempt (no backoff -> the crashed
    # lease-winner recovers in <= TTL + one tick, not 1 h) and posts NO
    # dispatch marker.
    import json
    from types import SimpleNamespace

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
        asw.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout="DISPATCH-LEASE HELD issue #483: in flight\n", stderr=""
        ),
    )
    asw.infra_drain_pass(dry_run=False, now=now, daemon_reachable=True)
    out = capsys.readouterr().out
    assert "INFRA-DRAIN SUPPRESSED issue #483" in out
    assert markers == []
    state = json.loads((isolated_registry / "infra-drain-state.json").read_text())
    assert state["attempts"] == {}  # suppressed -> no attempt, no backoff


def test_respawn_suppressed_output_not_booked_as_respawned(isolated_registry, monkeypatch, capsys):
    # Test 21 (helper level): every respawn-family helper detects a
    # suppressed rc-0 and returns "suppressed" with the loud
    # "suppressed — not respawned" line instead of RESPAWNED*.
    from types import SimpleNamespace

    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout="DISPATCH-LEASE HELD issue #7: in flight\n", stderr=""
        ),
    )
    assert asw._respawn({"issue": 7}, dry_run=False) == "suppressed"
    assert asw._respawn_stalled_session(7, 24.0, dry_run=False) == "suppressed"
    assert asw._respawn_orphan(7, 24.0, dry_run=False) == "suppressed"
    assert asw._redrive_capacity_retry(7, dry_run=False) == "suppressed"
    out = capsys.readouterr().out
    # RESPAWN + RESPAWN-STALLED + RESPAWN-ORPHAN all print the phrase.
    assert out.count("suppressed — not respawned (lease/collision)") == 3
    assert "RESPAWN-ORPHAN issue #7: suppressed — not respawned" in out
    assert "CAPACITY-RETRY issue #7: suppressed — not re-driven" in out
    assert "RESPAWNED" not in out and "CAPACITY-RETRIED" not in out


def test_stalled_handler_suppressed_books_nothing(isolated_registry, monkeypatch, capsys):
    # Test 21 (caller level, stalled respawn handler): a suppressed spawn ->
    # no respawn marker, no respawn_count bump, NO full state save (the
    # on-disk miss/stall state is left untouched). #845 addition: the
    # fence's stop_pending_* IS cleared on disk (surgically — the lease
    # collision proves a live driver owns the issue, so the fence episode
    # is over), but every other on-disk field stays byte-identical.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: True)
    monkeypatch.setattr(asw, "_stalled_cap_gpu_hours", lambda i: 24.0)
    monkeypatch.setattr(asw, "_respawn_stalled_session", lambda i, cap, dry: "suppressed")
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    # #1247 fence act-guard seam: confirm-active so the guard's live re-read
    # never shells the real `task.py view` subprocess (hermeticity).
    monkeypatch.setattr(asw, "_task_status", lambda _i: "running")
    markers: list[tuple] = []
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: markers.append((a, k)))
    saves: list[tuple] = []
    monkeypatch.setattr(asw, "_save_stalled_state", lambda *a, **k: saves.append((a, k)))
    # On-disk state mid-fence-episode: the stop already fired on a prior
    # tick (stop_pending_sid == the entry sid) and the sid is now absent
    # from live_ids, so the fence reaches its verified-dead spawn branch.
    (isolated_registry / "stalled-7.json").write_text(
        json.dumps({"missed": 2, "alerted": True, "stop_pending_sid": "sid-old"})
    )
    ctx = asw._StalledActionCtx(
        issue=7,
        happy_session_id="sid-old",
        prev_state={"missed": 2, "alerted": True, "stop_pending_sid": "sid-old"},
        alerted=True,
        respawn_count=1,
        exhausted=False,
        last_self_report_ts=None,
        self_gap="3.0h",
        marker_gap="3.0h",
        has_pod=False,
        task_status="running",
        in_active=True,
        threshold=2,
        dry_run=False,
        live_ids=set(),  # verified dead -> fence 'spawn' branch
        stop_pending_sid="sid-old",
    )
    asw._handle_stalled_respawn(ctx)
    assert markers == []  # no respawn marker
    assert saves == []  # no full state save — miss/stall state left untouched
    on_disk = json.loads((isolated_registry / "stalled-7.json").read_text())
    assert on_disk["stop_pending_sid"] is None  # fence episode over
    assert on_disk["missed"] == 2  # ...but nothing else was booked
    assert on_disk["alerted"] is True


# ─── #1247: act-time terminal-status guard (stalled fence spawn) ─────────────
#
# The fence's stop->verify->spawn spans ticks, so ctx.task_status is >=10 min
# old at spawn time by construction. _fence_spawn_stalled now re-reads the
# LIVE status and requires a positive ACTIVE confirmation before spawning or
# posting the respawn marker (the stalled-arm sibling of ORPHAN-ACT-GUARD).
# These tests drive the REAL _handle_stalled_respawn -> _fence_spawn_stalled
# path; only subprocess-boundary seams are monkeypatched.


def _drive_fence_spawn(asw, monkeypatch, isolated_registry, *, issue, task_status, dry_run=False):
    """Reach the fence's verified-dead spawn branch (pending sid set, sid
    absent from the live set) with recorder seams; returns (spawns, markers)."""
    import json

    spawns: list[int] = []
    markers: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_stalled_cap_gpu_hours", lambda i: 24.0)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    monkeypatch.setattr(
        asw, "_respawn_stalled_session", lambda i, cap, dry: spawns.append(i) or "spawned"
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda i, note, dry_run, label: markers.append((i, note)),
    )
    monkeypatch.setattr(asw, "_task_status", lambda _i: task_status)
    (isolated_registry / f"stalled-{issue}.json").write_text(
        json.dumps({"missed": 2, "alerted": True, "stop_pending_sid": "sid-old"})
    )
    ctx = asw._StalledActionCtx(
        issue=issue,
        happy_session_id="sid-old",
        prev_state={"missed": 2, "alerted": True, "stop_pending_sid": "sid-old"},
        alerted=True,
        respawn_count=0,
        exhausted=False,
        last_self_report_ts=None,
        self_gap="3.0h",
        marker_gap="3.0h",
        has_pod=False,
        task_status="running",
        in_active=True,
        threshold=2,
        dry_run=dry_run,
        live_ids=set(),  # verified dead -> fence 'spawn' branch
        stop_pending_sid="sid-old",
    )
    asw._handle_stalled_respawn(ctx)
    return spawns, markers


def test_stalled_fence_spawn_guard_aborts_on_terminal_live_status(
    isolated_registry, monkeypatch, capsys
):
    # Live re-read says "completed": no spawn, no marker, the fence's
    # stop_pending_* fields cleared (episode over) while the rest of the
    # stalled episode state is untouched.
    import json

    import autonomous_session_watch as asw

    spawns, markers = _drive_fence_spawn(
        asw, monkeypatch, isolated_registry, issue=90711, task_status="completed"
    )
    assert spawns == []
    assert markers == []
    on_disk = json.loads((isolated_registry / "stalled-90711.json").read_text())
    assert on_disk["stop_pending_sid"] is None  # fence state cleared
    assert on_disk["missed"] == 2  # episode state untouched (surgical clear)
    err = capsys.readouterr().err
    assert "STALLED-ACT-GUARD" in err and "'completed'" in err


def test_stalled_fence_spawn_guard_defers_on_unreadable_live_status(
    isolated_registry, monkeypatch, capsys
):
    # _task_status -> None (transient read failure): no spawn, no marker, and
    # the fence stays PENDING on disk — re-evaluated next tick.
    import json

    import autonomous_session_watch as asw

    spawns, markers = _drive_fence_spawn(
        asw, monkeypatch, isolated_registry, issue=90712, task_status=None
    )
    assert spawns == []
    assert markers == []
    on_disk = json.loads((isolated_registry / "stalled-90712.json").read_text())
    assert on_disk["stop_pending_sid"] == "sid-old"  # fence KEPT pending
    assert "STALLED-ACT-GUARD" in capsys.readouterr().err


def test_stalled_fence_spawn_guard_dry_run_never_mutates_state(
    isolated_registry, monkeypatch, capsys
):
    # Dry-run sibling: the guard's state-clear is dry_run-gated — the fence
    # file survives byte-for-byte, and nothing spawns or posts.
    import json

    import autonomous_session_watch as asw

    state_path = isolated_registry / "stalled-90713.json"
    spawns, markers = _drive_fence_spawn(
        asw, monkeypatch, isolated_registry, issue=90713, task_status="archived", dry_run=True
    )
    assert spawns == []
    assert markers == []
    on_disk = json.loads(state_path.read_text())
    assert on_disk["stop_pending_sid"] == "sid-old"  # untouched under dry-run
    assert "STALLED-ACT-GUARD" in capsys.readouterr().err


def test_stalled_fence_spawn_allows_live_active_and_note_carries_source_stamp(
    isolated_registry, monkeypatch
):
    # Regression floor + stamp: a positively-ACTIVE live read lets the fence
    # spawn proceed, and the respawn note self-identifies the posting process.
    import re

    import autonomous_session_watch as asw

    spawns, markers = _drive_fence_spawn(
        asw, monkeypatch, isolated_registry, issue=90714, task_status="running"
    )
    assert spawns == [90714]
    assert len(markers) == 1 and markers[0][0] == 90714
    assert re.search(r"\[src: host=\S+ user=\S+ pid=\d+ sha=\S+ root=/\S+\]$", markers[0][1])


def test_capacity_retry_suppressed_consumes_no_budget(isolated_registry, monkeypatch, capsys):
    # Test 21 (capacity-retry variant): a suppressed re-drive consumes NO
    # per-day retry budget and writes NO state.
    import autonomous_session_watch as asw

    now = _SWEEP_NOW
    monkeypatch.setattr(asw, "_task_events", lambda i: [])
    monkeypatch.setattr(
        asw,
        "_is_transient_capacity_block",
        lambda events: (True, "no_compute_available", now - 99999),
    )
    monkeypatch.setattr(asw, "_load_capacity_retry_state", lambda i: {})
    monkeypatch.setattr(asw, "_redrive_capacity_retry", lambda i, dry: "suppressed")
    saves: list[tuple] = []
    monkeypatch.setattr(asw, "_save_capacity_retry_state", lambda *a, **k: saves.append((a, k)))
    markers: list[tuple] = []
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: markers.append((a, k)))
    asw._process_capacity_retry(7, now, "2027-01-15", False, backoff_s=3600.0, max_per_day=4)
    assert saves == []  # retry budget NOT consumed
    assert markers == []


# ─── #845 stall-detection hardening — integration ─────────────────────────────


def _iso_845(epoch: float) -> str:
    """Canonical trailing-Z UTC ISO string (the shape _parse_event_ts parses)."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_stalled_state_845(reg_dir, issue):
    import json

    path = reg_dir / f"stalled-{issue}.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def test_fence_stop_only_first_tick(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (a-ii) plan test 3: the tick the respawn action trips issues the
    # STOP only — no spawn — and persists the pending sid for the next
    # tick's verification.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 971, "sess-971", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-971"]
    assert spawns == [] and markers == []
    state = _read_stalled_state_845(isolated_registry, 971)
    assert state["stop_pending_sid"] == "sess-971"
    assert state["stop_pending_ts"] == now
    assert state["stop_retried"] is False
    # Pinned at the threshold so the arm re-fires (and verifies) next tick.
    assert state["missed"] == 2


def test_fence_spawns_after_verified_dead(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (a-ii) plan test 4: with the pending sid absent from the live set
    # on the following tick, the fence spawns, bumps respawn_count, and
    # clears the pending state.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 972, "sess-972", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    for _ in range(3):  # miss -> stop -> verified-dead spawn
        asw.stalled_session_pass(
            dry_run=False, threshold=2, now=now, daemon_reachable=True, live_ids=set()
        )
    assert stops == ["sess-972"]
    assert spawns == [(972, 24.0)]
    assert markers == [(972, "session-auto-respawn")]
    state = _read_stalled_state_845(isolated_registry, 972)
    assert state["stop_pending_sid"] is None
    assert state["respawn_count"] == 1
    assert state["wt_hold_count"] == 0


def test_fence_spawn_suppressed_clears_pending(isolated_registry, monkeypatch, stalled_recorder):
    # #845 plan test 4b: the verified-dead spawn returning the #843
    # tri-state "suppressed" books NOTHING (no marker, no respawn_count
    # bump) but DOES clear the fence's stop_pending_* (the lease collision
    # proves a live driver owns the issue — episode over).
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 973, "sess-973", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-973"] and spawns == []

    monkeypatch.setattr(asw, "_respawn_stalled_session", lambda i, cap, dry: "suppressed")
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert spawns == [] and markers == []
    state = _read_stalled_state_845(isolated_registry, 973)
    assert state["stop_pending_sid"] is None  # fence episode over
    assert state["respawn_count"] == 0  # nothing booked
    assert state["missed"] == 2  # miss/stall state left untouched


def test_fence_state_clears_on_self_report_advance(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #845 plan test 6: a mid-fence self-report advancement (the session
    # resumed) ends the episode — every hardening field clears.
    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 974, "sess-974", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-974"]
    assert _read_stalled_state_845(isolated_registry, 974)["stop_pending_sid"] == "sess-974"

    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (30.0, "ts-z-advanced"))
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    state = _read_stalled_state_845(isolated_registry, 974)
    assert state["stop_pending_sid"] is None
    assert state["stop_retried"] is False
    assert state["wt_hold_count"] == 0
    assert state["daemon_blocked_ticks"] == 0


def test_fence_clears_on_sid_change_never_stops_fresh_sid(isolated_registry, monkeypatch):
    # #845 plan test 6b (the MF1 fence-race pin): a pending fence whose sid
    # no longer matches the registry entry's sid means a CONCURRENT respawn
    # (crash arm / #843-leased driver) replaced the session inside the
    # stop->verify gap — the fence clears itself and the fresh sid is NEVER
    # stopped.
    import autonomous_session_watch as asw

    stops: list[str] = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: stops.append(sid) or True)
    spawns: list = []
    monkeypatch.setattr(
        asw, "_respawn_stalled_session", lambda i, c, d: spawns.append(i) or "spawned"
    )
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    ctx = asw._StalledActionCtx(
        issue=975,
        happy_session_id="sid-fresh",
        prev_state={"stop_pending_sid": "sid-old"},
        alerted=True,
        respawn_count=0,
        exhausted=False,
        last_self_report_ts="ts-1",
        self_gap="90.0m",
        marker_gap="none",
        has_pod=False,
        task_status="running",
        in_active=True,
        threshold=2,
        dry_run=False,
        now=1_000_000.0,
        live_ids={"sid-fresh"},
        stop_pending_sid="sid-old",
    )
    asw._handle_stalled_respawn(ctx)
    assert stops == []  # the fresh sid was never stopped
    assert spawns == []  # and nothing was spawned next to it
    state = _read_stalled_state_845(isolated_registry, 975)
    assert state["stop_pending_sid"] is None  # fence cleared


def test_stalled_arm_skips_within_spawn_grace(isolated_registry, monkeypatch, stalled_recorder):
    # #845 plan test 6c: an entry (re)written within RESPAWN_SPAWN_GRACE_S
    # of the pass clock means a concurrent respawn owns the issue — the
    # stalled arm skips entirely (mirror of the crash arm's #759 grace).
    import json

    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    now = 1_000_000.0
    (isolated_registry / "issue-976.json").write_text(
        json.dumps(
            {
                "issue": 976,
                "happy_session_id": "sess-976",
                "cwd": "/repo",
                "auto_approve_gpu_hours": 24.0,
                "spawned_at": now - 60,  # 1 min ago on the PASS clock
                "missed": 0,
            }
        )
    )
    _patch_stale_signals(monkeypatch, asw, status="running")

    for _ in range(3):
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == []
    assert _read_stalled_state_845(isolated_registry, 976).get("stop_pending_sid") is None


def test_worktree_hold_defers_stalled_respawn(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (b) plan test 7: fresh worktree activity defers the stalled
    # respawn arm (no stop, no spawn) and increments the bounded hold count.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 977, "sess-977", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and spawns == []
    state = _read_stalled_state_845(isolated_registry, 977)
    assert state["wt_hold_count"] == 1
    assert state["missed"] == 2  # stays armed

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == []
    assert _read_stalled_state_845(isolated_registry, 977)["wt_hold_count"] == 2


def test_worktree_hold_bounded_six_ticks(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (b) plan test 7b: the hold is a BOUND, not a latch — at the cap
    # the arm proceeds (stop phase) despite ongoing activity.
    import json

    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 978, "sess-978", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    now = 1_000_000.0
    # Seed a mid-episode state at the hold cap.
    (isolated_registry / "stalled-978.json").write_text(
        json.dumps(
            {
                "happy_session_id": "sess-978",
                "missed": 1,
                "alerted": False,
                "last_self_report_ts": "ts-old",
                "wt_hold_count": asw.WT_HOLD_MAX_TICKS,
            }
        )
    )
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-978"]  # the 7th tick respawns (stop phase) anyway


def test_crash_pass_worktree_hold_preserves_missed_then_respawns(isolated_registry, monkeypatch):
    # #845 (b) plan test 8: the crash-recovery arm defers a dead-session
    # respawn while the worktree is active, pinning `missed` at the
    # threshold; the first quiet tick respawns.
    import json
    import time as _t

    import autonomous_session_watch as asw

    respawns: list = []
    monkeypatch.setattr(asw, "_respawn", lambda entry, dry_run: respawns.append(entry) or "spawned")
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    activity = {"fresh": True}
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: activity["fresh"])
    path = isolated_registry / "issue-979.json"
    path.write_text(
        json.dumps(
            {
                "issue": 979,
                "happy_session_id": "sess-979",
                "spawned_at": _t.time() - 3600,  # outside the spawn grace
                "missed": 1,
            }
        )
    )
    # Dead sid (not in live_ids) + threshold reached -> respawn action, but
    # the fresh worktree holds it: missed pinned, hold count incremented.
    asw._process_entry(path, live_ids=set(), dry_run=False, threshold=2)
    assert respawns == []
    entry = json.loads(path.read_text())
    assert entry["missed"] == 2  # pinned at the threshold — stays armed
    assert entry["wt_hold_count"] == 1

    # Activity quiets -> the very next tick respawns.
    activity["fresh"] = False
    asw._process_entry(path, live_ids=set(), dry_run=False, threshold=2)
    assert len(respawns) == 1


def test_crash_pass_worktree_hold_bounded(isolated_registry, monkeypatch):
    # #845 (b) plan test 8b: 7 consecutive fresh-activity ticks -> _respawn
    # fires EXACTLY ONCE, on the 7th tick (the cap is a bound, not a latch).
    import json
    import time as _t

    import autonomous_session_watch as asw

    respawns: list = []

    def fake_respawn(entry, dry_run):
        respawns.append(dict(entry))
        return "spawned"

    monkeypatch.setattr(asw, "_respawn", fake_respawn)
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    path = isolated_registry / "issue-980.json"
    path.write_text(
        json.dumps(
            {
                "issue": 980,
                "happy_session_id": "sess-980",
                "spawned_at": _t.time() - 3600,
                "missed": 2,
            }
        )
    )
    for _ in range(7):
        asw._process_entry(path, live_ids=set(), dry_run=False, threshold=2)
    assert len(respawns) == 1
    # The spawn happened on the 7th tick, after 6 held ticks.
    assert respawns[0]["wt_hold_count"] == asw.WT_HOLD_MAX_TICKS


def test_crash_pass_quiet_worktree_unchanged_timing(isolated_registry, monkeypatch):
    # #845 (b) plan test 9 (regression pin): with NO worktree activity, a
    # dead ACTIVE session respawns on exactly today's clock (2 misses).
    import json
    import time as _t

    import autonomous_session_watch as asw

    respawns: list = []
    monkeypatch.setattr(asw, "_respawn", lambda entry, dry_run: respawns.append(entry) or "spawned")
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    path = isolated_registry / "issue-981.json"
    path.write_text(
        json.dumps(
            {
                "issue": 981,
                "happy_session_id": "sess-981",
                "spawned_at": _t.time() - 3600,
                "missed": 0,
            }
        )
    )
    asw._process_entry(path, live_ids=set(), dry_run=False, threshold=2)
    assert respawns == []  # miss 1
    assert json.loads(path.read_text())["missed"] == 1
    asw._process_entry(path, live_ids=set(), dry_run=False, threshold=2)
    assert len(respawns) == 1  # miss 2 -> respawn, exactly as pre-#845


def test_daemon_probe_retries_then_succeeds(monkeypatch):
    # #845 (c) plan test 10: the retry wrapper survives two transient probe
    # failures (backoff sleeps monkeypatched — no real sleep) and reports
    # reachable on the third; all-fail reports unreachable.
    import autonomous_session_watch as asw

    sleeps: list[float] = []
    monkeypatch.setattr(asw.time, "sleep", lambda s: sleeps.append(s))
    seq = iter([False, False, True])
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: next(seq))
    assert asw._daemon_reachable_with_retry(attempts=3) is True
    assert sleeps == [5.0, 10.0]

    sleeps.clear()
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    assert asw._daemon_reachable_with_retry(attempts=3) is False
    assert sleeps == [5.0, 10.0]  # no sleep after the final attempt


def test_daemon_probe_attempts_env_override(monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_DAEMON_PROBE_ATTEMPTS", raising=False)
    assert asw._daemon_probe_attempts() == asw._DAEMON_PROBE_ATTEMPTS_DEFAULT
    monkeypatch.setenv("EPM_DAEMON_PROBE_ATTEMPTS", "1")
    assert asw._daemon_probe_attempts() == 1
    monkeypatch.setenv("EPM_DAEMON_PROBE_ATTEMPTS", "0")
    assert asw._daemon_probe_attempts() == asw._DAEMON_PROBE_ATTEMPTS_DEFAULT
    monkeypatch.setenv("EPM_DAEMON_PROBE_ATTEMPTS", "garbled")
    assert asw._daemon_probe_attempts() == asw._DAEMON_PROBE_ATTEMPTS_DEFAULT


def test_daemon_blocked_two_ticks_one_push(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (c) plan test 11: a respawn-worthy stall deferred by an
    # unreachable daemon for >= 2 consecutive ticks fires EXACTLY ONE push;
    # a reachable tick resets the counter.
    import autonomous_session_watch as asw

    _stops, _spawns, markers = stalled_recorder
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append(msg) or True)
    _write_autonomous_entry(isolated_registry, 982, "sess-982", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    # Tick 1: miss 1 (no alert yet, not counted — not yet respawn-worthy).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    assert pushes == []
    # Tick 2: threshold met, daemon down -> ALERT arm; blocked tick 1.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    assert markers == [(982, "session-stalled-alert")]
    assert pushes == []
    assert _read_stalled_state_845(isolated_registry, 982)["daemon_blocked_ticks"] == 1
    # Tick 3: still blocked -> 2 consecutive ticks -> ONE push.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    assert len(pushes) == 1
    assert "982" in pushes[0]
    # Tick 4: still blocked -> NO repeat push.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    assert len(pushes) == 1
    # Daemon back -> counter resets (and the escalation respawns via the
    # fence, which is covered elsewhere).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    state = _read_stalled_state_845(isolated_registry, 982)
    assert state["daemon_blocked_ticks"] == 0
    assert state["daemon_blocked_pushed"] is False


def test_alerted_escalates_on_first_daemon_up_tick(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #845 (c) plan test 12 (regression pin, extends
    # test_alerted_escalates_to_respawn_when_eligible): an episode alerted
    # while the daemon was down escalates to the respawn action on the very
    # FIRST daemon-up tick — the fence then issues the stop that same tick.
    import autonomous_session_watch as asw

    stops, _spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 983, "sess-983", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=False)
    assert markers == [(983, "session-stalled-alert")]
    assert stops == []

    asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == ["sess-983"]  # queued escalation fired immediately


def test_wedge_bypasses_debounce_but_not_fence(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (e) plan test 16: direct wedge evidence escalates to the respawn
    # arm on the FIRST stale tick (bypassing the 2-miss debounce), but the
    # stop-verify fence still applies (stop first, spawn next tick).
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 984, "sess-984", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: [dequeue] * 3)
    now = 1_000_000.0
    pids = {"sess-984": 4242}

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert stops == ["sess-984"]  # first tick: wedge -> respawn -> fence stop
    assert spawns == []
    state = _read_stalled_state_845(isolated_registry, 984)
    assert state["wedge_hits"] == 1
    assert state["live_consecutive"] == 0  # wedge respawn resets the K counter

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert spawns == [(984, 24.0)]  # verified dead -> spawn
    assert markers and markers[-1] == (984, "session-auto-respawn")


def _wedge_1074_api_error_tail():
    # #1104: the #1074 orchestrator-refusal shape — each refused wake is
    # dequeue x2 + prompt x2 + ONE api-error assistant row (top-level
    # `isApiErrorMessage: true`; sanitized, no refusal text), x3 wakes.
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    prompt = {"type": "user", "message": {"role": "user", "content": "/issue-tick 1074"}}
    api_error = {
        "type": "assistant",
        "isApiErrorMessage": True,
        "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
    }
    return [dequeue, dequeue, prompt, prompt, api_error] * 3


def test_wedge_api_error_tail_escalates_to_respawn(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1104 plan test 6 (mirrors test_wedge_bypasses_debounce_but_not_fence):
    # a #1074-shaped tail of refused wake turns escalates to the respawn arm
    # on the FIRST stale tick — pre-fix, every api-error row classified
    # "assistant" and reset the run, so this tail could never fire.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1074, "sess-1074", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1074_api_error_tail()
    )
    now = 1_000_000.0
    pids = {"sess-1074": 4242}

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert stops == ["sess-1074"]  # first tick: wedge -> respawn -> fence stop
    assert spawns == []
    state = _read_stalled_state_845(isolated_registry, 1074)
    assert state["wedge_hits"] == 1

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert spawns == [(1074, 24.0)]  # verified dead -> spawn
    assert markers and markers[-1] == (1074, "session-auto-respawn")


def test_wedge_api_error_env_kill_switch_disables_production_path(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1104 plan test 7 (reconciled-critic BINDING; the #1021 wiring-test
    # pattern): EPM_TICK_WEDGE_MIN_API_ERRORS=0 must disable the api-error
    # trigger THROUGH THE PRODUCTION PASS — proving
    # _apply_prompt_wedge_override actually calls _tick_wedge_min_api_errors()
    # (an unwired call site running at the keyword default would pass the
    # predicate-level tests and still fire here).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1075, "sess-1075", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1074_api_error_tail()
    )
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_API_ERRORS", "0")
    # #1127 plan-sanctioned edit (§7.13): the #1074 stale tail now ALSO trips
    # the new failed-turn-run trigger (3 failed wake-turns). This test pins
    # "each kill switch disables its OWN trigger class" — each class needs
    # its own switch thrown to silence this tail.
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", "0")
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", "0")
    now = 1_000_000.0

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-1075": 4242},
    )
    assert stops == [] and spawns == []  # kill switch: no wedge escalation
    state = _read_stalled_state_845(isolated_registry, 1075)
    assert state.get("wedge_hits", 0) == 0  # no wedge hit recorded


def _wedge_1127_partial_wake_unit(n_heartbeats=2):
    # #1127: the #1098 (5bdae5b8) / #1090 (5e464f3d) repeating tail unit —
    # [dequeue, prompt, prompt, assistant x n, api-error]: the wake posts
    # mid-turn heartbeats (resetting every ROW-level counter) then dies in an
    # api-error row (sanitized; no refusal text).
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    prompt = {"type": "user", "message": {"role": "user", "content": "/issue-tick 1098"}}
    assistant = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "on it"}]},
    }
    api_error = {
        "type": "assistant",
        "isApiErrorMessage": True,
        "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
    }
    return [dequeue, prompt, prompt, *([assistant] * n_heartbeats), api_error]


def test_wedge_fresh_self_report_failed_turn_tail_escalates(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1127 plan test 9 — THE headline: a FRESH self-report (age 5 min — a
    # dying-but-heartbeating wake keeps refreshing it) no longer gates the
    # turn-level probe; 3 partial-wake units (each with mid-turn heartbeats)
    # escalate to the respawn arm on the first tick (fence stop), spawn on
    # the second (mirrors test_wedge_api_error_tail_escalates_to_respawn).
    # Pre-#1127 this exact setup was a no-op twice over: the lazy gate never
    # probed a fresh session, and the row-level predicate was blind to the
    # partially-successful wake anyway.
    import autonomous_session_watch as asw

    stops, spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1098, "sess-1098", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    monkeypatch.setattr(
        asw,
        "_transcript_tail_rows",
        lambda pid, **_k: [row for _ in range(3) for row in _wedge_1127_partial_wake_unit()],
    )
    now = 1_000_000.0
    pids = {"sess-1098": 4242}

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert stops == ["sess-1098"]  # first tick: wedge -> respawn -> fence stop
    assert spawns == []
    state = _read_stalled_state_845(isolated_registry, 1098)
    assert state["wedge_hits"] == 1

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert spawns == [(1098, 24.0)]  # verified dead -> spawn
    assert markers and markers[-1] == (1098, "session-auto-respawn")


def _wedge_1127_alternating_storm_tail():
    # #1127: the c16b10ca structural shape — 7 timestamped failed turns
    # alternating with ok turns (5 min apart, ~every other wake lost), the
    # NEWEST completed turn failed. Sanitized structural rows only.
    from datetime import UTC, datetime

    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    prompt = {"type": "user", "message": {"role": "user", "content": "/issue-tick 1090"}}
    tail = []
    base = 1_780_000_000.0
    for i in range(13):  # f,o,f,o,...,f -> 7 failed, 6 ok
        ts = datetime.fromtimestamp(base + i * 300, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        resp = {
            "type": "assistant",
            "timestamp": ts,
            "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
        }
        if i % 2 == 0:
            resp["isApiErrorMessage"] = True
        tail.extend([dequeue, prompt, resp])
    return tail


def test_wedge_fresh_self_report_rate_tail_escalates(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1127 plan test 10: a fresh self-report with an alternating-storm tail
    # (>= 6 timestamped failed turns inside the 120-min window, newest
    # completed turn failed) escalates via the failed-turn-rate trigger.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1290, "sess-1290", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1127_alternating_storm_tail()
    )
    now = 1_000_000.0
    pids = {"sess-1290": 4242}

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert stops == ["sess-1290"]  # first tick: rate wedge -> fence stop
    assert spawns == []

    asw.stalled_session_pass(
        dry_run=False, threshold=2, now=now, daemon_reachable=True, pids_by_sid=pids
    )
    assert spawns == [(1290, 24.0)]


def test_wedge_fresh_path_kill_switches_restore_lazy_gate(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1127 plan test 11 (the #1021 wiring-test pattern):
    # EPM_TICK_WEDGE_MIN_FAILED_TURNS=0 + EPM_TICK_WEDGE_MIN_FAILED_TOTAL=0
    # restore the exact pre-#1127 lazy gate on a fresh self-report — no
    # stop, no spawn, wedge_hits == 0, AND the transcript is NEVER probed
    # (proving the production gate calls both env helpers and that 0+0 keeps
    # the zero-probe hot path).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1291, "sess-1291", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    probes: list[int] = []

    def _recording_tail(pid, **_k):
        probes.append(pid)
        return [row for _ in range(3) for row in _wedge_1127_partial_wake_unit()]

    monkeypatch.setattr(asw, "_transcript_tail_rows", _recording_tail)
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", "0")
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", "0")
    now = 1_000_000.0

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-1291": 4242},
    )
    assert stops == [] and spawns == []
    assert probes == []  # zero-probe hot path: _transcript_tail_rows never called
    state = _read_stalled_state_845(isolated_registry, 1291)
    assert state.get("wedge_hits", 0) == 0


def test_wedge_fresh_path_does_not_fire_stale_only_triggers(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1127 plan test 12: the ROW-level triggers stay STALENESS-GATED — on a
    # fresh self-report neither the #779 swallow tail nor a single wake's
    # multi-retry api-error rows escalate (their failure modes freeze the
    # self-report by construction, so the stale gate opens for them there).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    now = 1_000_000.0
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    prompt = {"type": "user", "message": {"role": "user", "content": "/issue-tick 779"}}
    api_error = {
        "type": "assistant",
        "isApiErrorMessage": True,
        "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
    }

    _write_autonomous_entry(isolated_registry, 1292, "sess-1292", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: [dequeue] * 3)
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-1292": 4242},
    )
    assert stops == [] and spawns == []  # #779 shape: dequeue-run is stale-only

    _write_autonomous_entry(isolated_registry, 1293, "sess-1293", cap=24.0)
    monkeypatch.setattr(
        asw,
        "_transcript_tail_rows",
        lambda pid, **_k: [dequeue, prompt, api_error, api_error, api_error],
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-1293": 4242},
    )
    assert stops == [] and spawns == []  # single-wake retries: api-error-run is stale-only


def test_wedge_self_report_age_none_routes_to_fresh_path(monkeypatch):
    # #1127 round-1 critique duty 4 (§7b): `self_report_age is None` routes
    # to the FRESH path (turn-level triggers only) instead of the old
    # no-probe return — a failed-turn tail fires, while the #779 swallow
    # tail (a stale-only row trigger) does not.
    import autonomous_session_watch as asw

    entry = {"happy_session_id": "sess-none", "issue": 1294}
    pids = {"sess-none": 4242}

    monkeypatch.setattr(
        asw,
        "_transcript_tail_rows",
        lambda pid, **_k: [row for _ in range(3) for row in _wedge_1127_partial_wake_unit()],
    )
    action, live, hits, note = asw._apply_prompt_wedge_override(
        issue=1294,
        entry=entry,
        action="keep",
        self_report_age=None,
        respawn_eligible=True,
        pids_by_sid=pids,
        live_consecutive=2,
        wedge_hits=0,
    )
    assert action == "respawn" and live == 0 and hits == 1
    assert note is not None and "failed-turn-run" in note

    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: [dequeue] * 3)
    action, live, hits, note = asw._apply_prompt_wedge_override(
        issue=1294,
        entry=entry,
        action="keep",
        self_report_age=None,
        respawn_eligible=True,
        pids_by_sid=pids,
        live_consecutive=2,
        wedge_hits=0,
    )
    assert action == "keep" and live == 2 and hits == 0 and note is None


def test_wedge_unresolvable_transcript_noop(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (e) plan test 16b: an unresolvable transcript fails toward
    # NO-WEDGE — the slow debounce path is unchanged.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 985, "sess-985", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: None)
    now = 1_000_000.0

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-985": 4242},
    )
    assert stops == [] and spawns == []  # first miss only — no bypass


def test_wedge_respects_worktree_hold(isolated_registry, monkeypatch, stalled_recorder):
    # #845 (e) plan test MF3: a wedge-forced respawn is STILL held by fresh
    # worktree activity (an in-flight implementer's edits win) — no stop.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 986, "sess-986", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: [dequeue] * 3)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    now = 1_000_000.0

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-986": 4242},
    )
    assert stops == [] and spawns == []
    state = _read_stalled_state_845(isolated_registry, 986)
    assert state["wt_hold_count"] == 1
    assert state["wedge_hits"] == 1


def test_wedge_bypasses_marker_window_reaches_fence(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #845 (e) plan test MF3b: a fresh (< 2h) non-watcher marker would keep
    # the slow path quiet, but direct wedge evidence bypasses the marker
    # window and reaches the hold/fence path (stop issued).
    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 987, "sess-987", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0
    # Fresh lifecycle marker 5 min ago (parseable ts, non-watcher).
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda issue: [
            {"kind": "epm:experiment-implementation", "ts": _iso_845(now - 300), "note": "impl"}
        ],
    )
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: [dequeue] * 3)

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-987": 4242},
    )
    assert stops == ["sess-987"]  # wedge beat the fresh-marker keep


def test_wedge_respects_park_exemptions_on_keep_path(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #845 r2 (concern wedge-bypasses-unprobed-park-exemptions): on the
    # fresh-marker keep(0) hot path decide() returns ("keep", 0), so the
    # LAZY park exemptions are never probed (`exempted` is vacuously
    # False) — the wedge escalation must re-probe them ONCE against the
    # escalated action. A firing provision-in-flight exemption VETOES the
    # wedge: no stop, no spawn, and no wedge hit recorded.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 989, "sess-989", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0
    # Fresh (< 2h) lifecycle marker 5 min ago -> the slow path stays quiet
    # (keep/0), exactly the shape where the exemptions go unprobed.
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda issue: [
            {"kind": "epm:experiment-implementation", "ts": _iso_845(now - 300), "note": "impl"}
        ],
    )
    # Wedge transcript signature (>= 3 promptless dequeue rows)...
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, **_k: [dequeue] * 3)
    # ...AND an in-flight provision — the escalation re-probe must fire it
    # (re-patches the stalled_recorder / _patch_stale_signals None stub).
    monkeypatch.setattr(
        asw,
        "_provision_in_flight_reason",
        lambda issue, now: "provision in flight (pod.py provision pid 4321)",
    )

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        pids_by_sid={"sess-989": 4242},
    )
    assert stops == [] and spawns == []  # exemption vetoed the wedge
    state = _read_stalled_state_845(isolated_registry, 989)
    assert state["wedge_hits"] == 0  # a vetoed wedge records no hit
    assert state.get("stop_pending_sid") is None  # fence never armed


# ── #1209 failed-turn-silence wiring (dead-wake stop + day cap) ──────────────


def _wedge_1209_dead_tail(anchor_ts: float):
    # The 8e9c371d / #1092 die-on-turn-1 shape: one delivery burst, mid-turn
    # heartbeat assistant rows, ONE final api-error row at anchor_ts (the
    # silence anchor), one ts-less trailing row. Exactly ONE completed turn
    # (failed); run=0, api_run=1 — below every pre-#1209 threshold.
    # Sanitized structural rows only (the #1104 refusal-safety contract).
    dequeue = {"type": "queue-operation", "operation": "dequeue"}
    prompt = {"type": "user", "message": {"role": "user", "content": "/issue 1092"}}
    heartbeat = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
    }
    api_error = {
        "type": "assistant",
        "isApiErrorMessage": True,
        "timestamp": _iso_845(anchor_ts),
        "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
    }
    return [dequeue, prompt, heartbeat, heartbeat, api_error, {"type": "last-prompt"}]


def _dead_silence_day_key(now: float) -> str:
    import time as _t

    return _t.strftime("%Y-%m-%d", _t.gmtime(now))


def _write_stalled_state_1209(reg_dir, issue, sid, **fields):
    """Seed a production-shaped stalled-<N>.json (every current field at its
    default; last_self_report_ts matches _patch_stale_signals' "ts-old" so a
    later pass reads NO advancement unless a test re-patches the report)."""
    import json

    payload = {
        "happy_session_id": sid,
        "missed": 0,
        "alerted": False,
        "respawn_count": 0,
        "exhausted": False,
        "refresh_attempted": False,
        "followups_child_alerted": False,
        "live_consecutive": 0,
        "stop_pending_sid": None,
        "stop_pending_ts": None,
        "stop_retried": False,
        "stop_failed_alerted": False,
        "wt_hold_count": 0,
        "daemon_blocked_ticks": 0,
        "daemon_blocked_pushed": False,
        "wedge_hits": 0,
        "dead_silence_respawn_day": None,
        "dead_silence_respawns_today": 0,
        "last_self_report_ts": "ts-old",
        "first_seen": 999_000.0,
    }
    payload.update(fields)
    (reg_dir / f"stalled-{issue}.json").write_text(json.dumps(payload))


def test_wedge_dead_silence_stop_then_crash_arm_handoff(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1209 T10 — the headline production thread: a FRESH boot self-report
    # (the die-on-turn-1 shape refreshes it once at boot) + the #1092 dead
    # tail escalate to the fence STOP on the first tick (bump + durable
    # stop marker), and on the next tick — the sid gone from the daemon
    # /list, hence absent from BOTH live_ids and pids_by_sid (production
    # derives both from ONE /list; a sid-dead-but-pid-live state is
    # production-impossible) — the STALLED lane issues NO spawn and leaves
    # the fence state intact: the CRASH-RECOVERY arm owns the respawn
    # (existing behavior). Also catches a forgotten now / dead_silence_s /
    # respawn_count production thread (an inert trigger -> no stop -> fail).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    notes: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: notes.append((issue, label, note)),
    )
    _write_autonomous_entry(isolated_registry, 1209, "sess-1209", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1209"},
        pids_by_sid={"sess-1209": 4242},
    )
    assert stops == ["sess-1209"]  # the trigger's act: the fence STOP
    assert spawns == []
    assert [n for n in notes if n[1] == "session-dead-silence-stop"]
    assert "[failed-turn-silence]" in notes[-1][2]
    state = _read_stalled_state_845(isolated_registry, 1209)
    assert state["dead_silence_respawns_today"] == 1  # stop-initiation bump
    assert state["dead_silence_respawn_day"] == _dead_silence_day_key(now)
    assert state["stop_pending_sid"] == "sess-1209"

    # Next tick: the stop landed — sid absent from BOTH daemon-derived maps.
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now + 600,
        daemon_reachable=True,
        live_ids=set(),
        pids_by_sid={},
    )
    assert spawns == []  # NO stalled-lane spawn — the crash arm owns it
    assert stops == ["sess-1209"]  # and no second stop
    state = _read_stalled_state_845(isolated_registry, 1209)
    assert state["stop_pending_sid"] == "sess-1209"  # fence state intact
    assert state["dead_silence_respawns_today"] == 1  # no re-bump


def test_wedge_dead_silence_caps_block_production_path(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1209 T11 — disk-seeded cap binding through the FULL production pass:
    # (a) day cap exhausted (dead_silence_respawns_today=3 under the
    # injected-now UTC day key) -> the trigger is disarmed, NO fence stop;
    # (b) episode belt (respawn_count=3, fresh episode fields) -> NO stop;
    # (c) dry_run=True on the FIRING path performs ZERO disk writes (every
    # new-path mutation rides _persist_stalled_ctx, which no-ops on dry_run).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )

    # (a) day cap exhausted.
    _write_autonomous_entry(isolated_registry, 1301, "sess-1301", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1301,
        "sess-1301",
        dead_silence_respawn_day=_dead_silence_day_key(now),
        dead_silence_respawns_today=3,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1301"},
        pids_by_sid={"sess-1301": 4242},
    )
    assert stops == [] and spawns == []
    state = _read_stalled_state_845(isolated_registry, 1301)
    assert state.get("stop_pending_sid") is None  # fence never armed
    assert state["dead_silence_respawns_today"] == 3  # keep-path preserved it

    # (b) episode belt: respawn_count at STALLED_MAX_RESPAWNS disarms too.
    _write_autonomous_entry(isolated_registry, 1302, "sess-1302", cap=24.0)
    _write_stalled_state_1209(isolated_registry, 1302, "sess-1302", respawn_count=3)
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1302"},
        pids_by_sid={"sess-1302": 4242},
    )
    assert stops == [] and spawns == []
    assert _read_stalled_state_845(isolated_registry, 1302).get("stop_pending_sid") is None

    # (c) dry-run write-safety on the FIRING path: zero disk writes.
    _write_autonomous_entry(isolated_registry, 1303, "sess-1303", cap=24.0)
    asw.stalled_session_pass(
        dry_run=True,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1303"},
        pids_by_sid={"sess-1303": 4242},
    )
    assert not list(isolated_registry.glob("stalled-1303*"))  # nothing persisted


def test_wedge_dead_silence_day_counter_bumps_once_per_episode_at_stop(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1209 T12 — the bump site is STOP-INITIATION (the stop_pending_sid
    # None -> sid transition) keyed on the bracketed [failed-turn-silence]
    # substring of the PRODUCTION wedge-note template: (a) a dead-silence
    # stop bumps once, day key derived from the injected ctx.now (never
    # wall-clock); (b) a retry-stop tick (pending sid already set) never
    # re-bumps; (c) a [failed-turn-run] wedge stop never bumps.
    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    notes: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: notes.append((issue, label, note)),
    )
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )
    # The PRODUCTION note builder pins the bracketed-reason format.
    action, _live, _hits, note = asw._apply_prompt_wedge_override(
        issue=1401,
        entry={"happy_session_id": "sess-1401", "issue": 1401},
        action="keep",
        self_report_age=300.0,
        respawn_eligible=True,
        pids_by_sid={"sess-1401": 4242},
        live_consecutive=0,
        wedge_hits=0,
        now=now,
    )
    assert action == "respawn"
    assert note is not None and note.startswith("prompt-wedge trigger [failed-turn-silence]")

    def _ctx(issue, sid, **kw):
        return asw._StalledActionCtx(
            issue=issue,
            happy_session_id=sid,
            prev_state={},
            alerted=False,
            respawn_count=0,
            exhausted=False,
            last_self_report_ts="ts-old",
            self_gap="5.0m",
            marker_gap="5.0m",
            has_pod=False,
            task_status="running",
            in_active=True,
            threshold=2,
            dry_run=False,
            now=now,
            **kw,
        )

    # (a) stop-initiation bumps ONCE; day key from the injected now.
    asw._handle_stalled_respawn(_ctx(1401, "sess-1401", live_ids={"sess-1401"}, wedge_note=note))
    assert stops == ["sess-1401"]
    state = _read_stalled_state_845(isolated_registry, 1401)
    assert state["dead_silence_respawns_today"] == 1
    assert state["dead_silence_respawn_day"] == _dead_silence_day_key(now)
    assert [n for n in notes if n[1] == "session-dead-silence-stop"]

    # (b) a RETRY-STOP tick (pending sid set) never re-bumps.
    notes.clear()
    asw._handle_stalled_respawn(
        _ctx(
            1401,
            "sess-1401",
            live_ids={"sess-1401"},
            wedge_note=note,
            stop_pending_sid="sess-1401",
            stop_pending_ts=now,
            dead_silence_respawn_day=_dead_silence_day_key(now),
            dead_silence_respawns_today=1,
        )
    )
    state = _read_stalled_state_845(isolated_registry, 1401)
    assert state["dead_silence_respawns_today"] == 1  # exactly-once per episode
    assert not [n for n in notes if n[1] == "session-dead-silence-stop"]

    # (c) a [failed-turn-run] wedge stop never bumps the dead-silence cap.
    asw._handle_stalled_respawn(
        _ctx(
            1402,
            "sess-1402",
            live_ids={"sess-1402"},
            wedge_note="prompt-wedge trigger [failed-turn-run] in the transcript tail (...)",
        )
    )
    state = _read_stalled_state_845(isolated_registry, 1402)
    assert state["dead_silence_respawns_today"] == 0
    assert not [n for n in notes if n[1] == "session-dead-silence-stop"]


def test_wedge_dead_silence_exemption_veto(isolated_registry, monkeypatch, stalled_recorder):
    # #1209 T13 (mirrors the #845 r2 park-exemption test): a firing
    # provision-in-flight exemption VETOES the dead-silence wedge — no stop,
    # no wedge hit, fence never armed, day counter untouched (it bumps only
    # at stop-initiation, which the veto prevents).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 1405, "sess-1405", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )
    monkeypatch.setattr(
        asw,
        "_provision_in_flight_reason",
        lambda issue, now: "provision in flight (pod.py provision pid 4321)",
    )

    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1405"},
        pids_by_sid={"sess-1405": 4242},
    )
    assert stops == [] and spawns == []
    state = _read_stalled_state_845(isolated_registry, 1405)
    assert state["wedge_hits"] == 0
    assert state.get("stop_pending_sid") is None
    assert state["dead_silence_respawns_today"] == 0


def test_wedge_dead_silence_day_state_survives_advancement(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1209 T14 — the day-cap fields are advancement-clear-EXEMPT: a boot
    # self-report ADVANCE (each die-on-turn-1 generation writes one) clears
    # the #845 episode fields but NOT the day counter, so a seeded
    # at-the-cap counter still disarms the trigger (a); a STALE day key
    # reads 0 and re-arms (b).
    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )
    # The self-report ADVANCED since the seeded state ("ts-boot-9" > "ts-boot-1").
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (300.0, "ts-boot-9"))

    # (a) same-day counter at the cap SURVIVES the advancement clear.
    _write_autonomous_entry(isolated_registry, 1406, "sess-1406", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1406,
        "sess-1406",
        last_self_report_ts="ts-boot-1",
        dead_silence_respawn_day=_dead_silence_day_key(now),
        dead_silence_respawns_today=3,
        stop_pending_sid="sess-1406",  # stale fence state, advancement-cleared
        respawn_count=2,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1406"},
        pids_by_sid={"sess-1406": 4242},
    )
    assert stops == []  # still disarmed: the day counter survived
    state = _read_stalled_state_845(isolated_registry, 1406)
    assert state["dead_silence_respawns_today"] == 3
    assert state.get("stop_pending_sid") is None  # the #845 fields DID clear

    # (b) a STALE day key reads 0 -> re-armed -> the stop fires.
    _write_autonomous_entry(isolated_registry, 1407, "sess-1407", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1407,
        "sess-1407",
        last_self_report_ts="ts-boot-1",
        dead_silence_respawn_day="1970-01-01",  # a rolled-over day
        dead_silence_respawns_today=3,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1407"},
        pids_by_sid={"sess-1407": 4242},
    )
    assert stops == ["sess-1407"]
    state = _read_stalled_state_845(isolated_registry, 1407)
    assert state["dead_silence_respawns_today"] == 1  # fresh day, first episode


def test_wedge_dead_silence_two_generation_loop_counts_and_cap_disarms(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1209 T15 — the cross-generation loop the day cap exists for: two
    # die-on-turn-1 generations (stalled-pass stop -> sid absent from both
    # daemon maps -> crash-arm spawn simulated -> new generation's boot
    # self-report + fresh dead tail) book dead_silence_respawns_today == 2,
    # and with the counter seeded at the cap generation 3 is NEVER
    # escalated. Daemon-map fixtures stay production-consistent (a sid is
    # in BOTH live_ids and pids_by_sid, or in NEITHER).
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )

    # Generation 1: boot self-report ts-gen1, dies on turn 1, goes silent.
    _write_autonomous_entry(isolated_registry, 1408, "sess-a", cap=24.0)
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (300.0, "ts-gen1"))
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-a"},
        pids_by_sid={"sess-a": 1111},
    )
    assert stops == ["sess-a"]
    assert _read_stalled_state_845(isolated_registry, 1408)["dead_silence_respawns_today"] == 1

    # The stop lands; the CRASH ARM (not this lane) spawns generation 2,
    # which writes a fresh boot self-report (ADVANCED) and dies on turn 1.
    _write_autonomous_entry(isolated_registry, 1408, "sess-b", cap=24.0)
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (300.0, "ts-gen2"))
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-b"},
        pids_by_sid={"sess-b": 2222},
    )
    assert stops == ["sess-a", "sess-b"]  # generation 2 stopped too
    state = _read_stalled_state_845(isolated_registry, 1408)
    assert state["dead_silence_respawns_today"] == 2  # cross-generation count

    # Generation 3 at the seeded cap: NEVER escalated.
    import json

    state_path = isolated_registry / "stalled-1408.json"
    payload = json.loads(state_path.read_text())
    payload["dead_silence_respawns_today"] = 3
    state_path.write_text(json.dumps(payload))
    _write_autonomous_entry(isolated_registry, 1408, "sess-c", cap=24.0)
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (300.0, "ts-gen3"))
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-c"},
        pids_by_sid={"sess-c": 3333},
    )
    assert stops == ["sess-a", "sess-b"]  # no third stop
    assert spawns == []  # the stalled lane never spawned in this whole loop
    assert _read_stalled_state_845(isolated_registry, 1408)["dead_silence_respawns_today"] == 3


# ── #1241 four-trigger cap parity (episode belt + shared day-keyed cap) ──────


def _wedge_1241_failed_turn_tail():
    # Three #1127 partial-wake units fire `failed-turn-run` on BOTH
    # self-report paths. The rows carry NO timestamps, so the #1209
    # dead-silence trigger stays NO-FIRE on this tail (ts-less anchor) —
    # letting the tests exercise the two day budgets independently.
    return [row for _ in range(3) for row in _wedge_1127_partial_wake_unit()]


def test_wedge_four_triggers_respect_episode_belt(isolated_registry, monkeypatch, stalled_recorder):
    # #1241 criterion 1 — the episode belt gates ALL the pre-#1209 triggers.
    # Kwarg truth-table legs (direct override calls) for the stale
    # (dequeue-run) and fresh (failed-turn-run) paths, plus a DISK-SEEDED
    # full-production belt leg. The belt disarms BOTH trigger families, so
    # the early quiet-return must also skip the 256 KB transcript read.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    calls = {"n": 0}

    def _tail(rows):
        def _fn(pid, **_k):
            calls["n"] += 1
            return rows

        return _fn

    entry = {"happy_session_id": "sess-1601", "issue": 1601}
    pids = {"sess-1601": 4242}
    dequeue = {"type": "queue-operation", "operation": "dequeue"}

    # Control (fire-capable seed shapes): belt open -> both tails fire.
    monkeypatch.setattr(asw, "_transcript_tail_rows", _tail([dequeue] * 3))
    action, _live, _hits, note = asw._apply_prompt_wedge_override(
        issue=1601,
        entry=entry,
        action="keep",
        self_report_age=STALLED_WINDOW_S + 60,  # STALE path
        respawn_eligible=True,
        pids_by_sid=pids,
        live_consecutive=0,
        wedge_hits=0,
        respawn_count=0,
    )
    assert action == "respawn" and note is not None and "dequeue-run" in note

    # Belt exhausted -> the STALE dequeue-run tail produces NO escalation,
    # and the transcript read is skipped (early quiet-return).
    calls["n"] = 0
    action, live, hits, note = asw._apply_prompt_wedge_override(
        issue=1601,
        entry=entry,
        action="keep",
        self_report_age=STALLED_WINDOW_S + 60,
        respawn_eligible=True,
        pids_by_sid=pids,
        live_consecutive=2,
        wedge_hits=0,
        respawn_count=asw.STALLED_MAX_RESPAWNS,
    )
    assert (action, live, hits, note) == ("keep", 2, 0, None)
    assert calls["n"] == 0  # quiet-return fired BEFORE the 256 KB read

    # Belt exhausted -> the FRESH failed-turn tail produces NO escalation.
    monkeypatch.setattr(asw, "_transcript_tail_rows", _tail(_wedge_1241_failed_turn_tail()))
    calls["n"] = 0
    action, live, hits, note = asw._apply_prompt_wedge_override(
        issue=1601,
        entry=entry,
        action="keep",
        self_report_age=300.0,  # FRESH path
        respawn_eligible=True,
        pids_by_sid=pids,
        live_consecutive=2,
        wedge_hits=0,
        respawn_count=asw.STALLED_MAX_RESPAWNS,
    )
    assert (action, live, hits, note) == ("keep", 2, 0, None)
    assert calls["n"] == 0

    # DISK-SEEDED full-production belt leg (the #1209 T11 shape): seeded
    # respawn_count == STALLED_MAX_RESPAWNS + a wedge-shaped tail -> NO
    # stop, fence never armed, and zero transcript reads.
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    monkeypatch.setattr(asw, "_transcript_tail_rows", _tail(_wedge_1241_failed_turn_tail()))
    now = 1_000_000.0
    _write_autonomous_entry(isolated_registry, 1601, "sess-1601", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry, 1601, "sess-1601", respawn_count=asw.STALLED_MAX_RESPAWNS
    )
    calls["n"] = 0
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1601"},
        pids_by_sid={"sess-1601": 4242},
    )
    assert stops == [] and spawns == []
    state = _read_stalled_state_845(isolated_registry, 1601)
    assert state.get("stop_pending_sid") is None  # fence never armed
    assert calls["n"] == 0  # belt disarms both families before the read


def test_wedge_four_triggers_respect_day_cap(isolated_registry, monkeypatch, stalled_recorder):
    # #1241 criterion 2 — DISK-SEEDED FULL-PRODUCTION-PASS day-cap binding
    # (the #1209 T11 mirror; NOT a direct-kwarg call): this pins the WHOLE
    # _day_scoped_count load -> exemption-probe kwarg -> override kwarg
    # thread — the kwarg default (0) is permissive/armed, so a dropped
    # threading link would ship the fix INERT-toward-uncapped while
    # direct-kwarg tests stay green.
    import autonomous_session_watch as asw

    stops, spawns, _markers = stalled_recorder
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1241_failed_turn_tail()
    )

    # (A) fire-capability control: no seeded counter -> the pass STOPS and
    # books the wedge day counter (proves the tail + thread are live, so
    # leg B's no-stop below is the CAP binding, not an inert fixture).
    _write_autonomous_entry(isolated_registry, 1611, "sess-1611", cap=24.0)
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1611"},
        pids_by_sid={"sess-1611": 4242},
    )
    assert stops == ["sess-1611"]
    state = _read_stalled_state_845(isolated_registry, 1611)
    assert state["wedge_respawns_today"] == 1
    assert state["wedge_respawn_day"] == _dead_silence_day_key(now)
    (isolated_registry / "issue-1611.json").unlink()  # keep later passes single-issue

    # (B) day cap exhausted under the injected-now UTC day key -> the four
    # triggers are disarmed: NO stop, fence never armed, counter preserved.
    _write_autonomous_entry(isolated_registry, 1612, "sess-1612", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1612,
        "sess-1612",
        wedge_respawn_day=_dead_silence_day_key(now),
        wedge_respawns_today=3,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1612"},
        pids_by_sid={"sess-1612": 4242},
    )
    assert stops == ["sess-1611"] and spawns == []  # no second stop
    state = _read_stalled_state_845(isolated_registry, 1612)
    assert state.get("stop_pending_sid") is None  # fence never armed
    assert state["wedge_respawns_today"] == 3  # keep-path preserved it
    (isolated_registry / "issue-1612.json").unlink()

    # (C) budget independence, direction 1: the wedge day cap exhausted +
    # a DEAD-SILENCE tail with dead_silence_respawns_today == 0 -> the
    # #1209 trigger still fires, and the bump lands on ITS budget only.
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1209_dead_tail(now - 21 * 60)
    )
    _write_autonomous_entry(isolated_registry, 1613, "sess-1613", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1613,
        "sess-1613",
        wedge_respawn_day=_dead_silence_day_key(now),
        wedge_respawns_today=3,
        dead_silence_respawns_today=0,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1613"},
        pids_by_sid={"sess-1613": 4242},
    )
    assert stops == ["sess-1611", "sess-1613"]
    state = _read_stalled_state_845(isolated_registry, 1613)
    assert state["dead_silence_respawns_today"] == 1  # #1209 budget bumped
    assert state["wedge_respawns_today"] == 3  # #1241 budget untouched


def test_wedge_day_cap_independent_of_dead_silence_cap(monkeypatch):
    # #1241 criterion 3 (budget independence, direction 2 — pure kwarg
    # truth table): the #1209 budget exhausted + the #1241 budget open ->
    # failed-turn-run still fires.
    import autonomous_session_watch as asw

    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1241_failed_turn_tail()
    )
    action, live, hits, note = asw._apply_prompt_wedge_override(
        issue=1621,
        entry={"happy_session_id": "sess-1621", "issue": 1621},
        action="keep",
        self_report_age=300.0,
        respawn_eligible=True,
        pids_by_sid={"sess-1621": 4242},
        live_consecutive=2,
        wedge_hits=0,
        respawn_count=0,
        dead_silence_respawns_today=3,  # #1209 budget exhausted
        wedge_respawns_today=0,  # #1241 budget open
    )
    assert action == "respawn" and live == 0 and hits == 1
    assert note is not None and "failed-turn-run" in note


def test_fence_stop_bumps_wedge_day_counter_once_per_episode(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1241 criterion 4 — DISK-SEEDED full-production-pass persistence: a
    # fence stop with a non-[failed-turn-silence] wedge note bumps the
    # #1241 counter ONCE at stop-initiation (day key from the injected
    # now), never touches the #1209 counter, a retry-stop tick does not
    # re-bump, and a k -> k+1 leg pins the LOAD (a broken load reading 0
    # would still pass a 0 -> 1-only assertion).
    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)  # FRESH
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1241_failed_turn_tail()
    )

    # (a) stop-initiation bumps ONCE; the #1209 budget is untouched.
    _write_autonomous_entry(isolated_registry, 1631, "sess-1631", cap=24.0)
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1631"},
        pids_by_sid={"sess-1631": 4242},
    )
    assert stops == ["sess-1631"]
    state = _read_stalled_state_845(isolated_registry, 1631)
    assert state["wedge_respawns_today"] == 1
    assert state["wedge_respawn_day"] == _dead_silence_day_key(now)
    assert state["dead_silence_respawns_today"] == 0  # other budget untouched
    assert state["stop_pending_sid"] == "sess-1631"

    # (b) a RETRY-STOP tick (pending sid set, sid still live) never re-bumps.
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now + 600,
        daemon_reachable=True,
        live_ids={"sess-1631"},
        pids_by_sid={"sess-1631": 4242},
    )
    assert stops == ["sess-1631", "sess-1631"]  # the one allowed stop retry
    state = _read_stalled_state_845(isolated_registry, 1631)
    assert state["wedge_respawns_today"] == 1  # exactly-once per episode
    assert state["stop_retried"] is True
    (isolated_registry / "issue-1631.json").unlink()

    # (c) k -> k+1: a PRIOR persisted count under today's key loads and the
    # bump lands k+1 (pins the _day_scoped_count load end-to-end).
    _write_autonomous_entry(isolated_registry, 1632, "sess-1632", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1632,
        "sess-1632",
        wedge_respawn_day=_dead_silence_day_key(now),
        wedge_respawns_today=1,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1632"},
        pids_by_sid={"sess-1632": 4242},
    )
    assert stops[-1] == "sess-1632"
    assert _read_stalled_state_845(isolated_registry, 1632)["wedge_respawns_today"] == 2


def test_wedge_day_counter_survives_advancement_clear(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #1241 criterion 5 — DISK-SEEDED full-production-pass persistence:
    # (a) a self-report ADVANCE clears the #845 hardening fields +
    # respawn_count but NOT wedge_respawns_today / wedge_respawn_day (the
    # at-cap counter still disarms the four triggers); (b) a rolled day
    # key reads as 0 and re-arms; (c) a pre-#1241 state file WITHOUT the
    # new fields loads as (None, 0) = armed (backward compat).
    import autonomous_session_watch as asw

    stops, _spawns, _markers = stalled_recorder
    _patch_stale_signals(monkeypatch, asw, status="running", age_s=300)
    now = 1_000_000.0
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, **_k: _wedge_1241_failed_turn_tail()
    )

    # (a) advancement ("ts-boot-9" > seeded "ts-boot-1") clears the episode
    # fields but the at-cap day counter SURVIVES and keeps the four
    # triggers disarmed.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (300.0, "ts-boot-9"))
    _write_autonomous_entry(isolated_registry, 1641, "sess-1641", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1641,
        "sess-1641",
        last_self_report_ts="ts-boot-1",
        wedge_respawn_day=_dead_silence_day_key(now),
        wedge_respawns_today=3,
        stop_pending_sid="sess-1641",  # stale fence state, advancement-cleared
        respawn_count=2,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1641"},
        pids_by_sid={"sess-1641": 4242},
    )
    assert stops == []  # still disarmed: the day counter survived
    state = _read_stalled_state_845(isolated_registry, 1641)
    assert state["wedge_respawns_today"] == 3
    assert state["wedge_respawn_day"] == _dead_silence_day_key(now)
    assert state.get("stop_pending_sid") is None  # the #845 fields DID clear
    (isolated_registry / "issue-1641.json").unlink()

    # (b) a STALE day key reads 0 -> re-armed -> the stop fires + bumps.
    _write_autonomous_entry(isolated_registry, 1642, "sess-1642", cap=24.0)
    _write_stalled_state_1209(
        isolated_registry,
        1642,
        "sess-1642",
        wedge_respawn_day="1970-01-01",  # a rolled-over day
        wedge_respawns_today=3,
    )
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1642"},
        pids_by_sid={"sess-1642": 4242},
    )
    assert stops == ["sess-1642"]
    assert _read_stalled_state_845(isolated_registry, 1642)["wedge_respawns_today"] == 1
    (isolated_registry / "issue-1642.json").unlink()

    # (c) absent-keys backward compat: a pre-#1241 stalled-<N>.json (the
    # seed helper's baseline payload carries NO wedge_respawn_* keys) loads
    # as (None, 0) = armed -> the stop fires.
    _write_autonomous_entry(isolated_registry, 1643, "sess-1643", cap=24.0)
    _write_stalled_state_1209(isolated_registry, 1643, "sess-1643")
    raw = asw._load_stalled_state(1643)
    assert "wedge_respawns_today" not in raw and "wedge_respawn_day" not in raw
    asw.stalled_session_pass(
        dry_run=False,
        threshold=2,
        now=now,
        daemon_reachable=True,
        live_ids={"sess-1643"},
        pids_by_sid={"sess-1643": 4242},
    )
    assert stops == ["sess-1642", "sess-1643"]
    assert _read_stalled_state_845(isolated_registry, 1643)["wedge_respawns_today"] == 1


def test_tick_wedge_respawns_per_day_knob(monkeypatch):
    # #1241 criterion 6 — env parse (mirrors the #1209 T9 helper contract):
    # default 3; positive int honored; `< 1 -> default` (disabling a
    # trigger is its own EPM_TICK_WEDGE_MIN_* knob's job, never the
    # cap's); malformed -> default. Never a kill switch.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_TICK_WEDGE_RESPAWNS_PER_DAY", raising=False)
    assert asw._tick_wedge_respawns_per_day() == 3
    monkeypatch.setenv("EPM_TICK_WEDGE_RESPAWNS_PER_DAY", "5")
    assert asw._tick_wedge_respawns_per_day() == 5
    monkeypatch.setenv("EPM_TICK_WEDGE_RESPAWNS_PER_DAY", "0")
    assert asw._tick_wedge_respawns_per_day() == 3
    monkeypatch.setenv("EPM_TICK_WEDGE_RESPAWNS_PER_DAY", "-2")
    assert asw._tick_wedge_respawns_per_day() == 3
    monkeypatch.setenv("EPM_TICK_WEDGE_RESPAWNS_PER_DAY", "junk")
    assert asw._tick_wedge_respawns_per_day() == 3


def test_marker_window_blocks_stalled_pass_alert_90min_marker(
    isolated_registry, monkeypatch, stalled_recorder
):
    # #845 (a-i) pass-level companion of the pure test: a 90-min-old REAL
    # marker (stale under the old 60-min window, fresh under the 2h one)
    # keeps the session quiet with a frozen self-report.
    import autonomous_session_watch as asw

    stops, _spawns, markers = stalled_recorder
    _write_autonomous_entry(isolated_registry, 988, "sess-988", cap=24.0)
    _patch_stale_signals(monkeypatch, asw, status="running")
    now = 1_000_000.0
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda issue: [{"kind": "epm:progress", "ts": _iso_845(now - 90 * 60), "note": "step"}],
    )
    for _ in range(3):
        asw.stalled_session_pass(dry_run=False, threshold=2, now=now, daemon_reachable=True)
    assert stops == [] and markers == []


# ── #845 (d) stale-registration pass ─────────────────────────────────────────


def _stale_reg_env(monkeypatch, asw, *, idle_s=16 * 3600, self_report_s=16 * 3600):
    """Common monkeypatching for the stale-registration pass tests."""
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (idle_s, None))
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (self_report_s, "ts"))
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")


def test_stale_registration_pass_unlinks_and_posts_marker(isolated_registry, monkeypatch):
    # #845 (d) plan test 14: a live 16h-transcript-idle registration is
    # unregistered (file unlinked) with a one-time marker naming the task
    # status; a dead-sid sibling stays (crash-recovery property); manual
    # registrations are in scope too.
    import autonomous_session_watch as asw

    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label, note)),
    )
    _stale_reg_env(monkeypatch, asw)
    _write_autonomous_entry(isolated_registry, 990, "sess-990")
    _write_manual_entry(isolated_registry, 991, "sess-991-manual")
    _write_autonomous_entry(isolated_registry, 992, "sess-992-dead")
    children = [
        {"happySessionId": "sess-990", "pid": 4242},
        {"happySessionId": "sess-991-manual", "pid": 4243},
        # sess-992-dead is NOT live.
    ]
    asw.stale_registration_pass(dry_run=False, children=children, now=1_000_000.0)

    assert not (isolated_registry / "issue-990.json").exists()
    assert not (isolated_registry / "manual-issue-991.json").exists()
    assert (isolated_registry / "issue-992.json").exists()  # dead sid kept
    labels = [(i, la) for i, la, _n in markers]
    assert (990, "stale-registration-unregister") in labels
    assert (991, "stale-registration-unregister") in labels
    note_990 = next(n for i, _la, n in markers if i == 990)
    assert "status=running" in note_990  # the marker logs the task status
    assert "NOT stopped" in note_990
    # Durable sidecar trace appended.
    events = (isolated_registry / "stale-registration-events.jsonl").read_text().splitlines()
    assert len(events) == 2


def test_stale_registration_pass_daemon_down_noop(isolated_registry, monkeypatch):
    # #845 (d) plan test 14b: children=None (daemon unreachable) => no-op.
    import autonomous_session_watch as asw

    _stale_reg_env(monkeypatch, asw)
    _write_autonomous_entry(isolated_registry, 990, "sess-990")
    asw.stale_registration_pass(dry_run=False, children=None, now=1_000_000.0)
    assert (isolated_registry / "issue-990.json").exists()


def test_stale_registration_pass_dry_run(isolated_registry, monkeypatch):
    # #845 (d) plan test 14c: dry-run reports but never unlinks.
    import autonomous_session_watch as asw

    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label, dry_run)),
    )
    _stale_reg_env(monkeypatch, asw)
    _write_autonomous_entry(isolated_registry, 990, "sess-990")
    asw.stale_registration_pass(
        dry_run=True, children=[{"happySessionId": "sess-990", "pid": 4242}], now=1_000_000.0
    )
    assert (isolated_registry / "issue-990.json").exists()  # never unlinked
    assert markers == [(990, "stale-registration-unregister", True)]
    assert not (isolated_registry / "stale-registration-events.jsonl").exists()


def test_stale_registration_pass_provision_in_flight_keeps(isolated_registry, monkeypatch):
    # #845 (d) plan test MF3c: a 16h-idle transcript with an in-flight
    # provision is WORKING, not stale — keep.
    import autonomous_session_watch as asw

    _stale_reg_env(monkeypatch, asw)
    monkeypatch.setattr(
        asw, "_provision_in_flight_reason", lambda issue, now: "live provision pid 1"
    )
    _write_autonomous_entry(isolated_registry, 990, "sess-990")
    asw.stale_registration_pass(
        dry_run=False, children=[{"happySessionId": "sess-990", "pid": 4242}], now=1_000_000.0
    )
    assert (isolated_registry / "issue-990.json").exists()


def test_stale_registration_pass_fresh_worktree_keeps(isolated_registry, monkeypatch):
    # #845 (d) plan test MF3d: fresh worktree activity = not stale — keep.
    import autonomous_session_watch as asw

    _stale_reg_env(monkeypatch, asw)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    _write_autonomous_entry(isolated_registry, 990, "sess-990")
    asw.stale_registration_pass(
        dry_run=False, children=[{"happySessionId": "sess-990", "pid": 4242}], now=1_000_000.0
    )
    assert (isolated_registry / "issue-990.json").exists()


def test_stale_registration_pass_unresolvable_transcript_keeps(isolated_registry, monkeypatch):
    # #845 (d): unresolvable transcript fails toward keep.
    import autonomous_session_watch as asw

    _stale_reg_env(monkeypatch, asw)
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (None, "no happy log"))
    _write_autonomous_entry(isolated_registry, 990, "sess-990")
    asw.stale_registration_pass(
        dry_run=False, children=[{"happySessionId": "sess-990", "pid": 4242}], now=1_000_000.0
    )
    assert (isolated_registry / "issue-990.json").exists()


def test_main_order_stale_registration_after_gate_push(isolated_registry, monkeypatch):
    # #845 (d) plan test 17: the stale-registration pass runs AFTER
    # gate_push_pass (the gate-push-before-reaper ordering is a documented
    # runaway-force-stop invariant), adjacent to the two reapers, and all
    # THREE consumers share the ONE post-gate-push /list snapshot (the
    # reaper_children computation is UNMOVED).
    import autonomous_session_watch as asw

    order: list[tuple[str, object]] = []
    snapshot = [{"happySessionId": "sid-shared", "pid": 12345}]
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    monkeypatch.setattr(asw, "_live_children", lambda: snapshot)
    # #1247 r3: shared hermeticity helper FIRST — adds cpu_guard /
    # verdict_disagree / vm_ledger_reap (absent from the loop below); the
    # helper also stubs gate_push_pass, but the gate_push recorder is patched
    # AFTER the helper, so the recorder wins.
    _stub_fleet_mutating_passes(asw, monkeypatch)
    for name in (
        "vm_disk_pass",
        "data_disk_pass",
        "happy_patch_pass",
        "triage_observer_pass",
        "program_orchestrator_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "proposed_infra_sweep_pass",
        "capacity_retry_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "gc_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(asw, "gate_push_pass", lambda *a, **kw: order.append(("gate_push", None)))
    monkeypatch.setattr(
        asw,
        "stale_registration_pass",
        lambda *a, **kw: order.append(("stale_registration", kw.get("children"))),
    )
    monkeypatch.setattr(
        asw, "zombie_wrapper_pass", lambda *a, **kw: order.append(("zombie", kw.get("children")))
    )
    monkeypatch.setattr(
        asw, "idle_unmapped_pass", lambda *a, **kw: order.append(("idle", kw.get("children")))
    )

    rc = asw.main([])
    assert rc == 0
    names = [n for n, _ in order]
    assert names == ["gate_push", "stale_registration", "zombie", "idle"]
    consumers = [c for n, c in order if n in ("stale_registration", "zombie", "idle")]
    assert all(c is snapshot for c in consumers)


# ── #1267 boot-death lane ─────────────────────────────────────────────────────
# A dispatched auto session whose transcript has ZERO response rows >= 30 min
# after spawned_at is STOPPED + surfaced instead of waiting 12h for the
# stale-registration pass. Conventions mirror the #845 (d) section above:
# isolated_registry + signal-helper monkeypatching (the conftest #1247 guards
# require _task_status / _post_progress_marker stubs on every firing path).

_BOOT_DEATH_NOW = 1_000_000.0


def _boot_death_rows_fixture(*, with_stderr=True, extra_row=None):
    """9-row replica of the live-captured #1251-family boot-death transcript
    shape (task #1267 plan §2): 2 queue-operation rows (enqueue + dequeue),
    3 attachment rows, 3 prompt-type user rows (one carrying the
    ``<local-command-stderr>`` skill-load diagnostic), 1 last-prompt row —
    ZERO assistant / api-error rows. ``extra_row`` appends a variant row
    (e.g. one assistant row) for the must-NOT-fire cases."""
    stderr_text = (
        "<command-message>issue</command-message>\n"
        "<local-command-stderr>Error: Shell command failed: exit status 1"
        "</local-command-stderr>"
    )
    rows = [
        {"type": "queue-operation", "operation": "enqueue"},
        {"type": "queue-operation", "operation": "dequeue"},
        {"type": "attachment", "attachment": {"kind": "skill"}},
        {"type": "attachment", "attachment": {"kind": "skill"}},
        {"type": "attachment", "attachment": {"kind": "skill"}},
        {"type": "user", "message": {"content": "/issue 1251"}},
        {"type": "user", "message": {"content": stderr_text if with_stderr else "second prompt"}},
        {"type": "user", "message": {"content": "third prompt"}},
        {"type": "last-prompt", "prompt": "/issue 1251"},
    ]
    if extra_row is not None:
        rows.append(extra_row)
    return rows


# Deliberately SHORT and mild (one line): the fixture needs a refusal-SHAPED
# api-error text for the sidecar-containment asserts, never real refusal prose.
_BOOT_REFUSAL_TEXT = "API Error: 400 request was blocked by our usage policy"


def _boot_refusal_rows_fixture(
    *, trailing_ok=False, with_prompt_evidence=True, leading_ok_turn=False
):
    """Reduced replica of the REAL #1277 boot-refusal transcript's
    classification sequence (#1287 plan §9 A2, verified at plan time: 40
    rows — 14 assistant / 1 api-error / 2 prompt / 1 dequeue / 22 other,
    sequence ``o d o o o p p o…a…a o`` with the trailing response row an
    api-error): a dequeue + prompt delivery burst (omitted under
    ``with_prompt_evidence=False`` — the oversize-tail shape where the
    prompt rows are truncated out and the #1127 leading-implicit-turn rule
    carries segmentation), interleaved ``other`` rows, 3 real assistant
    rows (the boot turn took real actions), and a trailing api-error row
    (row shape mirrors the live-captured #1074 api-error rows).
    ``trailing_ok=True`` appends a final plain assistant row instead — the
    recovered shape (last turn ``ok``). ``leading_ok_turn=True`` PREPENDS a
    completed OK delivery burst ([dequeue, prompt, assistant]) before the
    refusal burst — the two-burst ``[ok, failed]`` tail (the
    reduction-discrimination shape, the plan's test 9)."""
    rows: list[dict] = []
    if leading_ok_turn:
        rows += [
            {"type": "queue-operation", "operation": "dequeue"},
            {"type": "user", "message": {"content": "earlier prompt (ok turn)"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "earlier ok delivery"}]},
            },
        ]
    if with_prompt_evidence:
        rows += [
            {"type": "summary", "summary": "session meta"},  # other
            {"type": "queue-operation", "operation": "dequeue"},  # delivery burst
            {"type": "user", "message": {"content": "/issue 1277"}},  # prompt
            {"type": "user", "message": {"content": "second prompt"}},  # prompt (same burst)
        ]
    else:
        # Oversize-tail shape: the prompt-evidence rows are truncated out of
        # the 256 KB tail; the leading response rows form one implicit turn.
        rows += [{"type": "summary", "summary": "session meta"}]  # other
    rows += [
        {"type": "attachment", "attachment": {"kind": "skill"}},  # other
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "working"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "more work"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "heartbeat"}]}},
        {"type": "attachment", "attachment": {"kind": "file"}},  # other
    ]
    if trailing_ok:
        rows.append(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "recovered"}]}}
        )
    else:
        rows.append(
            {
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {"content": [{"type": "text", "text": _BOOT_REFUSAL_TEXT}]},
            }
        )
    rows.append({"type": "queue-operation", "operation": "enqueue"})  # trailing other (`…a o`)
    return rows


def _write_boot_death_entry(reg_dir, issue, session_id, spawned_at, *, manual=False):
    """Registration entry with an EXPLICIT spawned_at (the lane's age gate
    keys on it; `_write_autonomous_entry` pins spawned_at=now, too fresh)."""
    import json

    name = f"manual-issue-{issue}.json" if manual else f"issue-{issue}.json"
    (reg_dir / name).write_text(
        json.dumps(
            {
                "issue": issue,
                "happy_session_id": session_id,
                "cwd": "/repo",
                "auto_approve_gpu_hours": 12.0,
                "spawned_at": spawned_at,
                "missed": 0,
            }
        )
    )


def _boot_death_env(
    monkeypatch, asw, *, rows, tail_rows="mirror", idle_s=30 * 60, status="proposed", size=11368
):
    """Common signal monkeypatching for the boot-death pass tests. Returns
    the telegram-push recorder list. ``tail_rows`` stubs the arm-2 (#1287)
    ``_transcript_tail_rows`` seek-tail read; the default mirrors ``rows``
    (tail == whole file at <= cap), preserving every pre-#1287 call site's
    behavior."""
    if tail_rows == "mirror":
        tail_rows = rows
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (idle_s, None))
    monkeypatch.setattr(
        asw,
        "_boot_death_transcript_rows",
        lambda pid, max_bytes=None: (rows, "/fake/transcripts/boot-death.jsonl", size),
    )
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, max_bytes=None: tail_rows)
    pushes: list[tuple] = []
    monkeypatch.setattr(
        asw, "_telegram_push", lambda msg, dry_run: bool(pushes.append((msg, dry_run)))
    )
    return pushes


def test_decide_boot_death_fires_on_incident_shape():
    # Plan §6 acceptance 1: live sid + age 31 min + zero response rows +
    # idle 30 min + 0 stops today -> "stop". (Zero response rows => zero
    # completed turns => arm 2 reads False; the arms are mutually exclusive.)
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=31 * 60,
            response_row_seen=False,
            all_turns_failed=False,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == "stop"
    )


@pytest.mark.parametrize(
    "entry_age_s",
    [29 * 60, None, -120.0],  # young; missing/zero spawned_at; future-dated (negative age)
)
def test_decide_boot_death_keeps_young_entry(entry_age_s):
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=entry_age_s,
            response_row_seen=False,
            all_turns_failed=False,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == "keep"
    )


def test_decide_boot_death_keeps_on_response_row():
    # ANY response row (assistant — the healthy #1251 re-dispatch shape — OR
    # api-error) defeats ARM 1; the all-completed-turns-failed shape is ARM
    # 2's property (#1287, pinned by test_decide_boot_death_fires_on_
    # all_turns_failed + the boot-refusal pass tests). With arm 2 reading
    # False (an ok turn / zero completed turns), a response row keeps.
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=31 * 60,
            response_row_seen=True,
            all_turns_failed=False,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == "keep"
    )


def test_decide_boot_death_keeps_on_unresolvable_transcript():
    # response_row_seen=None + all_turns_failed=None: BOTH reads
    # unresolvable — fail toward keep (the 12h stale-registration pass
    # stays the backstop). The over-the-cap case where the TAIL still
    # resolves is arm 2's stop (test_decide_boot_death_fires_on_all_turns_
    # failed pins response_row_seen=None + all_turns_failed=True -> stop).
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=31 * 60,
            response_row_seen=None,
            all_turns_failed=None,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == "keep"
    )


@pytest.mark.parametrize(
    "transcript_idle_s",
    [9 * 60, None],  # in-flight first turn; unresolvable idle signal (fail toward keep)
)
def test_decide_boot_death_keeps_within_quiet_window(transcript_idle_s):
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=31 * 60,
            response_row_seen=False,
            all_turns_failed=False,
            transcript_idle_s=transcript_idle_s,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == "keep"
    )


def test_decide_boot_death_keeps_dead_sid():
    # A dead sid is the crash-recovery / sweep-grace passes' property.
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=False,
            entry_age_s=31 * 60,
            response_row_seen=False,
            all_turns_failed=False,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == "keep"
    )


@pytest.mark.parametrize(
    ("stops_today", "expected"),
    [
        (2, "stop"),  # cap boundary: the LAST permitted unit still stops (cap=3)
        (3, "cap-alert"),
        (4, "cap-alert"),
    ],
)
def test_decide_boot_death_cap_exhausted_returns_cap_alert(stops_today, expected):
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=31 * 60,
            response_row_seen=False,
            all_turns_failed=False,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=stops_today,
            stops_per_day=3,
        )
        == expected
    )


def test_boot_death_window_env_parse(monkeypatch):
    # Mirrors the _stale_registration_idle_s contract: default 1800.0;
    # minutes-valued env honored; malformed / non-positive -> default (a
    # typo'd var must never create an instant stopper).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_BOOT_DEATH_WINDOW_MIN", raising=False)
    assert asw._boot_death_window_s() == 1800.0
    monkeypatch.setenv("EPM_BOOT_DEATH_WINDOW_MIN", "45")
    assert asw._boot_death_window_s() == 2700.0
    for bad in ("0", "-5", "junk"):
        monkeypatch.setenv("EPM_BOOT_DEATH_WINDOW_MIN", bad)
        assert asw._boot_death_window_s() == 1800.0


def test_boot_death_stops_per_day_env_parse(monkeypatch):
    # Byte-parallel to _tick_wedge_respawns_per_day: default 3; positive int
    # honored; < 1 / malformed -> default. Never a kill switch.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_BOOT_DEATH_STOPS_PER_DAY", raising=False)
    assert asw._boot_death_stops_per_day() == 3
    monkeypatch.setenv("EPM_BOOT_DEATH_STOPS_PER_DAY", "5")
    assert asw._boot_death_stops_per_day() == 5
    for bad in ("0", "-2", "junk"):
        monkeypatch.setenv("EPM_BOOT_DEATH_STOPS_PER_DAY", bad)
        assert asw._boot_death_stops_per_day() == 3


def test_boot_death_transcript_rows_size_guard(tmp_path, monkeypatch):
    # Whole-file read with a hard size ceiling: a small file yields the FULL
    # row list (never a truncated tail); an over-cap file yields rows=None
    # (not-eligible -> keep) with the path/size kept as forensics; an
    # unresolvable transcript yields (None, None, None).
    import json

    import autonomous_session_watch as asw
    import session_resolver

    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(json.dumps({"type": "user", "message": {"content": f"p{i}"}}) for i in range(3))
        + "\n"
    )
    monkeypatch.setattr(
        session_resolver, "_resolve_transcript_via_happy_log", lambda pid: (str(p), None)
    )
    rows, path, size = asw._boot_death_transcript_rows(4242)
    assert path == str(p) and size == p.stat().st_size
    assert isinstance(rows, list) and len(rows) == 3
    rows2, path2, size2 = asw._boot_death_transcript_rows(4242, max_bytes=10)
    assert rows2 is None and path2 == str(p) and size2 == size
    monkeypatch.setattr(
        session_resolver, "_resolve_transcript_via_happy_log", lambda pid: (None, "no happy log")
    )
    assert asw._boot_death_transcript_rows(4242) == (None, None, None)


def test_boot_death_rows_incident_fixture_zero_response():
    # The live-captured 9-row incident shape classifies to ZERO response rows
    # (the firing precondition); a one-assistant-row variant AND an
    # api-error-row variant (#1209 family) both read response_row_seen=True.
    import autonomous_session_watch as asw

    rows = _boot_death_rows_fixture()
    assert len(rows) == 9
    classes = [asw._classify_wedge_row(r) for r in rows]
    assert classes.count("dequeue") == 1
    assert classes.count("prompt") == 3
    assert classes.count("other") == 5
    assert not any(c in ("assistant", "api-error") for c in classes)

    healthy = _boot_death_rows_fixture(
        extra_row={"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}
    )
    assert any(asw._classify_wedge_row(r) in ("assistant", "api-error") for r in healthy)
    refused = _boot_death_rows_fixture(
        extra_row={"type": "assistant", "isApiErrorMessage": True, "message": {}}
    )
    assert any(asw._classify_wedge_row(r) in ("assistant", "api-error") for r in refused)
    # #1287 arm mutual exclusivity: zero response rows => zero completed
    # turns => arm 2 reads False on exactly the shape arm 1 owns.
    assert asw._segment_wake_turns(_boot_death_rows_fixture()) == []


def test_boot_death_stderr_excerpt_bounded():
    # The forensic excerpt starts at the <local-command-stderr> tag, is
    # whitespace-collapsed, and never exceeds the 200-char bound; a fixture
    # without the tag yields None.
    import autonomous_session_watch as asw

    excerpt = asw._boot_death_stderr_excerpt(_boot_death_rows_fixture())
    assert excerpt is not None and excerpt.startswith("<local-command-stderr>Error:")
    assert len(excerpt) <= 200 and "\n" not in excerpt
    assert asw._boot_death_stderr_excerpt(_boot_death_rows_fixture(with_stderr=False)) is None
    long_rows = [
        {"type": "user", "message": {"content": "<local-command-stderr>" + "x" * 500}},
    ]
    assert len(asw._boot_death_stderr_excerpt(long_rows)) == 200
    # A PRESENT-but-non-str "text" value (null / int) still classifies as a
    # prompt row but carries no diagnostic text: the helper must skip it
    # WITHOUT raising (the docstring's never-raises claim; review concern
    # boot-death-stderr-excerpt-nonstr-text-raise). Pre-fix this shape
    # raised AttributeError on text.find().
    nonstr_rows = [
        {"type": "user", "message": {"content": [{"type": "text", "text": None}]}},
        {"type": "user", "message": {"content": [{"type": "text", "text": 42}]}},
    ]
    assert asw._boot_death_stderr_excerpt(nonstr_rows) is None
    # A non-str block alongside a real diagnostic block: the str block wins.
    mixed_rows = [
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": None},
                    {"type": "text", "text": "<local-command-stderr>Error: boom"},
                ]
            },
        },
    ]
    assert asw._boot_death_stderr_excerpt(mixed_rows) == "<local-command-stderr>Error: boom"


def test_boot_death_pass_stops_and_posts_marker(isolated_registry, monkeypatch):
    # Plan §6 pass-level headline: the incident shape stops the entry's sid,
    # posts ONE anti-liveness marker (label boot-death-stop, sentinel +
    # status= in the note), appends a sidecar row carrying the transcript=
    # and stderr_excerpt= forensic keys, bumps stops_today to 1, and KEEPS
    # the registration file (re-drive is the existing arms' job).
    import json

    import autonomous_session_watch as asw

    stops: list[tuple] = []
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: bool(stops.append((sid, dry_run))) or True
    )
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label, note, dry_run)),
    )
    pushes = _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-990", "pid": 4242}],
        now=_BOOT_DEATH_NOW,
    )

    assert stops == [("sess-990", False)]
    assert [(i, la) for i, la, _n, _d in markers] == [(990, "boot-death-stop")]
    note = markers[0][2]
    assert asw._BOOT_DEATH_STOP_NOTE_SENTINEL in note
    assert "shape=zero-response" in note  # #1287 shape tag: arm 1 owns this fixture
    assert "status=proposed" in note and "registration kept" in note
    assert len(pushes) == 1
    assert (isolated_registry / "issue-990.json").exists()  # KEPT, never unlinked
    state = json.loads((isolated_registry / "boot-death-990.json").read_text())
    assert state["stops_today"] == 1
    events = (isolated_registry / "boot-death-events.jsonl").read_text().splitlines()
    assert len(events) == 1
    row = json.loads(events[0])
    assert row["kind"] == "boot-death"
    assert "transcript=/fake/transcripts/boot-death.jsonl" in row["note"]
    assert "stderr_excerpt=<local-command-stderr>Error:" in row["note"]


def test_boot_death_pass_daemon_down_noop(isolated_registry, monkeypatch):
    # children=None (daemon unreachable) => no-op: liveness cannot be
    # established, and a false "live" read must not stop anything.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(dry_run=False, children=None, now=_BOOT_DEATH_NOW)
    assert stops == [] and (isolated_registry / "issue-990.json").exists()


def test_boot_death_pass_dry_run(isolated_registry, monkeypatch, capsys):
    # Reviewer concern 3, pinned test + code: a dry run stops NOTHING, posts
    # NO real task marker / push, and writes NO state / sidecar — log lines
    # only. _post_progress_marker / _stop_session / _telegram_push are left
    # REAL here with subprocess.run replaced by a recorder: dry_run=True must
    # short-circuit every one of them BEFORE any subprocess, so production
    # events.jsonl is provably never mutated by a dry run.
    import subprocess as _sp

    import autonomous_session_watch as asw

    calls: list = []

    def _no_subprocess(*a, **k):
        calls.append(a)
        raise AssertionError("dry-run must never reach subprocess.run")

    monkeypatch.setattr(_sp, "run", _no_subprocess)
    monkeypatch.setattr(asw, "_task_status", lambda issue: "proposed")
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (30 * 60, None))
    monkeypatch.setattr(
        asw,
        "_boot_death_transcript_rows",
        lambda pid, max_bytes=None: (_boot_death_rows_fixture(), "/fake/t.jsonl", 11368),
    )
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=True,
        children=[{"happySessionId": "sess-990", "pid": 4242}],
        now=_BOOT_DEATH_NOW,
    )
    assert calls == []  # no real stop / marker post / push subprocess
    assert (isolated_registry / "issue-990.json").exists()
    assert not (isolated_registry / "boot-death-990.json").exists()  # no state write
    assert not (isolated_registry / "boot-death-events.jsonl").exists()  # no sidecar
    out = capsys.readouterr().out
    assert "[dry-run] would stop session" in out
    assert "[dry-run] would post epm:progress" in out
    assert "[dry-run] would save boot-death state" in out


def test_boot_death_pass_manual_registration_excluded(isolated_registry, monkeypatch):
    # manual-issue-*.json is out of scope by design (#505 posture: a
    # user-driven session is never auto-stopped) — never probed, never
    # stopped, no marker.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    markers: list = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: markers.append(label)
    )
    _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    _write_boot_death_entry(
        isolated_registry, 991, "sess-991-manual", _BOOT_DEATH_NOW - 31 * 60, manual=True
    )
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-991-manual", "pid": 4243}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == [] and markers == []
    assert (isolated_registry / "manual-issue-991.json").exists()


def test_boot_death_pass_kill_switch(isolated_registry, monkeypatch, capsys):
    # EPM_DISABLE_BOOT_DEATH_PASS=1 -> logged no-op.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    monkeypatch.setenv("EPM_DISABLE_BOOT_DEATH_PASS", "1")
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-990", "pid": 4242}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == []
    assert "disabled via EPM_DISABLE_BOOT_DEATH_PASS" in capsys.readouterr().out


def test_boot_death_pass_cap_alert_once_per_day(isolated_registry, monkeypatch):
    # Plan §6 acceptance 3: the 4th qualifying detection in one UTC day gets
    # NO stop and exactly ONE cap push/marker; a same-day repeat tick is
    # quiet; the next UTC day re-arms (the day-keyed counter reads 0).
    import json
    import time as _t

    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)) or True)
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((label, note)),
    )
    pushes = _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    day_key = _t.strftime("%Y-%m-%d", _t.gmtime(_BOOT_DEATH_NOW))
    (isolated_registry / "boot-death-990.json").write_text(
        json.dumps({"stop_day": day_key, "stops_today": 3})
    )
    children = [{"happySessionId": "sess-990", "pid": 4242}]

    asw.boot_death_pass(dry_run=False, children=children, now=_BOOT_DEATH_NOW)
    assert stops == []
    assert [la for la, _n in markers] == ["boot-death-cap-exhausted"]
    assert asw._BOOT_DEATH_CAP_NOTE_SENTINEL in markers[0][1]
    assert len(pushes) == 1

    # Same-day repeat tick: quiet (cap_alerted_day dedup).
    asw.boot_death_pass(dry_run=False, children=children, now=_BOOT_DEATH_NOW + 600)
    assert len(markers) == 1 and len(pushes) == 1 and stops == []

    # Next UTC day: the day-keyed counter reads 0 -> the lane re-arms + stops.
    next_day = _BOOT_DEATH_NOW + 24 * 3600
    _write_boot_death_entry(isolated_registry, 990, "sess-990", next_day - 31 * 60)
    asw.boot_death_pass(dry_run=False, children=children, now=next_day)
    assert stops == ["sess-990"]
    assert [la for la, _n in markers] == ["boot-death-cap-exhausted", "boot-death-stop"]


def test_boot_death_pass_stop_failure_still_consumes_budget(isolated_registry, monkeypatch):
    # #1241 parity: the counter bumps at STOP-INITIATION, so a failed stop
    # still consumes a budget unit (conservative in the safe direction), and
    # the note records stop_ok=False.
    import json

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: False)
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((label, note)),
    )
    _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-990", "pid": 4242}],
        now=_BOOT_DEATH_NOW,
    )
    state = json.loads((isolated_registry / "boot-death-990.json").read_text())
    assert state["stops_today"] == 1  # bumped despite the failed stop
    assert "stop_ok=False" in markers[0][1]


def test_boot_death_pass_provision_in_flight_keeps(isolated_registry, monkeypatch):
    # An in-flight provision means something owns this issue — keep.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    monkeypatch.setattr(
        asw, "_provision_in_flight_reason", lambda issue, now: "live provision pid 1"
    )
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-990", "pid": 4242}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == []


def test_boot_death_pass_fresh_worktree_keeps(isolated_registry, monkeypatch):
    # Fresh worktree edits mean an implementer is mid-work — keep.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    _boot_death_env(monkeypatch, asw, rows=_boot_death_rows_fixture())
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    _write_boot_death_entry(isolated_registry, 990, "sess-990", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-990", "pid": 4242}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == []


# ── #1287 boot-death lane arm 2 (boot-refusal) ────────────────────────────────
# The #1277 shape: the boot turn RAN (assistant rows exist) then died on a
# refusal, on an 826 KB transcript over arm 1's whole-file cap. Arm 2 reads
# the 256 KB tail via _transcript_tail_rows + _segment_wake_turns and fires
# on >= 1 completed turn with EVERY completed turn failed.


def test_boot_refusal_fixture_segments_to_one_failed_turn():
    # Pure (#1287 plan test 1): the fixture segments to exactly ONE completed
    # turn, outcome "failed", for BOTH with_prompt_evidence=True AND False
    # (the oversize-tail shape — prompt rows truncated out, the #1127
    # leading-implicit-turn rule carries it — the real #1277 tail path);
    # trailing_ok=True flips the last (only) turn to "ok".
    import autonomous_session_watch as asw

    for wpe in (True, False):
        turns = asw._segment_wake_turns(_boot_refusal_rows_fixture(with_prompt_evidence=wpe))
        assert [o for o, _ts in turns] == ["failed"], f"with_prompt_evidence={wpe}: {turns}"
    turns_ok = asw._segment_wake_turns(_boot_refusal_rows_fixture(trailing_ok=True))
    assert [o for o, _ts in turns_ok] == ["ok"]


@pytest.mark.parametrize(
    ("response_row_seen", "all_turns_failed", "expected"),
    [
        (True, True, "stop"),  # arm 2 fires past arm 1's response-row keep
        (True, None, "keep"),  # tail unresolvable -> fail toward keep
        (True, False, "keep"),  # an ok turn / zero completed turns
        (None, True, "stop"),  # oversize: arm 1's unresolvable whole-file read does NOT veto arm 2
    ],
)
def test_decide_boot_death_fires_on_all_turns_failed(response_row_seen, all_turns_failed, expected):
    # Pure (#1287 plan test 2): the arm-2 predicate at age 31 min / idle
    # 30 min / 0 stops today.
    import autonomous_session_watch as asw

    assert (
        asw.decide_boot_death(
            sid_alive=True,
            entry_age_s=31 * 60,
            response_row_seen=response_row_seen,
            all_turns_failed=all_turns_failed,
            transcript_idle_s=30 * 60,
            window_s=asw.BOOT_DEATH_WINDOW_S,
            quiet_s=asw.BOOT_DEATH_QUIET_S,
            stops_today=0,
            stops_per_day=3,
        )
        == expected
    )


def test_boot_death_pass_fires_on_boot_refusal_oversize_transcript(isolated_registry, monkeypatch):
    # THE #1277 REGRESSION TEST (#1287 plan test 3): an oversize transcript
    # (rows=None at 825,591 B — arm 1 unresolvable) whose 256 KB tail
    # segments to one implicit completed turn, failed. The pass stops the
    # sid, tags the note shape=boot-refusal with the ALL-failed evidence,
    # bumps the day counter, and confines the refusal excerpt to the
    # SIDECAR (never the marker note, never the push).
    import json

    import autonomous_session_watch as asw

    stops: list[tuple] = []
    monkeypatch.setattr(
        asw, "_stop_session", lambda sid, dry_run: bool(stops.append((sid, dry_run))) or True
    )
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((issue, label, note)),
    )
    tail = _boot_refusal_rows_fixture(with_prompt_evidence=False)
    pushes = _boot_death_env(monkeypatch, asw, rows=None, tail_rows=tail, size=825_591)
    _write_boot_death_entry(isolated_registry, 992, "sess-992", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-992", "pid": 4244}],
        now=_BOOT_DEATH_NOW,
    )

    assert stops == [("sess-992", False)]
    assert [(i, la) for i, la, _n in markers] == [(992, "boot-death-stop")]
    note = markers[0][2]
    assert asw._BOOT_DEATH_STOP_NOTE_SENTINEL in note
    assert "shape=boot-refusal" in note
    assert "ALL failed" in note and "1 api-error row(s)" in note
    assert "task status=proposed" in note  # the real PARK-status shape recorded
    assert len(pushes) == 1 and "shape=boot-refusal" in pushes[0][0]
    # Sidecar-only containment: the refusal text reaches the sidecar's
    # api_error_excerpt= field and NOTHING else.
    assert _BOOT_REFUSAL_TEXT not in note
    assert _BOOT_REFUSAL_TEXT not in pushes[0][0]
    state = json.loads((isolated_registry / "boot-death-992.json").read_text())
    assert state["stops_today"] == 1
    events = (isolated_registry / "boot-death-events.jsonl").read_text().splitlines()
    assert len(events) == 1
    row = json.loads(events[0])
    assert f"api_error_excerpt={_BOOT_REFUSAL_TEXT[:20]}" in row["note"]
    assert (isolated_registry / "issue-992.json").exists()  # registration KEPT


def test_boot_death_pass_fires_on_boot_refusal_small_transcript(isolated_registry, monkeypatch):
    # Whole-file variant (#1287 plan test 4): at <= cap the whole-file rows
    # RESOLVE (arm 1 keeps: response_row_seen=True) and arm 2 fires on the
    # SAME rows — tail_rows is stubbed to None here, so a fire PROVES the
    # caller reused the resolved whole-file rows instead of the second read.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)) or True)
    markers: list[tuple] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: markers.append((label, note)),
    )
    rows = _boot_refusal_rows_fixture()
    _boot_death_env(monkeypatch, asw, rows=rows, tail_rows=None)
    _write_boot_death_entry(isolated_registry, 992, "sess-992", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-992", "pid": 4244}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == ["sess-992"]
    assert [la for la, _n in markers] == ["boot-death-stop"]
    assert "shape=boot-refusal" in markers[0][1]


@pytest.mark.parametrize("age_s", [31 * 60, 7 * 24 * 3600])
@pytest.mark.parametrize("shape", ["trailing_ok", "assistant_only"])
def test_boot_death_pass_keeps_on_trailing_ok_turn(isolated_registry, monkeypatch, shape, age_s):
    # Must-NOT-fire (#1287 plan test 5): (a) the recovered shape — the tail's
    # last completed turn is "ok" (trailing_ok fixture); (b) plain
    # assistant-row-only rows (one implicit ok turn). Parametrized over an
    # OLD entry age (7 days) too: arm 2 makes the lane reachable at ANY age,
    # so the healthy-long-lived-session keep is pinned, not only young ones.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    markers: list = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: markers.append(label)
    )
    if shape == "trailing_ok":
        rows = _boot_refusal_rows_fixture(trailing_ok=True)
    else:
        rows = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": f"t{i}"}]}}
            for i in range(3)
        ]
    _boot_death_env(monkeypatch, asw, rows=rows)
    _write_boot_death_entry(isolated_registry, 993, "sess-993", _BOOT_DEATH_NOW - age_s)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-993", "pid": 4245}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == [] and markers == []
    assert not (isolated_registry / "boot-death-993.json").exists()  # no state write
    assert not (isolated_registry / "boot-death-events.jsonl").exists()  # no sidecar


def test_boot_death_pass_refusal_keeps_within_quiet_window(isolated_registry, monkeypatch):
    # #1287 plan test 6: the mtime-quiet guard is wired for arm 2 — a
    # refusal-shaped tail with a FRESH transcript (idle 5 min < 10 min quiet)
    # keeps: a live retrying session keeps appending rows.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    markers: list = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: markers.append(label)
    )
    _boot_death_env(monkeypatch, asw, rows=_boot_refusal_rows_fixture(), idle_s=5 * 60)
    _write_boot_death_entry(isolated_registry, 993, "sess-993", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-993", "pid": 4245}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == [] and markers == []


def test_boot_death_pass_keeps_when_both_arms_unresolvable(isolated_registry, monkeypatch):
    # #1287 plan test 7: whole-file read unresolvable (rows=None) AND tail
    # read unresolvable (tail_rows=None) -> keep; no state write, no marker.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    markers: list = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: markers.append(label)
    )
    _boot_death_env(monkeypatch, asw, rows=None, tail_rows=None, size=None)
    _write_boot_death_entry(isolated_registry, 993, "sess-993", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-993", "pid": 4245}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == [] and markers == []
    assert not (isolated_registry / "boot-death-993.json").exists()
    assert not (isolated_registry / "boot-death-events.jsonl").exists()


def test_boot_death_api_error_excerpt_bounded():
    # #1287 plan test 8 — mirrors test_boot_death_stderr_excerpt_bounded for
    # the arm-2 forensic helper: LAST api-error row wins when two exist;
    # 200-char bound + whitespace collapse; None when no api-error rows / no
    # usable text; present-but-non-str "text" skipped without raising;
    # plain-str content handled.
    import autonomous_session_watch as asw

    excerpt = asw._boot_death_api_error_excerpt(_boot_refusal_rows_fixture())
    assert excerpt == _BOOT_REFUSAL_TEXT
    assert len(excerpt) <= 200 and "\n" not in excerpt
    two = [
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "message": {"content": [{"type": "text", "text": "API Error: first"}]},
        },
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "message": {"content": "API Error:   second\nline"},  # plain-str content
        },
    ]
    assert asw._boot_death_api_error_excerpt(two) == "API Error: second line"
    long_rows = [
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "message": {"content": "API  Error: " + "x " * 300},
        },
    ]
    assert len(asw._boot_death_api_error_excerpt(long_rows)) == 200
    # No api-error rows at all (the zero-response fixture) -> None.
    assert asw._boot_death_api_error_excerpt(_boot_death_rows_fixture()) is None
    nonstr_rows = [
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "message": {"content": [{"type": "text", "text": None}]},
        },
        {
            "type": "assistant",
            "isApiErrorMessage": True,
            "message": {"content": [{"type": "text", "text": 42}]},
        },
    ]
    assert asw._boot_death_api_error_excerpt(nonstr_rows) is None


def test_boot_death_pass_keeps_on_leading_ok_turn_then_failed(isolated_registry, monkeypatch):
    # #1287 plan test 9 — the reduction-discrimination must-not-fire (round-1
    # Statistics Must-Fix): a two-burst [ok, failed] tail is the ONE
    # configuration that distinguishes the correct all(outcome == "failed")
    # reduction from the rejected any(failed) / last-turn-failed
    # over-triggers (the #1104 single-refusal guard): an ok turn NOT last
    # must keep.
    import autonomous_session_watch as asw

    stops: list = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: bool(stops.append(sid)))
    markers: list = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: markers.append(label)
    )
    tail = _boot_refusal_rows_fixture(leading_ok_turn=True)
    assert [o for o, _ts in asw._segment_wake_turns(tail)] == ["ok", "failed"]
    _boot_death_env(monkeypatch, asw, rows=None, tail_rows=tail, size=825_591)
    _write_boot_death_entry(isolated_registry, 994, "sess-994", _BOOT_DEATH_NOW - 31 * 60)
    asw.boot_death_pass(
        dry_run=False,
        children=[{"happySessionId": "sess-994", "pid": 4246}],
        now=_BOOT_DEATH_NOW,
    )
    assert stops == [] and markers == []
    assert not (isolated_registry / "boot-death-994.json").exists()  # no state write
    assert not (isolated_registry / "boot-death-events.jsonl").exists()  # no sidecar


def test_boot_death_sentinels_in_watcher_note_sentinels():
    # Anti-liveness pin (durability pin for this task): both sentinels MUST
    # be members of _WATCHER_NOTE_SENTINELS or the lane's own notes would
    # refresh _latest_progress_ts and mask the orphan/stalled staleness
    # clocks (the line-1207 contract).
    import autonomous_session_watch as asw

    assert asw._BOOT_DEATH_STOP_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._BOOT_DEATH_CAP_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_main_order_boot_death_before_stale_registration(isolated_registry, monkeypatch):
    # #1267 wiring: boot_death_pass runs AFTER gate_push_pass (the gate-push-
    # before-reaper ordering invariant) and BEFORE stale_registration_pass,
    # and (reviewer concern 4) receives the SAME shared reaper `children`
    # snapshot OBJECT as the three sibling consumers — object identity, not
    # just equality.
    import autonomous_session_watch as asw

    order: list[tuple[str, object]] = []
    snapshot = [{"happySessionId": "sid-shared", "pid": 12345}]
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    monkeypatch.setattr(asw, "_live_children", lambda: snapshot)
    _stub_fleet_mutating_passes(asw, monkeypatch)
    for name in (
        "vm_disk_pass",
        "data_disk_pass",
        "happy_patch_pass",
        "triage_observer_pass",
        "program_orchestrator_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "proposed_infra_sweep_pass",
        "capacity_retry_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "gc_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(asw, "gate_push_pass", lambda *a, **kw: order.append(("gate_push", None)))
    monkeypatch.setattr(
        asw, "boot_death_pass", lambda *a, **kw: order.append(("boot_death", kw.get("children")))
    )
    monkeypatch.setattr(
        asw,
        "stale_registration_pass",
        lambda *a, **kw: order.append(("stale_registration", kw.get("children"))),
    )
    monkeypatch.setattr(
        asw, "zombie_wrapper_pass", lambda *a, **kw: order.append(("zombie", kw.get("children")))
    )
    monkeypatch.setattr(
        asw, "idle_unmapped_pass", lambda *a, **kw: order.append(("idle", kw.get("children")))
    )

    rc = asw.main([])
    assert rc == 0
    names = [n for n, _ in order]
    assert names == ["gate_push", "boot_death", "stale_registration", "zombie", "idle"]
    consumers = [c for n, c in order if n != "gate_push"]
    assert all(c is snapshot for c in consumers)


def test_gc_targets_include_boot_death_prefix():
    # The day-cap state file is reaped by the generalized GC at terminal
    # status (the `proposed` incident-class status is NOT terminal, so a live
    # loop's counter is never reset mid-episode).
    import autonomous_session_watch as asw

    assert (asw.BOOT_DEATH_STATE_PREFIX, "") in asw._GC_TARGETS


# ─── triage-observer pass (#967) ──────────────────────────────────────────────
#
# NON-GATING post-hoc audit of the pre-dispatch external-marker triage duty
# (origin incident #779). The pure predicate lives in task_workflow
# (tests/test_pre_dispatch_triage.py); these tests pin the DRIVER: kill
# switch, dry-run hygiene, the MF3 two-invocation fire-once round-trip
# through the REAL _save/_load state singleton, the two-pronged non-gating
# invariant, and the nudge-text trap strings.


def _triage_observer_sandbox(tmp_path, monkeypatch, events, *, status="running", issue=321):
    """Fully sandbox triage_observer_pass: tmp registry + task dir (fresh
    events.jsonl mtime), tmp state/sidecar singletons, recorded pushes.
    Returns (asw, state_path, sidecar_path, pushes)."""
    import json as _json

    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    reg_root = tmp_path / "repo"
    task_rel = f"tasks/{status}/{issue}"
    task_dir = reg_root / task_rel
    task_dir.mkdir(parents=True)
    (task_dir / "events.jsonl").write_text("")  # fresh mtime; list_events is patched
    reg_path = reg_root / "tasks" / "REGISTRY.json"
    reg_path.write_text(
        _json.dumps(
            {
                "tasks": {
                    str(issue): {
                        "status": status,
                        "path": task_rel,
                        "kind": "experiment",
                        "title": "synthetic",
                        "has_clean_result": False,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(task_workflow, "registry_path", lambda: reg_path)
    monkeypatch.setattr(task_workflow, "list_events", lambda _issue: list(events))
    state_path = tmp_path / "triage-observer.json"
    sidecar_path = tmp_path / "triage-observer-events.jsonl"
    monkeypatch.setattr(asw, "_triage_observer_state_path", lambda: state_path)
    monkeypatch.setattr(asw, "_triage_observer_sidecar_path", lambda: sidecar_path)
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry: pushes.append((msg, dry)))
    return asw, state_path, sidecar_path, pushes


def _matured_violating_events():
    """One post-epoch, MATURED (2h-old — older than the 30-min adjacency
    horizon), in-lookback launch marker with no triage line anywhere."""
    from datetime import UTC, datetime, timedelta

    ts = (datetime.now(tz=UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [{"ts": ts, "kind": "epm:run-launched", "by": "poll_pipeline", "note": "launched"}]


def _matured_violating_events_n(n):
    """``n`` distinct post-epoch, MATURED lone launch markers (ascending ts,
    ~35 min apart, 2h-6.5h old — all older than the 30-min adjacency horizon,
    all inside the 48h lookback). Each yields one ``launch-missing-line``
    warn: launch↔launch proximity grants no adjacency coverage — only a
    triage-LINE boundary neighbor does (#1167 push-cap fixtures)."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(tz=UTC)
    return [
        {
            "ts": (now - timedelta(minutes=120 + 35 * (n - 1 - i))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": "epm:run-launched",
            "by": "poll_pipeline",
            "note": f"launched {i}",
        }
        for i in range(n)
    ]


def test_triage_observer_kill_switch_skips_everything(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    monkeypatch.setenv("EPM_DISABLE_TRIAGE_OBSERVER", "1")
    state_path = tmp_path / "state.json"
    sidecar_path = tmp_path / "sidecar.jsonl"
    monkeypatch.setattr(asw, "_triage_observer_state_path", lambda: state_path)
    monkeypatch.setattr(asw, "_triage_observer_sidecar_path", lambda: sidecar_path)

    def _forbidden():
        raise AssertionError("registry must not be read under the kill switch")

    monkeypatch.setattr(task_workflow, "registry_path", _forbidden)
    assert asw.triage_observer_pass(dry_run=False) is False
    assert not state_path.exists() and not sidecar_path.exists()


def test_decide_triage_observer_actions_routing():
    import autonomous_session_watch as asw

    w1 = {
        "record_ts": "2026-07-10T10:00:00Z",
        "violation": "launch-missing-line",
        "severity": "warn",
    }
    w2 = {
        "record_ts": "2026-07-10T11:00:00Z",
        "violation": "breadcrumb-missing-line",
        "severity": "warn",
    }
    i1 = {
        "record_ts": "2026-07-10T09:00:00Z",
        "violation": "none-with-candidates",
        "severity": "info",
    }
    flagged = {"2026-07-10T10:00:00Z|launch-missing-line"}
    actions = asw.decide_triage_observer_actions([i1, w1, w2], flagged, marker_budget=1)
    # w1 deduped away; warn-before-info ordering; the budget=1 marker lands
    # on the surviving warn; info rows get neither push nor marker.
    assert [(a["violation"], a["push"], a["marker"]) for a in actions] == [
        ("breadcrumb-missing-line", True, True),
        ("none-with-candidates", False, False),
    ]
    assert all(a["sidecar"] for a in actions)
    # #1167 default pin: push_budget=None (uncapped) -> nothing suppressed.
    assert all(a["push_suppressed"] is False for a in actions)
    # The pure function never mutates the caller's flagged set.
    assert flagged == {"2026-07-10T10:00:00Z|launch-missing-line"}
    # Cap-overflow semantics: an over-budget warn keeps push=True and
    # marker=False (permanently sidecar+push-only — the caller still flags
    # it, so its marker is NEVER deferred to a later tick).
    w3 = dict(w1, record_ts="2026-07-10T12:00:00Z")
    actions = asw.decide_triage_observer_actions([w2, w3], set(), marker_budget=1)
    assert [(a["severity"], a["push"], a["marker"]) for a in actions] == [
        ("warn", True, True),
        ("warn", True, False),
    ]


def test_decide_triage_observer_actions_push_cap():
    # #1167: the pure decider caps the push channel INDEPENDENTLY of the
    # marker channel; over-push-budget warns get push=False,
    # push_suppressed=True (the caller rolls them into one summary push).
    import autonomous_session_watch as asw

    warns = [
        {
            "record_ts": f"2026-07-10T1{i}:00:00Z",
            "violation": "launch-missing-line",
            "severity": "warn",
        }
        for i in range(3)
    ]
    info = {
        "record_ts": "2026-07-10T09:00:00Z",
        "violation": "none-with-candidates",
        "severity": "info",
    }
    flagged: set[str] = set()
    actions = asw.decide_triage_observer_actions(
        [*warns, info], flagged, marker_budget=1, push_budget=2
    )
    # Budget INDEPENDENCE: the marker budget (1) and push budget (2) run out
    # at different points; the over-push-budget warn keeps marker semantics
    # (here: marker budget already spent); info consumes neither budget and
    # is never push_suppressed.
    assert [(a["push"], a["push_suppressed"], a["marker"]) for a in actions] == [
        (True, False, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ]
    assert all("push_suppressed" in a for a in actions)
    # The pure function never mutates the caller's flagged set.
    assert flagged == set()

    # Cap 0: every warn suppressed; info rows are never push_suppressed.
    actions = asw.decide_triage_observer_actions([*warns, info], set(), 5, push_budget=0)
    assert [(a["severity"], a["push"], a["push_suppressed"]) for a in actions] == [
        ("warn", False, True),
        ("warn", False, True),
        ("warn", False, True),
        ("info", False, False),
    ]

    # Back-compat pin: push_budget=None (the default) = uncapped.
    actions = asw.decide_triage_observer_actions(warns, set(), 5, push_budget=None)
    assert all(a["push"] is True and a["push_suppressed"] is False for a in actions)


def test_triage_observer_dry_run_performs_zero_writes(tmp_path, monkeypatch):
    # Backs the post-merge `--triage-observer-only --dry-run` smoke: a
    # dry-run must create no sidecar, write no state, and spawn no
    # subprocess (the #596/#607/#633 pattern).
    asw, state_path, sidecar_path, _pushes = _triage_observer_sandbox(
        tmp_path, monkeypatch, _matured_violating_events()
    )
    calls: list = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **kw: calls.append(a))
    asw.triage_observer_pass(dry_run=True)
    assert calls == []
    assert not state_path.exists()
    assert not sidecar_path.exists()


def test_triage_observer_fire_once_two_invocations_real_writes(tmp_path, monkeypatch):
    # MF3: tick 1 on a violating task emits exactly 1 sidecar row + 1
    # post-marker subprocess + 1 push; tick 2 on UNCHANGED events emits
    # nothing new. Exercises the REAL _save/_load state round-trip — a
    # broken round-trip would ship green on single-invocation tests and
    # produce the re-alert storm the kill criterion forbids.
    import json as _json
    import subprocess as _subprocess

    asw, state_path, sidecar_path, pushes = _triage_observer_sandbox(
        tmp_path, monkeypatch, _matured_violating_events()
    )
    argvs: list[list[str]] = []

    def _fake_run(cmd, *a, **kw):
        argvs.append([str(c) for c in cmd])
        return _subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)

    assert asw.triage_observer_pass(dry_run=False) is True
    rows = [_json.loads(line) for line in sidecar_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["issue"] == 321
    assert rows[0]["violation"] == "launch-missing-line"
    assert rows[0]["severity"] == "warn"
    marker_cmds = [c for c in argvs if "post-marker" in c]
    assert len(marker_cmds) == 1
    assert len(pushes) == 1
    assert state_path.exists()

    # Non-gating pin, prong (a): the recorded subprocess argv never mutates
    # task state — no set-status, no session stop, only post-marker.
    for cmd in argvs:
        joined = " ".join(cmd)
        assert "set-status" not in joined
        assert "spawn_session" not in joined
        assert any("task.py" in c for c in cmd) and "post-marker" in cmd

    # Tick 2: unchanged events -> 0 new rows / markers / pushes (cursor +
    # flagged-key dedup, reloaded from the REAL tmp state file).
    assert asw.triage_observer_pass(dry_run=False) is False
    assert len(sidecar_path.read_text().splitlines()) == 1
    assert len([c for c in argvs if "post-marker" in c]) == 1
    assert len(pushes) == 1


def test_triage_observer_push_cap_caps_pushes_and_sends_one_digest(tmp_path, monkeypatch):
    # #1167: 8 matured warns with push cap 3 + marker cap 5 -> exactly 3
    # individual pushes + ONE trailing "+5 more" summary push; the marker
    # and sidecar channels are unaffected by the push cap; tick 2 on
    # unchanged events emits nothing new (suppressed pushes are flagged,
    # never deferred — the summary never re-fires for old violations).
    import json as _json
    import subprocess as _subprocess

    asw, state_path, sidecar_path, pushes = _triage_observer_sandbox(
        tmp_path, monkeypatch, _matured_violating_events_n(8)
    )
    monkeypatch.setattr(asw, "TRIAGE_OBSERVER_PUSH_CAP", 3)
    monkeypatch.setattr(asw, "TRIAGE_OBSERVER_MARKER_CAP", 5)
    argvs: list[list[str]] = []

    def _fake_run(cmd, *a, **kw):
        argvs.append([str(c) for c in cmd])
        return _subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)

    assert asw.triage_observer_pass(dry_run=False) is True
    rows = [_json.loads(line) for line in sidecar_path.read_text().splitlines()]
    # Overflow guard (A7 hedge): the fixture MUST overflow the push cap for
    # this test to cover anything; the exact counts below are the primary pin.
    assert len(rows) > 3
    assert len(rows) == 8  # criterion 3: the cap never drops sidecar fidelity
    assert all(r["severity"] == "warn" for r in rows)

    assert len(pushes) == 4  # 3 individual + 1 summary
    summary = pushes[-1][0]
    assert "+5 more" in summary
    assert "push cap (3)" in summary
    assert ".claude/cache/triage-observer-events.jsonl" in summary
    assert all("more warn flag" not in msg for msg, _ in pushes[:-1])

    # Criterion 4: the marker channel is unaffected by the push cap.
    assert len([c for c in argvs if "post-marker" in c]) == 5  # min(8, cap 5)

    # Non-gating pin: the recorded subprocess argv never mutates task state.
    for cmd in argvs:
        joined = " ".join(cmd)
        assert "set-status" not in joined
        assert "spawn_session" not in joined
        assert any("task.py" in c for c in cmd) and "post-marker" in cmd

    # Tick 2 on unchanged events: 0 new pushes / markers / sidecar rows.
    assert asw.triage_observer_pass(dry_run=False) is False
    assert len(sidecar_path.read_text().splitlines()) == 8
    assert len([c for c in argvs if "post-marker" in c]) == 5
    assert len(pushes) == 4
    assert state_path.exists()


def test_triage_observer_push_budget_threads_cross_task(tmp_path, monkeypatch):
    # #1167 criterion 6: ONE shared push budget for the whole pass (like
    # marker_budget) and ONE pass-level summary. 3 warns on #321 + 3 warns
    # on #322 with push cap 2 -> both individual pushes land on #321
    # (issue-id STRING order), #322 is fully suppressed, total = K+1 = 3
    # pushes. A forgotten cross-task decrement would emit 2+2 individual
    # pushes; a per-TASK summary would emit two summaries.
    import json as _json
    import subprocess as _subprocess

    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    reg_root = tmp_path / "repo"
    events_by_issue = {
        321: _matured_violating_events_n(3),
        322: _matured_violating_events_n(3),
    }
    reg_tasks = {}
    for issue in events_by_issue:
        task_rel = f"tasks/running/{issue}"
        task_dir = reg_root / task_rel
        task_dir.mkdir(parents=True)
        (task_dir / "events.jsonl").write_text("")
        reg_tasks[str(issue)] = {
            "status": "running",
            "path": task_rel,
            "kind": "experiment",
            "title": "synthetic",
            "has_clean_result": False,
        }
    reg_path = reg_root / "tasks" / "REGISTRY.json"
    reg_path.write_text(_json.dumps({"tasks": reg_tasks}))
    monkeypatch.setattr(task_workflow, "registry_path", lambda: reg_path)
    monkeypatch.setattr(task_workflow, "list_events", lambda issue: list(events_by_issue[issue]))
    state_path = tmp_path / "triage-observer.json"
    sidecar_path = tmp_path / "triage-observer-events.jsonl"
    monkeypatch.setattr(asw, "_triage_observer_state_path", lambda: state_path)
    monkeypatch.setattr(asw, "_triage_observer_sidecar_path", lambda: sidecar_path)
    pushes: list[tuple[str, bool]] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry: pushes.append((msg, dry)))
    monkeypatch.setattr(asw, "TRIAGE_OBSERVER_PUSH_CAP", 2)
    monkeypatch.setattr(asw, "TRIAGE_OBSERVER_MARKER_CAP", 10)
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda cmd, *a, **kw: _subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    assert asw.triage_observer_pass(dry_run=False) is True
    assert len(pushes) == 3  # K + 1: 2 individual + ONE pass-level summary
    assert all("#321" in msg for msg, _ in pushes[:2])  # string-order consumption
    assert "+4 more" in pushes[-1][0] and "push cap (2)" in pushes[-1][0]
    assert len(sidecar_path.read_text().splitlines()) == 6  # fidelity intact


def test_triage_observer_push_cap_summary_dry_run_zero_writes(tmp_path, monkeypatch):
    # #1167: the summary-push branch under dry_run routes through the same
    # _telegram_push(msg, dry_run) helper with zero filesystem writes and
    # zero subprocess spawns (the dry-run hygiene pin, >K-warn fixture).
    asw, state_path, sidecar_path, pushes = _triage_observer_sandbox(
        tmp_path, monkeypatch, _matured_violating_events_n(8)
    )
    monkeypatch.setattr(asw, "TRIAGE_OBSERVER_PUSH_CAP", 3)
    monkeypatch.setattr(asw, "TRIAGE_OBSERVER_MARKER_CAP", 5)
    calls: list = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **kw: calls.append(a))
    asw.triage_observer_pass(dry_run=True)
    assert calls == []
    assert not state_path.exists()
    assert not sidecar_path.exists()
    assert pushes and pushes[-1][1] is True and "+5 more" in pushes[-1][0]
    assert all(dry is True for _, dry in pushes)


def test_triage_observer_never_calls_in_process_mutators(tmp_path, monkeypatch):
    # Non-gating pin, prong (b): the in-process mutator surfaces reachable
    # via the lazy task_workflow import are NEVER touched — all task-state
    # mutation from this pass goes through the post-marker subprocess only.
    import subprocess as _subprocess

    from explore_persona_space import task_workflow

    asw, _state, sidecar_path, _pushes = _triage_observer_sandbox(
        tmp_path, monkeypatch, _matured_violating_events()
    )

    def _forbidden(*a, **kw):
        raise AssertionError("triage_observer_pass must never mutate task state in-process")

    monkeypatch.setattr(task_workflow, "set_status", _forbidden)
    monkeypatch.setattr(task_workflow, "post_event", _forbidden)
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda cmd, *a, **kw: _subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    assert asw.triage_observer_pass(dry_run=False) is True
    assert sidecar_path.exists()


def test_triage_observer_nudge_trap_strings():
    # The nudge must not itself become a window-closing boundary record
    # (no triage-line prefix) nor a breadcrumb-shaped note (no lstripped
    # `stage-dispatch ` prefix), and carries the anti-liveness watcher
    # sentinel prefix.
    import autonomous_session_watch as asw

    from explore_persona_space.task_workflow import TRIAGE_LINE_PREFIX

    base = {
        "record_ts": "2026-07-10T10:00:00Z",
        "candidate_count": 12,
        "candidate_kinds": ["epm:progress", "epm:results"],
        "signature_hits": ["# Audit"],
    }
    for v in (
        {
            **base,
            "record_kind": "epm:run-launched",
            "stage": None,
            "violation": "launch-missing-line",
        },
        {
            **base,
            "record_kind": "epm:progress",
            "stage": "followup-grid",
            "violation": "breadcrumb-missing-line",
        },
        {
            **base,
            "record_kind": "epm:progress",
            "stage": None,
            "violation": "none-with-candidates",
        },
    ):
        text = asw._triage_observer_nudge(v)
        assert TRIAGE_LINE_PREFIX not in text
        assert not text.lstrip().startswith("stage-dispatch ")
        assert text.startswith("[autonomous_session_watch:triage-observer]")


# ---------------------------------------------------------------------------
# #966 emitter-side --by convention: every watcher/spawn-helper post-marker
# subprocess site carries a distinctive --by identity (a trustworthy-positive
# EXTERNAL signal for the /issue pre-dispatch triage read; never added to
# TRIAGE_MACHINE_BY — see tests/test_pre_dispatch_triage.py).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name",
    [
        "_post_progress_marker",
        "_post_failure_marker",
        "_post_followup_run_marker",
        "_post_campaign_marker",
    ],
)
def test_post_marker_helpers_carry_distinctive_by(fn_name):
    """#966: each watcher post-marker argv sets ``--by autonomous_session_watch``
    (source-inspection pin — robust to the helpers' fixture-heavy signatures)."""
    import inspect
    import re

    import autonomous_session_watch as asw

    src = inspect.getsource(getattr(asw, fn_name))
    assert '"--by"' in src and '"autonomous_session_watch"' in src
    # Adjacency: "--by" is immediately followed by the identity value in the
    # argv list (not two unrelated occurrences elsewhere in the source).
    assert re.search(r'"--by",\s*\n\s*"autonomous_session_watch",', src), src


def test_spawn_session_duplicate_suppressed_marker_carries_distinctive_by():
    """#966 sibling pin: the spawn-helper's duplicate-dispatch-suppression post
    sets ``--by spawn_session`` (distinct from ``cmd_stop``'s
    ``spawn_session-stop`` — a different pass)."""
    import inspect
    import re

    src = inspect.getsource(spawn_session._post_duplicate_suppressed_marker)
    assert '"--by"' in src and '"spawn_session"' in src
    assert re.search(r'"--by",\s*\n\s*"spawn_session",', src), src
