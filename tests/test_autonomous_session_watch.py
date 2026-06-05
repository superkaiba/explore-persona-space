"""Decision-matrix tests for the autonomous-session crash-recovery watcher.

The watcher re-spawns autonomous `/issue` sessions, and a wrong RESPAWN can
launch a duplicate session -> a duplicate pod -> real spend. The whole respawn
gate is the pure :func:`decide` function, so it is pinned exhaustively here.
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
    PARK,
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
# Stopping a pod is reversible (pod.py stop preserves the volume), but a wrong
# stop still interrupts a live experiment, so the gate (decide_pod_safety) is
# pinned exhaustively like decide().


@pytest.mark.parametrize("session_alive", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_pod_safety_ignores_non_running(session_alive, missed):
    # A pod that is not RUNNING is outside this pass's remit (pod_audit owns the
    # EXITED / stale buckets) — always ignore, miss counter reset to 0.
    assert decide_pod_safety(
        pod_running=False, managed=True, session_alive=session_alive, missed=missed
    ) == ("ignore", 0)


@pytest.mark.parametrize("session_alive", [True, False])
@pytest.mark.parametrize("missed", [0, 1, 5])
def test_pod_safety_ignores_unmanaged(session_alive, missed):
    # A non-managed pod name is never stopped by this pass — always ignore.
    assert decide_pod_safety(
        pod_running=True, managed=False, session_alive=session_alive, missed=missed
    ) == ("ignore", 0)


@pytest.mark.parametrize("missed", [0, 1, 5])
def test_pod_safety_keeps_when_session_alive(missed):
    # A live driving session means hands off — keep, and reset the miss counter
    # so a future episode of deadness starts clean.
    assert decide_pod_safety(pod_running=True, managed=True, session_alive=True, missed=missed) == (
        "keep",
        0,
    )


def test_pod_safety_needs_two_misses_before_stop():
    # First dead check only increments; the stop fires on the SECOND consecutive
    # miss (default threshold 2) — guards a transient daemon-list / cwd glitch.
    assert decide_pod_safety(
        pod_running=True, managed=True, session_alive=False, missed=0, threshold=2
    ) == ("keep", 1)
    assert decide_pod_safety(
        pod_running=True, managed=True, session_alive=False, missed=1, threshold=2
    ) == ("stop", 0)


def test_pod_safety_threshold_one_stops_immediately():
    assert decide_pod_safety(
        pod_running=True, managed=True, session_alive=False, missed=0, threshold=1
    ) == ("stop", 0)


def test_pod_safety_higher_threshold_delays_stop():
    assert decide_pod_safety(
        pod_running=True, managed=True, session_alive=False, missed=1, threshold=3
    ) == ("keep", 2)
    assert decide_pod_safety(
        pod_running=True, managed=True, session_alive=False, missed=2, threshold=3
    ) == ("stop", 0)


def test_pod_safety_resets_when_session_revives():
    # A pod that accumulated misses but whose session came back is kept with the
    # counter reset — so a brief liveness blip cannot push it over the threshold
    # on the next dead tick.
    assert decide_pod_safety(
        pod_running=True, managed=True, session_alive=True, missed=1, threshold=2
    ) == ("keep", 0)


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
