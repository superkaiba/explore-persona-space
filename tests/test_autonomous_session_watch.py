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


# ─── pod-safety I/O wrapper tests ────────────────────────────────────────────
# The pure decision function is pinned above; here we cover the wrappers that
# actually touch state — false-stop branches and orphan-cleanup, which are
# exactly the paths where a bug leaks GPU spend or weakens the 2-miss guard.


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


def _write_state(reg_dir, issue, pod_id, missed, first_seen):
    import json

    (reg_dir / f"pod-safety-{issue}.json").write_text(
        json.dumps({"pod_id": pod_id, "missed": missed, "first_seen": first_seen})
    )


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
    # Secondary backstop: even if the API is flaky and a vanished pod somehow
    # still appears "running" (or the running-set was empty this tick due to a
    # transport error), a state file older than POD_SAFETY_STATE_MAX_AGE_S is
    # dropped on the not-in-running path. Pin the path by passing a "now"
    # past the cap and treating the pod as if it WERE still running — the age
    # path can only trigger when NOT in the running set, per the implementation;
    # so the realistic test is: file is old AND pod no longer in running set.
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


def test_pod_safety_pass_api_error_does_not_stop(isolated_registry, monkeypatch):
    # When `_running_managed_issue_pods` returns [] (transport error case, or
    # genuinely no pods), `pod_safety_pass` MUST NOT call `_stop_pod`. This is
    # the fail-closed invariant for the destructive action.
    import autonomous_session_watch as asw

    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)

    asw.pod_safety_pass(live_ids=set(), live_cwds=set(), dry_run=False, threshold=2)

    assert stops == []


def test_pod_safety_pass_stops_only_on_second_miss_and_clears_state(isolated_registry, monkeypatch):
    # End-to-end of the 2-miss guard via the I/O wrapper: tick 1 increments to
    # missed=1 (no stop), tick 2 hits threshold and calls _stop_pod ONCE, and
    # the state file is cleared (so a re-used pod starts fresh).
    import autonomous_session_watch as asw

    stops: list[int] = []
    posts: list[tuple[int, str]] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(137, "abc123")])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(
        asw,
        "_post_pod_stopped_marker",
        lambda issue, pod_id, note, dry_run: posts.append((issue, pod_id)),
    )

    # Tick 1: session dead, missed 0 -> 1, no stop, state persisted.
    asw.pod_safety_pass(live_ids=set(), live_cwds=set(), dry_run=False, threshold=2)
    state_path = isolated_registry / "pod-safety-137.json"
    assert stops == []
    assert state_path.exists()
    import json

    assert json.loads(state_path.read_text())["missed"] == 1

    # Tick 2: session still dead, missed 1 -> threshold -> stop fires + state cleared.
    asw.pod_safety_pass(live_ids=set(), live_cwds=set(), dry_run=False, threshold=2)
    assert stops == [137]
    assert posts == [(137, "abc123")]
    assert not state_path.exists()


def test_pod_safety_pass_alive_session_clears_state_no_stop(isolated_registry, monkeypatch):
    # A pod that accumulated 1 miss but whose driving session came back: keep,
    # clear state (so a brief liveness blip cannot push it past threshold on
    # the next dead tick), no stop call.
    import autonomous_session_watch as asw

    _write_state(isolated_registry, 137, "abc123", missed=1, first_seen=__import__("time").time())
    stops: list[int] = []
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [(137, "abc123")])
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    # Worktree cwd matches issue-137 -> session_alive=True via _worktree_session_alive.
    asw.pod_safety_pass(
        live_ids=set(),
        live_cwds={"/repo/.claude/worktrees/issue-137"},
        dry_run=False,
        threshold=2,
    )

    assert stops == []
    assert not (isolated_registry / "pod-safety-137.json").exists()


def test_pod_safety_pass_gc_runs_even_with_no_running_pods(isolated_registry, monkeypatch):
    # GC must fire BEFORE the `if not running: return` early-out; otherwise a
    # tick where every managed pod has vanished would never clean up its state.
    import autonomous_session_watch as asw

    _write_state(isolated_registry, 99, "gone", missed=1, first_seen=__import__("time").time())
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda: [])

    asw.pod_safety_pass(live_ids=set(), live_cwds=set(), dry_run=False, threshold=2)

    assert not (isolated_registry / "pod-safety-99.json").exists()


def test_main_daemon_unreachable_short_circuits(isolated_registry, monkeypatch):
    # If the Happy daemon is unreachable, BOTH passes must skip — neither
    # respawn nor pod-stop is safe when liveness can't be judged. Verify by
    # spying on the pod-safety pass (the respawn pass guards itself by the
    # registry being empty in this tmp-dir setup).
    import autonomous_session_watch as asw

    pod_safety_calls: list[tuple] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: pod_safety_calls.append((a, kw)))

    rc = asw.main([])

    assert rc == 0
    assert pod_safety_calls == []  # daemon-unreachable short-circuit before pod-safety


def test_save_pod_safety_state_carries_first_seen_forward(isolated_registry):
    import json

    import autonomous_session_watch as asw

    asw._save_pod_safety_state(7, "pod-7", missed=1, prev={"first_seen": 1234.0})
    payload = json.loads((isolated_registry / "pod-safety-7.json").read_text())
    assert payload == {"pod_id": "pod-7", "missed": 1, "first_seen": 1234.0}

    # On a second save (passing the previous payload), first_seen must persist.
    asw._save_pod_safety_state(7, "pod-7", missed=2, prev=payload)
    payload2 = json.loads((isolated_registry / "pod-safety-7.json").read_text())
    assert payload2["first_seen"] == 1234.0
    assert payload2["missed"] == 2
