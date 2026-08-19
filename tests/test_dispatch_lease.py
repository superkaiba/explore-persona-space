"""Tests for the per-issue dispatch lease + registration collision-fail (#843).

Pins the M1/M2 contract in ``scripts/spawn_session.py``:

- **M1 lease primitive** — atomic create-or-fail claim (`acquire_dispatch_lease`,
  ``os.open(O_CREAT|O_EXCL)`` fast path; flock-sidecar-serialized single-winner
  stale takeover), per-issue granularity, TTL freshness (env override), a
  garbled-but-fresh-mtime lease fails CLOSED (blocks until TTL), token-verified
  release, and the 8-process barrier-synced race repro (exactly one winner —
  the acceptance smoke).
- **M1 chokepoint wiring** — `spawn-issue --auto` acquires BEFORE the daemon
  POST; a loser exits 0 with the loud ``DISPATCH-LEASE HELD`` line and no POST;
  a SUCCESSFUL spawn LEAVES the lease in place, so a sequential second attempt
  within TTL is suppressed (test 8b — the load-bearing no-release-on-success
  invariant; the #833 3-min duplicate shape); a spawn FAILURE releases it;
  manual spawns are not gated and create no lease.
- **M2 collision-fail** — a fresh (< 900 s) ``issue-<N>.json`` naming a
  DIFFERENT session makes registration raise; `cmd_spawn_issue` stops the
  just-spawned duplicate, keeps the first registration byte-identical, exits 0
  with ``REGISTRATION-COLLISION``, best-effort posts the suppression marker,
  and HOLDS the lease. An old entry / ``force=True`` overwrites as before, and
  the collision window is capped at the 900 s default even when
  ``EPM_DISPATCH_LEASE_TTL_S`` is raised (so a longer lease can never suppress
  a legitimate crash-recovery respawn).

Every daemon POST / task.py subprocess is mocked; the registry dir is a
tmp_path. The multiprocessing race test pins the ``fork`` start method AND
passes the tmp registry dir explicitly (hermeticity: under ``spawn`` a
re-import would silently race on the REAL ``~/.eps-autonomous``).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import spawn_session  # noqa: E402

# A high issue number with no worktree so cmd_spawn_issue resolves cwd to the
# repo root deterministically.
ISSUE = 424242


@pytest.fixture
def lease_registry(tmp_path, monkeypatch):
    """Point spawn_session.AUTONOMOUS_REGISTRY_DIR at a tmp dir (the lease
    helpers read the module global at call time)."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


# ─── M1 primitive ─────────────────────────────────────────────────────────────


def test_acquire_creates_lease_with_expected_fields(lease_registry):
    entry = spawn_session.acquire_dispatch_lease(ISSUE, holder="test-holder")
    assert entry is not None
    on_disk = json.loads(spawn_session.dispatch_lease_path(ISSUE).read_text())
    assert on_disk == entry
    assert on_disk["issue"] == ISSUE
    assert on_disk["holder"] == "test-holder"
    assert on_disk["pid"] == os.getpid()
    assert isinstance(on_disk["token"], str) and on_disk["token"]
    assert isinstance(on_disk["acquired_at"], float)


def test_second_acquire_fresh_lease_returns_none(lease_registry):
    first = spawn_session.acquire_dispatch_lease(ISSUE, holder="w1")
    assert first is not None
    before = spawn_session.dispatch_lease_path(ISSUE).read_bytes()
    assert spawn_session.acquire_dispatch_lease(ISSUE, holder="w2") is None
    # Loser must not have touched the winner's lease.
    assert spawn_session.dispatch_lease_path(ISSUE).read_bytes() == before


def test_per_issue_granularity(lease_registry):
    assert spawn_session.acquire_dispatch_lease(1, holder="a") is not None
    assert spawn_session.acquire_dispatch_lease(2, holder="b") is not None
    assert spawn_session.dispatch_lease_path(1).exists()
    assert spawn_session.dispatch_lease_path(2).exists()


def test_stale_lease_takeover(lease_registry):
    now = time.time()
    ttl = spawn_session._dispatch_lease_ttl_s()
    stale = {"issue": ISSUE, "holder": "old", "pid": 1, "token": "t0", "acquired_at": now - ttl - 1}
    spawn_session.dispatch_lease_path(ISSUE).write_text(json.dumps(stale))
    entry = spawn_session.acquire_dispatch_lease(ISSUE, holder="new", now=now)
    assert entry is not None and entry["holder"] == "new"
    on_disk = json.loads(spawn_session.dispatch_lease_path(ISSUE).read_text())
    assert on_disk["holder"] == "new"
    # The slow path leaves the PERMANENT flock sidecar behind (no tombstone
    # exists in this protocol).
    assert spawn_session._dispatch_lease_lock_path(ISSUE).exists()


def _race_worker(registry_dir: str, rounds: int, barrier, queue) -> None:
    """Barrier-synced acquire worker for the multiprocessing race repro.

    Sets the registry dir EXPLICITLY from the passed string (hermeticity: do
    not rely on inheriting the parent's monkeypatched module state), then
    races one acquire per round on a per-round issue number."""
    spawn_session.AUTONOMOUS_REGISTRY_DIR = Path(registry_dir)
    for r in range(rounds):
        barrier.wait()
        entry = spawn_session.acquire_dispatch_lease(1000 + r, holder=f"pid-{os.getpid()}")
        queue.put((r, entry is not None))


def test_concurrent_acquires_exactly_one_winner(lease_registry):
    # THE race repro / acceptance smoke: 8 barrier-synced processes, 20
    # rounds (a fresh issue per round) -> exactly 1 winner per round.
    # `fork` is pinned: under `spawn` the children would re-import
    # spawn_session and (without the explicit dir set) race on the REAL
    # ~/.eps-autonomous.
    ctx = multiprocessing.get_context("fork")
    n_workers, rounds = 8, 20
    barrier = ctx.Barrier(n_workers)
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_race_worker, args=(str(lease_registry), rounds, barrier, queue))
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()
    results: list[tuple[int, bool]] = []
    for _ in range(n_workers * rounds):
        results.append(queue.get(timeout=60))
    for w in workers:
        w.join(timeout=60)
        assert w.exitcode == 0
    for r in range(rounds):
        winners = sum(1 for rr, won in results if rr == r and won)
        assert winners == 1, f"round {r}: expected exactly 1 winner, got {winners}"


def _stale_race_worker(registry_dir: str, barrier, queue) -> None:
    """Worker for the stale-takeover contention repro: all workers race the
    takeover of ONE pre-seeded stale lease."""
    spawn_session.AUTONOMOUS_REGISTRY_DIR = Path(registry_dir)
    barrier.wait()
    entry = spawn_session.acquire_dispatch_lease(42, holder=f"pid-{os.getpid()}")
    queue.put(entry is not None)


def test_concurrent_stale_takeover_exactly_one_winner(lease_registry):
    # Single-winner takeover under contention (the round-1 TOCTOU fix): 8
    # processes race a pre-seeded STALE lease; the flock-serialized slow path
    # admits exactly one.
    ttl = spawn_session._dispatch_lease_ttl_s()
    stale = {"issue": 42, "holder": "old", "pid": 1, "token": "t0", "acquired_at": 1.0}
    spawn_session.dispatch_lease_path(42).write_text(json.dumps(stale))
    os.utime(spawn_session.dispatch_lease_path(42), (time.time() - ttl - 60,) * 2)
    ctx = multiprocessing.get_context("fork")
    n_workers = 8
    barrier = ctx.Barrier(n_workers)
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_stale_race_worker, args=(str(lease_registry), barrier, queue))
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()
    wins = [queue.get(timeout=60) for _ in range(n_workers)]
    for w in workers:
        w.join(timeout=60)
        assert w.exitcode == 0
    assert sum(wins) == 1
    assert json.loads(spawn_session.dispatch_lease_path(42).read_text())["holder"] != "old"


def test_garbled_lease_fresh_mtime_blocks_stale_mtime_takes_over(lease_registry):
    path = spawn_session.dispatch_lease_path(ISSUE)
    path.write_text("not json{{{")
    # Fresh mtime -> fail CLOSED (treated fresh; blocks dispatch).
    assert spawn_session.acquire_dispatch_lease(ISSUE, holder="w") is None
    assert path.read_text() == "not json{{{"
    # mtime pushed past TTL -> the takeover proceeds.
    ttl = spawn_session._dispatch_lease_ttl_s()
    os.utime(path, (time.time() - ttl - 60,) * 2)
    entry = spawn_session.acquire_dispatch_lease(ISSUE, holder="w")
    assert entry is not None
    assert json.loads(path.read_text())["holder"] == "w"


def test_release_only_with_matching_token(lease_registry):
    entry = spawn_session.acquire_dispatch_lease(ISSUE, holder="w")
    assert entry is not None
    spawn_session.release_dispatch_lease(ISSUE, "wrong-token")
    assert spawn_session.dispatch_lease_path(ISSUE).exists()
    spawn_session.release_dispatch_lease(ISSUE, entry["token"])
    assert not spawn_session.dispatch_lease_path(ISSUE).exists()


def test_ttl_env_override_and_malformed_fallback(monkeypatch):
    monkeypatch.setenv("EPM_DISPATCH_LEASE_TTL_S", "60")
    assert spawn_session._dispatch_lease_ttl_s() == 60.0
    monkeypatch.setenv("EPM_DISPATCH_LEASE_TTL_S", "not-a-number")
    assert spawn_session._dispatch_lease_ttl_s() == float(spawn_session.DISPATCH_LEASE_TTL_S)
    monkeypatch.delenv("EPM_DISPATCH_LEASE_TTL_S")
    assert spawn_session._dispatch_lease_ttl_s() == float(spawn_session.DISPATCH_LEASE_TTL_S)


def test_contended_takeover_lock_returns_none(lease_registry):
    # A held sidecar flock (another taker mid-takeover) -> acquire returns
    # None WITHOUT unlinking the stale lease (skip-this-tick semantics).
    import fcntl

    ttl = spawn_session._dispatch_lease_ttl_s()
    stale = {"issue": ISSUE, "holder": "old", "pid": 1, "token": "t0", "acquired_at": 1.0}
    path = spawn_session.dispatch_lease_path(ISSUE)
    path.write_text(json.dumps(stale))
    os.utime(path, (time.time() - ttl - 60,) * 2)
    lock_fd = os.open(spawn_session._dispatch_lease_lock_path(ISSUE), os.O_CREAT | os.O_WRONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert spawn_session.acquire_dispatch_lease(ISSUE, holder="w") is None
        assert path.exists()  # the stale lease was NOT unlinked
    finally:
        os.close(lock_fd)


# ─── cmd_spawn_issue integration (M1 wiring + M2 collision) ───────────────────


def _install_daemon_mock(monkeypatch, *, spawn_success=True):
    """Mock `spawn_session.post` + `_verify_happy_patch_or_die` + the marker
    subprocess. Returns ``(posts, marker_calls, lease_at_post)`` recorders:
    every ``post()`` call (path, body), every ``subprocess.run`` argv, and —
    for each /spawn-session POST — whether the lease file already existed."""
    posts: list[tuple[str, dict]] = []
    marker_calls: list[list[str]] = []
    lease_at_post: list[bool] = []

    def _fake_post(path, body):
        posts.append((path, body))
        if path == "/spawn-session":
            issue = body.get("_test_issue", ISSUE)
            lease_at_post.append(spawn_session.dispatch_lease_path(issue).exists())
            if not spawn_success:
                return {"success": False}
            return {"success": True, "sessionId": f"sid-new-{len(posts)}"}
        return {"success": True}

    monkeypatch.setattr(spawn_session, "post", _fake_post)
    monkeypatch.setattr(spawn_session, "_verify_happy_patch_or_die", lambda **kw: None)

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        spawn_session.subprocess,
        "run",
        lambda cmd, **kw: marker_calls.append(list(cmd)) or _FakeCompleted(),
    )
    return posts, marker_calls, lease_at_post


def _spawn_posts(posts):
    return [p for p in posts if p[0] == "/spawn-session"]


def _stop_posts(posts):
    return [p for p in posts if p[0] == "/stop-session"]


def test_auto_spawn_acquires_lease_before_post(lease_registry, monkeypatch):
    posts, _markers, lease_at_post = _install_daemon_mock(monkeypatch)
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])
    assert len(_spawn_posts(posts)) == 1
    assert lease_at_post == [True]  # the lease existed when the POST fired


def test_auto_spawn_success_leaves_lease_and_suppresses_sequential_second(
    lease_registry, monkeypatch, capsys
):
    # Test 8b — the load-bearing no-release-on-success invariant (the #833
    # sequential-duplicate shape): a release-on-success implementation would
    # pass tests 8/9 and still admit sequential duplicates.
    posts, _markers, _lap = _install_daemon_mock(monkeypatch)
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])
    assert spawn_session.dispatch_lease_path(ISSUE).exists()  # lease persists
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])
    assert len(_spawn_posts(posts)) == 1  # NO second daemon POST
    out = capsys.readouterr().out
    assert "DISPATCH-LEASE HELD" in out


def test_auto_spawn_on_fresh_lease_no_post_exit_zero_loud(lease_registry, monkeypatch, capsys):
    assert spawn_session.acquire_dispatch_lease(ISSUE, holder="other-dispatcher") is not None
    posts, _markers, _lap = _install_daemon_mock(monkeypatch)
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])  # no SystemExit
    assert _spawn_posts(posts) == []
    out = capsys.readouterr().out
    assert "DISPATCH-LEASE HELD" in out
    assert "holder=other-dispatcher" in out


def test_manual_spawn_ignores_lease(lease_registry, monkeypatch, capsys):
    # (a) With a fresh lease: proceeds (warn only).
    assert spawn_session.acquire_dispatch_lease(ISSUE, holder="auto-dispatcher") is not None
    posts, _markers, _lap = _install_daemon_mock(monkeypatch)
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE)])
    assert len(_spawn_posts(posts)) == 1
    out = capsys.readouterr().out
    assert "a fresh dispatch lease exists" in out
    # (b) With NO lease: a manual spawn creates none (acquisition is
    # --auto-gated).
    other = ISSUE + 1
    spawn_session.main(["spawn-issue", "--issue", str(other)])
    assert not spawn_session.dispatch_lease_path(other).exists()


def test_spawn_failure_releases_lease(lease_registry, monkeypatch):
    _posts, _markers, _lap = _install_daemon_mock(monkeypatch, spawn_success=False)
    with pytest.raises(SystemExit):
        spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])
    assert not spawn_session.dispatch_lease_path(ISSUE).exists()  # slot freed


def test_registration_collision_stops_duplicate(lease_registry, monkeypatch, capsys):
    # Pre-seed a FRESH registration naming a DIFFERENT session id.
    reg = lease_registry / f"issue-{ISSUE}.json"
    original = json.dumps(
        {"issue": ISSUE, "happy_session_id": "sid-first", "spawned_at": time.time(), "missed": 0}
    )
    reg.write_text(original)
    posts, marker_calls, _lap = _install_daemon_mock(monkeypatch)
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])  # exit 0, no raise
    # The duplicate we just spawned was stopped, with the NEW sid.
    stops = _stop_posts(posts)
    assert len(stops) == 1
    assert stops[0][1]["sessionId"] == "sid-new-1"
    # The FIRST registration is byte-identical.
    assert reg.read_text() == original
    out = capsys.readouterr().out
    assert "REGISTRATION-COLLISION" in out
    assert "sid-first" in out
    # The best-effort suppression marker subprocess was attempted.
    assert any("post-marker" in c for call in marker_calls for c in call)
    note = next(c for call in marker_calls for c in call if "duplicate" in c)
    assert spawn_session.DUPLICATE_DISPATCH_NOTE_SENTINEL in note
    # The lease is deliberately HELD after the collision exit (§3.B): a
    # session IS live and driving; holding suppresses re-spawn churn.
    assert spawn_session.dispatch_lease_path(ISSUE).exists()


def test_registration_overwrite_allowed_when_entry_old(lease_registry, monkeypatch):
    # An entry at/past the 900 s window is the crash-recovery respawn case —
    # overwrite proceeds exactly as before.
    reg = lease_registry / f"issue-{ISSUE}.json"
    reg.write_text(
        json.dumps(
            {
                "issue": ISSUE,
                "happy_session_id": "sid-old",
                "spawned_at": time.time() - 901.0,
                "missed": 2,
            }
        )
    )
    posts, _markers, _lap = _install_daemon_mock(monkeypatch)
    spawn_session.main(["spawn-issue", "--issue", str(ISSUE), "--auto"])
    assert _stop_posts(posts) == []  # no collision remediation
    assert json.loads(reg.read_text())["happy_session_id"] == "sid-new-1"


def test_register_current_force_overwrites(lease_registry):
    # force=True (the register-current path) bypasses the collision check for
    # an already-live session re-registration (#472 revival).
    reg = lease_registry / "issue-77.json"
    reg.write_text(
        json.dumps({"issue": 77, "happy_session_id": "sid-a", "spawned_at": time.time()})
    )
    spawn_session._register_autonomous_session(77, "sid-b", "/tmp", 100.0, force=True)
    assert json.loads(reg.read_text())["happy_session_id"] == "sid-b"
    # And without force, the same write raises.
    with pytest.raises(spawn_session.RegistrationCollisionError):
        spawn_session._register_autonomous_session(77, "sid-c", "/tmp", 100.0)


def test_collision_window_capped_at_default(lease_registry, monkeypatch):
    # Raising EPM_DISPATCH_LEASE_TTL_S lengthens the LEASE but must never
    # widen the M2 collision window past the 900 s respawn grace (round-1
    # env-decoupling hardening): an entry aged 1200 s overwrites fine even
    # under a 3600 s lease TTL; an entry aged 500 s still collides.
    monkeypatch.setenv("EPM_DISPATCH_LEASE_TTL_S", "3600")
    reg = lease_registry / "issue-78.json"
    reg.write_text(
        json.dumps({"issue": 78, "happy_session_id": "sid-a", "spawned_at": time.time() - 1200.0})
    )
    spawn_session._register_autonomous_session(78, "sid-b", "/tmp", 100.0)  # no raise
    assert json.loads(reg.read_text())["happy_session_id"] == "sid-b"
    reg.write_text(
        json.dumps({"issue": 78, "happy_session_id": "sid-c", "spawned_at": time.time() - 500.0})
    )
    with pytest.raises(spawn_session.RegistrationCollisionError):
        spawn_session._register_autonomous_session(78, "sid-d", "/tmp", 100.0)


# ─── spawn_output_suppressed (M1b shared helper) ──────────────────────────────


def test_spawn_output_suppressed_matches_both_sentinels():
    assert (
        spawn_session.spawn_output_suppressed("DISPATCH-LEASE HELD issue #7: in flight")
        == spawn_session.DISPATCH_LEASE_HELD_SENTINEL
    )
    assert (
        spawn_session.spawn_output_suppressed("...\nREGISTRATION-COLLISION issue #7: kept sid-a")
        == spawn_session.REGISTRATION_COLLISION_SENTINEL
    )
    assert spawn_session.spawn_output_suppressed("Issue #7 session spawned: sid") is None
    assert spawn_session.spawn_output_suppressed("") is None
    assert spawn_session.spawn_output_suppressed(None) is None


# ─── router Lease.free_lane_park_state persistence (#2161) ────────────────────
# NOTE: this pins the ROUTER's durable per-issue routing lease
# (``~/.eps-routing/issue-<N>.json``; ``explore_persona_space.backends.router
# .Lease``) — a DIFFERENT lease family from the spawn-session dispatch lease
# the rest of this file covers. Colocated here per the #2161 plan's test map.


def test_lease_free_lane_park_state_round_trip_and_tolerant_parse():
    """#2161: the resumable park state round-trips through to_json/from_json;
    a malformed (non-dict) payload parses tolerantly to None; a legacy
    payload without the key defaults to None."""
    from explore_persona_space.backends.router import Lease

    state = {
        "lane": "fellows",
        "job_id": "31337",
        "rung_idx": 1,
        "rung_park_elapsed_s": 123.5,
        "spec_hash": "abc123",
        "updated_ts": 1_700_000_000.0,
    }
    lease = Lease(issue=137, spec_hash="abc123", attempt_id="a1", free_lane_park_state=state)
    payload = lease.to_json()
    assert payload["free_lane_park_state"] == state
    round_tripped = Lease.from_json(json.loads(json.dumps(payload)))
    assert round_tripped.free_lane_park_state == state

    # job_id None (budget cut between rungs) round-trips too.
    pending = {**state, "job_id": None}
    lease_p = Lease(issue=137, spec_hash="abc123", attempt_id="a1", free_lane_park_state=pending)
    assert Lease.from_json(lease_p.to_json()).free_lane_park_state == pending

    # Tolerant parse: a malformed (non-dict) value reads as None, never raises.
    garbled = lease.to_json()
    garbled["free_lane_park_state"] = "not-a-dict"
    assert Lease.from_json(garbled).free_lane_park_state is None

    # Legacy payload (pre-#2161, key absent) defaults to None.
    legacy = lease.to_json()
    del legacy["free_lane_park_state"]
    assert Lease.from_json(legacy).free_lane_park_state is None


# ─── #2142 dispatch-loop freshness + cap re-check guards ─────────────────────
#
# Loop-driving tests below exercise the REAL infra_drain_pass /
# proposed_infra_sweep_pass guard loops with every task.py / daemon /
# occupancy seam stubbed. The watcher binds its OWN copy of
# AUTONOMOUS_REGISTRY_DIR (`from spawn_session import ...`), so these tests
# patch BOTH modules' globals (`watcher_registry`) — patching only
# spawn_session's would leave the watcher reading the real ~/.eps-autonomous.

W_ISSUE = 515151
W_ISSUE_B = 515152
W_ISSUE_C = 515153


def _iso(ts: float) -> str:
    """Epoch -> the canonical task-event ts format (%Y-%m-%dT%H:%M:%SZ)."""
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def watcher_registry(tmp_path, monkeypatch):
    """tmp registry for BOTH modules (#2142 plan §4 fixture note): the lease
    helpers read spawn_session's module global at call time; the watcher's
    registration / queue / state helpers read autonomous_session_watch's own
    imported binding. Also clears the dispatch-loop env knobs so a live
    operator override can't leak into the assertions."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    for var in (
        "EPM_DISABLE_INFRA_DRAIN",
        "EPM_DISABLE_PROPOSED_INFRA_SWEEP",
        "EPM_DISPATCH_REG_FRESH_S",
        "EPM_PROPOSED_INFRA_SWEEP_MARKER_FRESH_S",
        "EPM_INFRA_SWEEP_URGENT_BONUS",
        "EPM_INFRA_DRAIN_BACKOFF_S",
        "EPM_INFRA_DRAIN_MAX_ATTEMPTS",
        "EPM_PROPOSED_INFRA_SWEEP_BACKOFF_S",
        "EPM_PROPOSED_INFRA_SWEEP_MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write_queue(reg_dir: Path, ids: list[int], *, cap: int = 3) -> None:
    (reg_dir / "infra-drain-queue.json").write_text(
        json.dumps(
            {
                "ripe_oldest_first": ids,
                "cap": cap,
                "holds": {},
                "updated_ts": "2026-06-12T22:40:00Z",
                "updated_by": "pm-session-drain-tick",
                "comment": "#2142 test fixture",
            }
        )
    )


def _write_registration(reg_dir: Path, issue: int, *, spawned_at: float) -> None:
    (reg_dir / f"issue-{issue}.json").write_text(
        json.dumps({"happy_session_id": "owner-sid", "spawned_at": spawned_at, "cwd": "/tmp"})
    )


def _marker_recorder(monkeypatch) -> list[tuple[int, str]]:
    """Recorder for the _post_progress_marker seam (satisfies the #1247
    conftest hermeticity guard: a later test-level monkeypatch wins)."""
    markers: list[tuple[int, str]] = []

    def _record(issue, note, dry_run, *, label, by="autonomous_session_watch"):
        markers.append((issue, label))

    monkeypatch.setattr(asw, "_post_progress_marker", _record)
    return markers


def _stub_loop_seams(
    monkeypatch,
    ids: list[int],
    *,
    stale: frozenset[int] | set[int] = frozenset(),
    events: list[dict] | None = None,
    occupancy_seq: list | None = None,
    dispatch_result="spawned",
) -> tuple[list[int], list[tuple[int, str]]]:
    """Stub the task.py-backed seams BOTH dispatch loops consume; returns the
    (dispatched, markers) recorders. ``occupancy_seq`` is consumed one element
    per ``_infra_drain_occupancy`` call (the pass-level read first, then one
    per #2142 mid-batch re-check); the LAST element repeats once exhausted.
    ``dispatch_result`` is a tri-state string or a per-issue dict of them."""
    sk = {i: ("proposed", "infra") for i in ids}
    monkeypatch.setattr(
        asw, "_infra_drain_signals", lambda cand, holds, regs, now: (sk, set(stale), 0)
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: list(events or []))
    seq = list(occupancy_seq) if occupancy_seq is not None else [[]]
    calls = {"n": 0}

    def _occupancy():
        idx = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[idx]

    monkeypatch.setattr(asw, "_infra_drain_occupancy", _occupancy)
    dispatched: list[int] = []

    def _fake_dispatch(issue, slot_desc, dry_run, **kwargs):
        dispatched.append(issue)
        if isinstance(dispatch_result, dict):
            return dispatch_result.get(issue, "spawned")
        return dispatch_result

    monkeypatch.setattr(asw, "_dispatch_infra_drain", _fake_dispatch)
    markers = _marker_recorder(monkeypatch)
    return dispatched, markers


def _run_drain(reg_dir: Path, ids: list[int], *, cap: int = 3, now: float, dry_run: bool = False):
    _write_queue(reg_dir, ids, cap=cap)
    asw.infra_drain_pass(dry_run, now=now, daemon_reachable=True)


def _run_sweep(
    monkeypatch,
    candidates: list[int],
    *,
    urgent: set[int] | frozenset[int] = frozenset(),
    cap: int = 3,
    now: float,
    dry_run: bool = False,
):
    monkeypatch.setattr(
        asw, "_proposed_infra_candidates", lambda: (list(candidates), frozenset(urgent))
    )
    monkeypatch.setattr(
        asw,
        "_infra_drain_read_queue",
        lambda: {"ids": [], "cap": cap, "holds": {}, "updated_ts": None},
    )
    asw.proposed_infra_sweep_pass(dry_run, now=now, daemon_reachable=True)


def _drain_attempt_ids(reg_dir: Path) -> list[str]:
    path = reg_dir / "infra-drain-state.json"
    return sorted(json.loads(path.read_text())["attempts"]) if path.exists() else []


def _sweep_attempt_ids(reg_dir: Path) -> list[str]:
    path = reg_dir / "proposed-infra-sweep-state.json"
    return sorted(json.loads(path.read_text())["attempts"]) if path.exists() else []


# ─── §1: post-stagger lease re-check ──────────────────────────────────────────


def test_pre_spawn_lease_recheck_suppresses_after_stagger(watcher_registry, monkeypatch, capsys):
    """§1: a lease acquired DURING the stagger sleep is caught by the
    post-stagger re-check — result "suppressed" (NOT "failed"): no spawn
    subprocess, no dispatch marker, no attempt/backoff booked."""
    now = time.time()
    markers = _marker_recorder(monkeypatch)
    sk = {W_ISSUE: ("proposed", "infra")}
    monkeypatch.setattr(asw, "_infra_drain_signals", lambda *a: (sk, set(), 0))
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_auth_outage_spawn_gate", lambda *a, **k: None)
    # Force a full stagger window; the rival acquires the lease DURING it.
    monkeypatch.setattr(asw, "session_dispatch_stagger_s", lambda: 60.0)
    monkeypatch.setattr(asw, "last_session_dispatch_age_s", lambda: 0.0)
    monkeypatch.setattr(
        asw,
        "_stagger_sleep",
        lambda seconds: spawn_session.acquire_dispatch_lease(W_ISSUE, holder="rival"),
    )
    run_calls: list = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: run_calls.append(a))
    _run_drain(watcher_registry, [W_ISSUE], now=now)
    out = capsys.readouterr().out
    assert f"INFRA-DRAIN SUPPRESSED issue #{W_ISSUE} (lease acquired during" in out
    assert "DISPATCHED" not in out
    assert run_calls == []  # the spawn subprocess never ran
    assert markers == []  # M1b no-booking: a suppressed no-op posts no marker
    assert _drain_attempt_ids(watcher_registry) == []  # no attempt, no backoff


def test_pre_spawn_lease_recheck_reads_lease_file_fresh(watcher_registry, monkeypatch, capsys):
    """§1: the load-bearing element is the fresh FILE READ — with a ZERO
    stagger delay (no sleep at all) a lease already on disk still suppresses
    at the post-stagger re-check inside _dispatch_infra_drain (the caller
    loop's earlier verdict is not what protects)."""
    monkeypatch.setattr(asw, "_auth_outage_spawn_gate", lambda *a, **k: None)
    monkeypatch.setattr(asw, "last_session_dispatch_age_s", lambda: None)  # delay 0
    slept: list = []
    monkeypatch.setattr(asw, "_stagger_sleep", lambda seconds: slept.append(seconds))
    run_calls: list = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: run_calls.append(a))
    assert spawn_session.acquire_dispatch_lease(W_ISSUE, holder="rival") is not None
    result = asw._dispatch_infra_drain(W_ISSUE, "slot 1/3", False)
    assert result == "suppressed"
    assert slept == []  # no stagger elapsed — the fresh read did the work
    assert run_calls == []
    assert "lease acquired during" in capsys.readouterr().out


# ─── §2: registration-freshness guard ─────────────────────────────────────────


@pytest.mark.parametrize("age_s", [30.0, 59.0])
def test_fresh_registration_skipped_drain_loop(watcher_registry, monkeypatch, capsys, age_s):
    """§2: a registration at the incident ages (26-59s owner signals;
    #1771/#1997/#1988/#1992) skips the drain candidate with NO attempt, even
    when the staleness rule classified it stale (positive override of the
    `- stale` subtraction)."""
    now = time.time()
    _write_registration(watcher_registry, W_ISSUE, spawned_at=now - age_s)
    # Prefilter escape: a registered candidate early-exits the pass unless
    # classified possibly-stale — the incident channel this guard overrides.
    monkeypatch.setattr(asw, "_infra_drain_possibly_stale_ids", lambda *a: {W_ISSUE})
    dispatched, _markers = _stub_loop_seams(monkeypatch, [W_ISSUE], stale={W_ISSUE})
    _run_drain(watcher_registry, [W_ISSUE], now=now)
    out = capsys.readouterr().out
    assert dispatched == []
    assert f"INFRA-DRAIN SKIP issue #{W_ISSUE} (registration {age_s:.0f}s < 600s fresh)" in out
    assert _drain_attempt_ids(watcher_registry) == []


@pytest.mark.parametrize("age_s", [30.0, 59.0])
def test_fresh_registration_skipped_sweep_loop(watcher_registry, monkeypatch, capsys, age_s):
    """§2: same guard in the proposed-infra sweep loop."""
    now = time.time()
    _write_registration(watcher_registry, W_ISSUE, spawned_at=now - age_s)
    dispatched, _markers = _stub_loop_seams(monkeypatch, [W_ISSUE], stale={W_ISSUE})
    _run_sweep(monkeypatch, [W_ISSUE], now=now)
    out = capsys.readouterr().out
    assert dispatched == []
    assert (
        f"PROPOSED-INFRA-SWEEP SKIP issue #{W_ISSUE} (registration {age_s:.0f}s < 600s fresh)"
        in out
    )
    assert _sweep_attempt_ids(watcher_registry) == []


def test_registration_freshness_garbled_treated_fresh(watcher_registry):
    """§2: a present-but-garbled registration reads MAXIMALLY FRESH (0.0 —
    fail-closed toward not dispatching; deliberately NOT dispatch_lease_fresh's
    mtime-with-TTL convention), a missing one reads None (guard inert), and a
    well-formed spawned_at reads the exact age."""
    now = time.time()
    (watcher_registry / f"issue-{W_ISSUE}.json").write_bytes(b"{not json")
    assert asw._registration_age_s(W_ISSUE, now) == 0.0
    (watcher_registry / f"manual-issue-{W_ISSUE_B}.json").write_text("[]")  # non-dict JSON
    assert asw._registration_age_s(W_ISSUE_B, now) == 0.0
    assert asw._registration_age_s(W_ISSUE_C, now) is None
    _write_registration(watcher_registry, W_ISSUE_C, spawned_at=now - 42.0)
    assert asw._registration_age_s(W_ISSUE_C, now) == pytest.approx(42.0)


def test_registration_freshness_zero_window_disables_guard(watcher_registry, monkeypatch):
    """§2: EPM_DISPATCH_REG_FRESH_S=0 disables the guard (kill switch) — the
    30s-fresh registration dispatches again."""
    now = time.time()
    monkeypatch.setenv("EPM_DISPATCH_REG_FRESH_S", "0")
    assert asw._dispatch_reg_fresh_s() == 0.0
    _write_registration(watcher_registry, W_ISSUE, spawned_at=now - 30.0)
    monkeypatch.setattr(asw, "_infra_drain_possibly_stale_ids", lambda *a: {W_ISSUE})
    dispatched, markers = _stub_loop_seams(monkeypatch, [W_ISSUE], stale={W_ISSUE})
    _run_drain(watcher_registry, [W_ISSUE], now=now)
    assert dispatched == [W_ISSUE]
    assert markers == [(W_ISSUE, "infra-drain")]


# ─── §2b: owner-activity marker guard + drain M3 adoption ─────────────────────


def test_owner_activity_marker_skips_candidate(watcher_registry, monkeypatch, capsys):
    """§2b: a 26s-old NON-watcher progress note (the #1997 shape — owner alive
    with NO registration and NO lease) skips the sweep candidate, NO attempt."""
    now = float(int(time.time()))  # whole seconds: the event ts round-trips exactly
    events = [{"kind": "epm:progress", "note": "round 3 running", "ts": _iso(now - 26)}]
    dispatched, _markers = _stub_loop_seams(monkeypatch, [W_ISSUE], events=events)
    _run_sweep(monkeypatch, [W_ISSUE], now=now)
    out = capsys.readouterr().out
    assert dispatched == []
    assert (
        f"PROPOSED-INFRA-SWEEP SKIP issue #{W_ISSUE} (owner-activity marker 26s < 600s fresh)"
        in out
    )
    assert _sweep_attempt_ids(watcher_registry) == []


def test_owner_activity_guard_ignores_watcher_own_sentinel():
    """§2b: the watcher's OWN dispatch sentinels must NOT suppress its retries
    — only genuinely non-watcher markers count as owner activity."""
    now = 1_800_000_000.0
    for sentinel in (asw._INFRA_DRAIN_NOTE_SENTINEL, asw._PROPOSED_INFRA_SWEEP_NOTE_SENTINEL):
        events = [
            {
                "kind": "epm:progress",
                "note": f"{sentinel} watcher dispatched",
                "ts": _iso(now - 26),
            }
        ]
        assert asw._owner_activity_marker_age_s(events, now) is None
    events = [{"kind": "epm:progress", "note": "owner progress", "ts": _iso(now - 26)}]
    assert asw._owner_activity_marker_age_s(events, now) == pytest.approx(26.0)


def test_drain_loop_honors_recent_dispatch_marker(watcher_registry, monkeypatch, capsys):
    """§2b: the drain loop gains the sweep's #843 M3 dispatch-sentinel guard
    (it previously lacked it) — a 30s-old dispatch marker skips the candidate."""
    now = float(int(time.time()))
    events = [
        {
            "kind": "epm:progress",
            "note": f"{asw._INFRA_DRAIN_NOTE_SENTINEL} watcher dispatched autonomous session",
            "ts": _iso(now - 30),
        }
    ]
    dispatched, _markers = _stub_loop_seams(monkeypatch, [W_ISSUE], events=events)
    _run_drain(watcher_registry, [W_ISSUE], now=now)
    out = capsys.readouterr().out
    assert dispatched == []
    assert f"INFRA-DRAIN SKIP issue #{W_ISSUE} (recent-dispatch-marker 30s < 600s)" in out
    assert _drain_attempt_ids(watcher_registry) == []


# ─── §3: per-spawn cap re-check ───────────────────────────────────────────────


def test_per_spawn_cap_recheck_stops_batch_at_limit(watcher_registry, monkeypatch, capsys):
    """§3: occupancy that grew EXTERNALLY (rival dispatchers) after the first
    in-batch spawn is seen by the mid-batch re-read — the second candidate
    skips with NO attempt booked."""
    now = time.time()
    ids = [W_ISSUE, W_ISSUE_B]
    dispatched, _markers = _stub_loop_seams(monkeypatch, ids, occupancy_seq=[[], [101, 102, 103]])
    _run_drain(watcher_registry, ids, cap=3, now=now)
    out = capsys.readouterr().out
    assert dispatched == [W_ISSUE]
    assert f"INFRA-DRAIN SKIP issue #{W_ISSUE_B} (cap-full-recheck: live 3+0+1 >= cap 3)" in out
    assert _drain_attempt_ids(watcher_registry) == [str(W_ISSUE)]


def test_per_spawn_cap_recheck_counts_in_batch_dispatches(watcher_registry, monkeypatch, capsys):
    """§3 Must-Fix regression (success criterion 2): the `+ dispatched` term.

    live+pending < cap <= live+pending+dispatched: external growth of ONE
    (live=[101]) plus in-batch spawns. The corrected formula stops the third
    candidate (1 live + 0 pending + 2 in-batch = 3 >= cap 3 — total spawned
    this tick = cap); the pre-#2142 formula (len(live) + pending >= cap)
    reads 1 < 3 for BOTH later candidates and over-dispatches to 4 total."""
    now = time.time()
    ids = [W_ISSUE, W_ISSUE_B, W_ISSUE_C]
    dispatched, _markers = _stub_loop_seams(monkeypatch, ids, occupancy_seq=[[], [101]])
    _run_drain(watcher_registry, ids, cap=3, now=now)
    out = capsys.readouterr().out
    assert dispatched == [W_ISSUE, W_ISSUE_B]
    assert f"INFRA-DRAIN SKIP issue #{W_ISSUE_C} (cap-full-recheck: live 1+0+2 >= cap 3)" in out


def test_per_spawn_cap_recheck_preserves_urgent_bonus(watcher_registry, monkeypatch, capsys):
    """§3 hard non-goal 1: the re-check honors the #1853 urgent bonus — a
    bare-`cap` re-check would revoke the urgent candidate's sanctioned bonus
    slot mid-batch (0 live + 0 pending + 1 dispatched >= cap 1, but < limit
    2 = cap + bonus)."""
    now = time.time()
    monkeypatch.setenv("EPM_INFRA_SWEEP_URGENT_BONUS", "1")
    dispatched, _markers = _stub_loop_seams(monkeypatch, [W_ISSUE, W_ISSUE_B])
    _run_sweep(monkeypatch, [W_ISSUE, W_ISSUE_B], urgent={W_ISSUE_B}, cap=1, now=now)
    out = capsys.readouterr().out
    assert dispatched == [W_ISSUE, W_ISSUE_B]
    assert "cap-full-recheck" not in out


def test_per_spawn_cap_recheck_rechecks_after_failed_first_attempt(
    watcher_registry, monkeypatch, capsys
):
    """§3: the re-check keys on `attempted_any`, NOT `dispatched > 0` — a
    FAILED first attempt leaves dispatched at 0 with the stagger already
    elapsed, and the second candidate must still re-read occupancy."""
    now = time.time()
    ids = [W_ISSUE, W_ISSUE_B]
    dispatched, markers = _stub_loop_seams(
        monkeypatch,
        ids,
        occupancy_seq=[[], [101, 102, 103]],
        dispatch_result={W_ISSUE: "failed", W_ISSUE_B: "spawned"},
    )
    _run_drain(watcher_registry, ids, cap=3, now=now)
    out = capsys.readouterr().out
    assert dispatched == [W_ISSUE]  # attempt 1 failed; candidate 2 never attempted
    assert f"INFRA-DRAIN SKIP issue #{W_ISSUE_B} (cap-full-recheck: live 3+0+0 >= cap 3)" in out
    assert markers == []


def test_per_spawn_cap_recheck_fails_closed_on_occupancy_read_failure(
    watcher_registry, monkeypatch, capsys
):
    """§3 fail-loud pin: a None mid-batch occupancy read BREAKS the batch (no
    further dispatches this tick; nothing booked for the un-attempted
    remainder) — never coerced to an empty set and dispatched past the cap."""
    now = time.time()
    ids = [W_ISSUE, W_ISSUE_B, W_ISSUE_C]
    dispatched, _markers = _stub_loop_seams(monkeypatch, ids, occupancy_seq=[[], None])
    _run_drain(watcher_registry, ids, cap=3, now=now)
    out = capsys.readouterr().out
    assert dispatched == [W_ISSUE]
    assert "live occupancy read FAILED mid-batch" in out
    assert _drain_attempt_ids(watcher_registry) == [str(W_ISSUE)]


# ─── dry-run threading ────────────────────────────────────────────────────────


def test_dry_run_threads_through_new_guards(watcher_registry, monkeypatch, capsys):
    """Success criterion 5's zero-write property: dry_run=True through BOTH
    loops (guards + cap re-check exercised, REAL _dispatch_infra_drain) spawns
    nothing, acquires no lease, writes no attempt state, posts no marker."""
    now = time.time()
    ids = [W_ISSUE, W_ISSUE_B]
    sk = {i: ("proposed", "infra") for i in ids}
    monkeypatch.setattr(asw, "_infra_drain_signals", lambda *a: (sk, set(), 0))
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_infra_drain_occupancy", lambda: [])
    monkeypatch.setattr(asw, "_auth_outage_spawn_gate", lambda *a, **k: None)
    markers = _marker_recorder(monkeypatch)
    run_calls: list = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **k: run_calls.append(a))
    _run_drain(watcher_registry, ids, cap=3, now=now, dry_run=True)
    _run_sweep(monkeypatch, ids, cap=3, now=now, dry_run=True)
    out = capsys.readouterr().out
    # The REAL _dispatch_infra_drain dry-run branch ran for every candidate in
    # both loops; the cap re-check ran for each second candidate
    # (attempted_any after the first "failed" dry-run) and let it through.
    assert out.count("[dry-run] would dispatch infra-drain") == 4
    assert "cap-full-recheck" not in out
    assert run_calls == []  # no spawn subprocess
    assert markers == []  # no marker posted
    assert not spawn_session.dispatch_lease_path(W_ISSUE).exists()
    assert not spawn_session.dispatch_lease_path(W_ISSUE_B).exists()
    assert not (watcher_registry / "infra-drain-state.json").exists()
    assert not (watcher_registry / "proposed-infra-sweep-state.json").exists()
