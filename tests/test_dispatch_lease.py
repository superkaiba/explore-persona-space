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
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

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
