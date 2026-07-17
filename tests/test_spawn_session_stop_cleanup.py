"""Tests for the #1455 operator-stop one-shot cleanup in ``scripts/spawn_session.py``.

What this pins (plan #1455 §4; incident 2026-07-16 on #1090: a deliberate
operator stop left the crash-recovery registration + dispatch lease behind,
costing 1 manual unregister + 2 lease clears + 3 spawn attempts):

1. A confirmed-dead OPERATOR stop removes the sid-matched registration
   file(s) and releases the dispatch lease the stopped session's own
   dispatch minted (``acquired_at <= spawned_at``).
2. Watcher-sourced stops perform ZERO cleanup — registration + lease are
   BYTE-IDENTICAL after the stop (the watcher's boot-death / dead-wake /
   crash-recovery respawn arms REQUIRE the registration to survive its own
   stops). **Durability pin:** ``test_watcher_source_stop_never_cleans_up``.
3. Stopping session A never removes B's registration, and never removes a
   successor dispatch's (newer) lease.
4. Fail-soft: cleanup failures WARN and never change ``cmd_stop``'s exit
   behavior; a still-live poll AND a daemon-unreachable strict probe both
   SKIP cleanup (fail toward keep).
5. The ``--no-cleanup`` escape hatch restores today's behavior verbatim, and
   the ``eps_sessions.py`` bare-Namespace delegation shape (no ``kill`` /
   ``no_cleanup`` attrs) inherits the cleanup without raising.
6. Forward-drift gate: every ``spawn_session.py stop`` invocation constructed
   in ``scripts/autonomous_session_watch.py`` threads ``--stop-source
   watcher`` (an operator-sourced watcher stop would trigger the cleanup and
   delete the registration its own respawn arms need).

No daemon, no real registry, no real task state: ``AUTONOMOUS_REGISTRY_DIR``,
``_load_session_issue_map``, ``post``, and ``_live_children`` are all
monkeypatched (the seam style of ``test_spawn_session_stop_marker.py`` +
``test_dispatch_lease.py``); ``time.sleep`` is a no-op.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import session_resolver  # noqa: E402
import spawn_session  # noqa: E402

SID = "sess-1455-owner"
OTHER_SID = "sess-1455-other"
ISSUE = 5


def _seed_auto_registration(reg: Path, issue: int, sid: str, spawned_at: float) -> Path:
    """Write an ``issue-<N>.json`` crash-recovery registration recording ``sid``."""
    path = reg / f"issue-{issue}.json"
    path.write_text(
        json.dumps(
            {
                "issue": issue,
                "happy_session_id": sid,
                "cwd": "/tmp",
                "auto_approve_gpu_hours": 100.0,
                "spawned_at": spawned_at,
                "missed": 0,
            }
        )
    )
    return path


def _seed_lease(reg: Path, issue: int, acquired_at: float) -> Path:
    """Write a ``dispatch-lease-<N>.json`` in the acquire-site entry shape."""
    path = reg / f"dispatch-lease-{issue}.json"
    path.write_text(
        json.dumps(
            {
                "issue": issue,
                "holder": "spawn-issue --auto pid=1234",
                "pid": 1234,
                "token": "tok-1455",
                "acquired_at": acquired_at,
            }
        )
    )
    return path


@pytest.fixture
def stop_env(tmp_path, monkeypatch):
    """Hermetic ``cmd_stop`` seams: isolated registry dir, no real daemon, no
    real task state, no real sleeping. The daemon live list defaults to empty
    with the STRICT probe succeeding, so the dead-poll confirms immediately."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(spawn_session, "_load_session_issue_map", lambda: {})
    monkeypatch.setattr(spawn_session, "post", lambda path, body: {"success": True})
    monkeypatch.setattr(spawn_session, "_live_children", lambda *, strict=False: [])
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return tmp_path


# ── operator-stop cleanup (the one-shot Goal case) ──────────────────────────


def test_operator_stop_removes_sid_matched_registration_and_own_lease(stop_env, capsys):
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    spawn_session.main(["stop", "--session-id", SID])
    assert not reg_path.exists()
    assert not lease_path.exists()
    out = capsys.readouterr().out
    assert "REMOVED" in out
    assert "released" in out


def test_operator_stop_keeps_successor_lease(stop_env, capsys):
    """A lease acquired AFTER the stopped session's registration was written
    belongs to a successor dispatch mid-flight — never released by the stop."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t + 30)
    spawn_session.main(["stop", "--session-id", SID])
    assert not reg_path.exists()
    assert lease_path.exists()
    assert "kept-successor" in capsys.readouterr().out


def test_stop_of_a_never_deletes_bs_registration_or_lease(stop_env):
    """Sid-mismatch guard: assert FILE PRESENCE (+ lease presence) — in scan
    mode ``unregister_paths`` appends NO kept-sid-mismatch/missing rows, so
    rows/stdout carry no signal for the kept case."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, OTHER_SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    spawn_session.main(["stop", "--session-id", SID])
    assert reg_path.exists()
    assert lease_path.exists()


def test_manual_registration_removed_but_no_lease_release(stop_env, capsys):
    """Manual/campaign spawns never mint a lease — a removed
    ``manual-issue-<N>.json`` must not trigger any lease action."""
    t = time.time()
    manual_path = stop_env / f"manual-issue-{ISSUE}.json"
    manual_path.write_text(
        json.dumps(
            {
                "issue": ISSUE,
                "happy_session_id": SID,
                "cwd": "/tmp",
                "spawned_at": t,
                "mode": "manual",
            }
        )
    )
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    spawn_session.main(["stop", "--session-id", SID])
    assert not manual_path.exists()
    assert lease_path.exists()
    assert "dispatch-lease" not in capsys.readouterr().out


# ── gating: watcher source + --no-cleanup ───────────────────────────────────


def test_watcher_source_stop_never_cleans_up(stop_env):
    """DURABILITY PIN (acceptance criterion 2): after a watcher-sourced stop
    the registration + lease files are BYTE-IDENTICAL — the watcher's
    boot-death / dead-wake / crash-recovery respawn arms depend on the
    registration surviving the watcher's own stops."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    reg_before, lease_before = reg_path.read_bytes(), lease_path.read_bytes()
    spawn_session.main(["stop", "--session-id", SID, "--stop-source", "watcher"])
    assert reg_path.read_bytes() == reg_before
    assert lease_path.read_bytes() == lease_before


def test_no_cleanup_flag_skips_cleanup(stop_env):
    """``--no-cleanup`` = today's operator behavior verbatim (stop, leave
    state, let the watcher's crash-recovery resurrect)."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    reg_before, lease_before = reg_path.read_bytes(), lease_path.read_bytes()
    spawn_session.main(["stop", "--session-id", SID, "--no-cleanup"])
    assert reg_path.read_bytes() == reg_before
    assert lease_path.read_bytes() == lease_before


# ── dead-confirm precondition + fail-soft ───────────────────────────────────


def test_cleanup_skipped_when_sid_still_live_after_poll(stop_env, monkeypatch, capsys):
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    monkeypatch.setattr(
        spawn_session, "_live_children", lambda *, strict=False: [{"happySessionId": SID}]
    )
    # Resolved in-body at CALL time (never a def-time default), so this
    # monkeypatch shrinks the poll deadline to zero.
    monkeypatch.setattr(spawn_session, "STOP_CLEANUP_DEAD_POLL_S", 0.0)
    spawn_session.main(["stop", "--session-id", SID])
    assert reg_path.exists()
    assert lease_path.exists()
    err = capsys.readouterr().err
    assert f"unregister --session-id {SID}" in err


def test_cleanup_skipped_when_daemon_unreachable(stop_env, monkeypatch, capsys):
    """R5 hardening: a daemon that cannot be LISTED at all is not evidence the
    session is gone — the strict probe's RuntimeError SKIPS cleanup with a
    WARN, never firing off the lenient empty-set read."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)

    def raising_children(*, strict=False):
        raise RuntimeError("daemon /list failed: connection refused")

    monkeypatch.setattr(spawn_session, "_live_children", raising_children)
    spawn_session.main(["stop", "--session-id", SID])  # must NOT raise
    assert reg_path.exists()
    assert lease_path.exists()
    err = capsys.readouterr().err
    assert "daemon unreachable" in err
    assert f"unregister --session-id {SID}" in err


def test_cleanup_failure_is_failsoft(stop_env, monkeypatch, capsys):
    _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=time.time())

    def boom(**_k):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(spawn_session, "unregister_paths", boom)
    spawn_session.main(["stop", "--session-id", SID])  # must NOT raise / SystemExit
    err = capsys.readouterr().err
    assert "WARN: post-stop cleanup failed" in err


# ── _stop_fallback --kill path + the eps_sessions.py bare Namespace ─────────


def test_stop_fallback_kill_success_fires_cleanup(stop_env, monkeypatch):
    """The ``--kill`` fallback's SIGTERM-confirmed-dead branch (``_pid_alive``
    false) satisfies the confirmed-dead precondition without a daemon read —
    cleanup fires there and nowhere else in the fallback."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    lease_path = _seed_lease(stop_env, ISSUE, acquired_at=t - 3)
    monkeypatch.setattr(spawn_session, "post", lambda path, body: {"success": False})
    monkeypatch.setattr(spawn_session, "_live_session_ids", lambda: set())
    monkeypatch.setattr(session_resolver, "find_node_pid_for_session", lambda sid, now=None: 4242)
    monkeypatch.setattr(session_resolver, "_read_proc_comm", lambda pid: "node")
    monkeypatch.setattr(
        session_resolver,
        "_read_proc_cmdline",
        lambda pid: "node /home/u/.nvm/node_modules/happy-coder/dist/index.mjs claude --resume",
    )
    monkeypatch.setattr(session_resolver, "_happy_daemon_pid", lambda: None)
    monkeypatch.setattr(session_resolver, "resolve_claude_pid", lambda pid: None)
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    spawn_session.main(["stop", "--session-id", SID, "--kill"])
    assert not reg_path.exists()
    assert not lease_path.exists()


def test_eps_sessions_bare_namespace_inherits_cleanup(stop_env):
    """Pins the getattr defaults: ``eps_sessions.py cmd_stop`` delegates with
    a bare ``Namespace(session_id, reason, stop_source="operator")`` — NO
    ``kill``/``no_cleanup`` attrs — and, as an operator per-issue stop,
    inherits the cleanup without raising."""
    t = time.time()
    reg_path = _seed_auto_registration(stop_env, ISSUE, SID, spawned_at=t)
    ns = argparse.Namespace(session_id=SID, reason="test", stop_source="operator")
    spawn_session.cmd_stop(ns)  # must not AttributeError on the missing attrs
    assert not reg_path.exists()


# ── release_dispatch_lease_for_stopped_dispatch (pure helper) ───────────────


class TestReleaseDispatchLeaseForStoppedDispatch:
    def test_released_when_lease_predates_registration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
        t = time.time()
        lease = _seed_lease(tmp_path, ISSUE, acquired_at=t - 3)
        assert spawn_session.release_dispatch_lease_for_stopped_dispatch(ISSUE, t) == "released"
        assert not lease.exists()

    def test_missing_when_no_lease_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
        result = spawn_session.release_dispatch_lease_for_stopped_dispatch(ISSUE, time.time())
        assert result == "missing"

    def test_kept_successor_when_lease_newer_than_registration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
        t = time.time()
        lease = _seed_lease(tmp_path, ISSUE, acquired_at=t + 30)
        result = spawn_session.release_dispatch_lease_for_stopped_dispatch(ISSUE, t)
        assert result == "kept-successor"
        assert lease.exists()

    def test_kept_garbled_on_non_numeric_acquired_at(self, tmp_path, monkeypatch):
        monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
        lease = tmp_path / f"dispatch-lease-{ISSUE}.json"
        lease.write_text(json.dumps({"issue": ISSUE, "acquired_at": "yesterday"}))
        result = spawn_session.release_dispatch_lease_for_stopped_dispatch(ISSUE, time.time())
        assert result == "kept-garbled"
        assert lease.exists()

    def test_kept_contended_when_flock_held(self, tmp_path, monkeypatch):
        """A takeover mid-flight (LOCK_EX held on the permanent flock sidecar
        — flock conflicts across file descriptions, so a second fd in the
        same process contends) keeps the lease; the TTL owns it."""
        monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
        t = time.time()
        lease = _seed_lease(tmp_path, ISSUE, acquired_at=t - 3)
        lock_fd = os.open(
            spawn_session._dispatch_lease_lock_path(ISSUE), os.O_CREAT | os.O_WRONLY, 0o644
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            result = spawn_session.release_dispatch_lease_for_stopped_dispatch(ISSUE, t)
            assert result == "kept-contended"
            assert lease.exists()
        finally:
            os.close(lock_fd)


# ── forward-drift gate: watcher stop invocations ────────────────────────────


def test_watcher_stop_invocations_carry_stop_source_watcher():
    """Every ``spawn_session.py stop`` invocation constructed in
    ``scripts/autonomous_session_watch.py`` must thread ``--stop-source
    watcher`` — an operator-sourced watcher stop would trigger the #1455
    cleanup and delete the registration its own respawn arms depend on."""
    src = (SCRIPTS / "autonomous_session_watch.py").read_text()
    sites = [m.start() for m in re.finditer(r'spawn_session\.py",\s*"stop"', src)]
    assert sites, "expected >=1 spawn_session.py stop construction site in the watcher"
    for pos in sites:
        window = src[pos : src.index("]", pos)]
        assert '"--stop-source"' in window and '"watcher"' in window, (
            f"watcher stop invocation at char {pos} lacks --stop-source watcher:\n{window}"
        )
