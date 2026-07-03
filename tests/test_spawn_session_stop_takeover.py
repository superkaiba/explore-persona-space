"""Tests for the #903 deliberate-takeover spawn gate + `stop` fallback in
``scripts/spawn_session.py`` / ``scripts/session_resolver.py``.

What this pins (#866 incident class):

1. **Spawn gate.** A FRESH ``issue-<N>.json.paused-takeover-*`` sentinel makes
   ``spawn-issue --auto`` suppress with the rc-0 ``TAKEOVER-SENTINEL HELD``
   line — recognized by :func:`spawn_session.spawn_output_suppressed` — with
   NO daemon POST and NO dispatch-lease acquisition (the pre-lease placement
   is load-bearing: a gate landing after lease acquisition would leave a
   TTL-held lease suppressing crash recovery past the sentinel TTL). Manual
   (non-``--auto``) spawns warn-and-proceed (the #843 lease posture).
2. **Stop fallback.** ``stop --session-id`` on a ``{'success': False}`` reply
   never dies with the bare unactionable error: daemon-tracked sids get a
   structured retry message; daemon-untracked sids are resolved to a live
   happy node pid via the ``~/.happy/logs`` reverse map
   (:func:`session_resolver.find_node_pid_for_session`) and either produce a
   kill-by-pid recipe or — under ``--kill`` — a verified SIGTERM behind the
   stacked comm / happy-cmdline / not-the-daemon-pid identity refusals.
3. **Reverse map.** Newest-log-per-pid rule (an older log for a since-recycled
   pid cannot vouch — the wrong-kill vector) + the ``-daemon.log`` exclusion.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import session_resolver  # noqa: E402
import spawn_session  # noqa: E402

# An issue number with no live worktree so `cmd_spawn_issue` resolves
# cwd=PROJECT_ROOT (which passes `_assert_spawn_cwd`).
_FAKE_ISSUE = 9903
_SID = "happy-sess-903-takeover"


def _spawn_ns(*, auto: bool) -> argparse.Namespace:
    """A minimal `spawn-issue` Namespace covering every attribute
    `cmd_spawn_issue` reads before (and at) the takeover gate."""
    return argparse.Namespace(
        issue=_FAKE_ISSUE,
        auto=auto,
        initial_prompt=None,
        betas=None,
        model=None,
        effort=None,
    )


@pytest.fixture
def takeover_registry(tmp_path, monkeypatch):
    """Isolated AUTONOMOUS_REGISTRY_DIR with a FRESH takeover sentinel for
    :data:`_FAKE_ISSUE`.

    Clears ``EPS_TAKEOVER_TTL_H`` so the sentinel tests pin the DEFAULT 6h
    TTL — an operator shell exporting the fleet knob must not flip them."""
    monkeypatch.delenv("EPS_TAKEOVER_TTL_H", raising=False)
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    (tmp_path / f"issue-{_FAKE_ISSUE}.json.paused-takeover-20260702").write_text("{}")
    return tmp_path


# ── spawn gate ──────────────────────────────────────────────────────────────


def test_spawn_output_suppressed_recognizes_takeover_sentinel():
    out = (
        f"{spawn_session.TAKEOVER_HELD_SENTINEL} issue #903: a deliberate session "
        f"takeover is in flight; NOT spawning."
    )
    assert spawn_session.spawn_output_suppressed(out) == spawn_session.TAKEOVER_HELD_SENTINEL
    # A real spawn's output is still not suppressed.
    assert spawn_session.spawn_output_suppressed("Spawned session abc for issue #903") is None


def test_spawn_issue_auto_suppressed_on_fresh_takeover_sentinel(
    takeover_registry, monkeypatch, capsys
):
    # The gate must fire BEFORE the daemon POST and BEFORE lease acquisition
    # (pre-lease placement pin: a gate after the lease would leave a TTL-held
    # lease suppressing crash recovery past the sentinel TTL).
    monkeypatch.setattr(
        spawn_session, "post", lambda *a, **k: pytest.fail("daemon POST must not happen")
    )
    monkeypatch.setattr(
        spawn_session,
        "acquire_dispatch_lease",
        lambda *a, **k: pytest.fail("dispatch lease must not be acquired"),
    )
    monkeypatch.setattr(
        spawn_session,
        "_spawn_issue_session",
        lambda *a, **k: pytest.fail("spawn tail must not be reached"),
    )
    spawn_session.cmd_spawn_issue(_spawn_ns(auto=True))  # returns (exit 0), no SystemExit
    out = capsys.readouterr().out
    # Producer -> recognizer loop closed on the CAPTURED stdout (a hand-crafted
    # fixture string could drift from the actual print; #607 class).
    assert spawn_session.spawn_output_suppressed(out) == spawn_session.TAKEOVER_HELD_SENTINEL


def test_spawn_issue_manual_warns_and_proceeds_on_sentinel(takeover_registry, monkeypatch, capsys):
    reached: list[int] = []
    monkeypatch.setattr(
        spawn_session,
        "_spawn_issue_session",
        lambda args, issue, *rest, **k: reached.append(issue),
    )
    spawn_session.cmd_spawn_issue(_spawn_ns(auto=False))
    out = capsys.readouterr().out
    assert reached == [_FAKE_ISSUE]  # manual spawns are not gated
    assert "takeover sentinel" in out
    assert spawn_session.spawn_output_suppressed(out) is None  # warn line is NOT a suppression


def test_spawn_issue_auto_ignores_stale_sentinel(takeover_registry, monkeypatch):
    # A stale sentinel must NOT gate the spawn (fail open): the flow proceeds
    # to lease acquisition as today.
    sentinel = takeover_registry / f"issue-{_FAKE_ISSUE}.json.paused-takeover-20260702"
    stale = time.time() - 7 * 3600  # past the 6h default TTL
    os.utime(sentinel, (stale, stale))
    lease_calls: list[int] = []

    def fake_lease(issue, holder, now=None):
        lease_calls.append(issue)
        return {"token": "t"}

    monkeypatch.setattr(spawn_session, "acquire_dispatch_lease", fake_lease)
    monkeypatch.setattr(
        spawn_session, "_spawn_issue_session", lambda *a, **k: None
    )  # swallow the spawn tail
    spawn_session.cmd_spawn_issue(_spawn_ns(auto=True))
    assert lease_calls == [_FAKE_ISSUE]


# ── cmd_stop fallback ───────────────────────────────────────────────────────


def _stop_ns(*, kill: bool) -> argparse.Namespace:
    # stop_source="watcher" skips main's deliberate-stop breadcrumb block
    # (added concurrently on main), keeping these tests focused on the #903
    # _stop_fallback path; reason rides along for parser parity.
    return argparse.Namespace(
        session_id=_SID,
        kill=kill,
        stop_source="watcher",
        reason="test",
    )


@pytest.fixture
def failing_stop(monkeypatch):
    """`post` returns the #866 failure reply; sid is daemon-untracked by
    default (tests override `_live_session_ids` for the tracked case)."""
    monkeypatch.setattr(spawn_session, "post", lambda path, body: {"success": False})
    monkeypatch.setattr(spawn_session, "_live_session_ids", lambda: set())


def test_cmd_stop_tracked_failure_structured(monkeypatch):
    monkeypatch.setattr(spawn_session, "post", lambda path, body: {"success": False})
    monkeypatch.setattr(spawn_session, "_live_session_ids", lambda: {_SID})
    with pytest.raises(SystemExit) as exc:
        spawn_session.cmd_stop(_stop_ns(kill=False))
    assert "DAEMON-TRACKED" in str(exc.value)


def test_cmd_stop_untracked_no_pid_structured_error(failing_stop, monkeypatch):
    monkeypatch.setattr(session_resolver, "find_node_pid_for_session", lambda sid, now=None: None)
    with pytest.raises(SystemExit) as exc:
        spawn_session.cmd_stop(_stop_ns(kill=False))
    msg = str(exc.value)
    assert "kill -TERM" in msg
    assert "daemon-untracked" in msg
    assert "UNKNOWN" in msg


def test_cmd_stop_untracked_pid_found_names_recipe_without_kill(failing_stop, monkeypatch):
    monkeypatch.setattr(session_resolver, "find_node_pid_for_session", lambda sid, now=None: 4242)
    with pytest.raises(SystemExit) as exc:
        spawn_session.cmd_stop(_stop_ns(kill=False))
    msg = str(exc.value)
    assert "4242" in msg
    assert "--kill" in msg


def _arm_kill_identity(monkeypatch, *, comm="node", cmdline=None, daemon_pid=None):
    """Monkeypatch the resolver identity probes for the --kill path."""
    if cmdline is None:
        cmdline = "node /home/u/.nvm/node_modules/happy-coder/dist/index.mjs claude --resume"
    monkeypatch.setattr(session_resolver, "find_node_pid_for_session", lambda sid, now=None: 4242)
    monkeypatch.setattr(session_resolver, "_read_proc_comm", lambda pid: comm)
    monkeypatch.setattr(session_resolver, "_read_proc_cmdline", lambda pid: cmdline)
    monkeypatch.setattr(session_resolver, "_happy_daemon_pid", lambda: daemon_pid)


def test_cmd_stop_untracked_kill_sigterm_verified(failing_stop, monkeypatch, capsys):
    _arm_kill_identity(monkeypatch)
    monkeypatch.setattr(session_resolver, "resolve_claude_pid", lambda pid: None)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(time, "sleep", lambda s: None)  # host-independent, no real waiting
    # Live -> dead sequence through the ONE module-level seam the death-wait
    # loop calls (session_resolver._pid_alive; no inline /proc check).
    alive_calls: list[int] = []

    def fake_pid_alive(pid):
        alive_calls.append(pid)
        return len(alive_calls) < 2  # first probe: still alive; second: dead

    monkeypatch.setattr(session_resolver, "_pid_alive", fake_pid_alive)
    spawn_session.cmd_stop(_stop_ns(kill=True))  # returns normally (success)
    assert killed == [(4242, signal.SIGTERM)]
    assert len(alive_calls) >= 2
    assert "Stopped daemon-untracked session" in capsys.readouterr().out


def test_cmd_stop_kill_refuses_on_comm_mismatch(failing_stop, monkeypatch):
    _arm_kill_identity(monkeypatch, comm="python3")
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(SystemExit) as exc:
        spawn_session.cmd_stop(_stop_ns(kill=True))
    assert "refusing --kill" in str(exc.value)
    assert killed == []


def test_cmd_stop_kill_refuses_on_cmdline_mismatch(failing_stop, monkeypatch):
    # comm == "node" but the cmdline lacks the happy-wrapper signature — the
    # recycled-pid-to-unrelated-node case (Happy daemon, eps-dashboard, ...).
    _arm_kill_identity(monkeypatch, cmdline="node /opt/eps-dashboard/server.js")
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(SystemExit) as exc:
        spawn_session.cmd_stop(_stop_ns(kill=True))
    assert "refusing --kill" in str(exc.value)
    assert killed == []


def test_cmd_stop_kill_refuses_on_daemon_pid(failing_stop, monkeypatch):
    _arm_kill_identity(monkeypatch, daemon_pid=4242)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(SystemExit) as exc:
        spawn_session.cmd_stop(_stop_ns(kill=True))
    assert "DAEMON" in str(exc.value)
    assert killed == []


# ── happy-log reverse map (sid -> node pid) ─────────────────────────────────


def test_find_node_pid_for_session_from_happy_log(tmp_path, monkeypatch):
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    log = tmp_path / "2026-07-02-10-00-00-pid-4242.log"
    log.write_text(f'... "sessionId": "{_SID}" ...\n')
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: True)
    assert session_resolver.find_node_pid_for_session(_SID) == 4242
    # A dead pid never resolves (the caller degrades to a recipe).
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: False)
    assert session_resolver.find_node_pid_for_session(_SID) is None
    # A sid the log does not reference never resolves.
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: True)
    assert session_resolver.find_node_pid_for_session("some-other-sid") is None


def test_find_node_pid_reused_pid_older_log_does_not_vouch(tmp_path, monkeypatch):
    # TWO logs for pid 4242: the OLDER one embeds the sid, the NEWER one does
    # not (the pid was recycled to a different wrapper). Only the newest log
    # per pid may vouch, so the resolve must MISS — accepting the older log is
    # the wrong-kill vector. This is also the multi-log fall-through fixture.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    now = time.time()
    older = tmp_path / "2026-07-01-09-00-00-pid-4242.log"
    older.write_text(f'"sessionId": "{_SID}"\n')
    os.utime(older, (now - 7200, now - 7200))
    newer = tmp_path / "2026-07-02-10-00-00-pid-4242.log"
    newer.write_text('"sessionId": "a-DIFFERENT-session"\n')
    os.utime(newer, (now - 100, now - 100))
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: True)
    assert session_resolver.find_node_pid_for_session(_SID, now=now) is None


def test_find_node_pid_degenerate_sid_never_resolves(tmp_path, monkeypatch):
    # An EMPTY sid (an unset "$SID" in a caller script) bare-substring-matches
    # every log head and would resolve to the newest live wrapper — which IS a
    # happy wrapper, so it passes the comm/cmdline/daemon-pid refusals and gets
    # SIGTERMed under --kill. The degenerate-sid floor must return None even
    # with a perfectly matchable log present.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    log = tmp_path / "2026-07-02-10-00-00-pid-4242.log"
    log.write_text(f'... "sessionId": "{_SID}" ...\n')
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: True)
    assert session_resolver.find_node_pid_for_session("") is None
    # A short fragment below the 8-char floor (real Happy sids are ~25-char
    # cuids) is rejected even when it appears verbatim in the log content.
    assert session_resolver.find_node_pid_for_session("happy") is None


def test_find_node_pid_requires_quoted_form_match(tmp_path, monkeypatch):
    # The log carries the sid ONLY as a PREFIX of a longer id (and as a bare
    # path fragment). A bare `sid in head` substring match would vouch pid
    # 4242 for a session the log never bound; the quoted-form `"<sid>"` match
    # must miss.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    log = tmp_path / "2026-07-02-10-00-00-pid-4242.log"
    log.write_text(f'"sessionId": "{_SID}-suffix"\npath=/tmp/{_SID}\n')
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: True)
    assert session_resolver.find_node_pid_for_session(_SID) is None


def test_find_node_pid_daemon_log_excluded(tmp_path, monkeypatch):
    # The Happy DAEMON's own log (`...-pid-<pid>-daemon.log`) sits in the same
    # dir and is the one fatal wrong-kill target — pin the regex exclusion.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    log = tmp_path / "2026-07-02-10-00-00-pid-4242-daemon.log"
    log.write_text(f'"sessionId": "{_SID}"\n')
    monkeypatch.setattr(session_resolver, "_pid_alive", lambda pid: True)
    assert session_resolver.find_node_pid_for_session(_SID) is None
