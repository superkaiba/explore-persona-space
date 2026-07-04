"""Tests for the #956 webhook-timeout reap + retry leg in ``scripts/spawn_session.py``.

What this pins (plan #956 v2 §7):

1. ``post()``'s ``/spawn-session`` HTTPError leg reaps the half-spawned child
   and retries EXACTLY once, only after a confirmed reap; a failed /
   unconfirmed reap exits nonzero WITHOUT retrying (no double-spawn).
2. Non-matching HTTP errors and non-``/spawn-session`` routes keep today's
   exact ``sys.exit`` message and never invoke the reap.
3. The reap is daemon-mediated by default (sid-stop for late-handshaken
   children, ``PID-<pid>`` stop otherwise); the client-side SIGTERM survives
   only as the untracked-but-alive fallback behind FOUR identity refusals
   (comm / cmdline / daemon-pid / cwd), and a surviving inner claude after
   ANY kill leg blocks the retry.
4. Transport failures are never conflated with the daemon's ``success:false``
   verdict (strict /list + ``_stop_session_raw`` raise; lenient /list
   returns ``[]``).

No live daemon: every daemon touchpoint (``urlopen``, ``_live_children``,
``_stop_session_raw``) and every /proc seam (``session_resolver._pid_alive``,
``_read_proc_*``, ``resolve_claude_pid``, ``os.kill``, ``os.readlink``) is
monkeypatched. An autouse fixture patches ``spawn_session.time.sleep`` so no
test ever sleeps the 30s backoff or the 20 x 0.5s death poll.
"""

from __future__ import annotations

import io
import json
import signal
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import session_resolver  # noqa: E402
import spawn_session  # noqa: E402

HAPPY_CMDLINE = (
    "node --no-warnings --no-deprecation /usr/lib/node_modules/happy/dist/index.mjs "
    "claude --happy-starting-mode remote --started-by daemon"
)
SPAWN_DIR = "/repo/.claude/worktrees/issue-952"


# ── shared fixtures / helpers ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def sleeps(monkeypatch) -> list[float]:
    """Record every ``time.sleep`` so no test waits the 30s backoff or the
    20 x 0.5s death poll (plan §7 sleep-hygiene requirement)."""
    recorded: list[float] = []
    monkeypatch.setattr(spawn_session.time, "sleep", lambda s: recorded.append(s))
    return recorded


class _FakeResponse:
    """Minimal context-manager stand-in for ``urlopen``'s response."""

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code: int, body: object) -> urllib.error.HTTPError:
    """Real ``urllib.error.HTTPError`` with a readable body (plan §7)."""
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError(
        "http://127.0.0.1:1/spawn-session", code, "error", {}, io.BytesIO(raw)
    )


def _webhook_body(pid: int, *, tmux: bool = False) -> dict:
    suffix = " (tmux)" if tmux else ""
    return {"success": False, "error": f"Session webhook timeout for PID {pid}{suffix}"}


def _install_urlopen(monkeypatch, outcomes: list[object]) -> list[tuple]:
    """Sequence ``urlopen`` outcomes; an Exception outcome is raised, a dict
    is returned as a fake response. Returns the recorded call list."""
    calls: list[tuple] = []

    def fake_urlopen(req, timeout=None):
        idx = len(calls)
        calls.append((req, timeout))
        assert idx < len(outcomes), f"unexpected urlopen call #{idx + 1}"
        outcome = outcomes[idx]
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr(spawn_session.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(spawn_session, "daemon_port", lambda: 65001)
    return calls


def _install_reap(monkeypatch, results: list) -> list[tuple]:
    """Replace ``_reap_half_spawned_session`` with a recorder returning the
    per-call entries of ``results``. Returns the (pid, directory, is_tmux)
    call list."""
    calls: list[tuple] = []

    def fake_reap(pid, directory, *, is_tmux=False):
        calls.append((pid, directory, is_tmux))
        return results[min(len(calls) - 1, len(results) - 1)]

    monkeypatch.setattr(spawn_session, "_reap_half_spawned_session", fake_reap)
    return calls


def _install_kill(monkeypatch, on_kill=None) -> list[tuple[int, int]]:
    kills: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))
        if on_kill is not None:
            on_kill(pid, sig)

    monkeypatch.setattr(spawn_session.os, "kill", fake_kill)
    return kills


def _patch_resolver(
    monkeypatch,
    *,
    claude_pid=None,
    alive=None,
    comm="node",
    cmdline=HAPPY_CMDLINE,
    daemon_pid=999,
):
    """Patch every session_resolver seam the reap consults (plan §7: the
    documented ``_pid_alive`` + lazy-import seams of ``_stop_fallback``)."""
    monkeypatch.setattr(session_resolver, "resolve_claude_pid", lambda p: claude_pid)
    monkeypatch.setattr(
        session_resolver, "_pid_alive", alive if alive is not None else lambda p: False
    )
    monkeypatch.setattr(session_resolver, "_read_proc_comm", lambda p: comm)
    monkeypatch.setattr(session_resolver, "_read_proc_cmdline", lambda p: cmdline)
    monkeypatch.setattr(session_resolver, "_happy_daemon_pid", lambda: daemon_pid)


def _install_children(monkeypatch, children: list[dict] | Exception) -> list[bool]:
    """Replace ``_live_children`` with a recorder of its ``strict`` kwarg."""
    strict_calls: list[bool] = []

    def fake_children(*, strict=False):
        strict_calls.append(strict)
        if isinstance(children, Exception):
            raise children
        return children

    monkeypatch.setattr(spawn_session, "_live_children", fake_children)
    return strict_calls


def _install_stop_raw(monkeypatch, result) -> list[str]:
    """Replace ``_stop_session_raw``; ``result`` is a bool, an Exception, or
    a dict mapping sessionId -> bool. Returns the recorded sessionId list."""
    calls: list[str] = []

    def fake_stop(session_id):
        calls.append(session_id)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, dict):
            return result[session_id]
        return result

    monkeypatch.setattr(spawn_session, "_stop_session_raw", fake_stop)
    return calls


# ── 1. regex shape ─────────────────────────────────────────────────────────


def test_webhook_timeout_regex_matches_plain_and_tmux():
    m = spawn_session.WEBHOOK_TIMEOUT_RE.search("Session webhook timeout for PID 3698")
    assert m is not None
    assert m.group(1) == "3698"
    assert m.group("tmux") is None
    m2 = spawn_session.WEBHOOK_TIMEOUT_RE.search("Session webhook timeout for PID 42 (tmux)")
    assert m2 is not None
    assert m2.group(1) == "42"
    assert m2.group("tmux") is not None
    # Other daemon errors never match.
    assert (
        spawn_session.WEBHOOK_TIMEOUT_RE.search("Failed to spawn Happy process - no PID returned")
        is None
    )
    assert spawn_session.WEBHOOK_TIMEOUT_RE.search("Unsupported agent type: 'claude'") is None


# ── 2-6. post()-level routing ──────────────────────────────────────────────


def test_http_500_webhook_timeout_reaps_and_retries_success(monkeypatch, sleeps):
    urlopen_calls = _install_urlopen(
        monkeypatch,
        [_http_error(500, _webhook_body(3698)), {"success": True, "sessionId": "sess-new"}],
    )
    reap_calls = _install_reap(monkeypatch, [spawn_session._ReapOutcome(True, "reaped")])
    out = spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    assert out == {"success": True, "sessionId": "sess-new"}
    assert reap_calls == [(3698, SPAWN_DIR, False)]
    assert sleeps == [spawn_session.WEBHOOK_TIMEOUT_RETRY_BACKOFF_S]
    assert len(urlopen_calls) == 2


def test_http_500_webhook_timeout_double_failure_exits_after_reaping_both(monkeypatch):
    # Each attempt's 500 names its OWN fork: distinct PIDs 3698 then 3699.
    urlopen_calls = _install_urlopen(
        monkeypatch,
        [_http_error(500, _webhook_body(3698)), _http_error(500, _webhook_body(3699))],
    )
    reap_calls = _install_reap(
        monkeypatch,
        [
            spawn_session._ReapOutcome(True, "reaped-first"),
            spawn_session._ReapOutcome(True, "reaped-second"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    msg = str(exc.value)
    assert "webhook" in msg and "timeout" in msg
    assert "reaped-second" in msg  # exit message names the LAST reap outcome
    # Per-attempt reap args pinned (catches stale-PID reuse across attempts).
    assert reap_calls == [(3698, SPAWN_DIR, False), (3699, SPAWN_DIR, False)]
    assert len(urlopen_calls) == 2


def test_http_500_webhook_timeout_failed_reap_exits_without_retry(monkeypatch, sleeps):
    urlopen_calls = _install_urlopen(monkeypatch, [_http_error(500, _webhook_body(3698))])
    reap_calls = _install_reap(
        monkeypatch, [spawn_session._ReapOutcome(False, "daemon unreachable during PID-stop")]
    )
    with pytest.raises(SystemExit) as exc:
        spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    assert "NOT retrying" in str(exc.value)
    assert len(reap_calls) == 1
    assert len(urlopen_calls) == 1  # no retry
    assert sleeps == []  # no backoff either


def test_http_500_non_webhook_shape_exits_verbatim_no_reap(monkeypatch):
    urlopen_calls = _install_urlopen(
        monkeypatch, [_http_error(500, {"success": False, "error": "something else"})]
    )
    reap_calls = _install_reap(monkeypatch, [spawn_session._ReapOutcome(True, "never")])
    with pytest.raises(SystemExit) as exc:
        spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    # Today's EXACT message, string-asserted (behavioral no-op outside the shape).
    assert str(exc.value) == (
        "Happy daemon /spawn-session returned HTTP 500: "
        "{'success': False, 'error': 'something else'}"
    )
    assert reap_calls == []
    assert len(urlopen_calls) == 1


def test_http_error_on_non_spawn_route_never_reaps(monkeypatch):
    # A webhook-timeout-SHAPED body on a non-/spawn-session route stays verbatim.
    urlopen_calls = _install_urlopen(monkeypatch, [_http_error(500, _webhook_body(3698))])
    reap_calls = _install_reap(monkeypatch, [spawn_session._ReapOutcome(True, "never")])
    with pytest.raises(SystemExit) as exc:
        spawn_session.post("/stop-session", {"sessionId": "sess-x"})
    assert str(exc.value) == (
        "Happy daemon /stop-session returned HTTP 500: "
        "{'success': False, 'error': 'Session webhook timeout for PID 3698'}"
    )
    assert reap_calls == []
    assert len(urlopen_calls) == 1


# ── 7-17. reap internals ───────────────────────────────────────────────────


def test_reap_stops_via_daemon_sid_when_late_webhook_delivered(monkeypatch):
    strict_calls = _install_children(
        monkeypatch, [{"pid": 3698, "happySessionId": "sess-x", "startedBy": "daemon"}]
    )
    stop_calls = _install_stop_raw(monkeypatch, {"sess-x": True})
    _patch_resolver(monkeypatch, claude_pid=None, alive=lambda p: False)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is True
    assert stop_calls == ["sess-x"]  # sid-stop, NOT "PID-3698"
    assert strict_calls == [True]  # the late-handshake probe is strict
    assert kills == []


def test_reap_daemon_pid_stop_for_list_invisible_child(monkeypatch):
    # The corrected primary no-sid path: a never-handshaken child is
    # /list-invisible BY DESIGN -> daemon PID-stop, no client-side kill.
    _install_children(monkeypatch, [])
    stop_calls = _install_stop_raw(monkeypatch, {"PID-3698": True})
    _patch_resolver(monkeypatch, claude_pid=None, alive=lambda p: False)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is True
    assert stop_calls == ["PID-3698"]
    assert kills == []


@pytest.mark.parametrize(
    "case",
    ["comm-mismatch", "cmdline-not-happy", "pid-is-daemon", "cwd-mismatch"],
)
def test_reap_fallback_refuses_kill_on_identity_mismatch(monkeypatch, case):
    # Fallback leg reached via: /list-invisible, PID-stop success:false,
    # pid still alive. EVERY identity mismatch refuses (reaped=False, no kill).
    _install_children(monkeypatch, [])
    _install_stop_raw(monkeypatch, {"PID-3698": False})
    kwargs = {"claude_pid": None, "alive": lambda p: True}
    if case == "comm-mismatch":
        kwargs["comm"] = "python3"
    elif case == "cmdline-not-happy":
        kwargs["cmdline"] = "node /usr/lib/node_modules/other-tool/dist/index.mjs serve"
    elif case == "pid-is-daemon":
        kwargs["daemon_pid"] = 3698
    _patch_resolver(monkeypatch, **kwargs)
    if case == "cwd-mismatch":
        monkeypatch.setattr(spawn_session.os, "readlink", lambda p: "/some/other/dir")
    else:
        monkeypatch.setattr(spawn_session.os, "readlink", lambda p: SPAWN_DIR)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is False
    assert "refusing kill" in outcome.detail
    assert kills == []


def test_reap_daemon_list_unreachable_blocks_retry(monkeypatch):
    # (a) The REAL _live_children: lenient swallows to [], strict raises.
    def boom():
        raise SystemExit("daemon.state.json missing")

    monkeypatch.setattr(spawn_session, "daemon_port", boom)
    assert spawn_session._live_children() == []
    with pytest.raises(RuntimeError, match="daemon /list failed"):
        spawn_session._live_children(strict=True)
    # (b) A strict-probe failure blocks the reap (an unreachable daemon can
    # never certify anything).
    strict_calls = _install_children(monkeypatch, RuntimeError("daemon /list failed: boom"))
    _patch_resolver(monkeypatch, claude_pid=None)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is False
    assert "unreachable" in outcome.detail
    assert strict_calls == [True]
    assert kills == []


def test_reap_stop_transport_failure_is_not_success_false(monkeypatch):
    # The conflation guard: a /stop-session TRANSPORT failure must never be
    # read as the daemon's success:false verdict — no already-gone check
    # (_pid_alive untouched), no fallback kill.
    _install_children(monkeypatch, [])
    _install_stop_raw(monkeypatch, RuntimeError("daemon /stop-session transport failure: refused"))

    def fail_alive(p):
        raise AssertionError("_pid_alive must NOT be consulted on a transport failure")

    _patch_resolver(monkeypatch, claude_pid=None, alive=fail_alive)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is False
    assert "unreachable" in outcome.detail
    assert kills == []


def test_reap_already_gone_via_pid_stop_false_and_dead(monkeypatch):
    # The corrected already-gone verdict: daemon reachable + pid untracked
    # (success:false) + not alive.
    _install_children(monkeypatch, [])
    stop_calls = _install_stop_raw(monkeypatch, {"PID-3698": False})
    _patch_resolver(monkeypatch, claude_pid=None, alive=lambda p: False)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is True
    assert "already exited" in outcome.detail
    assert stop_calls == ["PID-3698"]
    assert kills == []


def test_reap_fallback_sigterm_on_untracked_alive_child(monkeypatch):
    # Untracked-but-alive anomaly: all four identity checks pass -> SIGTERM.
    _install_children(monkeypatch, [])
    _install_stop_raw(monkeypatch, {"PID-3698": False})
    state = {"killed": False}
    kills = _install_kill(monkeypatch, on_kill=lambda pid, sig: state.__setitem__("killed", True))
    _patch_resolver(
        monkeypatch,
        claude_pid=None,
        alive=lambda p: not state["killed"] if p == 3698 else False,
    )
    monkeypatch.setattr(spawn_session.os, "readlink", lambda p: SPAWN_DIR)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is True
    assert kills == [(3698, signal.SIGTERM)]


def test_reap_inner_claude_survivor_blocks_retry_after_fallback_sigterm(monkeypatch, sleeps):
    # MUST-FIX 3: a surviving inner claude after the fallback SIGTERM blocks
    # the retry (reaped=False, NOT reaped-with-warning). Threaded through
    # post() with the reap UNMOCKED: SystemExit, urlopen called exactly once.
    def setup(state):
        _install_children(monkeypatch, [])
        _install_stop_raw(monkeypatch, {"PID-3698": False})

        def alive(p):
            if p == 3698:
                return not state["killed"]
            return p == 4001  # the inner claude survives every poll

        _patch_resolver(monkeypatch, claude_pid=4001, alive=alive)
        monkeypatch.setattr(spawn_session.os, "readlink", lambda p: SPAWN_DIR)
        return _install_kill(
            monkeypatch, on_kill=lambda pid, sig: state.__setitem__("killed", True)
        )

    # (1) Direct reap-level assert.
    state = {"killed": False}
    kills = setup(state)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is False
    assert "4001" in outcome.detail
    assert "retry blocked" in outcome.detail
    assert kills == [(3698, signal.SIGTERM)]
    # (2) post()-level thread (reap unmocked): no retry over the survivor.
    state["killed"] = False
    urlopen_calls = _install_urlopen(monkeypatch, [_http_error(500, _webhook_body(3698))])
    with pytest.raises(SystemExit) as exc:
        spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    msg = str(exc.value)
    assert "NOT retrying" in msg
    assert "4001" in msg
    assert len(urlopen_calls) == 1


def test_reap_inner_claude_survivor_blocks_retry_after_daemon_pid_stop(monkeypatch):
    # MUST-FIX 3, daemon-leg twin: the daemon kills the WRAPPER; the inner
    # claude can survive there exactly as under the fallback SIGTERM.
    _install_children(monkeypatch, [])
    _install_stop_raw(monkeypatch, {"PID-3698": True})
    _patch_resolver(monkeypatch, claude_pid=4001, alive=lambda p: p == 4001)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is False
    assert "4001" in outcome.detail
    assert "retry blocked" in outcome.detail
    assert kills == []  # daemon-mediated: no client-side kill


def test_reap_daemon_ack_but_pid_survives_is_not_reaped(monkeypatch, sleeps):
    # stopSession swallows kill errors and untracks anyway — success:true is
    # NOT death proof; the bounded poll is the only proof, and once untracked
    # no daemon stop can be re-issued.
    _install_children(monkeypatch, [])
    _install_stop_raw(monkeypatch, {"PID-3698": True})
    _patch_resolver(monkeypatch, claude_pid=None, alive=lambda p: True)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(3698, SPAWN_DIR)
    assert outcome.reaped is False
    assert "survived" in outcome.detail
    assert kills == []
    # The full bounded poll ran (20 x 0.5s, all monkeypatched).
    assert sleeps == [spawn_session.REAP_PID_DEATH_POLL_INTERVAL_S] * (
        spawn_session.REAP_PID_DEATH_POLL_TRIES
    )


def test_reap_tmux_variant_refuses_fallback_with_tmux_hint(monkeypatch):
    # Daemon legs 1-3 run unchanged for tmux; only the client-side fallback
    # is refused (pane-PID /proc identity unverified for tmux).
    _install_children(monkeypatch, [])
    stop_calls = _install_stop_raw(monkeypatch, {"PID-42": False})
    _patch_resolver(monkeypatch, claude_pid=None, alive=lambda p: True)
    kills = _install_kill(monkeypatch)
    outcome = spawn_session._reap_half_spawned_session(42, SPAWN_DIR, is_tmux=True)
    assert outcome.reaped is False
    assert "tmux" in outcome.detail
    assert "clean up via tmux" in outcome.detail
    assert stop_calls == ["PID-42"]  # the daemon legs DID run
    assert kills == []
    # post()-level thread: the tmux-suffixed 500 passes is_tmux=True.
    urlopen_calls = _install_urlopen(
        monkeypatch,
        [
            _http_error(500, _webhook_body(42, tmux=True)),
            {"success": True, "sessionId": "sess-2"},
        ],
    )
    reap_calls = _install_reap(monkeypatch, [spawn_session._ReapOutcome(True, "reaped")])
    out = spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    assert out == {"success": True, "sessionId": "sess-2"}
    assert reap_calls == [(42, SPAWN_DIR, True)]
    assert len(urlopen_calls) == 2


# ── 18. error-body guard + mixed-exception retry ───────────────────────────


def test_non_dict_error_body_falls_back_to_raw_and_mixed_exception_retry(monkeypatch, sleeps):
    # (a) Non-JSON body -> err_body == {"raw": ...}, no crash, verbatim exit,
    # no reap (there is no "error" field to match).
    urlopen_calls = _install_urlopen(monkeypatch, [_http_error(502, b"<html>502</html>")])
    reap_calls = _install_reap(monkeypatch, [spawn_session._ReapOutcome(True, "never")])
    with pytest.raises(SystemExit) as exc:
        spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    assert str(exc.value).startswith("Happy daemon /spawn-session returned HTTP 502: {'raw': ")
    assert reap_calls == []
    assert len(urlopen_calls) == 1

    # (b) Mixed-exception sequence: attempt 1 = webhook-timeout HTTPError
    # (reap ok), attempt 2 = socket TimeoutError -> the TimeoutError
    # reconcile runs against attempt 2's OWN spawn_started_at window.
    ticks = {"n": 0}

    def fake_time():
        ticks["n"] += 1
        return 1000.0 * ticks["n"]  # attempt 1 -> 1000.0, attempt 2 -> 2000.0

    monkeypatch.setattr(spawn_session.time, "time", fake_time)
    urlopen_calls = _install_urlopen(
        monkeypatch,
        [_http_error(500, _webhook_body(3698)), TimeoutError("timed out")],
    )
    reap_calls = _install_reap(monkeypatch, [spawn_session._ReapOutcome(True, "reaped")])
    reconcile_calls: list[tuple[dict, float]] = []

    def fake_reconcile(body, spawn_started_at):
        reconcile_calls.append((body, spawn_started_at))
        return "adopted-sid"

    monkeypatch.setattr(spawn_session, "_reconcile_spawn_after_timeout", fake_reconcile)
    out = spawn_session.post("/spawn-session", {"directory": SPAWN_DIR})
    assert out == {"success": True, "sessionId": "adopted-sid"}
    assert len(reap_calls) == 1
    assert len(urlopen_calls) == 2
    assert spawn_session.WEBHOOK_TIMEOUT_RETRY_BACKOFF_S in sleeps
    # The per-attempt reset: attempt 2's window (2000.0), never attempt 1's.
    assert reconcile_calls == [({"directory": SPAWN_DIR}, 2000.0)]
