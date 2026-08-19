"""Unit tests for the #2140 ESCALATE-ONLY daemon-liveness pass.

The outage this pass closes: 2026-08-04T21:13Z -> 2026-08-05T04:30Z (3h17m)
the Happy daemon was unreachable, every autonomous spawn lane no-oped with
only stdout lines (no push, no sidecar, no marker), and detection came from
a human opening a PM session. Covers:

* the pure predicate ``decide_daemon_liveness_escalation`` (below-threshold
  silence; `open` exactly once at the threshold, no second `open`; `realert`
  at the TTL boundary and not before; `recovered` once then clean state; a
  reachable below-threshold tick leaves no event; garbled state -> safe
  defaults; duration stamped at the FIRST unreachable tick);
* the 20-tick #2140-replay arithmetic (exactly 1 open + 3 realert +
  1 recovered at threshold 2 / TTL 60 min / 10-min ticks; reported duration
  ~3h20m, not ~20 min);
* the channel split (critic Must-Fix 1): `open` + `recovered` fire the
  IMMEDIATE ``_telegram_push_urgent``, `realert` fires the DIGEST
  ``_telegram_push``, no event fires both;
* the escalate-only hard invariant (mutation surfaces raise-patched with a
  BaseException so the pass's own fail-soft `except Exception` cannot
  swallow a violation; pushes pinned with recorders per critic Should-Fix 4);
* save-BEFORE-emit storm guard (critic Should-Fix 3): a state save forced to
  fail on every tick fires the `open` PUSH at most once (measured: zero) while
  stderr + sidecar still record the episode;
* the kill switch, the dry-run zero-write contract (backs the
  ``--daemon-liveness-only --dry-run`` live smoke), enrichment fail-soft
  (unreadable queue -> ``?`` in the push, no raise), the real
  ``_telegram_push_urgent`` body against a tmp script (the code-style
  one-production-body-test rule), and a source pin that main() threads the
  tick's single probe result (never a second probe).

Follows ``tests/test_autonomous_session_watch_keep_running_owner.py``
conventions: patched state dirs, recorder seams, no network, no real marker
posts (the shared #1247/#1265 conftest autouse hermeticity guards cover this
module with zero ceremony).
"""

from __future__ import annotations

import inspect
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402

THRESHOLD = asw.DAEMON_LIVENESS_THRESHOLD  # 2 consecutive unreachable ticks
REALERT_S = 3600.0  # the 60-min default, in seconds
TICK_S = 600.0  # the 10-min cron cadence

# The genuine bodies, captured at import time (before any fixture stubs
# them) — the code-style "one production-body test per seam-stubbed
# function" rule: the urgent-push body test below executes the real
# shell-out against a tmp script.
_REAL_URGENT_PUSH = asw._telegram_push_urgent


class _MutationFired(BaseException):
    """Raise-patch payload for the escalate-only invariant test. Derives
    from BaseException ON PURPOSE: the pass's own top-level fail-soft
    ``except Exception`` would swallow an AssertionError and silently pass
    the very test that exists to catch a mutation."""


def _decide(*, reachable: bool, state: dict, now: float, realert_s: float = REALERT_S):
    return asw.decide_daemon_liveness_escalation(
        daemon_reachable=reachable, state=state, now=now, realert_s=realert_s
    )


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def liveness_dirs(tmp_path, monkeypatch):
    """Redirect the singleton state dir + sidecar root to tmp so the pass's
    REAL load/save/append bodies run against a hermetic tree."""
    reg = tmp_path / "eps-autonomous"
    reg.mkdir()
    root = tmp_path / "project"
    (root / ".claude" / "cache").mkdir(parents=True)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", reg)
    monkeypatch.setattr(asw, "PROJECT_ROOT", root)
    return SimpleNamespace(
        reg=reg,
        root=root,
        state=reg / "daemon-liveness.json",
        sidecar=root / ".claude" / "cache" / "daemon-liveness-events.jsonl",
    )


@pytest.fixture
def push_recorders(monkeypatch):
    """Recorder monkeypatches for BOTH push channels (critic Should-Fix 4:
    recorders, never raise-patched subprocess — the invariant test must be
    able to assert pushes DID fire). Signature-conformant by construction:
    both fakes mirror the real ``(msg, dry_run)`` positional shape."""
    urgent: list[tuple[str, bool]] = []
    digest: list[tuple[str, bool]] = []

    def _rec_urgent(msg, dry_run):
        urgent.append((msg, dry_run))
        return True

    def _rec_digest(msg, dry_run):
        digest.append((msg, dry_run))
        return True

    monkeypatch.setattr(asw, "_telegram_push_urgent", _rec_urgent)
    monkeypatch.setattr(asw, "_telegram_push", _rec_digest)
    return SimpleNamespace(urgent=urgent, digest=digest)


@pytest.fixture
def clock(monkeypatch):
    """Controllable wall clock for the pass-level tests."""
    box = {"now": 1_000_000.0}
    monkeypatch.setattr(asw.time, "time", lambda: box["now"])
    return box


# ─── the pure predicate ──────────────────────────────────────────────────────


def test_predicate_below_threshold_is_silent():
    state, event = _decide(reachable=False, state={}, now=100.0)
    assert event is None
    assert state["consecutive_unreachable"] == 1
    assert state["episode_open_ts"] == 100.0  # stamped at the FIRST unreachable tick
    assert state["escalated"] is False
    assert state["last_push_ts"] is None


def test_predicate_open_fires_exactly_at_threshold_and_never_twice():
    state, event = _decide(reachable=False, state={}, now=0.0)
    assert event is None
    state, event = _decide(reachable=False, state=state, now=TICK_S)
    assert event == "open"
    assert state["escalated"] is True
    assert state["last_push_ts"] == TICK_S
    assert state["consecutive_unreachable"] == THRESHOLD
    # tick 3, inside the re-alert TTL: NO second open, no realert.
    state, event = _decide(reachable=False, state=state, now=2 * TICK_S)
    assert event is None
    assert state["consecutive_unreachable"] == THRESHOLD + 1
    assert state["last_push_ts"] == TICK_S  # push stamp untouched on a None tick


def test_predicate_realert_fires_at_ttl_boundary_and_not_before():
    escalated = {
        "consecutive_unreachable": 5,
        "episode_open_ts": 0.0,
        "last_push_ts": 1000.0,
        "escalated": True,
    }
    state, event = _decide(reachable=False, state=escalated, now=1000.0 + REALERT_S - 1.0)
    assert event is None
    assert state["last_push_ts"] == 1000.0
    state, event = _decide(reachable=False, state=escalated, now=1000.0 + REALERT_S)
    assert event == "realert"
    assert state["last_push_ts"] == 1000.0 + REALERT_S


def test_predicate_recovered_fires_once_then_state_is_clean():
    escalated = {
        "consecutive_unreachable": 4,
        "episode_open_ts": 10.0,
        "last_push_ts": 700.0,
        "escalated": True,
    }
    state, event = _decide(reachable=True, state=escalated, now=2000.0)
    assert event == "recovered"
    assert state == {
        "consecutive_unreachable": 0,
        "episode_open_ts": None,
        "last_push_ts": None,
        "escalated": False,
    }
    # The next reachable tick is a no-op on clean state.
    state2, event2 = _decide(reachable=True, state=state, now=2600.0)
    assert event2 is None
    assert state2 == state


def test_predicate_reachable_below_threshold_leaves_no_trace():
    below = {
        "consecutive_unreachable": 1,
        "episode_open_ts": 50.0,
        "last_push_ts": None,
        "escalated": False,
    }
    state, event = _decide(reachable=True, state=below, now=700.0)
    assert event is None  # a single flapped tick leaves no trace
    assert state["consecutive_unreachable"] == 0
    assert state["episode_open_ts"] is None


def test_predicate_garbled_state_reads_as_safe_defaults():
    garbled = {
        "consecutive_unreachable": "nine",
        "episode_open_ts": "yesterday",
        "last_push_ts": [],
        "escalated": "yes",  # truthy string is NOT `is True`
    }
    state, event = _decide(reachable=False, state=garbled, now=42.0)
    # Fail-safe direction: worst case ONE delayed escalation, never spurious.
    assert event is None
    assert state["consecutive_unreachable"] == 1
    assert state["episode_open_ts"] == 42.0
    assert state["escalated"] is False
    # Bools must not masquerade as counters/timestamps (bool is an int
    # subclass) — a `True` counter reads as garbled, not as 1.
    state2, event2 = _decide(
        reachable=False,
        state={"consecutive_unreachable": True, "episode_open_ts": True},
        now=7.0,
    )
    assert event2 is None
    assert state2["consecutive_unreachable"] == 1
    assert state2["episode_open_ts"] == 7.0


def test_predicate_hand_edited_counter_above_threshold_still_opens():
    # The `>=` fail-soft guard: a counter already past the threshold with
    # escalated=False still escalates (a strict `==` would stay silent
    # forever on this shape).
    state, event = _decide(
        reachable=False,
        state={"consecutive_unreachable": 7, "episode_open_ts": 1.0, "escalated": False},
        now=9.0,
    )
    assert event == "open"
    assert state["escalated"] is True


def test_predicate_20_tick_outage_replay_counts_and_duration():
    """The #2140 window replay (plan success criterion): a 20-tick (3h20m)
    outage at threshold 2 / TTL 60 min / 10-min ticks produces exactly
    1 open + 3 realert + 1 recovered, and the duration is measured from the
    FIRST unreachable tick (~3h20m at recovery, never ~20 min)."""
    t0 = 100_000.0
    state: dict = {}
    events: list[str] = []
    for k in range(20):
        state, event = _decide(reachable=False, state=state, now=t0 + k * TICK_S)
        if event:
            events.append(event)
        assert state["episode_open_ts"] == t0  # stamped once, at tick 1
    recovery_now = t0 + 20 * TICK_S
    old_open_ts = state["episode_open_ts"]
    state, event = _decide(reachable=True, state=state, now=recovery_now)
    assert event == "recovered"
    events.append(event)
    assert events == ["open", "realert", "realert", "realert", "recovered"]
    duration_s = recovery_now - old_open_ts
    assert duration_s == 20 * TICK_S  # 3h20m, not ~20 min
    assert asw._daemon_liveness_duration_label(duration_s) == "3h20m"


def test_predicate_one_tick_flap_produces_zero_events():
    state, event = _decide(reachable=False, state={}, now=0.0)
    assert event is None
    state, event = _decide(reachable=True, state=state, now=TICK_S)
    assert event is None
    assert state["consecutive_unreachable"] == 0


# ─── the pass: channel split, invariants, guards ─────────────────────────────


def _run_cycle(clock, *, dry_run: bool = False) -> list[str | bool]:
    """Drive a full outage -> recovery cycle through the REAL pass body:
    2 unreachable ticks (open at tick 2), one tick past the re-alert TTL
    (realert), then a reachable tick (recovered). Returns the per-tick
    return values."""
    results = []
    for now, reachable in (
        (0.0, False),
        (TICK_S, False),
        (TICK_S + REALERT_S, False),
        (TICK_S + REALERT_S + TICK_S, True),
    ):
        clock["now"] = 1_000_000.0 + now
        results.append(asw.daemon_liveness_pass(dry_run, daemon_reachable=reachable))
    return results


def test_daemon_liveness_channel_split(liveness_dirs, push_recorders, clock):
    """Critic Must-Fix 1: `open` + `recovered` ride the IMMEDIATE channel,
    `realert` rides the DIGEST channel, and no event fires both."""
    results = _run_cycle(clock)
    assert results == [False, True, True, True]
    assert len(push_recorders.urgent) == 2
    assert "UNREACHABLE" in push_recorders.urgent[0][0]
    assert "reachable again" in push_recorders.urgent[1][0]
    assert len(push_recorders.digest) == 1
    assert "UNREACHABLE" in push_recorders.digest[0][0]
    # No event fired both channels: 3 events total, 3 push calls total.
    assert len(push_recorders.urgent) + len(push_recorders.digest) == 3
    # The alert is ACTIONABLE: suppressed-work counts + the login-shell
    # recovery command (empty tmp registry -> 0 registrations; missing
    # queue file -> "?").
    open_msg = push_recorders.urgent[0][0]
    assert "0 registered autonomous session(s)" in open_msg
    assert "? ripe infra task(s)" in open_msg
    assert "bash -lc 'happy daemon status; happy daemon start'" in open_msg
    assert "never restarts the daemon" in open_msg
    # Recovery names the measured outage duration (open stamped at tick 1;
    # the cycle's recovery tick lands 4800s = 1h20m after the first
    # unreachable tick).
    assert "1h20m" in push_recorders.urgent[1][0]
    # Durable channels: sidecar rows for all 3 events; state reset at the end.
    rows = [json.loads(line) for line in liveness_dirs.sidecar.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["open", "realert", "recovered"]
    final_state = json.loads(liveness_dirs.state.read_text())
    assert final_state["escalated"] is False
    assert final_state["consecutive_unreachable"] == 0


def test_daemon_liveness_pass_never_restarts_or_mutates(
    liveness_dirs, push_recorders, clock, monkeypatch
):
    """The ESCALATE-ONLY hard invariant (plan acceptance criterion 6): a
    full outage -> recovery cycle completes, pushes fire, and NO mutation
    surface is ever touched. Mutation surfaces are raise-patched with a
    BaseException (the pass's fail-soft `except Exception` cannot swallow
    it); the push channel stays on recorders (critic Should-Fix 4 — a
    raise-patched subprocess would also sever the push helpers' shell-out,
    so the test could not simultaneously assert pushes fired)."""

    def _boom(*_a, **_k):
        raise _MutationFired("escalate-only violated: a mutation surface fired")

    for surface in (
        "_stop_session",  # session stop
        "_respawn",  # crash-recovery spawn
        "_respawn_stalled_session",  # stalled-arm spawn
        "_completed_unmerged_respawn",  # bounded-respawn spawn
        "_post_progress_marker",  # task.py post-marker
    ):
        monkeypatch.setattr(asw, surface, _boom)

    results = _run_cycle(clock)
    assert results == [False, True, True, True]
    assert len(push_recorders.urgent) == 2  # the pass completed + escalated
    assert len(push_recorders.digest) == 1


def test_save_before_emit_storm_guard(liveness_dirs, push_recorders, clock, monkeypatch):
    """Critic Should-Fix 3: with the state save failing on EVERY tick, a
    5-tick outage fires the `open` PUSH at most once (measured: zero — the
    save-first ordering suppresses the push on a failed save), while
    stderr + the sidecar still record the episode."""
    # Seed one already-counted tick so each subsequent tick recomputes the
    # threshold crossing (the save-after-emit order would push every tick).
    liveness_dirs.state.write_text(
        json.dumps(
            {
                "consecutive_unreachable": 1,
                "episode_open_ts": 1_000_000.0,
                "last_push_ts": None,
                "escalated": False,
            }
        )
    )
    save_calls: list[dict] = []

    def _failing_save(state, dry_run):
        save_calls.append(state)
        return False

    monkeypatch.setattr(asw, "_save_daemon_liveness_state", _failing_save)
    for k in range(5):
        clock["now"] = 1_000_000.0 + (k + 1) * TICK_S
        fired = asw.daemon_liveness_pass(False, daemon_reachable=False)
        assert fired is True  # the event still registers (sidecar/stderr)
    assert len(push_recorders.urgent) <= 1  # the storm guard bound
    assert len(push_recorders.urgent) == 0  # realized: failed save never pushes
    assert len(push_recorders.digest) == 0
    assert len(save_calls) == 5  # save attempted BEFORE every emit
    rows = [json.loads(line) for line in liveness_dirs.sidecar.read_text().splitlines()]
    assert len(rows) == 5  # the durable record survives the save failure
    assert all(r["event"] == "open" and r["state_saved"] is False for r in rows)


def test_kill_switch_writes_nothing_and_pushes_nothing(
    liveness_dirs, push_recorders, clock, monkeypatch
):
    monkeypatch.setenv("EPM_DISABLE_DAEMON_LIVENESS_PASS", "1")
    for k in range(3):
        clock["now"] = 1_000_000.0 + k * TICK_S
        assert asw.daemon_liveness_pass(False, daemon_reachable=False) is False
    assert not liveness_dirs.state.exists()
    assert not liveness_dirs.sidecar.exists()
    assert push_recorders.urgent == []
    assert push_recorders.digest == []


def test_daemon_liveness_pass_dry_run_writes_nothing(liveness_dirs, push_recorders, clock):
    """The dry-run kwarg thread (plan acceptance criterion 7): a full
    outage -> recovery cycle under dry_run=True performs ZERO writes (state
    + sidecar paths both still absent) and fires no push — so the
    ``--daemon-liveness-only --dry-run`` live smoke can never mutate fleet
    state. (Under dry-run the state never advances, so no event ever
    fires — every recorded push call, if any, must itself carry
    dry_run=True.)"""
    _run_cycle(clock, dry_run=True)
    assert not liveness_dirs.state.exists()
    assert not liveness_dirs.sidecar.exists()
    assert all(dry is True for _msg, dry in push_recorders.urgent)
    assert all(dry is True for _msg, dry in push_recorders.digest)


def test_enrichment_failsoft_unreadable_queue_renders_question_mark(
    liveness_dirs, push_recorders, clock
):
    """Plan acceptance criterion 5 fail-soft leg: an unreadable infra-drain
    queue yields `?` in the push; a readable one yields the real counts —
    and neither path raises."""
    # Garbled queue + two registrations.
    (liveness_dirs.reg / asw.INFRA_DRAIN_QUEUE_BASENAME).write_text("{not json")
    (liveness_dirs.reg / "issue-101.json").write_text("{}")
    (liveness_dirs.reg / "issue-102.json").write_text("{}")
    work = asw._daemon_liveness_suppressed_work()
    assert work == {"registrations": 2, "ripe_infra": None}
    clock["now"] = 1_000_000.0
    asw.daemon_liveness_pass(False, daemon_reachable=False)
    clock["now"] = 1_000_000.0 + TICK_S
    asw.daemon_liveness_pass(False, daemon_reachable=False)
    assert len(push_recorders.urgent) == 1
    msg = push_recorders.urgent[0][0]
    assert "2 registered autonomous session(s)" in msg
    assert "? ripe infra task(s)" in msg
    # Valid queue -> the real depth.
    (liveness_dirs.reg / asw.INFRA_DRAIN_QUEUE_BASENAME).write_text(
        json.dumps({"ripe_oldest_first": [11, 12, 13], "cap": 5})
    )
    assert asw._daemon_liveness_suppressed_work() == {"registrations": 2, "ripe_infra": 3}


# ─── the real urgent-push body (one production-body test per stubbed seam) ───


def _tmp_push_script(
    tmp_path: Path, *, rc: int = 0, name: str = "telegram_push"
) -> tuple[Path, Path]:
    """A real executable stand-in at the EXTERNAL subprocess boundary: it
    records its argv to a file and exits ``rc``."""
    out = tmp_path / f"{name}-sent.txt"
    script = tmp_path / f"{name}.sh"
    script.write_text(f'#!/bin/bash\nprintf "%s" "$1" > "{out}"\nexit {rc}\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, out


def test_telegram_push_urgent_real_body_sends_via_script(tmp_path, monkeypatch):
    script, out = _tmp_push_script(tmp_path)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_URGENT_SCRIPT", str(script))
    assert _REAL_URGENT_PUSH("daemon down 1h10m", False) is True
    assert out.read_text() == "daemon down 1h10m"


def test_telegram_push_urgent_real_body_fail_soft_branches(tmp_path, monkeypatch, capsys):
    # Non-zero rc -> False, loud, never raises.
    script, _out = _tmp_push_script(tmp_path, rc=3)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_URGENT_SCRIPT", str(script))
    assert _REAL_URGENT_PUSH("msg", False) is False
    # Missing script -> False, loud.
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_URGENT_SCRIPT", str(tmp_path / "absent.sh"))
    assert _REAL_URGENT_PUSH("msg", False) is False
    # Dry run -> False, no send even with a working script.
    script2, out2 = _tmp_push_script(tmp_path, name="telegram_push_dry")
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_URGENT_SCRIPT", str(script2))
    assert _REAL_URGENT_PUSH("msg", True) is False
    assert not out2.exists()
    err = capsys.readouterr().err
    assert err.count("urgent telegram push") >= 2


def test_urgent_script_default_and_override():
    default = asw._TELEGRAM_PUSH_URGENT_SCRIPT_DEFAULT
    assert default.name == "telegram_push.sh"
    prev = os.environ.pop("EPM_TELEGRAM_PUSH_URGENT_SCRIPT", None)
    try:
        assert asw._telegram_push_urgent_script() == default
        os.environ["EPM_TELEGRAM_PUSH_URGENT_SCRIPT"] = "/tmp/x.sh"
        assert asw._telegram_push_urgent_script() == Path("/tmp/x.sh")
    finally:
        if prev is None:
            os.environ.pop("EPM_TELEGRAM_PUSH_URGENT_SCRIPT", None)
        else:
            os.environ["EPM_TELEGRAM_PUSH_URGENT_SCRIPT"] = prev


# ─── wiring pins ─────────────────────────────────────────────────────────────


def test_main_threads_the_ticks_single_probe_result():
    """main() feeds the pass the tick's already-computed probe result —
    never a second probe (plan kill criterion 1). The production call site
    passes the `daemon_reachable` local; only the --daemon-liveness-only
    debug branch takes a bare probe."""
    src = inspect.getsource(asw.main)
    assert "daemon_liveness_pass(args.dry_run, daemon_reachable=daemon_reachable)" in src
    assert (
        "daemon_liveness_pass(args.dry_run, daemon_reachable=_daemon_reachable())" in src
    )  # the --only branch, bare probe by convention


def test_pass_load_save_real_bodies_roundtrip(liveness_dirs):
    """The real load/save bodies (stubbed in the storm-guard test) round-trip
    through the atomic tmp+rename path; a garbled file loads as {}."""
    state = {
        "consecutive_unreachable": 3,
        "episode_open_ts": 5.0,
        "last_push_ts": 6.0,
        "escalated": True,
    }
    assert asw._save_daemon_liveness_state(state, False) is True
    assert asw._load_daemon_liveness_state() == state
    liveness_dirs.state.write_text("{broken")
    assert asw._load_daemon_liveness_state() == {}
    liveness_dirs.state.write_text(json.dumps([1, 2]))  # non-dict payload
    assert asw._load_daemon_liveness_state() == {}
