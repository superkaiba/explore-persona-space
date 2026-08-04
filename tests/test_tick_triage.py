"""Unit tests for ``scripts/tick_triage.py`` — the one-call tick triage.

What this pins:

1. **Issue-mode verdict table** — HEALTHY / TERMINAL / GATE-TRANSITION /
   STALE-REDRIVE across the status enum x marker freshness, including the
   over-cap ``plan_pending`` gate special-case.
2. **Runaway counter** — the 3rd consecutive teardown-verdict triage
   (TERMINAL or GATE-TRANSITION) writes ``tick-runaway-<N>.flag``; any other
   verdict resets the streak AND clears a stale flag.
3. **Campaign-mode verdicts** — stranded-cron teardown, results-landed wake,
   all-arms-in-flight quiet idle, decision-round-owed re-drive.
4. **Fail-loud contract** — any state-read failure exits non-zero (the tick
   skill treats that as STALE-REDRIVE: fail toward coverage).

All state I/O goes through tmp dirs via ``EPM_TICK_STATE_DIR``; task reads
are monkeypatched — no live sessions, no real task folders.
"""

from __future__ import annotations

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

import tick_triage  # noqa: E402

NOW = time.time()


def _iso(epoch: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event(kind: str, age_s: float, note: str = "") -> dict:
    return {"kind": kind, "ts": _iso(NOW - age_s), "note": note}


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_TICK_STATE_DIR", str(tmp_path))
    return tmp_path


# Saved BEFORE the autouse fixture below patches the module attribute, so the
# #1629 resolution-chain tests can opt back in to the REAL resolver.
_REAL_TRANSCRIPT_RESOLVER = tick_triage._own_session_transcript_path


@pytest.fixture(autouse=True)
def _neutralize_human_probe(monkeypatch):
    """AUTOUSE (#1629, consistency WARN 2): the human-activity probe reads
    AMBIENT state (/proc ancestry from ``os.getpid()``), so a pytest run
    inside an interactive Claude session would resolve the LIVE session
    transcript and could flip every pre-existing STALE-REDRIVE expectation
    to HEALTHY. Neutralize by default; the #1629 suppression tests opt back
    in by re-patching to their fixture path (or to
    ``_REAL_TRANSCRIPT_RESOLVER`` for the resolution-chain tests)."""
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: None)


# ── compute_issue_verdict ───────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_ACTIVE | tick_triage.ISSUE_PARK))
def test_issue_fresh_marker_is_healthy(status):
    verdict, _, _ = tick_triage.compute_issue_verdict(
        status, status, 60.0, False, stale_after_s=1500
    )
    assert verdict == "HEALTHY"


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_ACTIVE | tick_triage.ISSUE_PARK))
def test_issue_stale_marker_is_redrive(status):
    verdict, _, _ = tick_triage.compute_issue_verdict(
        status, status, 3600.0, False, stale_after_s=1500
    )
    assert verdict == "STALE-REDRIVE"


def test_issue_no_markers_counts_as_stale():
    verdict, _, _ = tick_triage.compute_issue_verdict(
        "running", "running", None, False, stale_after_s=1500
    )
    assert verdict == "STALE-REDRIVE"


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_TERMINAL))
def test_issue_steady_terminal_is_terminal(status):
    verdict, _, _ = tick_triage.compute_issue_verdict(
        status, status, 60.0, False, stale_after_s=1500
    )
    assert verdict == "TERMINAL"


def test_issue_status_sets_cover_runtime_enum():
    """Every runtime task status MUST be classified by exactly one of the
    tick's issue-mode sets (ACTIVE / PARK / TERMINAL); otherwise
    compute_issue_verdict raises on a real task. Incident: `on_hold` was
    added to STATUSES without a tick-set entry, crashing every tick fired on
    a parked task. ISSUE_GATE is an annotation subset of TERMINAL, not part
    of the partition."""
    from explore_persona_space.task_workflow import STATUSES

    classified = tick_triage.ISSUE_ACTIVE | tick_triage.ISSUE_PARK | tick_triage.ISSUE_TERMINAL
    assert classified == set(STATUSES), (
        "tick issue-mode sets disagree with runtime STATUSES: "
        f"missing={set(STATUSES) - classified}, extra={classified - set(STATUSES)}"
    )
    assert tick_triage.ISSUE_ACTIVE.isdisjoint(tick_triage.ISSUE_PARK)
    assert tick_triage.ISSUE_ACTIVE.isdisjoint(tick_triage.ISSUE_TERMINAL)
    assert tick_triage.ISSUE_PARK.isdisjoint(tick_triage.ISSUE_TERMINAL)
    assert tick_triage.ISSUE_GATE <= tick_triage.ISSUE_TERMINAL


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_GATE))
def test_issue_gate_transition_fires_on_status_change(status):
    verdict, _, _ = tick_triage.compute_issue_verdict(
        status, "running", 60.0, False, stale_after_s=1500
    )
    assert verdict == "GATE-TRANSITION"


def test_issue_gate_transition_on_missing_snapshot():
    # Previous status unknown + currently at a gate: fire the transition
    # branch (a duplicate push beats a missed one — the tick skill's rule).
    verdict, _, _ = tick_triage.compute_issue_verdict(
        "awaiting_promotion", None, 60.0, False, stale_after_s=1500
    )
    assert verdict == "GATE-TRANSITION"


def test_issue_completed_transition_is_plain_terminal():
    # completed/archived are terminal but NOT user gates — no push branch.
    verdict, _, _ = tick_triage.compute_issue_verdict(
        "completed", "reviewing", 60.0, False, stale_after_s=1500
    )
    assert verdict == "TERMINAL"


def test_issue_plan_pending_over_cap_is_gate():
    verdict, _, _ = tick_triage.compute_issue_verdict(
        "plan_pending", "planning", 60.0, True, stale_after_s=1500
    )
    assert verdict == "GATE-TRANSITION"
    verdict, _, _ = tick_triage.compute_issue_verdict(
        "plan_pending", "plan_pending", 60.0, True, stale_after_s=1500
    )
    assert verdict == "TERMINAL"


def test_issue_plan_pending_under_cap_is_park():
    verdict, _, _ = tick_triage.compute_issue_verdict(
        "plan_pending", "plan_pending", 3600.0, False, stale_after_s=1500
    )
    assert verdict == "STALE-REDRIVE"


def test_issue_unknown_status_raises():
    with pytest.raises(ValueError):
        tick_triage.compute_issue_verdict("clarifying", None, 60.0, False, stale_after_s=1500)


# ── plan_pending_over_cap ───────────────────────────────────────────────────


def test_over_cap_requires_spend_marker_newer_than_status_change():
    events = [
        _event("epm:status-changed v1", 600),
        _event("epm:awaiting-spend-approval v1", 60),
    ]
    assert tick_triage.plan_pending_over_cap(events)
    events = [
        _event("epm:awaiting-spend-approval v1", 600),
        _event("epm:status-changed v1", 60),
    ]
    assert not tick_triage.plan_pending_over_cap(events)
    assert not tick_triage.plan_pending_over_cap([_event("epm:status-changed v1", 60)])


# ── latest_event_ts ─────────────────────────────────────────────────────────


def test_latest_event_ts_ignores_watcher_sentinel_notes():
    events = [
        _event("epm:campaign-progress v1", 7200),
        _event("epm:campaign-progress v2", 60, note="[autonomous_session_watch:campaign] alert"),
    ]
    ts = tick_triage.latest_event_ts(events, prefix="epm:campaign")
    assert ts is not None and (NOW - ts) > 3600


def test_latest_event_ts_ignores_deliberate_stop_records():
    # #1053: both legs of the deliberate-stop predicate — the lstripped note
    # PREFIX (an issue-session-guard exit breadcrumb) and the
    # by="spawn_session-stop" identity (note text irrelevant) — must never
    # count as issue freshness.
    for row in (
        {
            "kind": "epm:progress",
            "ts": _iso(NOW - 60),
            "note": "deliberate-stop pid=n/a target=self reason=step0-session-collision "
            "owner=happy-session:abc123 — duplicate /issue 42 session exiting at Step 0; "
            "owner happy-session:abc123 remains the driver; no state mutated",
            "by": "issue-session-guard",
        },
        {
            "kind": "epm:progress",
            "ts": _iso(NOW - 60),
            "note": "stopping session",
            "by": "spawn_session-stop",
        },
    ):
        events = [_event("epm:progress v1", 7200, note="real work marker"), row]
        ts = tick_triage.latest_event_ts(events)
        assert ts is not None and (NOW - ts) > 3600, row


# ── #1053 end-to-end: exit breadcrumb must not mask staleness ───────────────


def test_step0_collision_exit_breadcrumb_does_not_mask_staleness(state_dir, monkeypatch):
    # #1053 MF-2 pin (end-to-end): a stale issue whose ONLY fresh event is the
    # prescribed Step 0 collision-exit (or stale-wake-yield) breadcrumb must
    # stay STALE-REDRIVE — the duplicate's death record must not flip the tick
    # verdict to HEALTHY and mask a dead owner chain.
    collision_note = (
        "deliberate-stop pid=n/a target=self reason=step0-session-collision "
        "owner=happy-session:abc123 — duplicate /issue 1053 session exiting at Step 0; "
        "owner happy-session:abc123 remains the driver; no state mutated"
    )
    yield_note = (
        "deliberate-stop pid=n/a target=self reason=stale-wake-yield "
        "replacement=happy-session:def456 — stale /issue 1053 session yielding on wake; "
        "the replacement owns the task; no state mutated"
    )
    stale_age = tick_triage.stale_s() + 600
    for note in (collision_note, yield_note):
        events = [
            _event("epm:progress v1", stale_age, note="real work marker"),
            {
                "kind": "epm:progress",
                "ts": _iso(NOW - 60),
                "note": note,
                "by": "issue-session-guard",
            },
        ]
        _patch_issue_state(monkeypatch, "running", events)
        verdict, reason = tick_triage.triage(1053, "issue")
        assert verdict == "STALE-REDRIVE", (verdict, reason, note)


# ── runaway streak (via triage end-to-end) ──────────────────────────────────


def _patch_issue_state(monkeypatch, status: str, events: list[dict]):
    monkeypatch.setattr(tick_triage, "load_task_state", lambda _n: (status, events))


def test_runaway_flag_on_third_consecutive_terminal(state_dir, monkeypatch):
    _patch_issue_state(monkeypatch, "awaiting_promotion", [_event("epm:progress v1", 60)])
    for i in range(1, 4):
        tick_triage.triage(42, "issue")
        snap = json.loads(tick_triage.snapshot_path(42).read_text())
        assert snap["terminal_streak"] == i
    flag = tick_triage.runaway_flag_path(42)
    assert flag.is_file(), "3rd consecutive terminal tick must write the runaway flag"
    payload = json.loads(flag.read_text())
    assert payload["issue"] == 42 and payload["terminal_streak"] == 3


def test_streak_resets_on_non_terminal(state_dir, monkeypatch):
    _patch_issue_state(monkeypatch, "awaiting_promotion", [])
    tick_triage.triage(7, "issue")
    tick_triage.triage(7, "issue")
    assert json.loads(tick_triage.snapshot_path(7).read_text())["terminal_streak"] == 2
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 60)])
    verdict, _ = tick_triage.triage(7, "issue")
    assert verdict == "HEALTHY"
    assert json.loads(tick_triage.snapshot_path(7).read_text())["terminal_streak"] == 0
    assert not tick_triage.runaway_flag_path(7).is_file()


def test_snapshot_keeps_legacy_shape(state_dir, monkeypatch):
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 60)])
    tick_triage.triage(9, "issue")
    snap = json.loads(tick_triage.snapshot_path(9).read_text())
    assert snap["issue"] == 9 and snap["status"] == "running" and "ts" in snap


def test_gate_transition_then_terminal_on_repeat(state_dir, monkeypatch):
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 60)])
    tick_triage.triage(11, "issue")
    _patch_issue_state(monkeypatch, "awaiting_promotion", [_event("epm:progress v1", 60)])
    verdict, _ = tick_triage.triage(11, "issue")
    assert verdict == "GATE-TRANSITION"
    verdict, _ = tick_triage.triage(11, "issue")
    assert verdict == "TERMINAL", "second tick at the same gate must not re-push"


def test_stale_runaway_flag_cleared_on_recovery(state_dir, monkeypatch):
    """Review major (2026-06-12): a flag written during an earlier
    teardown-whiff episode must not survive a recovery — otherwise it would
    force-stop the session on weeks-old corroboration at the NEXT park."""
    _patch_issue_state(monkeypatch, "blocked", [_event("epm:failure v1", 60)])
    for _ in range(3):
        tick_triage.triage(13, "issue")
    assert tick_triage.runaway_flag_path(13).is_file()
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 60)])
    verdict, _ = tick_triage.triage(13, "issue")
    assert verdict == "HEALTHY"
    assert not tick_triage.runaway_flag_path(13).is_file(), (
        "a streak reset must also unlink the stale runaway flag"
    )


def test_over_cap_plan_pending_whiff_writes_flag(state_dir, monkeypatch):
    """Review minor (2026-06-12): the streak counts TEARDOWN VERDICTS, not
    just terminal statuses — a teardown that whiffs forever at over-cap
    plan_pending gets the same parachute (watcher alert-only outside the
    DONE set)."""
    events = [
        _event("epm:status-changed v1", 600),
        _event("epm:awaiting-spend-approval v1", 60),
    ]
    _patch_issue_state(monkeypatch, "plan_pending", events)
    verdicts = [tick_triage.triage(17, "issue")[0] for _ in range(3)]
    assert verdicts[0] == "GATE-TRANSITION" and verdicts[1] == "TERMINAL"
    assert tick_triage.runaway_flag_path(17).is_file(), (
        "3 consecutive teardown-verdict ticks at over-cap plan_pending must flag"
    )


# ── campaign mode ───────────────────────────────────────────────────────────


def test_campaign_stranded_cron_is_terminal():
    verdict, reason = tick_triage.compute_campaign_verdict(
        "planning",
        None,
        None,
        landed_unreconciled=[],
        open_rows_all_in_flight=False,
        stale_after_s=1500,
    )
    assert verdict == "TERMINAL" and "stranded" in reason


def test_campaign_blocked_transition_pushes():
    verdict, _ = tick_triage.compute_campaign_verdict(
        "blocked",
        "running",
        None,
        landed_unreconciled=[],
        open_rows_all_in_flight=False,
        stale_after_s=1500,
    )
    assert verdict == "GATE-TRANSITION"
    verdict, _ = tick_triage.compute_campaign_verdict(
        "blocked",
        "blocked",
        None,
        landed_unreconciled=[],
        open_rows_all_in_flight=False,
        stale_after_s=1500,
    )
    assert verdict == "TERMINAL"


def test_campaign_landed_result_wakes_regardless_of_freshness():
    verdict, reason = tick_triage.compute_campaign_verdict(
        "running",
        "running",
        60.0,
        landed_unreconciled=[593],
        open_rows_all_in_flight=False,
        stale_after_s=1500,
    )
    assert verdict == "STALE-REDRIVE" and "#593" in reason


def test_campaign_fresh_marker_is_healthy():
    verdict, _ = tick_triage.compute_campaign_verdict(
        "running",
        "running",
        60.0,
        landed_unreconciled=[],
        open_rows_all_in_flight=False,
        stale_after_s=1500,
    )
    assert verdict == "HEALTHY"


def test_campaign_stale_but_all_arms_in_flight_is_healthy():
    verdict, _ = tick_triage.compute_campaign_verdict(
        "running",
        "running",
        7200.0,
        landed_unreconciled=[],
        open_rows_all_in_flight=True,
        stale_after_s=1500,
    )
    assert verdict == "HEALTHY"


def test_campaign_stale_with_open_rows_redrives():
    verdict, _ = tick_triage.compute_campaign_verdict(
        "running",
        "running",
        7200.0,
        landed_unreconciled=[],
        open_rows_all_in_flight=False,
        stale_after_s=1500,
    )
    assert verdict == "STALE-REDRIVE"


def test_campaign_open_rows_derivation():
    state = {
        "experiments": [
            {"id": "e1", "status": "ingested", "child_task": 100},  # finished — ignored
            {"id": "e2", "status": "running", "child_task": 101},  # in flight
            {"id": "e3", "status": "running", "child_task": 102},  # landed
            {"id": "e4", "status": "planned", "child_task": None},  # decision owed
        ]
    }
    children = [
        {"id": 101, "status": "running"},
        {"id": 102, "status": "awaiting_promotion"},
    ]
    landed, all_in_flight = tick_triage.campaign_open_rows(state, children)
    assert landed == [102]
    assert all_in_flight is False

    state["experiments"] = [{"id": "e2", "status": "running", "child_task": 101}]
    landed, all_in_flight = tick_triage.campaign_open_rows(state, children)
    assert landed == [] and all_in_flight is True


def test_campaign_zero_open_rows_owes_decision():
    """Review blocker (2026-06-12): zero open rows — missing/garbled state
    file, or every row ingested/abandoned — must NOT read as
    all-arms-in-flight; such a campaign owes a decision round."""
    children = [{"id": 101, "status": "running"}]
    landed, all_in_flight = tick_triage.campaign_open_rows({}, children)
    assert landed == [] and all_in_flight is False
    state = {
        "experiments": [
            {"id": "e1", "status": "ingested", "child_task": 100},
            {"id": "e2", "status": "abandoned", "child_task": 101},
        ]
    }
    landed, all_in_flight = tick_triage.campaign_open_rows(state, children)
    assert landed == [] and all_in_flight is False


def test_campaign_cold_start_no_state_file_redrives(state_dir, monkeypatch):
    """End-to-end pin for the blocker: an ACTIVE campaign with stale (or no)
    campaign markers and NO campaign-state.json yet must STALE-REDRIVE, not
    idle as HEALTHY (the died-between-arm-and-first-write cold-start class)."""
    monkeypatch.setattr(tick_triage, "load_task_state", lambda _n: ("running", []))
    monkeypatch.setattr(tick_triage, "load_children", lambda _n: [])
    monkeypatch.setattr(tick_triage, "load_campaign_state", lambda _n: {})
    verdict, reason = tick_triage.triage(21, "campaign")
    assert verdict == "STALE-REDRIVE", reason


# ── content invariant (#1000) ───────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["issue", "campaign"])
def test_snapshot_carries_no_task_text(state_dir, monkeypatch, kind):
    """#1000: the printed verdict line AND the snapshot stay digest-only —
    a fixed key set and no task/marker-note text anywhere (both print/persist
    into tick turns; free text is the #866/#906 refusal-kill surface). Pins
    the CONTENT INVARIANT comment above compute_issue_verdict."""
    sentinel = "TRIGGERSENTINELXYZ"
    event_kind = "epm:progress v1" if kind == "issue" else "epm:campaign-poll v1"
    events = [_event(event_kind, 60, note=sentinel * 30)]
    _patch_issue_state(monkeypatch, "running", events)
    if kind == "campaign":
        monkeypatch.setattr(tick_triage, "load_children", lambda _n: [])
        monkeypatch.setattr(tick_triage, "load_campaign_state", lambda _n: {})
    verdict, reason = tick_triage.triage(13, kind)
    assert sentinel not in f"{verdict} {reason}"
    snap_text = tick_triage.snapshot_path(13).read_text()
    snap = json.loads(snap_text)
    assert set(snap) <= {"issue", "status", "ts", "terminal_streak", "root_disk"}, snap
    assert sentinel not in snap_text


# ── fail-loud CLI contract ──────────────────────────────────────────────────


def test_main_exits_nonzero_on_read_failure(state_dir, monkeypatch, capsys):
    def boom(_n):
        raise FileNotFoundError("task #999 not found")

    monkeypatch.setattr(tick_triage, "load_task_state", boom)
    rc = tick_triage.main(["999"])
    assert rc != 0
    assert "FAILED" in capsys.readouterr().err


def test_main_prints_single_verdict_line(state_dir, monkeypatch, capsys):
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 60)])
    rc = tick_triage.main(["42"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and len(out) == 1 and out[0].startswith("HEALTHY ")


# ── detached-phase liveness screen (#1051) ──────────────────────────────────


def _crumb(
    age_s: float,
    pid: int | None = None,
    log: str | None = None,
    stage: str = "followup-running",
    extra: str = "",
) -> dict:
    """A ``stage-dispatch`` breadcrumb-shaped ``epm:progress`` event."""
    parts = [f"stage-dispatch stage={stage} round=1 subagent=detached-vm-phase"]
    if pid is not None:
        parts.append(f"pid={pid}")
    if log is not None:
        parts.append(f"log={log}")
    if extra:
        parts.append(extra)
    return _event("epm:progress v1", age_s, note=" ".join(parts))


def test_stale_with_live_detached_pid_is_healthy(state_dir, monkeypatch):
    """The #931 replay fixture: an in-flight pid-bearing breadcrumb + plain
    liveness notes straddling the 25-min stale window (27-min gap shape) must
    read HEALTHY when the pid probe verifies alive+identity."""
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    events = [
        _crumb(
            7200,
            pid=4025577,
            log="/tmp/i931.log",
            extra="choom=ok label=author-blocked-folds worktree=/x/.claude/worktrees/issue-931",
        ),
        _event("epm:progress v2", 3300, note="liveness: pid 4025577 verified, args match"),
        _event("epm:progress v3", 1680, note="liveness: child worker advancing"),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, reason = tick_triage.triage(101, "issue")
    assert verdict == "HEALTHY", reason
    assert "4025577" in reason


def test_stale_with_dead_pid_redrives(state_dir, monkeypatch):
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: False)
    events = [
        _crumb(7200, pid=4025577),
        _event("epm:progress v2", 3300, note="liveness: pid 4025577 verified, args match"),
        _event("epm:progress v3", 1680, note="liveness: child worker advancing"),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, _ = tick_triage.triage(102, "issue")
    assert verdict == "STALE-REDRIVE"


def test_recycled_pid_identity_mismatch_redrives(state_dir, monkeypatch, tmp_path):
    """Exercise the REAL start-time identity guard via the _PROC_ROOT seam: a
    live pid whose start-epoch postdates breadcrumb ts + slack is a recycled
    pid, never re-attached to."""
    clk = os.sysconf("SC_CLK_TCK")
    pid = 4025577
    btime = int(NOW - 100_000)
    proc = tmp_path / "proc"
    (proc / str(pid)).mkdir(parents=True)
    (proc / "stat").write_text(f"cpu  1 2 3 4\nbtime {btime}\nprocesses 5\n")
    # start-epoch ~= NOW (recycled just now) — AFTER breadcrumb ts + 120s.
    starttime_ticks = int((NOW - btime) * clk)
    after_comm = ["S"] + ["0"] * 18 + [str(starttime_ticks)] + ["0"] * 10
    # comm contains ') ' to exercise the parse-after-LAST-')' rule.
    (proc / str(pid) / "stat").write_text(f"{pid} (uv (run) x) " + " ".join(after_comm) + "\n")
    monkeypatch.setattr(tick_triage, "_PROC_ROOT", proc)
    assert tick_triage.pid_alive_with_identity(pid, NOW - 7200) is False
    _patch_issue_state(monkeypatch, "followups_running", [_crumb(7200, pid=pid)])
    verdict, _ = tick_triage.triage(103, "issue")
    assert verdict == "STALE-REDRIVE"


def test_proc_start_epoch_real_self():
    """#906 production-body rule: execute the real /proc read end-to-end on
    the test process itself."""
    start = tick_triage.proc_start_epoch(os.getpid())
    assert start is not None and 0 < start <= time.time()
    assert tick_triage.pid_alive_with_identity(os.getpid(), time.time()) is True
    # A launch deadline BEFORE the process started fails the identity guard.
    assert tick_triage.pid_alive_with_identity(os.getpid(), 0.0) is False


def test_cleared_breadcrumb_not_probed(state_dir, monkeypatch):
    """A breadcrumb with a LATER stage-clearing event is dead history: the pid
    probe is NEVER consulted. The stub is call-recording and returns True (NOT
    a raising stub — issue_liveness_reason's blanket ``except Exception``
    would swallow a raise and pass the test under the exact bug it pins): a
    wrongly-consulted probe returns True -> HEALTHY -> the verdict assert
    fails loud, and the empty-call-list assert catches the bug even if a
    future refactor changes the verdict path."""
    calls: list = []

    def recording_probe(pid, ts):
        calls.append((pid, ts))
        return True

    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", recording_probe)
    for clearing_kind in ("epm:upload-verification v1", "epm:failure v1"):
        calls.clear()
        events = [_crumb(7200, pid=999), _event(clearing_kind, 3600)]
        _patch_issue_state(monkeypatch, "followups_running", events)
        verdict, _ = tick_triage.triage(104, "issue")
        assert verdict == "STALE-REDRIVE", clearing_kind
        assert calls == [], f"cleared breadcrumb must never be probed ({clearing_kind})"


def test_breadcrumb_without_pid_ignored():
    # The newest PID-BEARING crumb decides even when a pid-less breadcrumb
    # (the interpreting/clean-result shape) is newer.
    events = [_crumb(7200, pid=4025577), _crumb(3600, pid=None)]
    crumb = tick_triage.newest_inflight_pid_breadcrumb(events)
    assert crumb is not None and crumb["pid"] == 4025577
    # A lone pid-less crumb yields None.
    assert tick_triage.newest_inflight_pid_breadcrumb([_crumb(3600, pid=None)]) is None


def test_breadcrumb_over_max_age_not_probed(state_dir, monkeypatch):
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    _patch_issue_state(monkeypatch, "followups_running", [_crumb(60 * 3600, pid=999)])
    verdict, _ = tick_triage.triage(105, "issue")
    assert verdict == "STALE-REDRIVE", "a 60h-old breadcrumb never grants HEALTHY (48h cap)"


def test_heartbeat_prefixed_note_grants_healthy(state_dir, monkeypatch):
    events = [_event("epm:progress v1", 3600, note="[long-phase-heartbeat] cell 2 in progress")]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, reason = tick_triage.triage(106, "issue")
    assert verdict == "HEALTHY", reason
    assert "heartbeat" in reason


def test_heartbeat_older_than_window_redrives(state_dir, monkeypatch):
    events = [_event("epm:progress v1", 6000, note="[long-phase-heartbeat] cell 2 in progress")]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, _ = tick_triage.triage(107, "issue")
    assert verdict == "STALE-REDRIVE"


def test_watcher_sentinel_heartbeat_ignored(state_dir, monkeypatch):
    events = [
        _event("epm:progress v1", 7200, note="plain progress"),
        _event(
            "epm:progress v2",
            600,
            note="[autonomous_session_watch:session-stalled-alert] [long-phase-heartbeat] x",
        ),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, _ = tick_triage.triage(108, "issue")
    assert verdict == "STALE-REDRIVE"


def test_heartbeat_future_ts_not_fresh():
    # Clock-skew guard parity with the watcher: a future ts is NOT fresh.
    events = [_event("epm:progress v1", -600, note="[long-phase-heartbeat] future")]
    assert tick_triage.issue_liveness_reason(events, NOW, 1500.0) is None


def test_log_mtime_fresh_grants_healthy(state_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: False)
    log = tmp_path / "i931.log"
    log.write_text("tick\n")  # mtime = now -> fresh
    _patch_issue_state(monkeypatch, "followups_running", [_crumb(7200, pid=999, log=str(log))])
    verdict, reason = tick_triage.triage(109, "issue")
    assert verdict == "HEALTHY", reason
    assert str(tmp_path) not in reason, "log paths are read internally, never printed"


def test_log_mtime_stale_redrives(state_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: False)
    log = tmp_path / "i931.log"
    log.write_text("tick\n")
    os.utime(log, times=(NOW - 7200, NOW - 7200))
    _patch_issue_state(monkeypatch, "followups_running", [_crumb(7200, pid=999, log=str(log))])
    verdict, _ = tick_triage.triage(110, "issue")
    assert verdict == "STALE-REDRIVE"


def test_probe_exception_falls_through_to_stale(state_dir, monkeypatch):
    def boom(_events):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(tick_triage, "newest_inflight_pid_breadcrumb", boom)
    _patch_issue_state(monkeypatch, "followups_running", [_crumb(7200, pid=999)])
    verdict, _ = tick_triage.triage(111, "issue")
    assert verdict == "STALE-REDRIVE"


def test_liveness_disabled_by_env(state_dir, monkeypatch):
    monkeypatch.setenv("EPM_TICK_LIVENESS_PROBE", "0")
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    _patch_issue_state(monkeypatch, "followups_running", [_crumb(7200, pid=999)])
    verdict, _ = tick_triage.triage(112, "issue")
    assert verdict == "STALE-REDRIVE"


def test_campaign_mode_never_probes(state_dir, monkeypatch):
    calls: list = []

    def recording_probe(pid, ts):
        calls.append(pid)
        return True

    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", recording_probe)
    events = [_event("epm:campaign-progress v1", 7200), _crumb(3600, pid=999)]
    monkeypatch.setattr(tick_triage, "load_task_state", lambda _n: ("running", events))
    monkeypatch.setattr(tick_triage, "load_children", lambda _n: [])
    monkeypatch.setattr(tick_triage, "load_campaign_state", lambda _n: {})
    verdict, _ = tick_triage.triage(113, "campaign")
    assert verdict == "STALE-REDRIVE"
    assert calls == [], "the liveness probe is issue-mode only"


def test_liveness_reason_carries_no_task_text(state_dir, monkeypatch):
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    events = [_crumb(7200, pid=999, log="/tmp/SECRETPATH.log", extra="label=SECRETLABEL")]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, reason = tick_triage.triage(114, "issue")
    assert verdict == "HEALTHY"
    assert "SECRET" not in f"{verdict} {reason}", reason


def test_heartbeat_constants_match_watcher():
    """Drift pin (text-level, no import — the watcher drags heavy deps): the
    prefix literal + the 90-min default + the shared env knob match, AND the
    heartbeat prefix is not a substring of any watcher-posted sentinel (nor
    does it contain tick's own watcher-sentinel exclusion substring — either
    would make the heartbeat leg exclude itself)."""
    src = (SCRIPTS / "autonomous_session_watch.py").read_text()
    m = re.search(r'_LONG_PHASE_HEARTBEAT_PREFIX\s*=\s*"([^"]+)"', src)
    assert m is not None and m.group(1) == tick_triage.LONG_PHASE_HEARTBEAT_PREFIX
    m = re.search(r"LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT\s*=\s*(\d+)\s*\*\s*60\b", src)
    assert m is not None
    assert float(m.group(1)) == tick_triage.LONG_PHASE_HEARTBEAT_FRESH_MIN_DEFAULT
    assert "EPM_LONG_PHASE_HEARTBEAT_FRESH_MIN" in src
    sentinels = re.findall(r'_[A-Z_]+_NOTE_SENTINEL\s*=\s*\(?\s*"(\[[^"]+\])"', src)
    assert sentinels, "watcher sentinel extraction regex found nothing — source drifted"
    for sentinel in sentinels:
        assert tick_triage.LONG_PHASE_HEARTBEAT_PREFIX not in sentinel, sentinel
        assert sentinel not in tick_triage.LONG_PHASE_HEARTBEAT_PREFIX, sentinel
    assert tick_triage._WATCHER_NOTE_SENTINEL not in tick_triage.LONG_PHASE_HEARTBEAT_PREFIX


def test_cleared_crumb_with_older_heartbeat_redrives(state_dir, monkeypatch):
    """Interaction pin (v2 leg precedence): a completed phase is never
    resurrected by its own stale heartbeat — a fresh (<90 min) heartbeat
    OLDER than the clearing event is invalidated."""
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    events = [
        _crumb(7200, pid=999),
        _event("epm:progress v2", 3600, note="[long-phase-heartbeat] cell 3"),
        _event("epm:upload-verification v1", 3000),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, _ = tick_triage.triage(115, "issue")
    assert verdict == "STALE-REDRIVE"
    # Variant: a heartbeat NEWER than the clearing event (a new
    # post-completion long phase legitimately re-attests), no in-flight pid
    # crumb -> HEALTHY via the fallback leg.
    events = [
        _event("epm:upload-verification v1", 3000),
        _event("epm:progress v3", 1800, note="[long-phase-heartbeat] post-completion phase"),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, reason = tick_triage.triage(115, "issue")
    assert verdict == "HEALTHY", reason
    assert "heartbeat" in reason


def test_dead_pid_with_fresh_heartbeat_redrives(state_dir, monkeypatch):
    """Interaction pin (v2 leg precedence): pid evidence is authoritative —
    a fresh heartbeat cannot rescue a dead detached phase (pins the 'first
    tick after the pid dies' property)."""
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: False)
    events = [
        _crumb(7200, pid=999),  # in-flight, <=48h, no log=
        # 30 min: past the 25-min stale window (base verdict STALE-REDRIVE)
        # yet well inside the 90-min heartbeat window — the heartbeat WOULD
        # rescue if the pid leg were not authoritative.
        _event("epm:progress v2", 1800, note="[long-phase-heartbeat] still going"),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, _ = tick_triage.triage(116, "issue")
    assert verdict == "STALE-REDRIVE"


def test_stale_heartbeat_with_live_pid_healthy(state_dir, monkeypatch):
    # A >90-min heartbeat neither grants nor blocks: the authoritative pid
    # leg decides.
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    events = [
        _crumb(7200, pid=4025577),
        _event("epm:progress v2", 6000, note="[long-phase-heartbeat] old heartbeat"),
    ]
    _patch_issue_state(monkeypatch, "followups_running", events)
    verdict, reason = tick_triage.triage(117, "issue")
    assert verdict == "HEALTHY", reason
    assert "4025577" in reason


# Verbatim #931 production breadcrumb note (events.jsonl 2026-07-04T20:24:12Z
# stage-dispatch row, copied as a string literal — not read from live task
# state, which moves).
_REAL_931_BREADCRUMB_NOTE = (
    "stage-dispatch stage=followup-running round=1 subagent=detached-vm-phase "
    "label=author-blocked-folds "
    "worktree=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-931 "
    "pid=4025577 "
    "log=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-931/logs/"
    "issue931_author_blocked_folds_production.log "
    "choom=ok external-markers triaged: 1 applied / 0 deferred (see epm:progress v39 "
    "pre-launch note; compute-character statement there too). Production command: "
    "uv run python scripts/issue931_author_blocked_folds.py (defaults: 20 nulls, 1000 boot, "
    "20 pseudo-draws, budget 4.5h) at branch b73c8f8484, env OMP/MKL/OPENBLAS/NUMEXPR=2, "
    "setsid-detached. Projected 4.6-5.2h wall; per-cell fingerprinted checkpoint/resume; "
    "success = [phase=done] + eval_results/issue_931/author_blocked_folds.json with "
    "registered_read.decision_row."
)


def test_breadcrumb_parse_on_real_931_note():
    """Parse-fidelity pin: _breadcrumb_fields behavior on the real #931 note
    shape (free text + tokens like 'OMP/MKL/OPENBLAS/NUMEXPR=2' and
    '[phase=done]' after the key=value block)."""
    events = [{"kind": "epm:progress", "ts": _iso(NOW - 7200), "note": _REAL_931_BREADCRUMB_NOTE}]
    crumb = tick_triage.newest_inflight_pid_breadcrumb(events)
    assert crumb is not None
    assert crumb["pid"] == 4025577
    assert crumb["log"] == (
        "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-931/logs/"
        "issue931_author_blocked_folds_production.log"
    )


# ── human-activity screen (#1629) ───────────────────────────────────────────
# Transcript fixtures use the millisecond ISO timestamp form real transcripts
# carry (`2026-07-23T06:27:15.951Z`; parse_event_ts handles it via
# fromisoformat after the Z replace).


def _iso_ms(epoch: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _transcript_row(kind: str, age_s: float, text: str = "hello") -> dict:
    """A transcript JSONL row in one of the measured §4.2 shapes:
    ``human`` / ``cron`` / ``meta`` / ``tool_result`` / ``interrupt``."""
    base: dict = {
        "type": "user",
        "isMeta": None,
        "userType": "external",  # measured NON-discriminator: external on human AND cron
        "timestamp": _iso_ms(NOW - age_s),
    }
    if kind == "human":
        base["message"] = {"role": "user", "content": text}
    elif kind == "cron":
        base["message"] = {
            "role": "user",
            "content": (
                "<command-message>issue-tick is running…</command-message>\n"
                "<command-name>/issue-tick</command-name>\n<command-args>1629</command-args>"
            ),
        }
    elif kind == "meta":
        base["isMeta"] = True
        base["message"] = {
            "role": "user",
            "content": [{"type": "text", "text": f"Base directory for this skill: {text}"}],
        }
    elif kind == "tool_result":
        base["message"] = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_x", "content": text}],
        }
    elif kind == "interrupt":
        base["message"] = {
            "role": "user",
            "content": [{"type": "text", "text": "[Request interrupted by user]"}],
        }
    elif kind == "notification":
        # Agent-tool completion notification / spawn brief (r2 Must-Fix):
        # a plain-STRING user row starting with the literal
        # <task-notification> — the dominant string-row class in autonomous
        # transcripts (149/149 measured across two transcripts).
        base["message"] = {
            "role": "user",
            "content": (
                f"<task-notification>Background task bash_1 completed</task-notification>\n{text}"
            ),
        }
    else:
        raise ValueError(kind)
    return base


def _write_transcript(tmp_path: Path, rows: list, name: str = "transcript.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_human_activity_suppresses_stale_redrive(state_dir, monkeypatch, tmp_path):
    """Durability pin (#1629): ACTIVE status + stale marker + a fresh human
    transcript row => HEALTHY with the `human-active` reason prefix."""
    path = _write_transcript(tmp_path, [_transcript_row("human", 900)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    verdict, reason = tick_triage.triage(1629, "issue")
    assert verdict == "HEALTHY", reason
    assert "human-active" in reason


@pytest.mark.parametrize("status", ["planning", "followups_running"])
def test_human_activity_suppresses_at_park_status(state_dir, monkeypatch, tmp_path, status):
    path = _write_transcript(tmp_path, [_transcript_row("human", 900)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, status, [_event("epm:progress v1", 3600)])
    verdict, reason = tick_triage.triage(1629, "issue")
    assert verdict == "HEALTHY", reason
    assert "human-active" in reason


def test_cron_prompt_only_transcript_does_not_suppress(state_dir, monkeypatch, tmp_path):
    """Fresh cron-injected slash-command, skill-load meta, and tool_result
    rows are ALL automation — a transcript with only those keeps the
    STALE-REDRIVE (the incident's cron row must never read as human)."""
    rows = [
        _transcript_row("cron", 900),
        _transcript_row("meta", 890),
        _transcript_row("tool_result", 880),
    ]
    path = _write_transcript(tmp_path, rows)
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    verdict, _ = tick_triage.triage(1629, "issue")
    assert verdict == "STALE-REDRIVE"


def test_agent_notification_row_does_not_suppress(state_dir, monkeypatch, tmp_path):
    """r2 Must-Fix pin (fresh-ts fixture): a FRESH Agent-tool
    `<task-notification>` plain-string user row — the near-continuous
    row class during agent-heavy autonomous work — classifies automation
    and never suppresses the STALE-REDRIVE."""
    path = _write_transcript(tmp_path, [_transcript_row("notification", 60)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    verdict, _ = tick_triage.triage(1629, "issue")
    assert verdict == "STALE-REDRIVE"


def test_debug_error_rung_on_raising_probe(state_dir, monkeypatch, capsys):
    """r2 Minor: with debug ON, a raising probe emits an `error=<ExcType>`
    rung (type name only — content invariant #1000) and still returns
    None (fail toward ticking)."""
    monkeypatch.setenv("EPM_TICK_HUMAN_PROBE_DEBUG", "1")

    def boom():
        raise RuntimeError("resolution blew up")

    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", boom)
    assert tick_triage.human_activity_reason(NOW) is None
    err = capsys.readouterr().err
    assert "[human-probe] error=RuntimeError" in err
    assert "resolution blew up" not in err  # exception TYPE only, never the message


def test_resolution_failure_fails_toward_ticking(state_dir, monkeypatch, capsys):
    """Resolution returning None AND raising both fall through to
    STALE-REDRIVE; the exit-code contract is intact (main() returns 0)."""
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    # (a) resolution miss (the autouse default: _own_session_transcript_path -> None)
    verdict, _ = tick_triage.triage(201, "issue")
    assert verdict == "STALE-REDRIVE"

    # (b) resolution RAISES -> the blanket guard eats it -> still STALE-REDRIVE
    def boom():
        raise RuntimeError("resolution blew up")

    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", boom)
    rc = tick_triage.main(["201"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out.startswith("STALE-REDRIVE")


def test_kill_switch_disables_human_probe(state_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("EPM_TICK_HUMAN_ACTIVE_PROBE", "0")
    path = _write_transcript(tmp_path, [_transcript_row("human", 60)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    verdict, _ = tick_triage.triage(1629, "issue")
    assert verdict == "STALE-REDRIVE"


def test_window_boundary(monkeypatch, tmp_path):
    """Per-row window test: age just over the window never suppresses, just
    under does, and a future-dated (clock-skewed) row never suppresses."""
    window = tick_triage.human_active_s()
    over = _write_transcript(tmp_path, [_transcript_row("human", window + 60)], name="over.jsonl")
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: over)
    assert tick_triage.human_activity_reason(NOW) is None
    under = _write_transcript(tmp_path, [_transcript_row("human", window - 60)], name="under.jsonl")
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: under)
    reason = tick_triage.human_activity_reason(NOW)
    assert reason is not None and "human-active" in reason
    future = _write_transcript(tmp_path, [_transcript_row("human", -120)], name="future.jsonl")
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: future)
    assert tick_triage.human_activity_reason(NOW) is None


def test_mixed_clock_skew_rows(monkeypatch, tmp_path):
    """Any-row-in-window semantics pin: a future-dated human row (last in the
    file — the position newest-row-wins would latch) can never mask a valid
    fresh row earlier in the tail."""
    rows = [_transcript_row("human", 300), _transcript_row("human", -600)]
    path = _write_transcript(tmp_path, rows)
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    reason = tick_triage.human_activity_reason(NOW)
    assert reason is not None and "human-active" in reason


def test_interrupt_row_counts_as_human(state_dir, monkeypatch, tmp_path):
    """The `[Request interrupted by user]` list-text row is a direct human
    action (the incident's 06:27:17 interrupt row) => suppression."""
    path = _write_transcript(tmp_path, [_transcript_row("interrupt", 300)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    verdict, reason = tick_triage.triage(1629, "issue")
    assert verdict == "HEALTHY", reason
    assert "human-active" in reason


def test_terminal_and_gate_verdicts_unchanged_when_human_active(state_dir, monkeypatch, tmp_path):
    """Teardown verdicts are deliberately NOT suppressed — the teardown is
    what stops future interruptions (pro-human)."""
    path = _write_transcript(tmp_path, [_transcript_row("human", 60)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "awaiting_promotion", [_event("epm:progress v1", 60)])
    verdict, _ = tick_triage.triage(301, "issue")
    assert verdict == "GATE-TRANSITION", "first gate crossing fires the push branch"
    verdict, _ = tick_triage.triage(301, "issue")
    assert verdict == "TERMINAL"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_transcript_row("human", 60), True),
        (_transcript_row("human", 60, text="what does this even mean"), True),
        (_transcript_row("interrupt", 60), True),
        (_transcript_row("cron", 60), False),
        (_transcript_row("meta", 60), False),
        (_transcript_row("tool_result", 60), False),
        ("not a dict", False),
        (42, False),
        (None, False),
        ({"type": "assistant", "message": {"role": "assistant", "content": "hi"}}, False),
        ({"type": "user"}, False),  # missing message
        ({"type": "user", "message": "not a dict"}, False),
        ({"type": "user", "message": {"role": "assistant", "content": "hi"}}, False),
        ({"type": "user", "message": {"role": "user", "content": ""}}, False),
        ({"type": "user", "message": {"role": "user", "content": "   "}}, False),
        ({"type": "user", "message": {"role": "user", "content": None}}, False),
        (
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": (
                        "<command-message>x</command-message>\n<command-name>/x</command-name>"
                    ),
                },
            },
            False,
        ),
        (
            {"type": "user", "message": {"role": "user", "content": "<local-command-stdout>x"}},
            False,
        ),
        (  # r2 Must-Fix pin: Agent-tool <task-notification> string rows are automation
            _transcript_row("notification", 60),
            False,
        ),
        (
            {
                "type": "user",
                "message": {"role": "user", "content": "<task-notification>Agent brief…"},
            },
            False,
        ),
        (
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "hello"}},
            False,
        ),
        (  # a plain list text block is NOT the interrupt row => automation
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": "plain text"}]},
            },
            False,
        ),
    ],
)
def test_is_human_transcript_row_classifier(row, expected):
    assert tick_triage.is_human_transcript_row(row) is expected


def test_human_reason_free_of_transcript_text(state_dir, monkeypatch, tmp_path):
    """Content invariant #1000: the verdict reason carries ages only — never
    the human message text (the reason line prints into an LLM tick turn)."""
    secret = "SECRET-HUMAN-MESSAGE-TEXT-xyzzy"
    path = _write_transcript(tmp_path, [_transcript_row("human", 300, text=secret)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 3600)])
    verdict, reason = tick_triage.triage(401, "issue")
    assert verdict == "HEALTHY" and "human-active" in reason
    assert secret not in reason


def test_tail_read_bounded_finds_recent_rows(monkeypatch, tmp_path):
    """Documents the 256 KB bound + the seek-from-END shape (#1287): a human
    row at EOF of an over-bound transcript IS seen; a human row only BEFORE
    the tail boundary (with an automation tail after it) is NOT."""
    filler = [_transcript_row("tool_result", 400, text="x" * 2048) for _ in range(140)]
    at_eof = _write_transcript(tmp_path, [*filler, _transcript_row("human", 300)], name="a.jsonl")
    assert os.stat(at_eof).st_size > tick_triage.TRANSCRIPT_TAIL_BYTES
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: at_eof)
    reason = tick_triage.human_activity_reason(NOW)
    assert reason is not None and "human-active" in reason
    before = _write_transcript(tmp_path, [_transcript_row("human", 300), *filler], name="b.jsonl")
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: before)
    assert tick_triage.human_activity_reason(NOW) is None


def test_liveness_screen_takes_precedence(state_dir, monkeypatch, tmp_path):
    """Ordering pin: the #1051 liveness screen runs FIRST — when both screens
    have fresh evidence the reason is the liveness one (byte-preserving the
    existing behavior; the human screen fires only on a liveness miss)."""
    monkeypatch.setattr(tick_triage, "pid_alive_with_identity", lambda *_a: True)
    path = _write_transcript(tmp_path, [_transcript_row("human", 300)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_crumb(7200, pid=4242)])
    verdict, reason = tick_triage.triage(501, "issue")
    assert verdict == "HEALTHY"
    assert "detached phase alive" in reason
    assert "human-active" not in reason


def test_campaign_kind_skips_human_probe(state_dir, monkeypatch, tmp_path):
    """Scope pin: the campaign branch never consults the human-activity
    screen (mirrors the #1051 campaign exclusion)."""
    path = _write_transcript(tmp_path, [_transcript_row("human", 300)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    monkeypatch.setattr(tick_triage, "load_campaign_state", lambda _n: {})
    monkeypatch.setattr(tick_triage, "load_children", lambda _n: [])
    _patch_issue_state(monkeypatch, "running", [_event("epm:campaign-decision v1", 7200)])
    verdict, _ = tick_triage.triage(601, "campaign")
    assert verdict == "STALE-REDRIVE"


def test_size_guard_boundary(monkeypatch, tmp_path):
    """REAL _own_session_transcript_path with only the OS boundary faked:
    a log at the byte cap resolves; one byte over the cap returns None
    (both sides of EPM_TICK_HUMAN_LOG_MAX_BYTES; fail SAFE = skip)."""
    import session_resolver

    transcript = _write_transcript(tmp_path, [_transcript_row("human", 300)])
    log = tmp_path / "2026-07-23-00-00-00-pid-12345.log"
    log.write_text(json.dumps({"transcript_path": transcript}) + "\n")
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", _REAL_TRANSCRIPT_RESOLVER)
    monkeypatch.setattr(tick_triage, "_own_node_wrapper_pid", lambda: 12345)
    monkeypatch.setattr(session_resolver, "_find_happy_log_for_node", lambda pid, now=None: log)
    size = log.stat().st_size
    monkeypatch.setenv("EPM_TICK_HUMAN_LOG_MAX_BYTES", str(size))  # log <= cap => resolves
    assert tick_triage._own_session_transcript_path() == transcript
    monkeypatch.setenv("EPM_TICK_HUMAN_LOG_MAX_BYTES", str(size - 1))  # log > cap => None
    assert tick_triage._own_session_transcript_path() is None


def test_transcript_extract_then_isfile_leg(monkeypatch, tmp_path):
    """REAL _own_session_transcript_path: the extracted transcript_path is
    returned iff it exists on disk; a missing file is a resolution miss."""
    import session_resolver

    transcript = _write_transcript(tmp_path, [_transcript_row("human", 300)])
    log = tmp_path / "2026-07-23-00-00-00-pid-777.log"
    log.write_text(json.dumps({"transcript_path": transcript}) + "\n")
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", _REAL_TRANSCRIPT_RESOLVER)
    monkeypatch.setattr(tick_triage, "_own_node_wrapper_pid", lambda: 777)
    monkeypatch.setattr(session_resolver, "_find_happy_log_for_node", lambda pid, now=None: log)
    assert tick_triage._own_session_transcript_path() == transcript
    log.write_text(json.dumps({"transcript_path": str(tmp_path / "gone.jsonl")}) + "\n")
    assert tick_triage._own_session_transcript_path() is None


def test_ancestry_walk_with_faked_proc(monkeypatch):
    """REAL _own_node_wrapper_pid over a synthetic /proc chain
    (python -> uv -> bash -> claude -> node): returns the node wrapper pid;
    a chain with no claude ancestor and a chain deeper than the cap both
    return None."""
    import session_resolver

    me = os.getpid()
    ppids = {me: 100, 100: 200, 200: 300, 300: 400, 400: 1}
    comms = {me: "python3", 100: "uv", 200: "bash", 300: "claude", 400: "node"}
    monkeypatch.setattr(tick_triage, "_proc_ppid", lambda pid: ppids.get(pid))
    monkeypatch.setattr(session_resolver, "_read_proc_comm", lambda pid: comms.get(pid))
    assert tick_triage._own_node_wrapper_pid() == 400

    no_claude = {me: "python3", 100: "uv", 200: "bash", 300: "sshd", 400: "systemd"}
    monkeypatch.setattr(session_resolver, "_read_proc_comm", lambda pid: no_claude.get(pid))
    assert tick_triage._own_node_wrapper_pid() is None

    deep_ppids: dict[int, int] = {}
    deep_comms: dict[int, str] = {}
    pid = me
    for i in range(tick_triage._ANCESTRY_MAX_DEPTH + 3):
        nxt = 10_000 + i
        deep_ppids[pid] = nxt
        deep_comms[pid] = "bash"
        pid = nxt
    deep_comms[pid] = "claude"  # claude sits DEEPER than the walk cap
    deep_ppids[pid] = 99_999
    monkeypatch.setattr(tick_triage, "_proc_ppid", lambda p: deep_ppids.get(p))
    monkeypatch.setattr(session_resolver, "_read_proc_comm", lambda p: deep_comms.get(p))
    assert tick_triage._own_node_wrapper_pid() is None


def test_debug_telemetry_line(monkeypatch, tmp_path, capsys):
    """EPM_TICK_HUMAN_PROBE_DEBUG=1 emits one stderr [human-probe] line with
    path/counts/ages and NEVER transcript message text; default OFF emits
    nothing (zero production output)."""
    secret = "DEBUG-SECRET-MESSAGE-TEXT"
    path = _write_transcript(tmp_path, [_transcript_row("human", 300, text=secret)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    monkeypatch.delenv("EPM_TICK_HUMAN_PROBE_DEBUG", raising=False)
    assert tick_triage.human_activity_reason(NOW) is not None
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("EPM_TICK_HUMAN_PROBE_DEBUG", "1")
    assert tick_triage.human_activity_reason(NOW) is not None
    err = capsys.readouterr().err
    assert "[human-probe]" in err
    assert path in err and "human=1" in err and "verdict=suppress" in err
    assert secret not in err


# ── api-error-after-marker screen (#1695) ───────────────────────────────────


def _api_error_row(age_s: float, text: str = "API Error: 500") -> dict:
    """An `isApiErrorMessage: true` assistant row in the shape Claude Code
    writes to the session transcript (verified live on #1687/#1689)."""
    return {
        "type": "assistant",
        "isApiErrorMessage": True,
        "timestamp": _iso_ms(NOW - age_s),
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def test_api_error_after_marker_returns_stale_redrive(state_dir, monkeypatch, tmp_path):
    """#1695 durability pin: a fresh assistant api-error row NEWER than the
    latest marker converts a would-be HEALTHY to STALE-REDRIVE with the
    exact `api-error-after-marker` reason substring."""
    # marker at 300s ago; api-error at 60s ago (newer) — HEALTHY would fire
    # by default (marker fresh), the api-error screen must flip it.
    path = _write_transcript(tmp_path, [_api_error_row(60)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 300)])
    verdict, reason = tick_triage.triage(1695, "issue")
    assert verdict == "STALE-REDRIVE", (verdict, reason)
    assert "api-error-after-marker" in reason


def test_api_error_after_marker_falls_open_on_missing_transcript(state_dir, monkeypatch):
    """Fail-toward-today: when `_own_session_transcript_path` returns None
    the predicate returns None and the initial verdict is preserved
    (HEALTHY on a fresh marker)."""
    # The autouse fixture already patches _own_session_transcript_path to
    # return None, but be explicit for the pin.
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: None)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 300)])
    verdict, _ = tick_triage.triage(1695, "issue")
    assert verdict == "HEALTHY"


def test_api_error_older_than_marker_returns_healthy(state_dir, monkeypatch, tmp_path):
    """Strict `>`: an api-error row OLDER than the latest marker (a
    resolved incident whose next successful turn posted a fresh event) is
    NOT a wedge — HEALTHY stays HEALTHY. Also covers the equal-ts tie
    (which stays HEALTHY by design; the marker likely followed the row)."""
    # api-error at 600s ago (older than the 60s-ago marker): the incident
    # was resolved before the last marker post.
    path = _write_transcript(tmp_path, [_api_error_row(600)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 60)])
    verdict, _ = tick_triage.triage(1695, "issue")
    assert verdict == "HEALTHY"


def test_api_error_probe_kill_switch(state_dir, monkeypatch, tmp_path):
    """`EPM_TICK_API_ERROR_PROBE=0` disables the predicate — a HEALTHY tick
    with a fresh api-error row and stale marker stays HEALTHY (pre-#1695
    behavior). Confirms the kill switch is honored."""
    monkeypatch.setenv("EPM_TICK_API_ERROR_PROBE", "0")
    path = _write_transcript(tmp_path, [_api_error_row(60)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 300)])
    verdict, _ = tick_triage.triage(1695, "issue")
    assert verdict == "HEALTHY"


def test_api_error_probe_incident_1689_content_string(state_dir, monkeypatch, tmp_path):
    """The verbatim #1689 incident content string (`"API Error: Repeated
    529 Overloaded errors. The API is at capacity"`) trips the predicate.
    The strongest "the fix actually addresses the incident that motivated
    it" invariant — a regression that changes the field or shape the
    predicate reads would break here."""
    text = "API Error: Repeated 529 Overloaded errors. The API is at capacity"
    path = _write_transcript(tmp_path, [_api_error_row(60, text=text)])
    monkeypatch.setattr(tick_triage, "_own_session_transcript_path", lambda: path)
    _patch_issue_state(monkeypatch, "running", [_event("epm:progress v1", 300)])
    verdict, reason = tick_triage.triage(1689, "issue")
    assert verdict == "STALE-REDRIVE", (verdict, reason)
    assert "api-error-after-marker" in reason
