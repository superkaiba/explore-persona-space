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


# ── compute_issue_verdict ───────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_ACTIVE | tick_triage.ISSUE_PARK))
def test_issue_fresh_marker_is_healthy(status):
    verdict, _ = tick_triage.compute_issue_verdict(status, status, 60.0, False, stale_after_s=1500)
    assert verdict == "HEALTHY"


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_ACTIVE | tick_triage.ISSUE_PARK))
def test_issue_stale_marker_is_redrive(status):
    verdict, _ = tick_triage.compute_issue_verdict(
        status, status, 3600.0, False, stale_after_s=1500
    )
    assert verdict == "STALE-REDRIVE"


def test_issue_no_markers_counts_as_stale():
    verdict, _ = tick_triage.compute_issue_verdict(
        "running", "running", None, False, stale_after_s=1500
    )
    assert verdict == "STALE-REDRIVE"


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_TERMINAL))
def test_issue_steady_terminal_is_terminal(status):
    verdict, _ = tick_triage.compute_issue_verdict(status, status, 60.0, False, stale_after_s=1500)
    assert verdict == "TERMINAL"


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_GATE))
def test_issue_gate_transition_fires_on_status_change(status):
    verdict, _ = tick_triage.compute_issue_verdict(
        status, "running", 60.0, False, stale_after_s=1500
    )
    assert verdict == "GATE-TRANSITION"


def test_issue_gate_transition_on_missing_snapshot():
    # Previous status unknown + currently at a gate: fire the transition
    # branch (a duplicate push beats a missed one — the tick skill's rule).
    verdict, _ = tick_triage.compute_issue_verdict(
        "awaiting_promotion", None, 60.0, False, stale_after_s=1500
    )
    assert verdict == "GATE-TRANSITION"


def test_issue_completed_transition_is_plain_terminal():
    # completed/archived are terminal but NOT user gates — no push branch.
    verdict, _ = tick_triage.compute_issue_verdict(
        "completed", "reviewing", 60.0, False, stale_after_s=1500
    )
    assert verdict == "TERMINAL"


def test_issue_plan_pending_over_cap_is_gate():
    verdict, _ = tick_triage.compute_issue_verdict(
        "plan_pending", "planning", 60.0, True, stale_after_s=1500
    )
    assert verdict == "GATE-TRANSITION"
    verdict, _ = tick_triage.compute_issue_verdict(
        "plan_pending", "plan_pending", 60.0, True, stale_after_s=1500
    )
    assert verdict == "TERMINAL"


def test_issue_plan_pending_under_cap_is_park():
    verdict, _ = tick_triage.compute_issue_verdict(
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
