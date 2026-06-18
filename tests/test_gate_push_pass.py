"""Tests for the watcher's gate-push + title-reconcile + tick-runaway pass.

What this pins (anti-stall redesign change 2 + the §4 runaway parachute):

1. **decide_gate_push** — fires exactly once per transition INTO a user gate
   (awaiting_promotion / blocked / over-cap plan_pending); steady state and
   non-gate transitions never push; an unknown previous status counts as a
   transition (duplicate-beats-missed).
2. **_telegram_push** — best-effort: missing/failing script logs loud and
   returns False, never raises; dry-run never executes.
3. **Runaway force-stop guards** — no-live-session clears the flag (daemon up
   only); non-DONE statuses alert-only; follow-up / pod / keep-running skips;
   a guarded stop removes the flag only on full ACK.
4. **Gate-notify state roundtrip** — atomic save/load of the transition key.
5. **Campaign coverage** — campaign-<N>.json registrations are gate-push
   candidates (same transition dedup, same guard posture: push-only, never
   stop), and main()'s pre-campaign_pass snapshot keeps the `blocked` push
   alive even after the campaign GC reaps the registration on the same tick
   (`blocked` IS campaign-terminal).
6. **Issue snapshot** — the sibling race on the issue side: main()'s
   pre-respawn-pass snapshot keeps the `awaiting_promotion` push alive even
   after _process_entry reaps issue-<N>.json on the same daemon-up tick
   (`awaiting_promotion` IS respawn-terminal).
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402


@pytest.fixture
def reg_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


# ── decide_gate_push ────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(asw.GATE_PUSH_STATUSES))
def test_push_on_transition_into_gate(status):
    assert asw.decide_gate_push(status, "running", False)


@pytest.mark.parametrize("status", sorted(asw.GATE_PUSH_STATUSES))
def test_no_push_at_steady_state(status):
    assert not asw.decide_gate_push(status, status, False)


def test_push_when_previous_status_unknown():
    # Never-observed issue parked at a gate: duplicate-beats-missed.
    assert asw.decide_gate_push("awaiting_promotion", None, False)


@pytest.mark.parametrize("status", ["running", "planning", "completed", "archived", "approved"])
def test_no_push_on_non_gate_transition(status):
    assert not asw.decide_gate_push(status, "running", False)


def test_plan_pending_pushes_only_over_cap():
    assert asw.decide_gate_push("plan_pending", "planning", True)
    assert not asw.decide_gate_push("plan_pending", "planning", False)
    assert not asw.decide_gate_push("plan_pending", "plan_pending", True)


def test_no_push_on_unreadable_status():
    assert not asw.decide_gate_push(None, "running", False)


# ── _telegram_push ──────────────────────────────────────────────────────────


def _make_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "push.sh"
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_telegram_push_success(tmp_path, monkeypatch):
    script = _make_script(tmp_path, "echo sent.; exit 0")
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert asw._telegram_push("hello", dry_run=False)


def test_telegram_push_missing_script_is_loud_false(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(tmp_path / "absent.sh"))
    assert not asw._telegram_push("hello", dry_run=False)
    assert "missing" in capsys.readouterr().err


def test_telegram_push_failing_script_is_loud_false(tmp_path, monkeypatch, capsys):
    script = _make_script(tmp_path, "echo nope >&2; exit 1")
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert not asw._telegram_push("hello", dry_run=False)
    assert "failed" in capsys.readouterr().err


def test_telegram_push_dry_run_never_executes(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    script = _make_script(tmp_path, f"touch {marker}; exit 0")
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert not asw._telegram_push("hello", dry_run=True)
    assert not marker.exists()


# ── gate-notify state roundtrip ─────────────────────────────────────────────


def test_gate_notify_state_roundtrip(reg_dir):
    assert asw._load_gate_notify_state(5) == {}
    asw._save_gate_notify_state(5, last_status="awaiting_promotion")
    state = asw._load_gate_notify_state(5)
    assert state["last_status"] == "awaiting_promotion"
    # Garbled file degrades to {} (fresh start), never raises.
    asw._gate_notify_state_path(5).write_text("{nope")
    assert asw._load_gate_notify_state(5) == {}


# ── runaway flags ───────────────────────────────────────────────────────────


def _flag(reg_dir: Path, issue: int) -> Path:
    path = reg_dir / f"tick-runaway-{issue}.flag"
    path.write_text(json.dumps({"issue": issue, "terminal_streak": 3}))
    return path


def test_runaway_flags_enumeration(reg_dir):
    _flag(reg_dir, 42)
    _flag(reg_dir, 7)
    (reg_dir / "tick-runaway-garbled.flag").write_text("{}")
    assert [issue for issue, _ in asw._runaway_flags()] == [7, 42]


def test_runaway_no_live_session_clears_flag(reg_dir):
    flag = _flag(reg_dir, 42)
    asw._process_runaway_flag(42, flag, [], set(), daemon_reachable=True, dry_run=False)
    assert not flag.exists()


def test_runaway_no_live_session_daemon_down_keeps_flag(reg_dir):
    flag = _flag(reg_dir, 42)
    asw._process_runaway_flag(42, flag, [], set(), daemon_reachable=False, dry_run=False)
    assert flag.exists(), "liveness is unknowable during a daemon outage — keep the flag"


def test_runaway_non_done_status_alerts_only(reg_dir, monkeypatch, capsys):
    flag = _flag(reg_dir, 42)
    monkeypatch.setattr(asw, "_task_status", lambda *_a, **_k: "blocked")
    stops: list[str] = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, _d: stops.append(sid) or True)
    asw._process_runaway_flag(42, flag, ["sid-1"], set(), daemon_reachable=True, dry_run=False)
    assert flag.exists() and not stops
    assert "alert only" in capsys.readouterr().err


def test_runaway_guards_skip_followup_pod_keep_running(reg_dir, monkeypatch):
    flag = _flag(reg_dir, 42)
    monkeypatch.setattr(asw, "_task_status", lambda *_a, **_k: "awaiting_promotion")
    stops: list[str] = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, _d: stops.append(sid) or True)
    # Live follow-up → skip.
    monkeypatch.setattr(asw, "_task_followup_active", lambda *_a, **_k: True)
    asw._process_runaway_flag(42, flag, ["sid-1"], set(), daemon_reachable=True, dry_run=False)
    assert flag.exists() and not stops
    # RUNNING pod → skip.
    monkeypatch.setattr(asw, "_task_followup_active", lambda *_a, **_k: False)
    asw._process_runaway_flag(42, flag, ["sid-1"], {42}, daemon_reachable=True, dry_run=False)
    assert flag.exists() and not stops
    # keep-running tag → skip.
    monkeypatch.setattr(asw, "_task_keep_running", lambda *_a, **_k: True)
    asw._process_runaway_flag(42, flag, ["sid-1"], set(), daemon_reachable=True, dry_run=False)
    assert flag.exists() and not stops


def test_runaway_force_stop_removes_flag_on_full_ack(reg_dir, monkeypatch):
    flag = _flag(reg_dir, 42)
    monkeypatch.setattr(asw, "_task_status", lambda *_a, **_k: "awaiting_promotion")
    monkeypatch.setattr(asw, "_task_followup_active", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_task_keep_running", lambda *_a, **_k: False)
    markers: list[str] = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda _i, note, _d, **_k: markers.append(note)
    )
    stops: list[str] = []
    monkeypatch.setattr(asw, "_stop_session", lambda sid, _d: stops.append(sid) or True)
    asw._process_runaway_flag(
        42, flag, ["sid-1", "sid-2"], set(), daemon_reachable=True, dry_run=False
    )
    assert stops == ["sid-1", "sid-2"]
    assert not flag.exists()
    assert markers and "runaway" in markers[0]


def test_runaway_partial_ack_keeps_flag(reg_dir, monkeypatch):
    flag = _flag(reg_dir, 42)
    monkeypatch.setattr(asw, "_task_status", lambda *_a, **_k: "awaiting_promotion")
    monkeypatch.setattr(asw, "_task_followup_active", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_task_keep_running", lambda *_a, **_k: False)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(asw, "_stop_session", lambda sid, _d: sid == "sid-1")
    asw._process_runaway_flag(
        42, flag, ["sid-1", "sid-2"], set(), daemon_reachable=True, dry_run=False
    )
    assert flag.exists(), "a partial ACK must keep the flag so the next pass retries"


def test_runaway_dry_run_never_unlinks(reg_dir, monkeypatch):
    flag = _flag(reg_dir, 42)
    asw._process_runaway_flag(42, flag, [], set(), daemon_reachable=True, dry_run=True)
    assert flag.exists()


# ── gate_push_pass candidate filtering ──────────────────────────────────────


def test_pass_skips_completed_and_archived(reg_dir, monkeypatch):
    """completed/archived candidates must be skipped ENTIRELY: the
    terminal-status GC reaps their gate-notify state each tick, so acting on
    them would re-create state + re-refresh the self-report every pass —
    keeping the self-report permanently fresh and structurally disabling the
    session-reconcile idle signal for done tasks."""
    (reg_dir / "issue-620.json").write_text(json.dumps({"happy_session_id": "sid-x"}))
    monkeypatch.setattr(asw, "_task_status", lambda *_a, **_k: "completed")
    refreshed: list[int] = []
    monkeypatch.setattr(
        asw, "_refresh_self_report", lambda issue, *_a, **_k: refreshed.append(issue)
    )
    monkeypatch.setattr(asw, "_task_events", lambda *_a, **_k: [])
    asw.gate_push_pass(False, daemon_reachable=False)
    assert not refreshed
    assert not asw._gate_notify_state_path(620).exists()


# ── campaign coverage ───────────────────────────────────────────────────────


def _campaign_entry(reg_dir: Path, issue: int, sid: str = "sid-c") -> Path:
    path = reg_dir / f"campaign-{issue}.json"
    path.write_text(json.dumps({"issue": issue, "happy_session_id": sid}))
    return path


def test_campaign_gate_candidates_enumeration(reg_dir):
    _campaign_entry(reg_dir, 590)
    # Watch-state files match the campaign-*.json glob but are NOT
    # registrations (stem fails the int parse / no issue key).
    (reg_dir / "campaign-watch-590.json").write_text(json.dumps({"stalled_checks": 0}))
    (reg_dir / "campaign-garbled.json").write_text("{nope")
    # Issue registrations are a separate candidate source, not this one's.
    (reg_dir / "issue-620.json").write_text(json.dumps({"happy_session_id": "sid-x"}))
    assert asw._campaign_gate_candidates() == {590}


def _run_pass_recording_pushes(monkeypatch, status: str, **kwargs) -> list[str]:
    """gate_push_pass with the task/push surfaces stubbed; returns push msgs."""
    monkeypatch.setattr(asw, "_task_status", lambda *_a, **_k: status)
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda *_a, **_k: [{"kind": "epm:failure v1", "note": "budget exhausted"}],
    )
    monkeypatch.setattr(asw, "_task_title", lambda *_a, **_k: "")
    monkeypatch.setattr(asw, "_refresh_self_report", lambda *_a, **_k: None)
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, _d: pushes.append(msg) or True)
    asw.gate_push_pass(False, daemon_reachable=False, **kwargs)
    return pushes


def test_campaign_blocked_transition_pushes_once(reg_dir, monkeypatch):
    _campaign_entry(reg_dir, 590)
    pushes = _run_pass_recording_pushes(monkeypatch, "blocked")
    assert len(pushes) == 1 and "BLOCKED" in pushes[0] and "#590" in pushes[0]
    assert asw._load_gate_notify_state(590)["last_status"] == "blocked"
    # Steady state on the next tick: no second push (transition dedup).
    assert _run_pass_recording_pushes(monkeypatch, "blocked") == []


def test_campaign_snapshot_param_survives_same_tick_reap(reg_dir, monkeypatch):
    # The race the snapshot exists for: campaign_pass already stop-then-reaped
    # the blocked campaign's registration before gate_push_pass ran. main()'s
    # pre-campaign_pass snapshot (campaign_issues=...) must still push.
    assert not list(reg_dir.glob("campaign-*.json"))
    pushes = _run_pass_recording_pushes(monkeypatch, "blocked", campaign_issues={590})
    assert len(pushes) == 1 and "BLOCKED" in pushes[0] and "#590" in pushes[0]


def test_campaign_active_status_never_pushes(reg_dir, monkeypatch):
    # `running` is the held status for the whole campaign — no push, but the
    # transition key is recorded so the eventual blocked push has a baseline.
    _campaign_entry(reg_dir, 590)
    assert _run_pass_recording_pushes(monkeypatch, "running") == []
    assert asw._load_gate_notify_state(590)["last_status"] == "running"


def test_campaign_completed_skipped_entirely(reg_dir, monkeypatch):
    # Same posture as issues: completed/archived are never push targets and
    # acting on them would churn against the terminal-status GC.
    _campaign_entry(reg_dir, 590)
    assert _run_pass_recording_pushes(monkeypatch, "completed") == []
    assert not asw._gate_notify_state_path(590).exists()


def test_main_snapshots_campaign_candidates_before_campaign_pass():
    """Source pin on main()'s wiring (reviewer minor, 2026-06-12): the
    campaign snapshot must be taken BEFORE campaign_pass (whose terminal GC
    reaps a blocked campaign's registration on the same tick) and handed to
    gate_push_pass via the kwarg — gate_push_pass's None fallback would mask
    a dropped kwarg / reordered snapshot with green behavior tests."""
    import inspect

    src = inspect.getsource(asw.main)
    assert "campaign_issues=campaign_gate_candidates" in src
    assert src.index("_campaign_gate_candidates()") < src.index("campaign_pass(")


# ── issue-side snapshot (sibling of the campaign one) ───────────────────────


def test_issue_snapshot_param_survives_same_tick_reap(reg_dir, monkeypatch):
    # The race the snapshot exists for: the respawn pass already reaped the
    # parked issue's registration (`awaiting_promotion` IS respawn-terminal,
    # so _process_entry deletes issue-<N>.json on the first daemon-up tick
    # observing the park) before gate_push_pass ran. main()'s pre-respawn
    # snapshot (issue_snapshot=...) must still push.
    assert not list(reg_dir.glob("issue-*.json"))
    pushes = _run_pass_recording_pushes(monkeypatch, "awaiting_promotion", issue_snapshot={620})
    assert len(pushes) == 1 and "promote" in pushes[0] and "#620" in pushes[0]
    # Steady state on the next tick: no second push (transition dedup).
    assert _run_pass_recording_pushes(monkeypatch, "awaiting_promotion", issue_snapshot={620}) == []


def test_main_snapshots_issue_candidates_before_respawn_pass():
    """Source pin on main()'s wiring, mirroring the campaign pin above: the
    issue snapshot must be taken BEFORE the respawn pass (whose
    _process_entry reaps a parked issue's registration on the same daemon-up
    tick) and handed to gate_push_pass via the kwarg — gate_push_pass's None
    fallback would mask a dropped kwarg / reordered snapshot with green
    behavior tests."""
    import inspect

    src = inspect.getsource(asw.main)
    assert "issue_snapshot=issue_gate_candidates" in src
    assert src.index("set(_issue_registrations())") < src.index("_process_entry(")


# ── GC integration ──────────────────────────────────────────────────────────


def test_gate_notify_state_is_in_gc_sweep_set():
    assert ("gate-notify-", "") in asw._GC_TARGETS, (
        "gate-notify state files must be reaped at completed/archived by the terminal-status GC"
    )


# ── push message shape ──────────────────────────────────────────────────────


def test_gate_push_message_no_double_space_on_missing_title(monkeypatch):
    # Review minor (2026-06-12): a failed title read must not produce
    # "#42  · ..." (double space) in the phone push.
    monkeypatch.setattr(asw, "_task_title", lambda *_a, **_k: "")
    msg = asw._gate_push_message(42, "awaiting_promotion", [], False)
    assert msg.startswith("#42 ·") and "  " not in msg
