"""Tests for the alive-but-stalled detector + generalized GC of per-issue
``~/.eps-autonomous/`` state files.

Two failure modes the prior watcher missed and Piece 2b/3b now cover:

1. **Stalled detector** — a session whose Happy id is in the live set (so the
   respawn pass leaves it alone) but whose bg-Bash chain quietly died: no
   self-report advances, no new task markers. Autonomous entries escalate to
   AUTO-RESPAWN when eligible (ACTIVE status + reachable daemon, capped);
   manual entries are ALERT-ONLY by design (#505 round-2, 2026-06-10). Pin
   the decision matrix + the 2-miss guard + the dedup-within-episode + the
   sentinel-exclusion contract.
2. **Generalized GC** — for every per-issue state-file prefix under
   ``~/.eps-autonomous/``, terminal tasks (``completed`` / ``archived``)
   must drop the state file; ``awaiting_promotion`` / ``blocked`` /
   ``running`` / etc. must KEEP it. Garbled / unresolvable / very-old
   entries follow the age backstop.
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

import spawn_session  # noqa: E402
from autonomous_session_watch import (  # noqa: E402
    MAX_ENTRY_AGE_S,
    STALLED_STATE_PREFIX,
    STALLED_WINDOW_S,
    TERMINAL_FOR_GC,
    decide_session_stalled,
)

# ─── decide_session_stalled — pure decision matrix ────────────────────────────


def test_no_self_report_keeps():
    # A MISSING self-report (None) is the caller's signal to skip — never alert.
    # Targets autonomous sessions; interactive sessions don't self-report.
    assert decide_session_stalled(
        self_report_age_s=None,
        marker_progress_age_s=None,
        has_pod=False,
        missed=0,
        alerted=False,
    ) == ("keep", 0)


def test_fresh_self_report_keeps_and_resets():
    # Any fresh signal resets the miss counter.
    assert decide_session_stalled(
        self_report_age_s=60.0,  # 1 min ago, way under window
        marker_progress_age_s=None,
        has_pod=True,
        missed=3,
        alerted=False,
    ) == ("keep", 0)


def test_fresh_marker_keeps_even_if_self_report_stale():
    # Bg chain is still posting -> not stalled, even if self-report is late.
    stale_self = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale_self,
        marker_progress_age_s=60.0,  # 1 min ago, fresh
        has_pod=False,
        missed=1,
        alerted=False,
    ) == ("keep", 0)


def test_all_signals_stale_first_miss_increments():
    # First stale check only increments; alert fires on SECOND consecutive miss
    # (default threshold 2) — guards a flaky markers-fetch.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=0,
        alerted=False,
        threshold=2,
    ) == ("keep", 1)


def test_all_signals_stale_second_miss_alerts():
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=1,
        alerted=False,
        threshold=2,
    ) == ("alert", 0)


def test_already_alerted_stays_quiet_when_respawn_not_eligible():
    # Dedup within episode: once we've alerted and respawn is NOT eligible
    # (default for this no-eligibility call), subsequent stale ticks don't
    # re-alert (caller clears `alerted` when self-report advances).
    # Escalation to a respawn from alerted is covered by
    # `test_alerted_escalates_to_respawn_when_eligible` below; this case
    # only pins the dedup-of-repeat-alerts behavior.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=5,
        alerted=True,
        # respawn_eligible defaults to False — no escalation possible.
    ) == ("keep", 0)


# ─── alerted → respawn escalation (regression for incident #506) ─────────────


def test_alerted_escalates_to_respawn_when_eligible():
    # Incident #506 (2026-06-08): a Phase-1 alert set alerted=True ~11h
    # before respawn became eligible. The prior `if alerted: return keep`
    # short-circuit then suppressed the respawn on every subsequent tick
    # for 10+ hours while an 8xH200 pod idle-burned ~$460. An already-
    # alerted episode MUST still escalate to a respawn the moment it
    # becomes eligible — the alert flag dedups REPEAT ALERTS only, never
    # the stronger respawn action. The alert already required >= threshold
    # consecutive stale checks, so escalation needn't re-accumulate.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=0,  # caller may have reset; escalation must not depend on miss count
        alerted=True,
        respawn_eligible=True,
        respawn_count=0,
        threshold=2,
    ) == ("respawn", 0)


def test_alerted_at_cap_stays_quiet_no_phantom_respawn():
    # Exhausted-cap respected from the alerted branch: if respawn_count
    # is already at the cap (i.e. the exhausted marker has been posted),
    # the new escalation path must NOT resurrect a respawn. Stay quiet —
    # the caller's `exhausted` flag handles the exhausted-marker dedup
    # separately; here we just refuse to spawn past the cap.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=0,
        alerted=True,
        respawn_eligible=True,
        respawn_count=3,  # default STALLED_MAX_RESPAWNS == 3 -> at the cap
        threshold=2,
    ) == ("keep", 0)


def test_alerted_above_cap_stays_quiet_no_phantom_respawn():
    # Defensive: if respawn_count drifts > max (cap lowered between ticks,
    # state file hand-edited), still refuse to respawn from the alerted
    # branch. Mirrors the non-alerted defensive test
    # `test_session_stalled_respawn_above_cap_returns_exhausted` in
    # test_autonomous_session_watch.py.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=0,
        alerted=True,
        respawn_eligible=True,
        respawn_count=10,  # well above the cap
        threshold=2,
    ) == ("keep", 0)


def test_alerted_eligibility_false_stays_quiet():
    # Alerted + respawn NOT eligible (non-ACTIVE status, or daemon
    # unreachable this tick) -> stay quiet. No spurious alert escalation
    # — the prior alert already deduped, and a respawn would crash on the
    # missing prerequisite. The next tick that flips eligibility back on
    # is where the escalation fires.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=0,
        alerted=True,
        respawn_eligible=False,
        respawn_count=0,
        threshold=2,
    ) == ("keep", 0)


def test_stale_self_no_marker_at_all_is_stale():
    # No marker means marker_age=None, which the decision treats as stale —
    # a pod-active autonomous session that's never posted progress IS a signal.
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=None,
        has_pod=True,
        missed=1,
        alerted=False,
        threshold=2,
    ) == ("alert", 0)


def test_no_pod_still_triggers_if_both_other_signals_stale():
    # has_pod=False does NOT save a session from alerting — signals 1+2 alone
    # are sufficient. Some autonomous sessions are not pod-driving (interp /
    # review phases reading from WandB/HF).
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=1,
        alerted=False,
        threshold=2,
    ) == ("alert", 0)


def test_threshold_one_alerts_immediately():
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=0,
        alerted=False,
        threshold=1,
    ) == ("alert", 0)


def test_higher_threshold_delays_alert():
    stale = STALLED_WINDOW_S + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=1,
        alerted=False,
        threshold=3,
    ) == ("keep", 2)
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=2,
        alerted=False,
        threshold=3,
    ) == ("alert", 0)


def test_window_boundary_exact():
    # At exactly window_s the self-report is treated as stale (>= window).
    # Just under is treated as fresh.
    assert decide_session_stalled(
        self_report_age_s=STALLED_WINDOW_S - 1,
        marker_progress_age_s=STALLED_WINDOW_S + 60,
        has_pod=False,
        missed=1,
        alerted=False,
    ) == ("keep", 0)
    assert decide_session_stalled(
        self_report_age_s=STALLED_WINDOW_S,
        marker_progress_age_s=STALLED_WINDOW_S,
        has_pod=False,
        missed=1,
        alerted=False,
    ) == ("alert", 0)


# ─── state file roundtrip ────────────────────────────────────────────────────


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point AUTONOMOUS_REGISTRY_DIR at a tmp dir in BOTH spawn_session and
    autonomous_session_watch (the import re-binds the constant), and isolate
    the #573 ALIVE-BUT-STALLED provision-in-flight probe from real VM state:
    a real ``.claude/cache/poll-pipeline-<N>.json`` (or a live ``pod.py
    provision --issue <N>`` process on this VM) would otherwise fire the
    exemption inside ``stalled_session_pass`` and swallow the miss-counter
    increments these tests assert on (3 tests flaked env-dependently on any
    VM carrying a real poll-pipeline-489.json; surfaced by task #572).
    ``_POLL_STATE_DIR`` points at a nonexistent tmp subdir so the REAL probe
    still runs and exercises its missing-file branch."""
    import autonomous_session_watch as asw

    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "_POLL_STATE_DIR", tmp_path / "poll-state")
    monkeypatch.setattr(asw, "_find_provision_process", lambda *_a, **_k: None)
    return tmp_path


def test_save_and_load_stalled_state_roundtrip(isolated_registry):
    import autonomous_session_watch as asw

    asw._save_stalled_state(
        491,
        "sess-abc",
        missed=1,
        alerted=False,
        last_self_report_ts="2026-06-05T18:22:14Z",
        prev={"first_seen": 1234.0},
    )
    state = asw._load_stalled_state(491)
    assert state["happy_session_id"] == "sess-abc"
    assert state["missed"] == 1
    assert state["alerted"] is False
    assert state["last_self_report_ts"] == "2026-06-05T18:22:14Z"
    assert state["first_seen"] == 1234.0
    # first_seen persists on a subsequent save (no fresh stamping).
    asw._save_stalled_state(
        491,
        "sess-abc",
        missed=2,
        alerted=True,
        last_self_report_ts="2026-06-05T18:30:00Z",
        prev=state,
    )
    state2 = asw._load_stalled_state(491)
    assert state2["first_seen"] == 1234.0
    assert state2["missed"] == 2
    assert state2["alerted"] is True


def test_load_stalled_state_empty_on_missing(isolated_registry):
    import autonomous_session_watch as asw

    assert asw._load_stalled_state(999) == {}


def test_load_stalled_state_empty_on_garbled_json(isolated_registry):
    import autonomous_session_watch as asw

    (isolated_registry / f"{STALLED_STATE_PREFIX}33.json").write_text("not-json{{")
    assert asw._load_stalled_state(33) == {}


def test_clear_stalled_state_removes_file(isolated_registry):
    import autonomous_session_watch as asw

    asw._save_stalled_state(
        7,
        "x",
        missed=0,
        alerted=False,
        last_self_report_ts=None,
        prev=None,
    )
    assert (isolated_registry / f"{STALLED_STATE_PREFIX}7.json").exists()
    asw._clear_stalled_state(7)
    assert not (isolated_registry / f"{STALLED_STATE_PREFIX}7.json").exists()
    # idempotent
    asw._clear_stalled_state(7)


# ─── _latest_progress_ts — sentinel exclusion ────────────────────────────────


def test_latest_progress_ts_excludes_both_watcher_sentinels():
    # A mixed events list with BOTH the pod-safety alert sentinel AND the
    # session-stalled-alert sentinel must filter both out; the newest
    # non-watcher event wins.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-06-05T10:00:00Z", "note": "step 100"},
        {
            "kind": "epm:progress",
            "ts": "2026-06-05T18:00:00Z",
            "note": f"{asw._ALERT_NOTE_SENTINEL} pod stale alert ...",
        },
        {
            "kind": "epm:progress",
            "ts": "2026-06-05T19:00:00Z",
            "note": f"{asw._STALLED_ALERT_NOTE_SENTINEL} session stalled alert ...",
        },
        {"kind": "epm:results", "ts": "2026-06-05T11:00:00Z", "note": "done"},
    ]
    ts = asw._latest_progress_ts(events)
    # Newest non-watcher event is 11:00 results; the 18:00 + 19:00 watcher
    # alerts are filtered out.
    assert ts == asw._parse_event_ts("2026-06-05T11:00:00Z")


def test_watcher_note_sentinels_contains_both():
    # Pin the frozenset membership so future watcher-posted markers are added
    # explicitly (drift-prevention).
    import autonomous_session_watch as asw

    assert asw._ALERT_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._STALLED_ALERT_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


# ─── stalled_session_pass — top-level driver ─────────────────────────────────


def _write_autonomous_entry(reg_dir, issue, session_id, *, spawned_at=None):
    """Mimic spawn_session._register_autonomous_session output shape."""
    if spawned_at is None:
        spawned_at = time.time()
    (reg_dir / f"issue-{issue}.json").write_text(
        json.dumps(
            {
                "issue": issue,
                "happy_session_id": session_id,
                "cwd": "/repo",
                "auto_approve_gpu_hours": 24.0,
                "spawned_at": spawned_at,
                "missed": 0,
            }
        )
    )


def test_stalled_pass_alerts_after_two_consecutive_stale_ticks(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    # All three signals stale (self-report + marker).
    stale_age = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale_age, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # Tick 1: increments to missed=1 (no alert yet).
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now)
    assert posts == []
    state_path = isolated_registry / f"{STALLED_STATE_PREFIX}489.json"
    assert json.loads(state_path.read_text())["missed"] == 1

    # Tick 2: alert fires, miss counter resets, alerted=True persisted.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now)
    assert posts == [(489, "session-stalled-alert")]
    state = json.loads(state_path.read_text())
    assert state["alerted"] is True
    assert state["missed"] == 0


def test_stalled_pass_dedups_within_episode(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    stale_age = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale_age, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # 3 ticks while stale -> exactly 1 alert (threshold=1 so it fires on tick 1).
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    assert posts == [(489, "session-stalled-alert")]


def test_stalled_pass_clears_alerted_when_self_report_advances(isolated_registry, monkeypatch):
    # Episode 1: alert fires while frozen at ts_a. Self-report advances to
    # ts_b -> alerted clears (session recovered). Goes stale again -> NEW
    # alert episode.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    stale_age = STALLED_WINDOW_S + 600
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # Tick 1: stale at ts_a -> alert (threshold=1).
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale_age, "2026-06-05T10:00:00Z"),
    )
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)

    # Tick 2: self-report advanced to ts_b AND is fresh -> keep, alerted clears.
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (60.0, "2026-06-05T11:00:00Z"),
    )
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)

    # Tick 3: stale again at ts_c (LATER than ts_b — self-report advanced to
    # ts_c, then froze) -> alerted is cleared from tick 2, so we re-alert.
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale_age, "2026-06-05T12:00:00Z"),
    )
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)

    assert posts == [
        (489, "session-stalled-alert"),
        (489, "session-stalled-alert"),
    ]


def test_stalled_pass_skips_when_no_self_report(isolated_registry, monkeypatch):
    # Interactive (or just-spawned) sessions have no self-report file —
    # the pass treats that as "doesn't apply" and never alerts.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    assert posts == []


def test_stalled_pass_never_respawns_or_stops(isolated_registry, monkeypatch):
    # Hard contract: the stalled pass is ALERT-ONLY this round. It MUST NOT
    # call _respawn or _stop_pod. Pin the contract.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    respawns: list = []
    stops: list = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    stale = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(asw, "_respawn", lambda entry, dry_run: respawns.append(entry) or True)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    assert respawns == []
    assert stops == []


def test_stalled_pass_manual_session_alert_only_never_respawns(isolated_registry, monkeypatch):
    # Manual sessions (``manual-issue-<N>.json``, bare ``spawn-issue``) get
    # the SAME staleness detection in ALERT-ONLY mode (#505 round-2,
    # 2026-06-10): a stalled user-driven session posts the one-time alert
    # instead of orphaning silently, but is NEVER auto-respawned —
    # restarting a session the user drives by hand is the user's call.
    import autonomous_session_watch as asw

    posts: list = []
    respawns: list = []
    stops: list = []
    (isolated_registry / "manual-issue-100.json").write_text(
        json.dumps(
            {
                "issue": 100,
                "happy_session_id": "manual-sess",
                "cwd": "/repo",
                "spawned_at": time.time(),
                "mode": "manual",
            }
        )
    )

    stale = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    # ACTIVE status + reachable daemon: respawn WOULD be eligible were the
    # entry autonomous — the alert-only routing must come from manual=True.
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_respawn", lambda entry, dry_run: respawns.append(entry) or True)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: posts.append(a))

    asw.stalled_session_pass(dry_run=False, threshold=1, daemon_reachable=True)

    assert respawns == []
    assert stops == []
    assert len(posts) == 1
    issue, note, _dry_run = posts[0]
    assert issue == 100
    assert asw._STALLED_ALERT_NOTE_SENTINEL in note
    assert "STALLED manual issue session" in note
    # Manual entries share the per-issue stalled-state file: the one-time
    # alert is recorded so the next tick dedups instead of re-alerting.
    state = json.loads((isolated_registry / f"{STALLED_STATE_PREFIX}100.json").read_text())
    assert state["alerted"] is True


def test_stalled_pass_dry_run_no_state_write(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    now = 1_000_000.0
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    stale = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.stalled_session_pass(dry_run=True, threshold=1, now=now)
    # No state file written in dry-run.
    assert not (isolated_registry / f"{STALLED_STATE_PREFIX}489.json").exists()


# ─── generalized GC — terminal-status reaping ────────────────────────────────


def _populate_gc_targets(reg_dir: Path, issue: int) -> dict[str, Path]:
    """Drop one file under each GC target prefix for ``issue`` and return the
    paths so the test can assert presence/absence."""
    paths: dict[str, Path] = {}
    # manual-issue-<N>.json — top level
    paths["manual"] = reg_dir / f"manual-issue-{issue}.json"
    paths["manual"].write_text(json.dumps({"issue": issue, "mode": "manual"}))
    # stalled-<N>.json — top level
    paths["stalled"] = reg_dir / f"{STALLED_STATE_PREFIX}{issue}.json"
    paths["stalled"].write_text(json.dumps({"issue": issue, "missed": 0}))
    # issue-progress/<N>.json — nested
    (reg_dir / "issue-progress").mkdir(exist_ok=True)
    paths["progress"] = reg_dir / "issue-progress" / f"{issue}.json"
    paths["progress"].write_text(json.dumps({"issue": issue, "text": "x"}))
    # issue-tick-last-status/<N>.json — nested (Piece 1 file)
    (reg_dir / "issue-tick-last-status").mkdir(exist_ok=True)
    paths["tick"] = reg_dir / "issue-tick-last-status" / f"{issue}.json"
    paths["tick"].write_text(json.dumps({"issue": issue, "status": "running"}))
    return paths


@pytest.mark.parametrize("status", sorted(TERMINAL_FOR_GC))
def test_gc_reaps_all_targets_for_terminal_status(isolated_registry, monkeypatch, status):
    import autonomous_session_watch as asw

    paths = _populate_gc_targets(isolated_registry, 137)
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)

    counts = asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=False)

    # All four target paths should be gone.
    for p in paths.values():
        assert not p.exists(), f"{p} should have been reaped"
    # Counts dict should account for at least one drop per prefix.
    assert sum(counts.values()) == 4


@pytest.mark.parametrize(
    "status",
    [
        "awaiting_promotion",
        "blocked",
        "running",
        "approved",
        "planning",
        "proposed",
        "interpreting",
    ],
)
def test_gc_keeps_park_and_active_status(isolated_registry, monkeypatch, status):
    import autonomous_session_watch as asw

    paths = _populate_gc_targets(isolated_registry, 137)
    monkeypatch.setattr(asw, "_task_status", lambda issue: status)

    asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=False)

    # Every target stays — conservative on park / active / awaiting_promotion.
    for p in paths.values():
        assert p.exists(), f"{p} should NOT have been reaped (status={status})"


def test_gc_dry_run_does_not_delete(isolated_registry, monkeypatch):
    import autonomous_session_watch as asw

    paths = _populate_gc_targets(isolated_registry, 137)
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")

    counts = asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=True)

    # Counts report what WOULD have been cleared, but the files still exist.
    assert sum(counts.values()) == 4
    for p in paths.values():
        assert p.exists()


def test_gc_unresolvable_status_uses_age_backstop_drop(isolated_registry, monkeypatch):
    # Task status unresolvable (e.g. task folder deleted elsewhere) + mtime
    # past MAX_ENTRY_AGE_S -> drop. Recent file with unresolvable status -> keep.
    import autonomous_session_watch as asw

    paths_old = _populate_gc_targets(isolated_registry, 700)
    paths_new = _populate_gc_targets(isolated_registry, 701)

    very_old_mtime = time.time() - MAX_ENTRY_AGE_S - 3600
    for p in paths_old.values():
        # Backdate mtime past the age backstop.
        import os

        os.utime(p, (very_old_mtime, very_old_mtime))

    monkeypatch.setattr(asw, "_task_status", lambda issue: None)

    asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=False)

    # The OLD set is gone (age backstop fired); the NEW set stays.
    for p in paths_old.values():
        assert not p.exists()
    for p in paths_new.values():
        assert p.exists()


def test_gc_ignores_garbled_filenames(isolated_registry, monkeypatch):
    # A hand-debug file with a non-int issue stem is left in place — none of
    # the GC's business.
    import autonomous_session_watch as asw

    (isolated_registry / "manual-issue-foo.json").write_text('{"junk": true}')
    (isolated_registry / f"{STALLED_STATE_PREFIX}bar.json").write_text('{"junk": true}')
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")

    asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=False)

    assert (isolated_registry / "manual-issue-foo.json").exists()
    assert (isolated_registry / f"{STALLED_STATE_PREFIX}bar.json").exists()


def test_gc_does_not_touch_pod_safety_files(isolated_registry, monkeypatch):
    # The pod-safety GC owns pod-safety-<N>.json (it keys on the RUNNING set,
    # a different question than task terminal status). Even with the task in
    # TERMINAL_FOR_GC, the generalized GC must NOT touch pod-safety files.
    import autonomous_session_watch as asw

    (isolated_registry / "pod-safety-789.json").write_text(
        json.dumps({"pod_id": "p789", "missed": 0, "alerted": False})
    )
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")

    asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=False)

    assert (isolated_registry / "pod-safety-789.json").exists()


def test_gc_does_not_touch_autonomous_registry_entries(isolated_registry, monkeypatch):
    # issue-<N>.json (autonomous registry) is handled by the respawn pass's
    # per-entry status check; the generalized GC must NOT compete with that
    # path. Drop one with a terminal-status task and confirm it stays.
    import autonomous_session_watch as asw

    _write_autonomous_entry(isolated_registry, 850, "sess-850")
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")

    asw._gc_orphaned_eps_autonomous_files(now=time.time(), dry_run=False)

    assert (isolated_registry / "issue-850.json").exists()


# ─── main() wiring ───────────────────────────────────────────────────────────


def test_main_runs_stalled_and_gc_after_pod_safety(isolated_registry, monkeypatch):
    # Pin the call order: vm-disk -> (respawn, inline) -> pod-safety ->
    # stalled -> orphan-sweep -> (session-reconcile) -> zombie-wrapper ->
    # idle-unmapped -> gc. The pin protects the docstring's documented order
    # + ensures a refactor doesn't accidentally drop one of the passes. (The
    # respawn pass is inlined in main() over the registry glob — empty here —
    # so it has no patchable call to record.)
    import autonomous_session_watch as asw

    calls: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: calls.append("vm_disk"))
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: calls.append("pod_safety"))
    monkeypatch.setattr(asw, "stalled_session_pass", lambda *a, **kw: calls.append("stalled"))
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: calls.append("orphan_sweep"))
    monkeypatch.setattr(asw, "zombie_wrapper_pass", lambda *a, **kw: calls.append("zombie_wrapper"))
    monkeypatch.setattr(asw, "idle_unmapped_pass", lambda *a, **kw: calls.append("idle_unmapped"))
    monkeypatch.setattr(asw, "gc_pass", lambda *a, **kw: calls.append("gc"))

    rc = asw.main([])
    assert rc == 0
    assert calls == [
        "vm_disk",
        "pod_safety",
        "stalled",
        "orphan_sweep",
        "zombie_wrapper",
        "idle_unmapped",
        "gc",
    ]


def test_gc_only_short_circuits_other_passes(isolated_registry, monkeypatch):
    # --gc-only must skip vm-disk / respawn / pod-safety / stalled /
    # orphan-sweep entirely.
    import autonomous_session_watch as asw

    calls: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: calls.append("vm_disk"))
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: calls.append("pod_safety"))
    monkeypatch.setattr(asw, "stalled_session_pass", lambda *a, **kw: calls.append("stalled"))
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: calls.append("orphan_sweep"))
    monkeypatch.setattr(asw, "zombie_wrapper_pass", lambda *a, **kw: calls.append("zombie_wrapper"))
    monkeypatch.setattr(asw, "idle_unmapped_pass", lambda *a, **kw: calls.append("idle_unmapped"))
    monkeypatch.setattr(asw, "gc_pass", lambda *a, **kw: calls.append("gc"))

    rc = asw.main(["--gc-only"])
    assert rc == 0
    assert calls == ["gc"]
