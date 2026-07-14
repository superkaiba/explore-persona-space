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
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.conftest import _stub_fleet_mutating_passes

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402
from autonomous_session_watch import (  # noqa: E402
    MAX_ENTRY_AGE_S,
    STALE_BLOCKED_STATE_PREFIX,
    STALLED_MARKER_WINDOW_S_DEFAULT,
    STALLED_STATE_PREFIX,
    STALLED_WINDOW_S,
    TERMINAL_FOR_GC,
    decide_session_stalled,
)

# The #1247 hermeticity guards (`_forbid_real_marker_posts`,
# `_forbid_real_task_status_reads`) are now shared autouse fixtures in
# tests/conftest.py (task #1265) — they apply here automatically.
# The `_stub_fleet_mutating_passes` helper this file calls is likewise the
# shared conftest copy as of task #1278 (imported above).


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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=0,
        alerted=False,
        threshold=2,
    ) == ("keep", 1)


def test_all_signals_stale_second_miss_alerts():
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=1,
        alerted=False,
        threshold=2,
    ) == ("alert", 0)


def test_threshold_one_alerts_immediately():
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=False,
        missed=0,
        alerted=False,
        threshold=1,
    ) == ("alert", 0)


def test_higher_threshold_delays_alert():
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
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
    # At exactly its window each signal is treated as stale (>= window);
    # just under is fresh. The self-report uses window_s (60 min); the
    # marker uses its own marker_window_s (2h default) — #845 (a-i).
    assert decide_session_stalled(
        self_report_age_s=STALLED_WINDOW_S - 1,
        marker_progress_age_s=STALLED_MARKER_WINDOW_S_DEFAULT + 60,
        has_pod=False,
        missed=1,
        alerted=False,
    ) == ("keep", 0)
    assert decide_session_stalled(
        self_report_age_s=STALLED_WINDOW_S,
        marker_progress_age_s=STALLED_MARKER_WINDOW_S_DEFAULT,
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


# ─── stalled-detector signal 2 counts ANY non-watcher marker (#661) ───────────


def test_stalled_signal2_counts_pre_run_lifecycle_marker():
    # Regression for #661: the alive-but-stalled detector's signal 2 must count
    # a pre-run lifecycle marker (`epm:experiment-implementation`,
    # `epm:review-reconcile`, ...) as a sign of life. The narrow
    # `_latest_progress_ts` allowlist (run/upload/interpret-oriented) IGNORES
    # those kinds, so a session actively implementing code looked "stale" and
    # was falsely respawned. The detector now reads `_latest_nonwatcher_event_ts`
    # (markers of ANY kind), which DOES see them.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:status-changed", "ts": "2026-06-25T00:39:00Z", "note": "running"},
        {
            "kind": "epm:experiment-implementation",
            "ts": "2026-06-25T01:15:00Z",
            "note": "implemented dispatch script",
        },
        {"kind": "epm:review-reconcile", "ts": "2026-06-25T01:31:00Z", "note": "PASS"},
    ]
    # The narrow allowlist sees only the 00:39 status-changed (the very thing
    # that read 64 min stale in the #661 incident); the broad helper sees the
    # 01:31 reconcile, the session's real last sign of life.
    assert asw._latest_progress_ts(events) == asw._parse_event_ts("2026-06-25T00:39:00Z")
    assert asw._latest_nonwatcher_event_ts(events) == asw._parse_event_ts("2026-06-25T01:31:00Z")


def test_stalled_signal2_still_excludes_watcher_sentinel_markers():
    # The broad signal must STILL ignore the watcher's own alert/automation
    # posts (otherwise an alert would reset the staleness clock it measures).
    # The note-substring filter — not a kind allowlist — is what enforces that.
    import autonomous_session_watch as asw

    events = [
        {
            "kind": "epm:experiment-implementation",
            "ts": "2026-06-25T01:15:00Z",
            "note": "real progress",
        },
        {
            "kind": "epm:progress",
            "ts": "2026-06-25T01:50:00Z",
            "note": f"{asw._STALLED_ALERT_NOTE_SENTINEL} session stalled alert ...",
        },
    ]
    # Newest non-watcher marker is the 01:15 lifecycle event; the 01:50 watcher
    # alert is filtered out.
    assert asw._latest_nonwatcher_event_ts(events) == asw._parse_event_ts("2026-06-25T01:15:00Z")


# ─── #1053 exit breadcrumbs are anti-liveness on BOTH watcher clocks ──────────


def test_1053_exit_breadcrumbs_advance_neither_staleness_clock():
    # #1053 MF-1 pin: the EXACT prescribed Step 0 collision-exit and
    # stale-wake-yield breadcrumbs (by="issue-session-guard", SKILL.md Step 0)
    # advance NEITHER `_latest_progress_ts` NOR `_latest_nonwatcher_event_ts`
    # — a duplicate session's death record must not shield the OWNER from the
    # stalled / orphan-respawn / reconcile-idle / campaign-freshness reads.
    import autonomous_session_watch as asw

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
    for note in (collision_note, yield_note):
        events = [
            {"kind": "epm:progress", "ts": "2026-07-04T10:00:00Z", "note": "step 100"},
            {
                "kind": "epm:progress",
                "ts": "2026-07-04T12:00:00Z",
                "note": note,
                "by": "issue-session-guard",
            },
        ]
        expected = asw._parse_event_ts("2026-07-04T10:00:00Z")
        assert asw._latest_progress_ts(events) == expected, note
        assert asw._latest_nonwatcher_event_ts(events) == expected, note


def test_1053_nonwatcher_clock_still_counts_normal_markers():
    # Non-vacuity companion to the exclusion above: an ordinary marker (no
    # deliberate-stop prefix, benign by) still advances
    # `_latest_nonwatcher_event_ts` after the #1053 exclusion landed.
    import autonomous_session_watch as asw

    events = [
        {"kind": "epm:progress", "ts": "2026-07-04T10:00:00Z", "note": "step 100"},
        {
            "kind": "epm:experiment-implementation",
            "ts": "2026-07-04T12:00:00Z",
            "note": "implemented dispatch script",
            "by": "test",
        },
    ]
    assert asw._latest_nonwatcher_event_ts(events) == asw._parse_event_ts("2026-07-04T12:00:00Z")


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


@pytest.fixture
def hermetic_provision_probes(tmp_path, monkeypatch):
    """Isolate the ALIVE-BUT-STALLED provisioning exemption (refs #573) from
    real VM state. Two environment probes can suppress the alert these tests
    assert: (a) `_POLL_STATE_DIR` points at the repo's REAL `.claude/cache/`,
    where a real `poll-pipeline-<N>.json` (e.g. issue 489's) has a 2026 mtime
    that reads as negative age — i.e. "fresh" — against the mocked
    `now=1_000_000.0` clock; (b) `_find_provision_process` scans /proc and can
    match a live `pod.py provision --issue <N>` on the VM. Point the poll-state
    dir at an empty tmp dir and stub the process probe to None so the tests
    exercise the detector logic, not the host's state.
    """
    import autonomous_session_watch as asw

    poll_dir = tmp_path / "poll-state-empty"
    poll_dir.mkdir()
    monkeypatch.setattr(asw, "_POLL_STATE_DIR", poll_dir)
    monkeypatch.setattr(asw, "_find_provision_process", lambda issue: None)
    return poll_dir


def test_stalled_pass_alerts_after_two_consecutive_stale_ticks(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
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
    # #1247 r2 hermeticity: _process_stalled_session re-reads the task status
    # via a REAL `task.py view 489` subprocess. Stub a TERMINAL status (the
    # value the live host historically returned for #489) so respawn stays
    # ineligible and the pass exercises the alert-only arm this test pins.
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
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


def _iso_from_epoch(epoch: float) -> str:
    """Canonical trailing-Z UTC ISO string for an epoch float (the shape
    `_parse_event_ts` round-trips)."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_stalled_pass_no_respawn_when_fresh_lifecycle_marker_present(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
    # Regression for #661: a frozen self-report (the long-running
    # experiment-implementer subagent doesn't update the parent's per-issue
    # self-report) MUST NOT trigger an alert/respawn while a RECENT pre-run
    # lifecycle marker (`epm:experiment-implementation`) shows the session is
    # actively working. Before the fix, signal 2 read `_latest_progress_ts`,
    # whose run/upload/interpret allowlist ignored that kind, so BOTH signals
    # looked stale and the session was falsely respawned mid-implementation.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 661, "sess-661")

    # Self-report frozen (stale) — the #661 condition.
    stale_age = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale_age, "2026-06-25T00:38:58Z"),
    )
    # ACTIVE status (the strictest case: respawn-eligible). Stubbed so the test
    # is hermetic, not dependent on task 661's real status on the host.
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    # ...but a lifecycle marker landed 5 min ago: the session is alive.
    fresh_ts = _iso_from_epoch(now - 300)
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda issue: [
            {"kind": "epm:status-changed", "ts": _iso_from_epoch(now - 3600), "note": "running"},
            {"kind": "epm:experiment-implementation", "ts": fresh_ts, "note": "impl"},
        ],
    )
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )
    # Make the respawn path OBSERVABLE: on the buggy line (`_latest_progress_ts`)
    # the marker reads stale -> the 2nd tick respawns, which only posts its
    # `session-auto-respawn` marker if BOTH _stop_session and
    # _respawn_stalled_session succeed. In the hermetic env both return False,
    # so without these stubs the respawn early-returns and `assert posts == []`
    # would pass even on the buggy code (the false-confidence gap the reviewer
    # caught). Stub both True so a respawn — if it fires — is visible here.
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: True)
    monkeypatch.setattr(asw, "_respawn_stalled_session", lambda issue, cap, dry_run: "spawned")

    # Two consecutive ticks while the self-report stays frozen: a fresh marker
    # resets the miss counter every tick, so nothing ever fires (no alert AND
    # no respawn). On the pre-fix `_latest_progress_ts` line this respawns on
    # tick 2 and posts `session-auto-respawn`.
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=2, now=now)
    assert posts == []
    state_path = isolated_registry / f"{STALLED_STATE_PREFIX}661.json"
    assert json.loads(state_path.read_text())["missed"] == 0


def test_stalled_pass_alerts_when_newest_nonwatcher_marker_truly_old(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
    # The fix must NOT mask a genuinely stalled session: when the self-report
    # is frozen AND the newest NON-watcher marker (of any kind) is itself old,
    # the detector still flags. A recent WATCHER-sentinel'd `epm:progress` does
    # not rescue it (the note-substring filter excludes it).
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 661, "sess-661")

    stale_age = STALLED_WINDOW_S + 600
    # The MARKER must be older than the #845 2h marker window (not just the
    # 60-min self-report window) for the detector to corroborate the stall.
    stale_marker_age = STALLED_MARKER_WINDOW_S_DEFAULT + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale_age, "2026-06-25T00:38:58Z"),
    )
    # Non-ACTIVE status -> respawn ineligible -> the deterministic ALERT arm
    # (a respawn would also be a valid "flag", but stub the status so the
    # asserted recovery action does not depend on task 661's real host status).
    monkeypatch.setattr(asw, "_task_status", lambda issue: None)
    old_ts = _iso_from_epoch(now - stale_marker_age)
    recent_watcher_ts = _iso_from_epoch(now - 60)
    monkeypatch.setattr(
        asw,
        "_task_events",
        lambda issue: [
            {"kind": "epm:experiment-implementation", "ts": old_ts, "note": "impl"},
            {
                "kind": "epm:progress",
                "ts": recent_watcher_ts,
                "note": f"{asw._STALLED_ALERT_NOTE_SENTINEL} session stalled alert ...",
            },
        ],
    )
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    # threshold=1 -> the single stale tick flags. status stubbed None ->
    # respawn-ineligible -> ALERT arm.
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    assert posts == [(661, "session-stalled-alert")]


def test_stalled_pass_dedups_within_episode(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
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
    # #1247 r2 hermeticity: terminal status (see the two-tick alert test).
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
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


def test_stalled_pass_clears_alerted_when_self_report_advances(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
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
    # #1247 r2 hermeticity: terminal status (see the two-tick alert test).
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
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


def test_stalled_pass_skips_when_no_self_report(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
    # Interactive (or just-spawned) sessions have no self-report file —
    # the pass treats that as "doesn't apply" and never alerts.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str]] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    # #1247 r2 hermeticity: terminal status (see the two-tick alert test).
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, label)),
    )

    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    assert posts == []


def test_stalled_pass_never_respawns_or_stops(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
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
    # #1247 r2 hermeticity: terminal status (see the two-tick alert test).
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_respawn", lambda entry, dry_run: respawns.append(entry) or True)
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    asw.stalled_session_pass(dry_run=False, threshold=1, now=now)
    assert respawns == []
    assert stops == []


def test_stalled_pass_manual_session_alert_only_never_respawns(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
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


def test_stalled_pass_dry_run_no_state_write(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
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
    # #1247 r2 hermeticity: terminal status (see the two-tick alert test).
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)

    asw.stalled_session_pass(dry_run=True, threshold=1, now=now)
    # No state file written in dry-run.
    assert not (isolated_registry / f"{STALLED_STATE_PREFIX}489.json").exists()


# ─── generalized GC — terminal-status reaping ────────────────────────────────


def _populate_gc_targets(reg_dir: Path, issue: int) -> dict[str, Path]:
    """Drop one file under a representative subset of the GC target prefixes
    for ``issue`` and return the paths so the test can assert
    presence/absence."""
    paths: dict[str, Path] = {}
    # manual-issue-<N>.json — top level
    paths["manual"] = reg_dir / f"manual-issue-{issue}.json"
    paths["manual"].write_text(json.dumps({"issue": issue, "mode": "manual"}))
    # stalled-<N>.json — top level
    paths["stalled"] = reg_dir / f"{STALLED_STATE_PREFIX}{issue}.json"
    paths["stalled"].write_text(json.dumps({"issue": issue, "missed": 0}))
    # stale-blocked-<N>.json — top level (#1021 flag-pass dedup state)
    paths["stale_blocked"] = reg_dir / f"{STALE_BLOCKED_STATE_PREFIX}{issue}.json"
    paths["stale_blocked"].write_text(
        json.dumps({"flagged_run_launched_ts": 1.0, "alerted_ts": 2.0})
    )
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

    # All five target paths should be gone.
    for p in paths.values():
        assert not p.exists(), f"{p} should have been reaped"
    # Counts dict should account for at least one drop per prefix.
    assert sum(counts.values()) == 5


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
    assert sum(counts.values()) == 5
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
    # #1247 round 2: with _daemon_reachable forced True below, the unstubbed
    # sweep/observer passes would reach live VM state (this exact test shelled
    # a real `task.py list-by-status --status proposed` via
    # proposed_infra_sweep_pass). Called FIRST so the test's own gc_pass
    # recorder below overrides the helper's stub and keeps recording.
    _stub_fleet_mutating_passes(asw, monkeypatch)
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    # main() fetches the shared reaper-pass /list snapshot + the #845 wedge
    # pid map directly; patch both so the wiring test never RPCs the real
    # daemon.
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    monkeypatch.setattr(asw, "vm_disk_pass", lambda *a, **kw: calls.append("vm_disk"))
    # #967: never sweep the LIVE registry/events tree from a unit test.
    monkeypatch.setattr(asw, "triage_observer_pass", lambda *a, **kw: None)
    # #1021: the stale-blocked flag pass shells task.py against the LIVE
    # blocked set (daemon-independent) — never run it from a unit test.
    monkeypatch.setattr(asw, "stale_blocked_flag_pass", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "pod_safety_pass", lambda *a, **kw: calls.append("pod_safety"))
    monkeypatch.setattr(asw, "stalled_session_pass", lambda *a, **kw: calls.append("stalled"))
    monkeypatch.setattr(asw, "orphan_sweep_pass", lambda *a, **kw: calls.append("orphan_sweep"))
    monkeypatch.setattr(
        asw, "stale_registration_pass", lambda *a, **kw: calls.append("stale_registration")
    )
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
        "stale_registration",
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


# ─── #845 stall-detection hardening — pure predicates ─────────────────────────


def test_marker_window_2h_blocks_stall_declaration():
    # #845 (a-i) — the #761/#763 false-positive class: self-report 90 min
    # stale AND newest non-watcher marker 90 min old. Under the pre-#845
    # single 60-min window this corroborated a stall (alert on the 2nd miss);
    # under the dedicated 2h marker window the still-fresh-enough marker
    # blocks the declaration outright.
    ninety_min = 90 * 60
    assert decide_session_stalled(
        self_report_age_s=ninety_min,
        marker_progress_age_s=ninety_min,
        has_pod=False,
        missed=1,
        alerted=False,
        threshold=2,
    ) == ("keep", 0)


def test_marker_window_env_override(monkeypatch):
    # EPM_STALLED_MARKER_HEARTBEAT_MIN parses minutes; malformed falls back.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_STALLED_MARKER_HEARTBEAT_MIN", raising=False)
    assert asw._stalled_marker_window_s() == float(asw.STALLED_MARKER_WINDOW_S_DEFAULT)

    monkeypatch.setenv("EPM_STALLED_MARKER_HEARTBEAT_MIN", "45")
    assert asw._stalled_marker_window_s() == 45 * 60.0

    monkeypatch.setenv("EPM_STALLED_MARKER_HEARTBEAT_MIN", "garbled")
    assert asw._stalled_marker_window_s() == float(asw.STALLED_MARKER_WINDOW_S_DEFAULT)


def test_decide_respawn_fence_matrix():
    # #845 (a-ii): the stop-verify fence's full decision table.
    import autonomous_session_watch as asw

    f = asw.decide_respawn_fence
    # First tick of an episode: no pending stop -> stop only, never spawn.
    assert f(stop_pending_sid=None, current_sid="a", sid_alive=True, stop_retried=False) == "stop"
    assert f(stop_pending_sid=None, current_sid="a", sid_alive=False, stop_retried=False) == "stop"
    # Later tick, pending sid verified dead -> safe to spawn.
    assert f(stop_pending_sid="a", current_sid="a", sid_alive=False, stop_retried=False) == "spawn"
    # Still alive on the verify tick, retry unused -> exactly one retry.
    assert (
        f(stop_pending_sid="a", current_sid="a", sid_alive=True, stop_retried=False) == "retry-stop"
    )
    # Still alive after the retry -> loud one-time alert, never spawn.
    assert (
        f(stop_pending_sid="a", current_sid="a", sid_alive=True, stop_retried=True) == "stop-failed"
    )
    # Pending sid no longer matches the registry entry (a concurrent respawn
    # replaced the session) -> clear the fence, never stop the fresh sid.
    assert (
        f(stop_pending_sid="a", current_sid="b", sid_alive=True, stop_retried=False) == "clear-keep"
    )
    assert (
        f(stop_pending_sid="a", current_sid=None, sid_alive=False, stop_retried=True)
        == "clear-keep"
    )


def test_decide_worktree_hold_bounds():
    # #845 (b): hold iff activity is fresh AND under the tick cap — a bound,
    # not a latch.
    import autonomous_session_watch as asw

    assert asw.decide_worktree_hold(True, 0) is True
    assert asw.decide_worktree_hold(True, asw.WT_HOLD_MAX_TICKS - 1) is True
    assert asw.decide_worktree_hold(True, asw.WT_HOLD_MAX_TICKS) is False
    assert asw.decide_worktree_hold(False, 0) is False
    assert asw.decide_worktree_hold(True, 1, max_holds=1) is False


def test_decide_daemon_blocked_escalation_matrix():
    # #845 (c): increments on daemon-blocked alerted ACTIVE stale ticks,
    # fires ONCE at the 2-tick threshold, resets on a reachable tick.
    import autonomous_session_watch as asw

    f = asw.decide_daemon_blocked_escalation
    base = dict(in_active=True, manual=False, alerted=True, stale=True, daemon_reachable=False)
    # Tick 1 blocked: count 1, below threshold -> no push.
    assert f(**base, blocked_ticks=0, already_pushed=False) == (1, False)
    # Tick 2 blocked: threshold met -> fire exactly once.
    assert f(**base, blocked_ticks=1, already_pushed=False) == (2, True)
    # Tick 3 blocked, already pushed -> keep counting, no repeat push.
    assert f(**base, blocked_ticks=2, already_pushed=True) == (3, False)
    # Daemon reachable -> full reset (the alerted->eligible escalation
    # respawns on this same tick).
    assert f(**{**base, "daemon_reachable": True}, blocked_ticks=5, already_pushed=True) == (
        0,
        False,
    )
    # Manual / non-ACTIVE / not-alerted / not-stale episodes freeze (no
    # count-up, no push) — the deferral is not a respawn-worthy stall.
    assert f(**{**base, "manual": True}, blocked_ticks=1, already_pushed=False) == (1, False)
    assert f(**{**base, "in_active": False}, blocked_ticks=1, already_pushed=False) == (1, False)
    assert f(**{**base, "alerted": False}, blocked_ticks=1, already_pushed=False) == (1, False)
    assert f(**{**base, "stale": False}, blocked_ticks=1, already_pushed=False) == (1, False)


# ── #845 (e) prompt-wedge row fixtures — shapes match live transcripts ───────


def _wedge_prompt_row():
    return {"type": "user", "message": {"role": "user", "content": "/issue 779 tick"}}


def _wedge_dequeue_row():
    # VERBATIM real row shape captured from a live session transcript
    # (2026-07-02; plan §6 verification): dequeue rows carry NO content field.
    return {
        "type": "queue-operation",
        "operation": "dequeue",
        "timestamp": "2026-07-01T18:57:16.499Z",
        "sessionId": "d63ed59b-d534-497c-95ae-e30de63a3112",
    }


def _wedge_enqueue_row(content="/issue 813"):
    # VERBATIM real enqueue shape (carries content; classified 'other').
    return {
        "type": "queue-operation",
        "operation": "enqueue",
        "timestamp": "2026-07-01T18:57:16.498Z",
        "sessionId": "d63ed59b-d534-497c-95ae-e30de63a3112",
        "content": content,
    }


def _wedge_assistant_row():
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "on it"}]},
    }


def _wedge_tool_result_row():
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    }


def _wedge_api_error_row():
    # VERBATIM sanitized real row shape captured from the live #1074 incident
    # transcript (session 6f682c18, 2026-07-06; 38/38 refusal rows had this
    # shape). Pins the TOP-LEVEL placement of `isApiErrorMessage` — a future
    # transcript-format drift that NESTS the flag (e.g. under `message`) must
    # fail these tests visibly, not silently reclassify to "assistant".
    # Content/error text omitted or benign by construction (#1104
    # refusal-safety: never copy refusal text into the repo).
    return {
        "type": "assistant",
        "isApiErrorMessage": True,
        "error": "sanitized",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "sanitized"}]},
        "uuid": "00000000-0000-0000-0000-000000000000",
        "parentUuid": "00000000-0000-0000-0000-000000000001",
        "sessionId": "6f682c18-5370-4593-85a0-03f4aed6f810",
        "timestamp": "2026-07-06T21:48:37.000Z",
        "version": "2.1.128",
        "gitBranch": "main",
        "cwd": "/home/user/explore-persona-space",
        "userType": "external",
        "isSidechain": False,
        "entrypoint": "sdk-ts",
    }


def test_decide_prompt_wedge_three_promptless_rows():
    import autonomous_session_watch as asw

    tail = [_wedge_assistant_row(), _wedge_prompt_row(), _wedge_prompt_row(), _wedge_prompt_row()]
    assert asw.decide_prompt_wedge(tail, 3) is True
    # Two trailing prompts are under the N=3 floor.
    assert asw.decide_prompt_wedge(tail[:-1], 3) is False


def test_decide_prompt_wedge_assistant_resets():
    # An assistant row means the session took a turn — the run resets, so a
    # busy-but-responding session never trips the wedge.
    import autonomous_session_watch as asw

    tail = [
        _wedge_prompt_row(),
        _wedge_prompt_row(),
        _wedge_prompt_row(),
        _wedge_assistant_row(),
        _wedge_prompt_row(),
    ]
    assert asw.decide_prompt_wedge(tail, 3) is False


def test_decide_prompt_wedge_tool_result_rows_skipped():
    # tool_result user rows are 'other': skipped WITHOUT resetting the run
    # (a wedged session's queue keeps interleaving bg-task results).
    import autonomous_session_watch as asw

    tail = [
        _wedge_assistant_row(),
        _wedge_prompt_row(),
        _wedge_tool_result_row(),
        _wedge_prompt_row(),
        _wedge_tool_result_row(),
        _wedge_prompt_row(),
    ]
    assert asw.decide_prompt_wedge(tail, 3) is True


def test_decide_prompt_wedge_malformed_rows_other():
    # Malformed rows (non-dict, missing/unknown fields) classify 'other' —
    # never a crash, never a reset.
    import autonomous_session_watch as asw

    tail = [
        _wedge_prompt_row(),
        "garbage-line",
        None,
        {"type": 42},
        {"no-type": True},
        _wedge_prompt_row(),
        _wedge_prompt_row(),
    ]
    assert asw.decide_prompt_wedge(tail, 3) is True
    assert asw._classify_wedge_row("garbage-line") == "other"
    assert asw._classify_wedge_row({"type": "queue-operation", "operation": "enqueue"}) == "other"


def test_decide_prompt_wedge_counts_dequeue_ops():
    # CO-PRIMARY evidence: verified queue-operation dequeue records count
    # toward the trailing run — alone AND mixed with promptless user rows.
    import autonomous_session_watch as asw

    alone = [_wedge_assistant_row()] + [_wedge_dequeue_row()] * 3
    assert asw.decide_prompt_wedge(alone, 3) is True
    mixed = [
        _wedge_assistant_row(),
        _wedge_dequeue_row(),
        _wedge_prompt_row(),
        _wedge_dequeue_row(),
    ]
    assert asw.decide_prompt_wedge(mixed, 3) is True


def test_decide_prompt_wedge_779_replay_queue_operation_rows():
    # #779 replay: a tail built from the VERBATIM live enqueue/dequeue row
    # shapes — repeated tick prompts enqueued AND dequeued with no assistant
    # turn after the first. The enqueue rows are 'other' (skipped); the 3
    # dequeue records trip the wedge.
    import autonomous_session_watch as asw

    tail = [
        _wedge_assistant_row(),
        _wedge_enqueue_row("/issue-tick 779"),
        _wedge_dequeue_row(),
        _wedge_enqueue_row("/issue-tick 779"),
        _wedge_dequeue_row(),
        _wedge_enqueue_row("/issue-tick 779"),
        _wedge_dequeue_row(),
    ]
    assert asw.decide_prompt_wedge(tail, 3) is True
    # The same tail with a trailing assistant turn is NOT wedged.
    assert asw.decide_prompt_wedge([*tail, _wedge_assistant_row()], 3) is False


def test_tick_wedge_min_dequeued_env_override(monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_TICK_WEDGE_MIN_DEQUEUED", raising=False)
    assert asw._tick_wedge_min_dequeued() == asw.TICK_WEDGE_MIN_DEQUEUED
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_DEQUEUED", "5")
    assert asw._tick_wedge_min_dequeued() == 5
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_DEQUEUED", "0")
    assert asw._tick_wedge_min_dequeued() == asw.TICK_WEDGE_MIN_DEQUEUED
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_DEQUEUED", "garbled")
    assert asw._tick_wedge_min_dequeued() == asw.TICK_WEDGE_MIN_DEQUEUED


# ── #1104 api-error (orchestrator-refusal) wedge widening ────────────────────


def test_classify_wedge_row_api_error_assistant():
    # #1104 plan test 1: an assistant row with TOP-LEVEL
    # `isApiErrorMessage: true` classifies "api-error"; absent / False /
    # non-True values keep the plain "assistant" class (the reset row).
    import autonomous_session_watch as asw

    assert asw._classify_wedge_row(_wedge_api_error_row()) == "api-error"
    assert asw._classify_wedge_row(_wedge_assistant_row()) == "assistant"
    explicit_false = {**_wedge_assistant_row(), "isApiErrorMessage": False}
    assert asw._classify_wedge_row(explicit_false) == "assistant"
    truthy_non_bool = {**_wedge_assistant_row(), "isApiErrorMessage": "yes"}
    assert asw._classify_wedge_row(truthy_non_bool) == "assistant"


def test_decide_prompt_wedge_1074_replay_refusal_turns_fire():
    # #1104 plan test 2 — the #1074 replay: each refused wake contributes
    # dequeue x2 + prompt x2 + one api-error turn; three consecutive refused
    # wakes trip the wedge at defaults. The SAME tail with min_api_errors=0
    # (the kill switch) reproduces the pre-fix blindness: no fire.
    import autonomous_session_watch as asw

    refused_wake = [
        _wedge_dequeue_row(),
        _wedge_dequeue_row(),
        _wedge_prompt_row(),
        _wedge_prompt_row(),
        _wedge_api_error_row(),
    ]
    tail = [_wedge_assistant_row()] + refused_wake * 3
    assert asw.decide_prompt_wedge(tail, 3, min_api_errors=3) is True
    # #1127 plan-sanctioned edit (§7.13): this tail segments to 3 trailing
    # FAILED wake-turns, so under the new default min_failed_turns=3 the same
    # call fires the failed-turn-run trigger. This assertion pins "the
    # api-error kill switch disables the API-ERROR trigger" — each trigger
    # class needs its OWN switch thrown to silence this tail, so the turn
    # knobs are zeroed here to preserve the original pin.
    assert (
        asw.decide_prompt_wedge(tail, 3, min_api_errors=0, min_failed_turns=0, min_failed_total=0)
        is False
    )


def test_decide_prompt_wedge_api_error_resets_dequeue_run():
    # #1104 plan test 3: an api-error turn RESETS the dequeue/prompt run —
    # the prompt DID get a (failed) response, so a single refused wake's
    # 2+2 rows must not trip the EXISTING run >= 3 threshold.
    import autonomous_session_watch as asw

    tail = [
        _wedge_dequeue_row(),
        _wedge_dequeue_row(),
        _wedge_api_error_row(),
        _wedge_dequeue_row(),
        _wedge_dequeue_row(),
    ]
    assert asw.decide_prompt_wedge(tail, 3, min_api_errors=3) is False


def test_decide_prompt_wedge_real_assistant_resets_api_run():
    # #1104 plan test 4: a REAL assistant turn resets the api-error counter —
    # interleaved one-off refusals (the 18+/day transient class) never
    # accumulate across successful turns.
    import autonomous_session_watch as asw

    tail = [
        _wedge_api_error_row(),
        _wedge_api_error_row(),
        _wedge_assistant_row(),
        _wedge_api_error_row(),
        _wedge_api_error_row(),
    ]
    assert asw.decide_prompt_wedge(tail, 3, min_api_errors=3) is False


def test_tick_wedge_min_api_errors_env_override(monkeypatch):
    # #1104 plan test 5: mirror of test_tick_wedge_min_dequeued_env_override
    # with the ONE deliberate divergence — "0" DISABLES (returns 0), it does
    # not fall back to the default.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_TICK_WEDGE_MIN_API_ERRORS", raising=False)
    assert asw._tick_wedge_min_api_errors() == asw.TICK_WEDGE_MIN_API_ERRORS
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_API_ERRORS", "5")
    assert asw._tick_wedge_min_api_errors() == 5
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_API_ERRORS", "0")
    assert asw._tick_wedge_min_api_errors() == 0
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_API_ERRORS", "-2")
    assert asw._tick_wedge_min_api_errors() == asw.TICK_WEDGE_MIN_API_ERRORS
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_API_ERRORS", "garbled")
    assert asw._tick_wedge_min_api_errors() == asw.TICK_WEDGE_MIN_API_ERRORS


# ── #1127 turn-level failed-wake wedge (partial wakes + alternating storms) ──


def _wedge_iso_1127(epoch: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _wedge_api_error_row_at(ts: float):
    # Same sanitized structural shape as _wedge_api_error_row (no refusal
    # text — the #1104/#1127 refusal-safety contract), timestamp overridden
    # so the #1127 rate trigger's window arithmetic is testable.
    return {**_wedge_api_error_row(), "timestamp": _wedge_iso_1127(ts)}


def _wedge_assistant_row_at(ts: float):
    return {**_wedge_assistant_row(), "timestamp": _wedge_iso_1127(ts)}


def _wedge_partial_wake_unit(n_heartbeats: int, ts: float | None = None):
    # #1127: the #1098 (5bdae5b8) / #1090 (5e464f3d) repeating tail unit —
    # [dequeue, prompt, prompt, assistant x n, api-error]: the wake does SOME
    # work (mid-turn heartbeat assistant rows) then DIES in an api-error row,
    # so every row-level counter resets each cycle. Sanitized structural rows
    # only (same contract as _wedge_api_error_row).
    unit = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_prompt_row()]
    unit.extend(_wedge_assistant_row() for _ in range(n_heartbeats))
    unit.append(_wedge_api_error_row() if ts is None else _wedge_api_error_row_at(ts))
    return unit


def test_segment_wake_turns_1098_partial_wake_shape():
    # #1127 plan test 1 — the #1098/#1090 blindness pin: 8 partial-wake units
    # segment to 8 FAILED turns, while the OLD row-level counters stay quiet
    # (run=0, api_run<=1 — every unit's assistant heartbeats reset them), so
    # only the new failed-turn-run trigger can fire.
    import autonomous_session_watch as asw

    tail = [row for _ in range(8) for row in _wedge_partial_wake_unit(2)]
    turns = asw._segment_wake_turns(tail)
    assert [outcome for outcome, _ts in turns] == ["failed"] * 8
    # Pre-fix blindness: replay the row-level counters directly.
    run = api_run = 0
    for row in tail:
        cls = asw._classify_wedge_row(row)
        if cls == "assistant":
            run = api_run = 0
        elif cls == "api-error":
            run = 0
            api_run += 1
        elif cls in ("dequeue", "prompt"):
            run += 1
    assert run == 0 and api_run <= 1
    # Row-trigger-only read (the pre-#1127 predicate): NO fire.
    assert (
        asw.decide_prompt_wedge_reason(
            tail, 3, min_api_errors=3, min_failed_turns=0, min_failed_total=0
        )
        is None
    )
    # The fresh-path #1127 read fires the new trigger.
    assert asw.decide_prompt_wedge_reason(tail, 0, min_api_errors=0) == "failed-turn-run"


def test_segment_wake_turns_midturn_api_error_retry_is_ok():
    # #1127 plan test 2: a turn [prompt, api-error, assistant] ends in a
    # SUCCESSFUL row -> outcome "ok" (the healthy retried-429 shape); five
    # such turns never fire any trigger.
    import autonomous_session_watch as asw

    turn = [_wedge_prompt_row(), _wedge_api_error_row(), _wedge_assistant_row()]
    assert [o for o, _ts in asw._segment_wake_turns(turn)] == ["ok"]
    tail = turn * 5
    assert asw.decide_prompt_wedge_reason(tail, 0, min_api_errors=0) is None
    assert asw.decide_prompt_wedge_reason(tail, 3, min_api_errors=3) is None


def test_decide_prompt_wedge_single_refusal_turn_guard():
    # #1127 plan test 3 — the single-refusal guard at TURN granularity: two
    # ok turns then ONE failed turn is below every threshold; and a SINGLE
    # wake with 3 same-turn retry api-error rows on the fresh path
    # (min_api_errors=0) is ONE failed turn, not three.
    import autonomous_session_watch as asw

    ok_turn = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_assistant_row()]
    failed_turn = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_api_error_row()]
    assert asw.decide_prompt_wedge_reason([*ok_turn, *ok_turn, *failed_turn], 3) is None
    single_multi_retry = [
        _wedge_dequeue_row(),
        _wedge_prompt_row(),
        _wedge_api_error_row(),
        _wedge_api_error_row(),
        _wedge_api_error_row(),
    ]
    assert asw.decide_prompt_wedge_reason(single_multi_retry, 0, min_api_errors=0) is None


def test_decide_prompt_wedge_failed_turn_run_fires_at_three():
    # #1127 plan test 4: exactly 3 trailing failed turns fire failed-turn-run
    # (heartbeat units keep api_run quiet, so the TURN trigger is what fires
    # even at default row knobs); 2 do not; an ok turn between resets.
    import autonomous_session_watch as asw

    unit = _wedge_partial_wake_unit(1)
    assert asw.decide_prompt_wedge_reason(unit * 3, 3) == "failed-turn-run"
    assert asw.decide_prompt_wedge_reason(unit * 2, 3) is None
    ok_turn = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_assistant_row()]
    interleaved = [*unit, *unit, *ok_turn, *unit, *unit]
    assert asw.decide_prompt_wedge_reason(interleaved, 3) is None


def _wedge_alternating_storm_tail(base_ts: float, n_failed: int = 7, step_s: float = 300.0):
    # #1127: the c16b10ca structural shape — failed turns ALTERNATING with ok
    # turns (~every other wake lost), all timestamped, NEWEST completed turn
    # failed. n_failed failed turns interleaved with (n_failed - 1) ok turns.
    tail: list[dict] = []
    ts = base_ts
    for i in range(2 * n_failed - 1):
        tail.extend([_wedge_dequeue_row(), _wedge_prompt_row()])
        if i % 2 == 0:
            tail.append(_wedge_api_error_row_at(ts))
        else:
            tail.append(_wedge_assistant_row_at(ts))
        ts += step_s
    return tail


def test_decide_prompt_wedge_failed_turn_rate_alternating():
    # #1127 plan test 5 — a SYNTHETIC dense alternating storm (denser than
    # the measured c16b10ca tail, which held only 4-5 windowed failed turns
    # — below threshold by design, plan v4 §3.3): 7 timestamped
    # failed turns (5 min apart, interleaved with ok turns, newest completed
    # turn FAILED) fire failed-turn-rate; newest-turn-ok, ts-stripped, and
    # back-shifted (> 120 min behind the newest row ts) variants do NOT.
    import autonomous_session_watch as asw

    base = 1_780_000_000.0
    tail = _wedge_alternating_storm_tail(base)
    assert asw.decide_prompt_wedge_reason(tail, 0, min_api_errors=0) == "failed-turn-rate"
    # Newest completed turn ok -> a recovered session is not respawned.
    recovered = [
        *tail,
        _wedge_dequeue_row(),
        _wedge_prompt_row(),
        _wedge_assistant_row_at(base + 4000),
    ]
    assert asw.decide_prompt_wedge_reason(recovered, 0, min_api_errors=0) is None
    # All timestamps stripped -> no anchor -> the rate trigger is inert.
    stripped = [{k: v for k, v in row.items() if k != "timestamp"} for row in tail]
    assert asw.decide_prompt_wedge_reason(stripped, 0, min_api_errors=0) is None
    # Failures back-shifted > 120 min behind the newest row ts: a trailing
    # in-flight delivery row carries a much newer ts, aging every failure
    # out of the window (the newest completed turn is still failed).
    newest_failed_ts = base + 300.0 * (2 * 7 - 2)
    aged = [*tail, {**_wedge_dequeue_row(), "timestamp": _wedge_iso_1127(newest_failed_ts + 7300)}]
    assert asw.decide_prompt_wedge_reason(aged, 0, min_api_errors=0) is None


def test_decide_prompt_wedge_swallowed_deliveries_neither_reset_nor_count():
    # #1127 plan test 6: swallowed delivery bursts (prompt evidence, no
    # response) interleaved between failed turns produce NO turns — the turn
    # counters are unaffected; the #779 dequeue-run trigger still owns the
    # pure-swallow tail, and min_dequeued=0 disables it (the new fresh-path
    # semantics).
    import autonomous_session_watch as asw

    failed_turn = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_api_error_row()]
    swallow = [_wedge_dequeue_row(), _wedge_dequeue_row()]
    tail = [*failed_turn, *swallow, *failed_turn, *swallow, *failed_turn]
    turns = asw._segment_wake_turns(tail)
    assert [o for o, _ts in turns] == ["failed"] * 3
    assert asw.decide_prompt_wedge_reason(tail, 0, min_api_errors=0) == "failed-turn-run"
    pure_779 = [_wedge_dequeue_row()] * 5
    assert asw.decide_prompt_wedge_reason(pure_779, 3, min_api_errors=0) == "dequeue-run"
    assert asw.decide_prompt_wedge_reason(pure_779, 0, min_api_errors=0) is None


def test_decide_prompt_wedge_bool_wrapper_backcompat():
    # #1127 plan test 7: the thin bool wrapper returns True/False identically
    # to the pre-refactor predicate on the existing #779 and #1074 fixture
    # tails (guards the decide_prompt_wedge -> decide_prompt_wedge_reason
    # refactor).
    import autonomous_session_watch as asw

    tail_779 = [
        _wedge_assistant_row(),
        _wedge_enqueue_row("/issue-tick 779"),
        _wedge_dequeue_row(),
        _wedge_enqueue_row("/issue-tick 779"),
        _wedge_dequeue_row(),
        _wedge_enqueue_row("/issue-tick 779"),
        _wedge_dequeue_row(),
    ]
    assert asw.decide_prompt_wedge(tail_779, 3) is True
    assert asw.decide_prompt_wedge([*tail_779, _wedge_assistant_row()], 3) is False
    refused_wake = [
        _wedge_dequeue_row(),
        _wedge_dequeue_row(),
        _wedge_prompt_row(),
        _wedge_prompt_row(),
        _wedge_api_error_row(),
    ]
    tail_1074 = [_wedge_assistant_row()] + refused_wake * 3
    assert asw.decide_prompt_wedge(tail_1074, 3) is True
    assert asw.decide_prompt_wedge([*tail_1074, _wedge_assistant_row()], 3) is False


def test_tick_wedge_failed_turns_env_helpers(monkeypatch):
    # #1127 plan test 8: the three new env helpers — defaults, 0-DISABLES on
    # the two MIN knobs (the kill switches), malformed/negative -> default,
    # and the rate window's minutes-env -> seconds conversion with
    # non-positive -> default (a WINDOW, not a trigger).
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", raising=False)
    assert asw._tick_wedge_min_failed_turns() == asw.TICK_WEDGE_MIN_FAILED_TURNS
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", "5")
    assert asw._tick_wedge_min_failed_turns() == 5
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", "0")
    assert asw._tick_wedge_min_failed_turns() == 0
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", "-1")
    assert asw._tick_wedge_min_failed_turns() == asw.TICK_WEDGE_MIN_FAILED_TURNS
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TURNS", "garbled")
    assert asw._tick_wedge_min_failed_turns() == asw.TICK_WEDGE_MIN_FAILED_TURNS

    monkeypatch.delenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", raising=False)
    assert asw._tick_wedge_min_failed_total() == asw.TICK_WEDGE_MIN_FAILED_TOTAL
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", "8")
    assert asw._tick_wedge_min_failed_total() == 8
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", "0")
    assert asw._tick_wedge_min_failed_total() == 0
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", "-3")
    assert asw._tick_wedge_min_failed_total() == asw.TICK_WEDGE_MIN_FAILED_TOTAL
    monkeypatch.setenv("EPM_TICK_WEDGE_MIN_FAILED_TOTAL", "garbled")
    assert asw._tick_wedge_min_failed_total() == asw.TICK_WEDGE_MIN_FAILED_TOTAL

    monkeypatch.delenv("EPM_TICK_WEDGE_RATE_WINDOW_MIN", raising=False)
    assert asw._tick_wedge_rate_window_s() == float(asw.TICK_WEDGE_RATE_WINDOW_S)
    monkeypatch.setenv("EPM_TICK_WEDGE_RATE_WINDOW_MIN", "60")
    assert asw._tick_wedge_rate_window_s() == 3600.0  # minutes -> seconds
    monkeypatch.setenv("EPM_TICK_WEDGE_RATE_WINDOW_MIN", "0")
    assert asw._tick_wedge_rate_window_s() == float(asw.TICK_WEDGE_RATE_WINDOW_S)
    monkeypatch.setenv("EPM_TICK_WEDGE_RATE_WINDOW_MIN", "-5")
    assert asw._tick_wedge_rate_window_s() == float(asw.TICK_WEDGE_RATE_WINDOW_S)
    monkeypatch.setenv("EPM_TICK_WEDGE_RATE_WINDOW_MIN", "garbled")
    assert asw._tick_wedge_rate_window_s() == float(asw.TICK_WEDGE_RATE_WINDOW_S)


# ── #1209 failed-turn-silence (die-on-turn-1 dead-wake trigger) ──────────────


def _wedge_1092_tail():
    # Structural digest of the real #1092 incident transcript (session
    # 8e9c371d, 39 rows): one delivery burst, ~13 plain assistant rows
    # interleaved with tool_result rows, ONE final api-error row (the
    # silence anchor — the verbatim _wedge_api_error_row fixture, ts
    # 2026-07-06T21:48:37.000Z stands in for the incident's 02:54:29.726Z),
    # then one timestamp-less trailing row (the incident's `last-prompt`
    # row, classified "other" and excluded from the anchor scan). Sanitized
    # structural rows only (the #1104 refusal-safety contract).
    tail = [_wedge_dequeue_row(), _wedge_prompt_row()]
    for _ in range(13):
        tail.extend([_wedge_assistant_row(), _wedge_tool_result_row()])
    tail.append(_wedge_api_error_row())
    tail.append({"type": "last-prompt"})
    return tail


def _wedge_1092_anchor_ts():
    import autonomous_session_watch as asw

    ts = asw._row_ts(_wedge_api_error_row())
    assert ts is not None
    return ts


def test_decide_prompt_wedge_dead_silence_fires_on_1092_replay():
    # #1209 T1 — the incident replay: under the CURRENT classifiers this
    # shape reads run=0, api_run=1 and exactly ONE completed turn (failed),
    # so NO pre-#1209 trigger can ever fire; 21 min of silence past the
    # final api-error row fires the new failed-turn-silence trigger.
    import autonomous_session_watch as asw

    tail = _wedge_1092_tail()
    run, api_run = asw._wedge_trailing_row_runs(tail)
    assert (run, api_run) == (0, 1)  # below every row-level threshold
    assert [o for o, _ts in asw._segment_wake_turns(tail)] == ["failed"]
    # Every pre-#1209 trigger at production defaults: no fire (now omitted).
    assert asw.decide_prompt_wedge_reason(tail, 3) is None
    t = _wedge_1092_anchor_ts()
    assert asw.decide_prompt_wedge_reason(tail, 3, now=t + 21 * 60) == "failed-turn-silence"


def test_decide_prompt_wedge_dead_silence_below_threshold_no_fire():
    # #1209 T2: the same tail 10 min after death is NOT yet dead-silent
    # (a fresh api-error younger than the threshold never escalates).
    import autonomous_session_watch as asw

    tail = _wedge_1092_tail()
    assert asw.decide_prompt_wedge_reason(tail, 3, now=_wedge_1092_anchor_ts() + 10 * 60) is None


def test_decide_prompt_wedge_dead_silence_prior_ok_turn_blocks():
    # #1209 T3: an earlier ok-completed turn in the tail blocks the trigger
    # regardless of silence — the all-completed-turns-failed condition
    # confines it to sessions with ZERO successful history (a healthy-idle
    # session whose last wake ended in one transient api-error never fires).
    import autonomous_session_watch as asw

    ok_turn = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_assistant_row()]
    failed_turn = [_wedge_dequeue_row(), _wedge_prompt_row(), _wedge_api_error_row()]
    tail = [*ok_turn, *failed_turn]
    assert [o for o, _ts in asw._segment_wake_turns(tail)] == ["ok", "failed"]
    assert asw.decide_prompt_wedge_reason(tail, 3, now=_wedge_1092_anchor_ts() + 21 * 60) is None


def test_decide_prompt_wedge_dead_silence_no_completed_turns_no_fire():
    # #1209 T4 — the vacuous-all guard: a swallow-shaped tail (prompt
    # evidence, ZERO completed turns) never fires the silence trigger even
    # when dead-silent; the #779 dequeue-run trigger owns swallows.
    import autonomous_session_watch as asw

    tail = [_wedge_dequeue_row(), _wedge_dequeue_row(), _wedge_prompt_row()]
    assert asw._segment_wake_turns(tail) == []
    t = _wedge_1092_anchor_ts()  # dequeue fixture rows carry parseable ts
    assert asw.decide_prompt_wedge_reason(tail, 0, now=t + 10**9) is None


def test_decide_prompt_wedge_dead_silence_inert_without_now():
    # #1209 T5: `now=None` (every pre-existing pure call shape) leaves the
    # trigger inert — the bool wrapper's behavior on the incident tail is
    # byte-identical to pre-#1209.
    import autonomous_session_watch as asw

    tail = _wedge_1092_tail()
    assert asw.decide_prompt_wedge_reason(tail, 3) is None
    assert asw.decide_prompt_wedge(tail, 3) is False


def test_decide_prompt_wedge_dead_silence_zero_disables():
    # #1209 T6: dead_silence_s=0 is the kill switch even with `now` set
    # (the _tick_wedge_dead_silence_s 0-DISABLES convention).
    import autonomous_session_watch as asw

    tail = _wedge_1092_tail()
    now = _wedge_1092_anchor_ts() + 10**9
    assert asw.decide_prompt_wedge_reason(tail, 3, dead_silence_s=0, now=now) is None


def test_decide_prompt_wedge_dead_silence_tsless_tail_no_fire():
    # #1209 T7: a fully ts-less tail leaves the silence anchor undefined ->
    # NO-FIRE; a future-dated anchor (clock jump: now < anchor) also fails
    # toward NO-FIRE.
    import autonomous_session_watch as asw

    tail = _wedge_1092_tail()
    stripped = [
        {k: v for k, v in row.items() if k != "timestamp"} if isinstance(row, dict) else row
        for row in tail
    ]
    assert asw.decide_prompt_wedge_reason(stripped, 3, now=_wedge_1092_anchor_ts() + 10**9) is None
    # Future-dated anchor: `now` BEFORE the newest row ts (negative silence).
    assert asw.decide_prompt_wedge_reason(tail, 3, now=_wedge_1092_anchor_ts() - 100) is None


def test_tick_wedge_dead_silence_env_helper(monkeypatch):
    # #1209 T8: minutes-env -> seconds; 0 DISABLES (the trigger-arm window);
    # malformed / negative -> default.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_TICK_WEDGE_DEAD_SILENCE_MIN", raising=False)
    assert asw._tick_wedge_dead_silence_s() == 1200.0
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_SILENCE_MIN", "35")
    assert asw._tick_wedge_dead_silence_s() == 2100.0
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_SILENCE_MIN", "0")
    assert asw._tick_wedge_dead_silence_s() == 0.0
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_SILENCE_MIN", "-5")
    assert asw._tick_wedge_dead_silence_s() == 1200.0
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_SILENCE_MIN", "junk")
    assert asw._tick_wedge_dead_silence_s() == 1200.0


def test_tick_wedge_dead_respawns_per_day_env_helper(monkeypatch):
    # #1209 T9: default 3; override; `<1 -> default` is THIS helper's own
    # addition (do NOT literal-copy _orphan_max_respawns_per_day, which
    # guards only malformed — disabling the trigger is the SILENCE knob's
    # job, never the cap's); malformed -> default.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY", raising=False)
    assert asw._tick_wedge_dead_respawns_per_day() == 3
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY", "5")
    assert asw._tick_wedge_dead_respawns_per_day() == 5
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY", "0")
    assert asw._tick_wedge_dead_respawns_per_day() == 3
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY", "-2")
    assert asw._tick_wedge_dead_respawns_per_day() == 3
    monkeypatch.setenv("EPM_TICK_WEDGE_DEAD_RESPAWNS_PER_DAY", "junk")
    assert asw._tick_wedge_dead_respawns_per_day() == 3


def test_decide_prompt_wedge_dead_silence_armed_with_turn_knobs_zero():
    # #1209 T16: with BOTH #1127 turn knobs at 0 but the silence trigger
    # armed (dead_silence_s > 0 + explicit now), the predicate's turn-lane
    # early exit must still run _segment_wake_turns and fire — pins the
    # modified early exit in decide_prompt_wedge_reason /
    # _wedge_turn_lane_reason (the override-side fresh-path gate is
    # deliberately UNCHANGED and is pinned by the production-thread wiring
    # tests).
    import autonomous_session_watch as asw

    tail = _wedge_1092_tail()
    now = _wedge_1092_anchor_ts() + 21 * 60
    assert (
        asw.decide_prompt_wedge_reason(
            tail, 0, min_api_errors=0, min_failed_turns=0, min_failed_total=0, now=now
        )
        == "failed-turn-silence"
    )


def test_decide_prompt_wedge_dead_silence_implicit_turn_residual_pinned():
    # #1209 T17 — the ACCEPTED tail-truncation residual (plan §4.2),
    # deliberately pinned so any future guard change fails visibly: a tail
    # that is ONLY a leading implicit turn (response rows with the turn
    # start cut off by the 256 KB window — here a lone api-error row)
    # segments to one implicit FAILED turn and FIRES after the silence
    # window. Rationale: the session's last visible turn genuinely failed
    # and it has been silent >= 20 min, so the bounded fresh-context
    # respawn is the accepted recovery (CLAUDE.md refusal-ladder (f));
    # bounded by the day cap + the stop-first fence.
    import autonomous_session_watch as asw

    tail = [_wedge_api_error_row()]
    assert [o for o, _ts in asw._segment_wake_turns(tail)] == ["failed"]
    assert (
        asw.decide_prompt_wedge_reason(tail, 3, now=_wedge_1092_anchor_ts() + 21 * 60)
        == "failed-turn-silence"
    )


def test_decide_stale_registration_matrix():
    # #845 (d): the per-entry decision table (the #665 replay + every
    # fail-toward-keep guard).
    import autonomous_session_watch as asw

    f = asw.decide_stale_registration
    t = 12 * 3600
    sixteen_h = 16 * 3600
    # #665 replay: live sid, 16h transcript idle, self-report equally stale.
    assert (
        f(
            sid_alive=True,
            transcript_idle_s=sixteen_h,
            self_report_age_s=sixteen_h,
            idle_threshold_s=t,
        )
        == "unregister"
    )
    # A MISSING self-report (manual session — never self-reports) does not
    # rescue: the transcript idle IS the direct signal.
    assert (
        f(sid_alive=True, transcript_idle_s=sixteen_h, self_report_age_s=None, idle_threshold_s=t)
        == "unregister"
    )
    # Dead sid: the crash-recovery pass's property -> keep registered.
    assert (
        f(
            sid_alive=False,
            transcript_idle_s=sixteen_h,
            self_report_age_s=sixteen_h,
            idle_threshold_s=t,
        )
        == "keep"
    )
    # Unresolvable transcript -> keep (fail toward keep).
    assert (
        f(sid_alive=True, transcript_idle_s=None, self_report_age_s=sixteen_h, idle_threshold_s=t)
        == "keep"
    )
    # Fresh transcript OR fresh self-report -> keep.
    assert (
        f(sid_alive=True, transcript_idle_s=60.0, self_report_age_s=sixteen_h, idle_threshold_s=t)
        == "keep"
    )
    assert (
        f(sid_alive=True, transcript_idle_s=sixteen_h, self_report_age_s=60.0, idle_threshold_s=t)
        == "keep"
    )


def test_stale_registration_idle_env_override(monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_STALE_REGISTRATION_IDLE_H", raising=False)
    assert asw._stale_registration_idle_s() == float(asw.STALE_REGISTRATION_IDLE_S)
    monkeypatch.setenv("EPM_STALE_REGISTRATION_IDLE_H", "6")
    assert asw._stale_registration_idle_s() == 6 * 3600.0
    monkeypatch.setenv("EPM_STALE_REGISTRATION_IDLE_H", "-1")
    assert asw._stale_registration_idle_s() == float(asw.STALE_REGISTRATION_IDLE_S)
    monkeypatch.setenv("EPM_STALE_REGISTRATION_IDLE_H", "garbled")
    assert asw._stale_registration_idle_s() == float(asw.STALE_REGISTRATION_IDLE_S)


def test_watcher_note_sentinels_contains_845_sentinels():
    # Drift-prevention pin: the #845 stop-failed + stale-registration marker
    # notes must never reset the staleness clocks they inform.
    import autonomous_session_watch as asw

    assert asw._STALLED_STOP_FAILED_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    assert asw._STALE_REGISTRATION_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


def test_stalled_hardening_fields_legacy_defaults_and_advancement_clear():
    # Backward compat: a pre-#845 on-disk file (no hardening keys) loads the
    # defaults; a fully-populated file carries forward; self-report
    # advancement clears EVERYTHING (episode over).
    import autonomous_session_watch as asw

    legacy = {"missed": 2, "alerted": True}
    assert asw._stalled_hardening_fields(legacy, advanced=False) == dict(
        asw._STALLED_HARDENING_DEFAULTS
    )
    populated = {
        "stop_pending_sid": "sid-a",
        "stop_pending_ts": 5.0,
        "stop_retried": True,
        "stop_failed_alerted": True,
        "wt_hold_count": 3,
        "daemon_blocked_ticks": 2,
        "daemon_blocked_pushed": True,
        "wedge_hits": 1,
    }
    carried = asw._stalled_hardening_fields(populated, advanced=False)
    assert carried == populated
    assert asw._stalled_hardening_fields(populated, advanced=True) == dict(
        asw._STALLED_HARDENING_DEFAULTS
    )
    # Garbled field types fall back to the defaults (never a crash).
    garbled = {"stop_pending_sid": 42, "wt_hold_count": "three", "stop_retried": "yes"}
    fields = asw._stalled_hardening_fields(garbled, advanced=False)
    assert fields["stop_pending_sid"] is None
    assert fields["wt_hold_count"] == 0
    assert fields["stop_retried"] is True  # truthy string -> bool() semantics


# ─── #1071 — evidence-based alert reasons + post-stop hold gate ───────────────
#
# Both #813 incidents (2026-07-03/04) had daemon_reachable=True on every tick,
# yet the alert marker said "Happy daemon unreachable" — the pre-#759 reason
# ladder's else-branch fabricated a daemon outage for the live-corroboration
# downgrade, and the resulting "stop+respawn manually" instruction produced two
# manual respawns racing an in-flight auto-recovery. These tests pin the
# evidence-based ladder (reason + truthful next-step per branch), the
# corroboration helper's downgrade_note, the daemon-outage retry semantics
# (persisted `alerted` + eligibility flip == execute-on-next-reachable-tick),
# and the post-stop worktree-hold gate.


def _make_stalled_ctx(asw, **overrides):
    """Construct a minimal ``_StalledActionCtx`` for direct handler tests.

    Defaults model the #813 shape: autonomous entry, ACTIVE status, no pod,
    ``dry_run=True`` (no state writes; the marker post is monkeypatched by the
    caller). Override per test."""
    kwargs = dict(
        issue=813,
        happy_session_id="sess-813",
        prev_state={},
        alerted=False,
        respawn_count=0,
        exhausted=False,
        last_self_report_ts="2026-07-03T20:00:00Z",
        self_gap="131.8m",
        marker_gap="131.8m",
        has_pod=False,
        task_status="running",
        in_active=True,
        threshold=2,
        dry_run=True,
        manual=False,
    )
    kwargs.update(overrides)
    return asw._StalledActionCtx(**kwargs)


def test_stalled_alert_reason_corroboration_downgrade_not_daemon(monkeypatch):
    # Acceptance criterion 1: a corroboration-downgraded alert names the
    # debounce (episode n/K + the auto-escalation next-step) and NEVER
    # contains "daemon unreachable" while daemon_reachable=True. Full
    # note-CONTENT assertions, not just the sentinel label.
    import autonomous_session_watch as asw

    posts: list[str] = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: posts.append(note)
    )
    downgrade = (
        "live-session corroboration debounce: consecutive live-stall "
        "episode 1/2; the session id is still in the daemon live set"
    )
    ctx = _make_stalled_ctx(asw, daemon_reachable=True, downgrade_note=downgrade)
    asw._handle_stalled_alert(ctx)

    assert len(posts) == 1
    note = posts[0]
    assert asw._STALLED_ALERT_NOTE_SENTINEL in note
    assert "live-session corroboration debounce" in note
    assert "episode 1/2" in note
    assert "auto-escalates" in note
    assert "daemon unreachable" not in note
    assert "watcher bug" not in note
    # The manual-intervention invitation that raced the auto-recovery in both
    # #813 incidents survives ONLY in the manual branch.
    assert "stop+respawn manually" not in note


def test_stalled_alert_reason_daemon_outage_states_auto_retry(monkeypatch):
    # Acceptance criterion 2, BOTH halves: a genuine daemon-outage alert
    # states the persisted auto-retry ("next daemon-reachable tick") AND the
    # #845 (c) phone-push escalation clause ("2 blocked ticks").
    import autonomous_session_watch as asw

    posts: list[str] = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: posts.append(note)
    )
    ctx = _make_stalled_ctx(asw, daemon_reachable=False, downgrade_note=None)
    asw._handle_stalled_alert(ctx)

    assert len(posts) == 1
    note = posts[0]
    assert "Happy daemon unreachable" in note
    assert "next daemon-reachable tick" in note
    assert "2 blocked ticks" in note
    assert "watcher bug" not in note
    assert "stop+respawn manually" not in note


def test_stalled_alert_unexpected_cause_flags_bug(monkeypatch):
    # A future alert producer that reaches the handler without evidence
    # (non-manual, ACTIVE, daemon up, no downgrade note) must self-identify
    # as a watcher bug instead of fabricating a daemon outage (the exact
    # failure mode behind this task's incorrect premise).
    import autonomous_session_watch as asw

    posts: list[str] = []
    monkeypatch.setattr(
        asw, "_post_progress_marker", lambda issue, note, dry_run, label: posts.append(note)
    )
    ctx = _make_stalled_ctx(asw, daemon_reachable=True, downgrade_note=None)
    asw._handle_stalled_alert(ctx)

    assert len(posts) == 1
    assert "watcher bug" in posts[0]
    assert "daemon unreachable" not in posts[0]


def test_daemon_outage_remediation_executes_on_recovery_tick():
    # Goal part 2's "queued and executed on the next daemon-reachable tick"
    # semantics, pinned as a two-tick decide() SEQUENCE (#1071): tick 1 — the
    # threshold trips while the daemon is DOWN (respawn_eligible=False), so
    # the episode fires the ALERT (the caller persists alerted=True); tick 2
    # — the first reachable tick (respawn_eligible=True) escalates the
    # alerted episode straight to the respawn, no re-accumulation. Extends
    # (does not duplicate) the single-call #506-branch assertions in
    # test_alerted_escalates_to_respawn_when_eligible above: the persisted
    # `alerted` flag IS the queue — nothing is dropped during an outage.
    stale = STALLED_MARKER_WINDOW_S_DEFAULT + 60
    # Tick 1: threshold trip, daemon down -> alert.
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=1,
        alerted=False,
        respawn_eligible=False,
        threshold=2,
    ) == ("alert", 0)
    # Tick 2: daemon back -> the alerted episode respawns the same tick.
    assert decide_session_stalled(
        self_report_age_s=stale,
        marker_progress_age_s=stale,
        has_pod=True,
        missed=0,
        alerted=True,
        respawn_eligible=True,
        respawn_count=0,
        threshold=2,
    ) == ("respawn", 0)


def test_corroboration_returns_downgrade_note(isolated_registry, monkeypatch):
    # Arity pin for the #1071 triple return: the downgrade branch returns a
    # note carrying the {live_consecutive}/{k} episode fragment (criterion
    # 1's n/K disclosure); the escalation / dead-sid / daemon-down /
    # other-action branches all return None.
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "2")
    entry = {"issue": 5, "happy_session_id": "sid-live"}

    # Downgrade branch (live sid, episode 1 < K=2).
    action, live, note = asw._apply_stalled_live_corroboration(
        issue=5,
        entry=entry,
        action="respawn",
        daemon_reachable=True,
        live_ids={"sid-live"},
        live_consecutive=0,
        dry_run=True,
    )
    assert (action, live) == ("alert", 1)
    assert note is not None
    assert "live-session corroboration debounce" in note
    assert "1/2" in note

    # Escalation branch (Kth consecutive live stall) -> respawn, no note.
    assert asw._apply_stalled_live_corroboration(
        issue=5,
        entry=entry,
        action="respawn",
        daemon_reachable=True,
        live_ids={"sid-live"},
        live_consecutive=1,
        dry_run=True,
    ) == ("respawn", 0, None)

    # Dead-sid branch -> respawn passes through, no note.
    assert asw._apply_stalled_live_corroboration(
        issue=5,
        entry=entry,
        action="respawn",
        daemon_reachable=True,
        live_ids={"sid-other"},
        live_consecutive=1,
        dry_run=True,
    ) == ("respawn", 0, None)

    # Daemon-down branch -> respawn passes through, no note.
    assert asw._apply_stalled_live_corroboration(
        issue=5,
        entry=entry,
        action="respawn",
        daemon_reachable=False,
        live_ids=None,
        live_consecutive=1,
        dry_run=True,
    ) == ("respawn", 0, None)

    # Non-respawn action -> passthrough + counter reset, no note.
    assert asw._apply_stalled_live_corroboration(
        issue=5,
        entry=entry,
        action="keep",
        daemon_reachable=True,
        live_ids={"sid-live"},
        live_consecutive=1,
        dry_run=True,
    ) == ("keep", 0, None)


def test_wt_hold_skipped_when_fence_stop_pending(isolated_registry, monkeypatch):
    # Acceptance criterion 4, both halves. (a) Once stop_pending_sid is set
    # a stop has already been ISSUED — the #812 mid-edit hold protects the
    # STOP, so holding post-stop only delays the fence's verified-dead spawn
    # (incident #813: 3 held ticks post-stop, issue driverless). The hold is
    # skipped mid-fence, still applied pre-stop (persisting wt_hold_count).
    # (b) The spawn-grace skip stays UNCONDITIONAL — it guards concurrent
    # respawns, which is still valid mid-fence.
    import autonomous_session_watch as asw

    monkeypatch.delenv("EPM_RESPAWN_SPAWN_GRACE_MIN", raising=False)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda *_a, **_k: True)
    now = 1_000_000.0

    # Mid-fence (stop issued): hold does NOT apply — the arm proceeds.
    ctx = _make_stalled_ctx(
        asw, dry_run=False, now=now, stop_pending_sid="sidX", entry_spawned_at=None
    )
    assert asw._stalled_arm_deferral(ctx) is False

    # Pre-stop: hold applies and persists wt_hold_count=1.
    ctx = _make_stalled_ctx(
        asw, dry_run=False, now=now, stop_pending_sid=None, entry_spawned_at=None
    )
    assert asw._stalled_arm_deferral(ctx) is True
    state = json.loads((isolated_registry / f"{STALLED_STATE_PREFIX}813.json").read_text())
    assert state["wt_hold_count"] == 1

    # Mid-grace AND mid-fence: the spawn-grace skip is unconditional.
    ctx = _make_stalled_ctx(
        asw, dry_run=False, now=now, stop_pending_sid="sidX", entry_spawned_at=now - 60.0
    )
    assert asw._stalled_arm_deferral(ctx) is True


def test_stalled_pass_corroboration_downgrade_note_threaded(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
    # r1 Must-Fix (production-path threading, corroboration): drive the REAL
    # stalled_session_pass so the #759 downgrade fires through the live
    # _apply_stalled_live_corroboration call site and the _StalledActionCtx
    # construction. A threading miss there (downgrade_note not passed into
    # the ctx) falls through to the else-branch and posts "watcher bug";
    # the pre-#1071 code posted "daemon unreachable". Both are asserted
    # absent; the debounce reason is asserted present.
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str, str]] = []
    respawns: list[int] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    monkeypatch.setenv("EPM_STALLED_LIVE_ESCALATION_K", "2")
    stale = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    # ACTIVE status + reachable daemon + live sid: decide() wants a respawn;
    # the corroboration downgrades episode 1/2 to an alert.
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    # Make a wrong-path respawn observable (mirror of the #661 test's stubs).
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: True)
    monkeypatch.setattr(
        asw,
        "_respawn_stalled_session",
        lambda issue, cap, dry_run: respawns.append(issue) or "spawned",
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, note, label)),
    )

    asw.stalled_session_pass(
        dry_run=False, threshold=1, now=now, daemon_reachable=True, live_ids={"sess-489"}
    )

    assert respawns == []
    assert len(posts) == 1
    issue, note, label = posts[0]
    assert issue == 489
    assert label == "session-stalled-alert"
    assert "live-session corroboration debounce" in note
    assert "episode 1/2" in note
    assert "daemon unreachable" not in note
    assert "watcher bug" not in note


def test_stalled_pass_daemon_outage_note_threaded(
    isolated_registry, hermetic_provision_probes, monkeypatch
):
    # r1 Must-Fix (production-path threading, outage): the same pass-level
    # pattern with daemon_reachable=False on an ACTIVE task at the threshold
    # trip — the posted note must state the persisted auto-retry, proving
    # the pass-level daemon_reachable flag is threaded into the ctx (a miss
    # keeps the True default and posts the "watcher bug" else-branch).
    import autonomous_session_watch as asw

    now = 1_000_000.0
    posts: list[tuple[int, str, str]] = []
    respawns: list[int] = []
    _write_autonomous_entry(isolated_registry, 489, "sess-489")

    stale = STALLED_WINDOW_S + 600
    monkeypatch.setattr(
        asw,
        "_self_report_age_seconds",
        lambda issue, now: (stale, "2026-06-05T10:00:00Z"),
    )
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_running_managed_issue_pods", lambda *_a, **_k: [])
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_telegram_push", lambda *_a, **_k: None)
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry_run: True)
    monkeypatch.setattr(
        asw,
        "_respawn_stalled_session",
        lambda issue, cap, dry_run: respawns.append(issue) or "spawned",
    )
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label: posts.append((issue, note, label)),
    )

    asw.stalled_session_pass(
        dry_run=False, threshold=1, now=now, daemon_reachable=False, live_ids=None
    )

    assert respawns == []
    assert len(posts) == 1
    issue, note, label = posts[0]
    assert issue == 489
    assert label == "session-stalled-alert"
    assert "Happy daemon unreachable" in note
    assert "next daemon-reachable tick" in note
    assert "watcher bug" not in note
