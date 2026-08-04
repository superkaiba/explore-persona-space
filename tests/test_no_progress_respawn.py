"""Tests for the #2058 no-progress respawn lane.

Unit A (this file, LANDED): the pure `compute_issue_verdict` extension,
the `_HEARTBEAT_NOTE_SENTINELS` frozenset, and `compute_progress_fingerprint`.
Assertions 1-7 + assertion 23 (heartbeat-sentinel round-trip against
SKILL.md) are IMPLEMENTED here; assertions 8-22 + 24-25 live in the
Unit B follow-up (the watcher-pass predicate + state-file + smoke arm)
and are skipped with an explicit pytest.skip pointing at the unit they
belong to.
"""

from __future__ import annotations

import pytest

from scripts.autonomous_session_watch import (
    _NO_PROGRESS_RESPAWN_CAP_NOTE_SENTINEL,
    _NO_PROGRESS_RESPAWN_NOTE_SENTINEL,
    _WATCHER_NOTE_SENTINELS,
    decide_no_progress_respawn,
)
from scripts.tick_triage import (
    _HEARTBEAT_NOTE_SENTINELS,
    compute_issue_verdict,
    compute_progress_fingerprint,
    latest_nonwatcher_nonheartbeat_ts,
)


def _mk_predicate_kwargs(**overrides) -> dict:
    """Baseline all-positive kwargs for `decide_no_progress_respawn`. Every
    field is at its fire-eligible default; a test overrides just the arm it
    exercises. Threshold = 3, tick_streak = 3 (>= threshold), all vetoes
    off, fingerprint pair matching (no advance)."""
    defaults = {
        "kill_switch": False,
        "status_class": "active",
        "tick_streak": 3,
        "tick_threshold": 3,
        "respawns_today": 0,
        "respawns_per_day": 3,
        "episode_respawn_count": 0,
        "episode_belt_max": 3,
        "worktree_activity_fresh": False,
        "park_exemption_fires": False,
        "daemon_reachable": True,
        "transcript_resolvable": True,
        "live_pod_present": False,
        "prev_fingerprint": "abcd",
        "curr_fingerprint": "abcd",
    }
    defaults.update(overrides)
    return defaults


# ── Unit A — the pure predicate + fingerprint helper ────────────────────────


def _mk_fresh() -> dict:
    """Kwargs common to every age-fresh HEALTHY call."""
    return {
        "status": "running",
        "prev_status": "running",
        "marker_age_s": 60.0,  # 1 min, well within the default 25 min window
        "over_cap": False,
        "stale_after_s": 25 * 60,
    }


def test_fingerprint_unchanged_N_ticks_fires_no_progress_respawn():
    """Assertion 1: age-fresh + fingerprint unchanged N=3 consecutive times
    returns NO-PROGRESS-RESPAWN with streak=3. The fourth tick with the
    same unchanged fingerprint STILL returns the verdict (idempotent —
    the pass's cap gates the ACTION)."""
    fp = "1722900000|null|running"
    # Tick 1: fresh episode, prev_fp seeded equal, streak starts at 0.
    v1, _, s1 = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint=fp,
        prev_fingerprint=fp,
        no_progress_streak=0,
        no_progress_threshold=3,
    )
    assert v1 == "HEALTHY"
    assert s1 == 1

    # Tick 2: still unchanged, streak advances.
    v2, _, s2 = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint=fp,
        prev_fingerprint=fp,
        no_progress_streak=s1,
        no_progress_threshold=3,
    )
    assert v2 == "HEALTHY"
    assert s2 == 2

    # Tick 3: threshold reached ⇒ NO-PROGRESS-RESPAWN, streak=3.
    v3, reason3, s3 = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint=fp,
        prev_fingerprint=fp,
        no_progress_streak=s2,
        no_progress_threshold=3,
    )
    assert v3 == "NO-PROGRESS-RESPAWN"
    assert s3 == 3
    assert "unchanged across 3 ticks" in reason3

    # Tick 4: still unchanged, still emits the verdict (the watcher owns
    # the cap; the pure predicate stays idempotent).
    v4, _, s4 = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint=fp,
        prev_fingerprint=fp,
        no_progress_streak=s3,
        no_progress_threshold=3,
    )
    assert v4 == "NO-PROGRESS-RESPAWN"
    assert s4 == 4


def test_age_stale_returns_stale_redrive_regardless_of_fingerprint():
    """Assertion 2: an age-STALE tick returns STALE-REDRIVE regardless of
    the fingerprint state. Regression: existing behavior preserved."""
    for fp_pair in [(None, None), ("a|b|c", "a|b|c"), ("a|b|c", "different")]:
        current, prev = fp_pair
        v, reason, s = compute_issue_verdict(
            status="running",
            prev_status="running",
            marker_age_s=60 * 60,  # 60 min > 25 min window
            over_cap=False,
            stale_after_s=25 * 60,
            progress_fingerprint=current,
            prev_fingerprint=prev,
            no_progress_streak=99,  # deliberate high value; STALE resets to 0
            no_progress_threshold=3,
        )
        assert v == "STALE-REDRIVE"
        assert s == 0, f"STALE-REDRIVE must reset streak (got {s}) for {fp_pair!r}"


def test_fingerprint_changed_returns_healthy_streak_zero():
    """Assertion 3: fingerprint that ADVANCED returns HEALTHY with streak=0
    (regression: real progress preserved and resets any accumulated streak)."""
    v, _, s = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint="new|sha|running",
        prev_fingerprint="old|sha|running",
        no_progress_streak=2,  # deliberate mid-streak reset
        no_progress_threshold=3,
    )
    assert v == "HEALTHY"
    assert s == 0


def test_none_fingerprint_returns_healthy_streak_zero():
    """Assertion 4: progress_fingerprint or prev_fingerprint None returns
    HEALTHY with streak=0 (fail-open when fingerprint uncomputable — first
    tick of an episode, or a git failure zero'd the sha)."""
    # Current None (fingerprint uncomputable this tick).
    v1, _, s1 = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint=None,
        prev_fingerprint="a|b|c",
        no_progress_streak=2,
        no_progress_threshold=3,
    )
    assert v1 == "HEALTHY"
    assert s1 == 0

    # Prev None (first-tick episode).
    v2, _, s2 = compute_issue_verdict(
        **_mk_fresh(),
        progress_fingerprint="a|b|c",
        prev_fingerprint=None,
        no_progress_streak=0,
        no_progress_threshold=3,
    )
    assert v2 == "HEALTHY"
    assert s2 == 0


def test_terminal_gate_gate_transition_ignore_fingerprint():
    """Assertion 5: TERMINAL / GATE / GATE-TRANSITION verdicts return
    regardless of fingerprint state (regression: the teardown/gate branches
    are unaffected by the no-progress arm)."""
    fp_same = ("x|y|z", "x|y|z")
    # TERMINAL: task at completed, marker age N/A.
    v_t, _, s_t = compute_issue_verdict(
        status="completed",
        prev_status="completed",
        marker_age_s=None,
        over_cap=False,
        stale_after_s=25 * 60,
        progress_fingerprint=fp_same[0],
        prev_fingerprint=fp_same[1],
        no_progress_streak=99,
        no_progress_threshold=3,
    )
    assert v_t == "TERMINAL"
    assert s_t == 0

    # GATE-TRANSITION: reaching awaiting_promotion from a non-gate prev.
    v_g, _, s_g = compute_issue_verdict(
        status="awaiting_promotion",
        prev_status="reviewing",
        marker_age_s=60.0,
        over_cap=False,
        stale_after_s=25 * 60,
        progress_fingerprint=fp_same[0],
        prev_fingerprint=fp_same[1],
        no_progress_streak=99,
        no_progress_threshold=3,
    )
    assert v_g == "GATE-TRANSITION"
    assert s_g == 0

    # Over-cap plan_pending: TERMINAL (or GATE-TRANSITION when new).
    v_p, _, _ = compute_issue_verdict(
        status="plan_pending",
        prev_status="planning",
        marker_age_s=60.0,
        over_cap=True,
        stale_after_s=25 * 60,
        progress_fingerprint=fp_same[0],
        prev_fingerprint=fp_same[1],
    )
    assert v_p == "GATE-TRANSITION"


def test_heartbeat_note_sentinels_membership():
    """Assertion 6: `_HEARTBEAT_NOTE_SENTINELS` membership — the three
    canonical heartbeat prefixes are all members. Pinning this set is what
    the fingerprint helper reads to distinguish durable markers from
    slow-phase heartbeats."""
    assert "tick heartbeat:" in _HEARTBEAT_NOTE_SENTINELS
    assert "[long-phase-heartbeat]" in _HEARTBEAT_NOTE_SENTINELS
    assert "progress: none" in _HEARTBEAT_NOTE_SENTINELS


def test_progress_token_short_circuits_to_advance():
    """Assertion 7: the fingerprint helper reads a `progress: <not-none>`
    line inside a heartbeat's note as an ADVANCE signal (short-circuits the
    ts+sha+status computation). Synthesize a heartbeat row carrying
    `progress: commit=abc123` and confirm the helper returns a fingerprint
    containing the payload."""
    events = [
        {
            "ts": "2026-08-04T05:00:00Z",
            "kind": "epm:progress",
            "note": "tick heartbeat: job verified alive (pid 12345, log mtime foo)\nprogress: commit=abc123def456",
        }
    ]
    fp = compute_progress_fingerprint(events, head_sha="feedface", status="running")
    assert fp is not None
    assert "commit=abc123def456" in fp

    # A `progress: none` heartbeat DOES NOT advance the fingerprint (it is
    # the explicit no-durable-work declaration).
    events_none = [
        {
            "ts": "2026-08-04T05:00:00Z",
            "kind": "epm:progress",
            "note": "tick heartbeat: job verified alive (pid 12345, log mtime foo)\nprogress: none",
        }
    ]
    fp_none = compute_progress_fingerprint(events_none, head_sha="feedface", status="running")
    # None progress token ⇒ falls through to the marker-ts + sha + status
    # computation. Because the only event is a heartbeat, the
    # non-heartbeat ts is None, so the returned fingerprint carries
    # `none|<sha>|<status>`.
    assert fp_none == "none|feedface|running"


def test_latest_nonwatcher_nonheartbeat_ts_filters_heartbeats():
    """Regression: the helper that feeds the fingerprint's marker-ts
    component excludes heartbeat-class notes."""
    events = [
        {
            "ts": "2026-08-04T04:00:00Z",
            "kind": "epm:experiment-implementation",
            "note": "Unit A landed",
        },
        {
            "ts": "2026-08-04T05:00:00Z",
            "kind": "epm:progress",
            "note": "tick heartbeat: job verified alive (pid 1, log mtime 2)",
        },
        {
            "ts": "2026-08-04T06:00:00Z",
            "kind": "epm:progress",
            "note": "[long-phase-heartbeat] phase=analysis_fit still computing",
        },
    ]
    ts = latest_nonwatcher_nonheartbeat_ts(events)
    # The 04:00 durable marker wins; the 05:00 tick heartbeat and 06:00
    # long-phase heartbeat are filtered.
    assert ts is not None
    import datetime as _dt

    expected = _dt.datetime(2026, 8, 4, 4, 0, 0, tzinfo=_dt.UTC).timestamp()
    assert abs(ts - expected) < 1.0


# ── Assertion 23 — heartbeat-sentinel round-trip against SKILL.md ────────────


def test_heartbeat_sentinel_set_matches_skill_md():
    """Assertion 23: pin `_HEARTBEAT_NOTE_SENTINELS` to the ACTUAL heartbeat
    strings the /issue-tick skill emits. This surfaces drift the moment the
    heartbeat wording changes in the SKILL — update BOTH surfaces together.

    The two SKILL.md-quoted heartbeat surfaces are:
      * The /issue-tick ACTIVE-status slow-phase heartbeat, currently:
        ``tick heartbeat: job verified alive (pid <pid>, log mtime <ts>);
        slow phase, no state change``
      * The detached-VM long-phase heartbeat, prefixed
        ``[long-phase-heartbeat]``.

    The third member, ``progress: none``, is the canonical no-durable-work
    token the /issue-tick heartbeat emits under the #2058 Unit C SKILL.md
    extension — it is a token, not a heartbeat leader, so its round-trip
    check is against SKILL.md's Unit C documentation of the token.
    """
    # Locate the worktree's SKILL.md (repo-root-relative resolution — the
    # test may run from repo root or from the worktree).
    from explore_persona_space.task_workflow import repo_root

    skill = repo_root() / ".claude" / "skills" / "issue-tick" / "SKILL.md"
    assert skill.exists(), f"SKILL.md not found at {skill!s}"
    text = skill.read_text(encoding="utf-8")

    # `tick heartbeat:` — the exact SKILL.md quoted heartbeat lead.
    assert "tick heartbeat:" in text, (
        "SKILL.md no longer emits `tick heartbeat:` — update _HEARTBEAT_NOTE_SENTINELS."
    )
    # `[long-phase-heartbeat]` — the detached-VM heartbeat prefix.
    # (Absent-from-SKILL is TOLERATED here: the /issue-tick SKILL owns the
    # tick heartbeat; the long-phase heartbeat is emitted by other
    # detached-VM code paths under a different SKILL — the sentinel-set
    # member still binds via `LONG_PHASE_HEARTBEAT_PREFIX` in tick_triage.)
    # This clause pins the intent: the sentinel token is what those emit
    # sites use; sibling-SKILL edits must keep the token stable.

    # The `progress: none` token: SKILL.md Unit C landing must document it.
    # Until Unit C ships, this arm is SKIPPED — the sentinel member is
    # correct-by-construction (planned) but not yet SKILL.md-visible.
    if "progress: none" not in text:
        pytest.skip(
            "Unit C (SKILL.md heartbeat extension documenting `progress: none`) "
            "has not landed yet; the sentinel member is correct-by-construction "
            "for the Unit A landing round."
        )


# ── Unit B assertions — deferred to the watcher-pass follow-up ──────────────


def test_decide_no_progress_respawn_fires_after_n_ticks():
    """Assertion 8 (durability pin — the plan §10 Reproducibility Card
    names this test as the durability pin). All-positive kwargs at the
    tick_streak >= tick_threshold boundary must return ``("respawn", ...)``.
    """
    action, reason = decide_no_progress_respawn(**_mk_predicate_kwargs())
    assert action == "respawn", (action, reason)
    # Idempotent at streak > threshold too — the predicate keeps firing.
    action2, _ = decide_no_progress_respawn(**_mk_predicate_kwargs(tick_streak=10))
    assert action2 == "respawn", action2


def test_decide_no_progress_respawn_holds_on_worktree_activity():
    """Assertion 9 — an implementer mid-edit freezes the streak, not fires."""
    action, reason = decide_no_progress_respawn(
        **_mk_predicate_kwargs(worktree_activity_fresh=True)
    )
    assert action == "hold" and "worktree" in reason, (action, reason)


def test_decide_no_progress_respawn_holds_on_daemon_unreachable():
    """Assertion 10 — a daemon-probe failure (None) freezes the streak."""
    action, reason = decide_no_progress_respawn(**_mk_predicate_kwargs(daemon_reachable=None))
    assert action == "hold" and "daemon" in reason, (action, reason)


def test_decide_no_progress_respawn_holds_on_unresolvable_transcript():
    """Assertion 11 — a transcript-resolver failure (None) freezes the streak."""
    action, reason = decide_no_progress_respawn(**_mk_predicate_kwargs(transcript_resolvable=None))
    assert action == "hold" and "transcript" in reason, (action, reason)


def test_decide_no_progress_respawn_holds_on_park_exemption():
    """Assertion 12 — any park-exemption re-probe at act time freezes."""
    action, reason = decide_no_progress_respawn(**_mk_predicate_kwargs(park_exemption_fires=True))
    assert action == "hold" and "park exemption" in reason, (action, reason)


def test_decide_no_progress_respawn_clears_on_fingerprint_advance():
    """Assertion 13 — a fingerprint that ADVANCED since the tick's read
    ENDS the episode (streak resets), even if the tick verdict said fire."""
    action, reason = decide_no_progress_respawn(
        **_mk_predicate_kwargs(prev_fingerprint="abcd", curr_fingerprint="EFGH")
    )
    assert action == "clear" and "fingerprint" in reason, (action, reason)


def test_decide_no_progress_respawn_cap_exhausted():
    """Assertion 14 — fire conditions ALL hold but the per-UTC-day cap is
    exhausted (respawns_today >= respawns_per_day) => ``"cap-exhausted"``.
    Also covers the episode-belt exhaustion arm."""
    day_cap, _ = decide_no_progress_respawn(
        **_mk_predicate_kwargs(respawns_today=3, respawns_per_day=3)
    )
    assert day_cap == "cap-exhausted", day_cap
    belt_cap, _ = decide_no_progress_respawn(
        **_mk_predicate_kwargs(episode_respawn_count=3, episode_belt_max=3)
    )
    assert belt_cap == "cap-exhausted", belt_cap


def test_decide_no_progress_respawn_kill_switch():
    """Assertion 15 — ``kill_switch=True`` returns ``("clear", "kill switch")``
    unconditionally, regardless of every other input."""
    action, reason = decide_no_progress_respawn(**_mk_predicate_kwargs(kill_switch=True))
    assert action == "clear" and "kill switch" in reason, (action, reason)


def test_decide_no_progress_respawn_park_status_act_guard():
    """Assertion 16 (#1247) — a non-active live-status re-read at act time
    ends the episode without firing (the park-status act guard)."""
    for status in ("park", "terminal", "unknown"):
        action, reason = decide_no_progress_respawn(**_mk_predicate_kwargs(status_class=status))
        assert action == "clear" and "park-status" in reason, (status, action, reason)


def test_no_progress_markers_in_watcher_note_sentinels():
    """Assertion 17 — both no-progress-respawn note sentinels are members
    of ``_WATCHER_NOTE_SENTINELS`` so they never reset the very
    staleness/orphan-progress clocks the pass measures against."""
    assert _NO_PROGRESS_RESPAWN_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS
    assert _NO_PROGRESS_RESPAWN_CAP_NOTE_SENTINEL in _WATCHER_NOTE_SENTINELS


def test_decide_no_progress_respawn_holds_on_live_pod():
    """Assertion 24 — a live workload's orchestrator is a healthy monitor
    and a ``progress: none`` heartbeat is expected there; freeze the streak
    (never respawn against a live pod / instance). Unresolvable pod probe
    (None) freezes for the same reason."""
    live, reason = decide_no_progress_respawn(**_mk_predicate_kwargs(live_pod_present=True))
    assert live == "hold" and "live-pod" in reason, (live, reason)
    unresolvable, _ = decide_no_progress_respawn(**_mk_predicate_kwargs(live_pod_present=None))
    assert unresolvable == "hold", unresolvable


@pytest.mark.skip(reason="Unit B: NEVER-MUTATES-STATUS invariant argv-spy")
def test_no_progress_respawn_never_mutates_status():
    """Assertion 18."""


@pytest.mark.skip(reason="Unit B: idempotency — one bump per episode")
def test_no_progress_respawn_bumps_respawns_today_once_per_episode():
    """Assertion 19."""


@pytest.mark.skip(reason="Unit B: state-file GC at TERMINAL_FOR_GC only")
def test_no_progress_state_file_gc_terminal_only():
    """Assertion 20."""


@pytest.mark.skip(reason="Unit B: sidecar row shape")
def test_no_progress_sidecar_row_shape():
    """Assertion 21."""


@pytest.mark.skip(reason="Unit B: dry-run kwarg thread — zero side effects")
def test_no_progress_respawn_dry_run_writes_nothing():
    """Assertion 22."""


@pytest.mark.skip(reason="Unit B: degraded-key (sha-null) freeze — streak not reset, not advanced")
def test_no_progress_respawn_freezes_on_sha_null_transition():
    """Assertion 25."""
