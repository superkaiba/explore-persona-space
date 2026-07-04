"""Tests for the per-stage dispatch dedup predicate (#820, #778 replay).

``stage_dispatch_should_skip`` is the mechanical form of the /issue Step 9
entry guard's per-stage backwards scan: the most recent ``stage-dispatch``
breadcrumb matching the queried raw stage token + round is in flight unless a
stage-matching result marker (or ``epm:failure``) landed after it, with a
liveness-refreshed freshness window bounding staleness. The positive fixture
replays the #778 incident (events.jsonl lines 116-130): a second orchestrator
posted a byte-identical ``stage=followup-implementing round=1`` breadcrumb at
22:34:32Z because 8 intervening markers buried the 22:28:53Z original.
"""

from __future__ import annotations

from datetime import datetime

import explore_persona_space.task_workflow as tw


def _ev(ts: str, kind: str, note: str = "", version: int = 1, by: str = "test") -> dict:
    return {"ts": ts, "kind": kind, "version": version, "by": by, "note": note}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


_CRUMB_778 = (
    "stage-dispatch stage=followup-implementing round=1 subagent=experiment-implementer "
    "worktree=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-778"
)


def test_778_replay_duplicate_skipped():
    events = [
        _ev("2026-07-01T22:28:53Z", "epm:progress", _CRUMB_778, version=28),
        _ev(
            "2026-07-01T22:28:54Z",
            "epm:codex-task-spawned",
            "Codex job_id=task-mr2ndn0r-zboe5s effort=high write=True poll_interval=30s",
        ),
        _ev(
            "2026-07-01T22:29:25Z",
            "epm:codex-task-completed",
            "Codex job_id=task-mr2ndn0r-zboe5s phase=done after 30s.",
        ),
        _ev(
            "2026-07-01T22:30:44Z",
            "epm:codex-task-spawned",
            "Codex job_id=task-mr2ng0g6-0xj291 effort=high write=True poll_interval=30s",
        ),
        _ev(
            "2026-07-01T22:31:16Z",
            "epm:codex-task-completed",
            "Codex job_id=task-mr2ng0g6-0xj291 phase=done after 30s.",
        ),
        _ev(
            "2026-07-01T22:31:46Z",
            "epm:progress",
            "corrected-monitoring-8prompt-ladder round 1: plan + existing #778 drivers read; "
            "corrected_eval_prompts.md located in worktree. Starting implementation",
            version=29,
        ),
        _ev("2026-07-01T22:31:56Z", "epm:smoke-architecture-check", "verdict: PASS_UNIFIED", 2),
        _ev(
            "2026-07-01T22:33:50Z",
            "epm:plan",
            "Plan v5 (amendment for same-issue follow-up round corrected-monitoring-8prompt-"
            "ladder) written to plans/plan.md (gpu_hours_total=4).",
            version=3,
        ),
        _ev(
            "2026-07-01T22:33:58Z",
            "epm:plan-approved",
            "Auto-approved by the code-enforced autonomous plan-gate: gpu_hours_total=4.0",
        ),
    ]
    reason = tw.stage_dispatch_should_skip(
        events, "followup-implementing", 1, 15, now=_dt("2026-07-01T22:34:32Z")
    )
    assert reason is not None
    assert "followup-implementing" in reason


def test_stalled_past_window_allows_redispatch():
    events = [_ev("2026-07-01T10:00:00Z", "epm:progress", _CRUMB_778)]
    assert (
        tw.stage_dispatch_should_skip(
            events, "followup-implementing", 1, 15, now=_dt("2026-07-01T10:16:00Z")
        )
        is None
    )


def test_liveness_refresh_extends_then_expires():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev("2026-07-01T10:25:00Z", "epm:codex-task-completed", "Codex job phase=done"),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:35:00Z")
        )
        is not None
    )
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:41:00Z")
        )
        is None
    )


def test_concurrent_stages_independent():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=verifying round=1 subagent=upload-verifier",
        ),
        _ev(
            "2026-07-01T10:00:05Z",
            "epm:progress",
            "stage-dispatch stage=interpreting round=1 subagent=analyzer",
        ),
        _ev("2026-07-01T10:05:00Z", "epm:upload-verification", "verdict: PASS"),
    ]
    now = _dt("2026-07-01T10:06:00Z")
    assert tw.stage_dispatch_should_skip(events, "verifying", 1, 15, now=now) is None
    assert tw.stage_dispatch_should_skip(events, "interpreting", 1, 15, now=now) is not None


def test_round_boundary():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev("2026-07-01T10:10:00Z", "epm:experiment-implementation", "round 1 landed"),
    ]
    now = _dt("2026-07-01T10:12:00Z")
    assert tw.stage_dispatch_should_skip(events, "implementing", 2, 15, now=now) is None
    assert tw.stage_dispatch_should_skip(events, "implementing", 1, 15, now=now) is None


def test_failure_clears_any_stage():
    events = [
        _ev("2026-07-01T10:00:00Z", "epm:progress", "stage-dispatch stage=p6-judge round=1"),
        _ev("2026-07-01T10:05:00Z", "epm:failure", "failure_class: infra"),
    ]
    assert (
        tw.stage_dispatch_should_skip(events, "p6-judge", 1, 15, now=_dt("2026-07-01T10:06:00Z"))
        is None
    )


def test_quoted_breadcrumb_not_matched():
    quoting_note = (
        "DUAL-DISPATCH DETECTED — implementer #2 yielding. Two 'stage-dispatch "
        "stage=followup-implementing round=1 subagent=experiment-implementer' markers "
        "were posted for the SAME round"
    )
    crumb = _ev("2026-07-01T22:00:00Z", "epm:progress", _CRUMB_778)
    quoting = _ev("2026-07-01T22:20:00Z", "epm:progress", quoting_note, version=32)
    now = _dt("2026-07-01T22:21:00Z")
    # The quoting note is NOT a breadcrumb (anchor = lstripped note STARTS with
    # "stage-dispatch ") but IS a non-breadcrumb progress -> refreshes the window.
    assert (
        tw.stage_dispatch_should_skip([crumb, quoting], "followup-implementing", 1, 15, now=now)
        is not None
    )
    # Without the refreshing note the breadcrumb is 21m old -> stalled -> re-dispatch.
    assert tw.stage_dispatch_should_skip([crumb], "followup-implementing", 1, 15, now=now) is None


def test_followup_prefix_normalized_clearing():
    now = _dt("2026-07-01T10:06:00Z")
    events_a = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=followup-code-reviewing round=1 subagent=code-reviewer",
        ),
        _ev("2026-07-01T10:05:00Z", "epm:code-review", "verdict: PASS"),
    ]
    assert (
        tw.stage_dispatch_should_skip(events_a, "followup-code-reviewing", 1, 15, now=now) is None
    )
    events_b = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=followup-implementing round=1 subagent=experiment-implementer",
        ),
        _ev("2026-07-01T10:05:00Z", "epm:experiment-implementation", "round 1 landed"),
    ]
    assert tw.stage_dispatch_should_skip(events_b, "followup-implementing", 1, 15, now=now) is None


def test_raw_token_dedup_no_cross_match():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        )
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "followup-implementing", 1, 15, now=_dt("2026-07-01T10:01:00Z")
        )
        is None
    )


def test_unknown_token_conservative():
    events = [_ev("2026-07-01T10:00:00Z", "epm:progress", "stage-dispatch stage=p6-judge round=1")]
    assert (
        tw.stage_dispatch_should_skip(events, "p6-judge", 1, 15, now=_dt("2026-07-01T10:10:00Z"))
        is not None
    )
    assert (
        tw.stage_dispatch_should_skip(events, "p6-judge", 1, 15, now=_dt("2026-07-01T10:16:00Z"))
        is None
    )


def test_extra_fields_and_order_robust():
    note = (
        "stage-dispatch round=1 stage=implementing subagent=experiment-implementer worktree=/abs/x"
    )
    events = [_ev("2026-07-01T10:00:00Z", "epm:progress", note)]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:05:00Z")
        )
        is not None
    )


def test_round_missing_never_matches():
    events = [
        _ev("2026-07-01T10:00:00Z", "epm:progress", "stage-dispatch stage=implementing subagent=x")
    ]
    for round_num in (0, 1, 2):
        assert (
            tw.stage_dispatch_should_skip(
                events, "implementing", round_num, 15, now=_dt("2026-07-01T10:01:00Z")
            )
            is None
        )


def test_unrelated_progress_refreshes_window():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev("2026-07-01T10:14:00Z", "epm:progress", "pod-provision waiting on RunPod capacity..."),
    ]
    # Documented over-skip direction: any non-anti-liveness non-breadcrumb progress
    # refreshes the window.
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
        )
        is not None
    )


def test_no_breadcrumb_returns_none():
    events = [
        _ev("2026-07-01T10:00:00Z", "epm:plan", "Plan v1 written"),
        _ev("2026-07-01T10:01:00Z", "epm:progress", "phase tick: waiting on pod"),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:02:00Z")
        )
        is None
    )


def test_547_replay_clean_result_critic_after_analyzer_revision():
    # #547 real sequence: analyzer crumb 13:29:31 -> epm:interpretation (the analyzer's
    # revise output) 13:48:23 -> critic dispatch attempt 13:49:13. The intermediate
    # epm:interpretation is a clearing kind for stage=clean-result (the LAST dispatched
    # subagent finished), so the next in-round dispatch must be ALLOWED.
    events = [
        _ev(
            "2026-06-10T13:29:31Z",
            "epm:progress",
            "stage-dispatch stage=clean-result round=2 subagent=analyzer",
        ),
        _ev("2026-06-10T13:48:23Z", "epm:interpretation", "revise round 2 landed"),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "clean-result", 2, 30, now=_dt("2026-06-10T13:49:13Z")
        )
        is None
    )


def test_547_replay_interpreting_reconciler_after_critique():
    # #547 sibling replay: interpretation-critic crumb -> epm:interp-critique 19m later;
    # the reconciler dispatch 21m in (< 30m window) must be ALLOWED because the critique
    # marker clears the critic's crumb.
    events = [
        _ev(
            "2026-06-10T14:00:00Z",
            "epm:progress",
            "stage-dispatch stage=interpreting round=1 subagent=interpretation-critic",
        ),
        _ev("2026-06-10T14:19:00Z", "epm:interp-critique", "verdict: REVISE"),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "interpreting", 1, 30, now=_dt("2026-06-10T14:21:00Z")
        )
        is None
    )


def test_quoted_breadcrumb_behaviorally_pinned():
    # Discriminating fixture: [real crumb t0, its clearing result marker t0+5m, a note at
    # t0+10m that CONTAINS but does not START WITH the breadcrumb text]. A substring-matching
    # impl would read the quoting note as the freshest crumb with no result after it -> skip;
    # the startswith anchor keeps the real crumb (already cleared) -> dispatch allowed.
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev("2026-07-01T10:05:00Z", "epm:experiment-implementation", "round 1 landed"),
        _ev(
            "2026-07-01T10:10:00Z",
            "epm:progress",
            # No quote chars around the tokens — they must parse cleanly (round=1 as int) so a
            # substring-anchored impl WOULD accept this note as a fresh crumb and fail the test.
            "note: earlier stage-dispatch stage=implementing round=1 crumb was cleared",
        ),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:11:00Z")
        )
        is None
    )


def test_non_integer_round_token_never_matches():
    events = [
        _ev("2026-07-01T10:00:00Z", "epm:progress", "stage-dispatch stage=implementing round=abc")
    ]
    for round_num in (0, 1, 2):
        assert (
            tw.stage_dispatch_should_skip(
                events, "implementing", round_num, 15, now=_dt("2026-07-01T10:01:00Z")
            )
            is None
        )


def test_malformed_breadcrumb_ts_fails_toward_dispatch():
    events = [
        _ev(
            "not-a-date",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        )
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:01:00Z")
        )
        is None
    )


def test_age_exactly_window_allows_dispatch():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        )
    ]
    # age == window (15.0m exactly) -> >= is dispatch-allowed (stalled boundary).
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:15:00Z")
        )
        is None
    )


def test_810_replay_deliberate_stop_note_does_not_refresh_window():
    # #810 (2026-07-03): the stopped session's deliberate-stop record refreshed
    # the followup-implementing window; the replacement was told to skip dead work.
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev(
            "2026-07-01T10:14:00Z",
            "epm:progress",
            "deliberate-stop pid=n/a target=happy-session:abc123 reason=operator-replace",
            by="spawn_session-stop",
        ),
    ]
    # Past the breadcrumb's own window: the stop record must NOT have refreshed.
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
        )
        is None
    )
    # Inside the breadcrumb's own window: exclusion never CLEARS in-flight state.
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:10:00Z")
        )
        is not None
    )


def test_spawn_session_stop_by_field_does_not_refresh():
    # The by-field leg alone (note text variant) is sufficient to exclude — ANY
    # note posted with by="spawn_session-stop" is a stop record, whatever it says.
    for stop_note in ("stopping session", "session teardown requested by operator"):
        events = [
            _ev(
                "2026-07-01T10:00:00Z",
                "epm:progress",
                "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
            ),
            _ev("2026-07-01T10:14:00Z", "epm:progress", stop_note, by="spawn_session-stop"),
        ]
        assert (
            tw.stage_dispatch_should_skip(
                events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
            )
            is None
        ), stop_note


def test_deliberate_stop_note_prefix_alone_does_not_refresh():
    # The note-prefix leg ALONE (by is an ordinary poster, not
    # "spawn_session-stop") must exclude — isolates the first half of the OR
    # predicate so a regression dropping the prefix leg while keeping the
    # by-field leg fails this test (Codex code-review r1, Minor).
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev(
            "2026-07-01T10:14:00Z",
            "epm:progress",
            "deliberate-stop pid=n/a target=happy-session:abc123 reason=operator-replace",
        ),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
        )
        is None
    )


def test_watcher_telemetry_notes_do_not_refresh_window():
    # Third-party bracketed telemetry (watcher + spawn-session bookkeeping)
    # is not stage liveness; the watcher's own progress clock excludes the
    # same set (_WATCHER_NOTE_SENTINELS).
    for telemetry_note in (
        "[autonomous_session_watch:session-stalled-alert] session idle 2.1h",
        "[autonomous_session_watch:session-auto-respawn] respawned via spawn-issue --auto",
        "[spawn-session:duplicate-dispatch-suppressed] duplicate --auto dispatch suppressed",
    ):
        events = [
            _ev(
                "2026-07-01T10:00:00Z",
                "epm:progress",
                "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
            ),
            _ev("2026-07-01T10:14:00Z", "epm:progress", telemetry_note),
        ]
        assert (
            tw.stage_dispatch_should_skip(
                events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
            )
            is None
        ), telemetry_note


def test_long_phase_heartbeat_note_still_refreshes_window():
    # Inverse boundary: [long-phase-heartbeat] is stamped by the stage's OWN
    # long-running phase — it IS liveness and must keep refreshing.
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev("2026-07-01T10:14:00Z", "epm:progress", "[long-phase-heartbeat] batch poll tick 3"),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
        )
        is not None
    )


def test_note_quoting_deliberate_stop_mid_text_still_refreshes():
    # Prefix boundary: the deliberate-stop exclusion matches the lstripped note
    # PREFIX "deliberate-stop " only — an ordinary progress note that merely
    # QUOTES the string mid-text is still genuine liveness and refreshes.
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            "epm:progress",
            "stage-dispatch stage=implementing round=1 subagent=experiment-implementer",
        ),
        _ev(
            "2026-07-01T10:14:00Z",
            "epm:progress",
            "noting the earlier deliberate-stop was expected; resuming phase 2",
        ),
    ]
    assert (
        tw.stage_dispatch_should_skip(
            events, "implementing", 1, 15, now=_dt("2026-07-01T10:20:00Z")
        )
        is not None
    )
