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


def _ev(ts: str, kind: str, note: str = "", version: int = 1) -> dict:
    return {"ts": ts, "kind": kind, "version": version, "by": "test", "note": note}


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
    # Documented over-skip direction: ANY non-breadcrumb progress refreshes the window.
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
