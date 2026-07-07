"""Pure-helper tests for ``task_workflow.upload_verification_currency_blocker`` (#1026).

The helper is the verifier-currency gate behind ``dispatch_issue.py finalize``:
it refuses teardown when the upload-verification evidence is not a CURRENT
PASS — a dispatched-but-unresolved verifier round (in-flight / stalled), an
unattributable late verdict (ambiguous, MF-B), results newer than the latest
verdict (stale), or a FAIL as the latest verification (failed-current, MF-A).
Synthetic event lists, no registry — the pattern of
``tests/test_stage_dispatch_dedup.py``. The incident replayed throughout is
#778 (pod finalized on a stale fallback while the verifier was in flight; its
verdict later came back FAIL).
"""

from __future__ import annotations

from datetime import datetime

import explore_persona_space.task_workflow as tw


def _ev(ts: str, kind: str, note: str = "", version: int = 1, by: str = "test") -> dict:
    return {"ts": ts, "kind": kind, "version": version, "by": by, "note": note}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


_NOW = _dt("2026-07-02T19:00:00Z")

_CRUMB = "stage-dispatch stage=verifying round=1 subagent=upload-verifier worktree=repo-root"
_CRUMB_FOLLOWUP = (
    "stage-dispatch stage=followup-verifying round=1 subagent=upload-verifier worktree=repo-root"
)
_PASS_NOTE = "## Upload Verification\n\n**Verdict: PASS**\n\n11 files verified."
_FAIL_NOTE = "## Upload Verification\n\n**Verdict: FAIL**\n\n2 of 11 files missing."


def _blocker(events: list[dict], *, now: datetime = _NOW) -> dict | None:
    return tw.upload_verification_currency_blocker(events, now=now)


# ---------------------------------------------------------------------------
# Rule 1 — in-flight / stalled
# ---------------------------------------------------------------------------


def test_p1_fresh_crumb_with_genuine_prior_verdict_is_in_flight() -> None:
    """P1: [R_prior, V_prior-PASS, R, crumb(fresh)] → in_flight.

    Includes a GENUINE prior verdict + prior results so the
    ``verdict_idxs[-1] < crumb_idxs[-1]`` ordering branch is exercised —
    an implementation blocking only on "no verdict ever" fails this.
    """
    events = [
        _ev("2026-07-02T03:00:00Z", "epm:results", "round-1 results"),
        _ev("2026-07-02T03:36:37Z", "epm:upload-verification", _PASS_NOTE),
        _ev("2026-07-02T18:40:00Z", "epm:results", "round-2 results"),
        _ev("2026-07-02T18:55:00Z", "epm:progress", _CRUMB),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verifier_in_flight"
    assert blocker["state"] == "in-flight"
    assert blocker["stage"] == "verifying"
    assert blocker["round"] == 1
    assert blocker["breadcrumb_ts"] == "2026-07-02T18:55:00Z"


def test_p2_followup_verifying_raw_token_is_covered() -> None:
    """P2: the production ``stage=followup-verifying`` token (observed on
    #778 at 2026-07-02T03:29:51Z) normalizes to verifying and blocks."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:results", "results"),
        _ev("2026-07-02T18:55:00Z", "epm:progress", _CRUMB_FOLLOWUP),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verifier_in_flight"
    assert blocker["stage"] == "followup-verifying"


def test_p3_crumb_resolved_by_later_pass_verification_clears() -> None:
    """P3: a crumb with a LATER PASS verification verdict clears (None)."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:results", "results"),
        _ev("2026-07-02T18:45:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T18:55:00Z", "epm:upload-verification", _PASS_NOTE),
    ]
    assert _blocker(events) is None


def test_p4_crumb_resolved_by_later_sticky_clears() -> None:
    """P4: the sticky ``epm:upload-verified`` marker also resolves a crumb."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:results", "results"),
        _ev("2026-07-02T18:45:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T18:55:00Z", "epm:upload-verified", "sticky PASS"),
    ]
    assert _blocker(events) is None


def test_p5_crumb_past_window_with_no_verdict_is_stalled() -> None:
    """P5: #778-replay timestamps — crumb hours old, no verdict → stalled."""
    events = [
        _ev("2026-07-02T03:20:00Z", "epm:results", "results"),
        _ev("2026-07-02T03:29:51Z", "epm:progress", _CRUMB_FOLLOWUP),
    ]
    blocker = _blocker(events)  # now = 19:00Z, ~15.5h later
    assert blocker is not None
    assert blocker["reason"] == "upload_verifier_stalled"
    assert blocker["state"] == "stalled"
    assert blocker["age_minutes"] is not None and blocker["age_minutes"] > 15


def test_p6_failure_after_crumb_is_still_stalled_never_clear() -> None:
    """P6: ``epm:failure`` clears the DISPATCH dedup (may I re-dispatch?)
    but never the currency gate — the round died with NO verdict."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:results", "results"),
        _ev("2026-07-02T18:45:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T18:50:00Z", "epm:failure", "failure_class: infra"),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verifier_stalled"


def test_p11_malformed_crumb_ts_is_stalled() -> None:
    """P11: a malformed crumb ``ts`` fails toward stalled (blocked), never
    toward a silent clear."""
    events = [_ev("not-a-timestamp", "epm:progress", _CRUMB)]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verifier_stalled"


def test_p12_liveness_marker_refreshes_the_in_flight_window() -> None:
    """P12: crumb 20 m old + ``epm:codex-task-spawned`` 5 m ago → in_flight
    (delegation to ``stage_dispatch_should_skip``'s liveness refresh)."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T18:55:00Z", "epm:codex-task-spawned", "job spawned"),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verifier_in_flight"


def test_p17_crumb_with_missing_or_non_integer_round_is_stalled() -> None:
    """P17: a fresh crumb whose ``round=`` token is missing / non-integer
    can never be liveness-classified → stalled (blocked)."""
    no_round = "stage-dispatch stage=verifying subagent=upload-verifier"
    bad_round = "stage-dispatch stage=verifying round=abc subagent=upload-verifier"
    for note in (no_round, bad_round):
        events = [_ev("2026-07-02T18:55:00Z", "epm:progress", note)]
        blocker = _blocker(events)
        assert blocker is not None, note
        assert blocker["reason"] == "upload_verifier_stalled", note
        assert blocker["round"] is None, note


def test_p10_non_verifying_stage_crumb_is_ignored() -> None:
    """P10: an ``stage=interpreting`` crumb newer than the verdict is NOT a
    verifier round — the gate ignores it."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:upload-verification", _PASS_NOTE),
        _ev(
            "2026-07-02T18:55:00Z",
            "epm:progress",
            "stage-dispatch stage=interpreting round=1 subagent=analyzer",
        ),
    ]
    assert _blocker(events) is None


# ---------------------------------------------------------------------------
# Rule 2 (MF-B) — ambiguous late verdict across a results boundary
# ---------------------------------------------------------------------------


def test_p14_unresolved_crumb_across_results_boundary_blocks_ambiguous() -> None:
    """P14 (MF-B block): [R0, C1, R1, C2, V-late] → ambiguous — the late
    verdict cannot be attributed to the current results-epoch."""
    events = [
        _ev("2026-07-02T10:00:00Z", "epm:results", "round-1 results"),
        _ev("2026-07-02T10:05:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T12:00:00Z", "epm:results", "round-2 results"),
        _ev("2026-07-02T12:05:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T12:10:00Z", "epm:upload-verification", _PASS_NOTE),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verification_ambiguous"
    assert blocker["breadcrumb_ts"] == "2026-07-02T10:05:00Z"


def test_p15_respawn_recovery_same_epoch_clears() -> None:
    """P15 (MF-B recovery): [R, C1, C2, V] — the stalled→re-spawn recovery
    has no results between crumbs (same epoch), so any late verdict covers
    both crumbs → None."""
    events = [
        _ev("2026-07-02T10:00:00Z", "epm:results", "results"),
        _ev("2026-07-02T10:05:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T11:00:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T11:10:00Z", "epm:upload-verification", _PASS_NOTE),
    ]
    assert _blocker(events) is None


def test_p16_post_block_recovery_fresh_round_clears_no_deadlock() -> None:
    """P16 (MF-B deadlock-freedom): after an ambiguous block, ONE verifier
    re-run — [.., C3, V'-PASS] — resolves every earlier crumb by inclusion."""
    events = [
        _ev("2026-07-02T10:00:00Z", "epm:results", "round-1 results"),
        _ev("2026-07-02T10:05:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T12:00:00Z", "epm:results", "round-2 results"),
        _ev("2026-07-02T12:05:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T12:10:00Z", "epm:upload-verification", _PASS_NOTE),
        _ev("2026-07-02T12:20:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T12:30:00Z", "epm:upload-verification", _PASS_NOTE),
    ]
    assert _blocker(events) is None


# ---------------------------------------------------------------------------
# Rule 3 — stale (results postdating the latest verdict)
# ---------------------------------------------------------------------------


def test_p7_results_after_pass_verdict_is_stale() -> None:
    """P7: [V-PASS, R] → the newest results are unverified → stale."""
    events = [
        _ev("2026-07-02T03:36:37Z", "epm:upload-verification", _PASS_NOTE),
        _ev("2026-07-02T18:40:00Z", "epm:results", "round-2 results"),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verification_stale"
    assert "latest verdict at 2026-07-02T03:36:37Z" in blocker["detail"]


def test_p7b_results_with_no_verdict_ever_is_stale() -> None:
    """P7b: results exist and NO verdict marker exists at all → stale via
    the "no verdict marker exists" branch."""
    events = [_ev("2026-07-02T18:40:00Z", "epm:results", "results")]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verification_stale"
    assert "no verdict marker exists" in blocker["detail"]


def test_p8_verdict_postdating_results_and_crumb_clears() -> None:
    """P8: the normal happy path [R, crumb, V-PASS] → None."""
    events = [
        _ev("2026-07-02T18:30:00Z", "epm:results", "results"),
        _ev("2026-07-02T18:40:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T18:50:00Z", "epm:upload-verification", _PASS_NOTE),
    ]
    assert _blocker(events) is None


def test_p9_pure_sticky_legacy_clears_vacuously() -> None:
    """P9: bare sticky, no crumb / results / verification → None (the
    pure-sticky legacy shape stays vacuously clear — acceptance 6)."""
    events = [_ev("2026-07-02T18:40:00Z", "epm:upload-verified", "sticky PASS")]
    assert _blocker(events) is None


def test_empty_events_clears() -> None:
    """No verifier / results activity at all → None."""
    assert _blocker([]) is None


# ---------------------------------------------------------------------------
# Rule 4 (MF-A) — the current verification is a FAIL
# ---------------------------------------------------------------------------


def test_p13_sticky_then_fail_after_results_is_failed_current() -> None:
    """P13 (MF-A): [sticky, R, crumb, V-FAIL] → failed_current — the
    sticky-anywhere prior PASS can no longer green-light teardown after a
    FAILED current verification (the #778 shape)."""
    events = [
        _ev("2026-07-02T03:40:00Z", "epm:upload-verified", "sticky PASS"),
        _ev("2026-07-02T18:30:00Z", "epm:results", "round-2 results"),
        _ev("2026-07-02T18:40:00Z", "epm:progress", _CRUMB),
        _ev("2026-07-02T18:50:00Z", "epm:upload-verification", _FAIL_NOTE),
    ]
    blocker = _blocker(events)
    assert blocker is not None
    assert blocker["reason"] == "upload_verification_failed_current"


def test_p13b_sticky_after_fail_reads_as_subsequent_pass_record() -> None:
    """P13b: [V-FAIL, sticky] → None — a sticky posted AFTER a FAIL is the
    skill's subsequent-PASS record (latest verdict-kind wins)."""
    events = [
        _ev("2026-07-02T18:40:00Z", "epm:upload-verification", _FAIL_NOTE),
        _ev("2026-07-02T18:50:00Z", "epm:upload-verified", "sticky PASS after fix"),
    ]
    assert _blocker(events) is None


def test_p13c_rules_3_and_4_are_index_disjoint() -> None:
    """P13c: a FAIL postdating results reads failed_current; a FAIL
    predating results reads stale — never both, never neither."""
    fail_after_results = [
        _ev("2026-07-02T18:30:00Z", "epm:results", "results"),
        _ev("2026-07-02T18:40:00Z", "epm:upload-verification", _FAIL_NOTE),
    ]
    blocker = _blocker(fail_after_results)
    assert blocker is not None
    assert blocker["reason"] == "upload_verification_failed_current"

    fail_before_results = [
        _ev("2026-07-02T18:30:00Z", "epm:upload-verification", _FAIL_NOTE),
        _ev("2026-07-02T18:40:00Z", "epm:results", "results"),
    ]
    blocker = _blocker(fail_before_results)
    assert blocker is not None
    assert blocker["reason"] == "upload_verification_stale"


# ---------------------------------------------------------------------------
# P18 — canonical regex parity with dispatch_issue's private copy
# ---------------------------------------------------------------------------


def test_pass_regex_parity_with_dispatch_issue() -> None:
    """P18: the canonical ``task_workflow.UPLOAD_VERIFICATION_PASS_RE`` and
    dispatch_issue's private ``_UPLOAD_VERIFICATION_PASS_RE`` stay
    pattern-identical (duplication-drift guard)."""
    import scripts.dispatch_issue as di

    assert tw.UPLOAD_VERIFICATION_PASS_RE.pattern == di._UPLOAD_VERIFICATION_PASS_RE.pattern
    # And the shared shape actually matches / rejects the schema forms.
    assert tw.UPLOAD_VERIFICATION_PASS_RE.search("**Verdict: PASS**")
    assert not tw.UPLOAD_VERIFICATION_PASS_RE.search("**Verdict: FAIL**")
    assert not tw.UPLOAD_VERIFICATION_PASS_RE.search("the checks pass")
