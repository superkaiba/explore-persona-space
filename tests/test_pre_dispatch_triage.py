"""Tests for ``task_workflow.triage_candidates_since_last_dispatch`` (#889).

Pre-dispatch external-marker triage: before any /issue compute-stage dispatch,
the orchestrator enumerates events.jsonl markers posted since the previous
DUTY-BOUND dispatch record — a compute-launch marker (``epm:run-launched`` /
``epm:cluster-launched``) or a record carrying the ``external-markers
triaged:`` line — and triages each candidate (SKILL.md Step 9 entry guard
§ Pre-dispatch external-marker triage). The helper is pure enumeration /
bounding; externality classification is LLM-side.

Fixture extraction recipe (``tests/fixtures/issue779_predispatch_window.jsonl``
— JSONL carries no comments, so the recipe lives here): from the repo root,
with the #779 task folder resolved via ``uv run python scripts/task.py find
779`` (never a hand-built ``tasks/<status>/...`` path), extracted 2026-07-03
from the live #779 events.jsonl:

    jq -c 'select(.ts >= "2026-07-02T14:27:53Z" and .ts <= "2026-07-02T20:46:04Z")
           | .note |= (if . != null then .[0:200] else . end)' <events.jsonl>

39 rows — #779's dispatch window from the 14:27:53Z review-stage breadcrumb
through the 20:46:04Z incident grid breadcrumb inclusive, each note truncated
to its first 200 chars (truncation preserves every filter input: kind, by,
first-line shape, the ``stage-dispatch `` prefix, triage-line absence; no
secrets, audit prose only). Frozen because the live events.jsonl keeps growing
and the task folder moves across status folders — a live read would be flaky
and would violate the no-hand-built-``tasks/...``-paths rule.
"""

import json
from pathlib import Path

from explore_persona_space.task_workflow import (
    TRIAGE_LINE_PREFIX,
    TRIAGE_MACHINE_BY,
    triage_candidates_since_last_dispatch,
)

FIXTURE = Path(__file__).parent / "fixtures" / "issue779_predispatch_window.jsonl"

# The 10 external audit/directive markers of the #779 incident window
# (plan §1 acceptance criterion 2).
ISSUE779_EXTERNAL_TS = {
    "2026-07-02T16:12:37Z",
    "2026-07-02T16:12:58Z",
    "2026-07-02T16:50:17Z",
    "2026-07-02T17:15:08Z",
    "2026-07-02T17:29:39Z",
    "2026-07-02T17:34:38Z",
    "2026-07-02T17:36:43Z",
    "2026-07-02T17:40:57Z",
    "2026-07-02T17:44:31Z",
    "2026-07-02T17:47:06Z",
}


def _ev(
    ts: str,
    kind: str = "epm:progress",
    by: str = "unknown",
    note: str = "advisory: please read me",
) -> dict:
    """Build a minimal events.jsonl row for synthetic windows."""
    return {"ts": ts, "kind": kind, "by": by, "note": note}


def test_no_dispatch_record_returns_whole_history_filtered():
    events = [
        _ev("2026-07-01T10:00:00Z", note="advisory A"),
        _ev("2026-07-01T10:01:00Z", kind="epm:status-changed", note="proposed -> planning"),
        _ev("2026-07-01T10:02:00Z", note="advisory B"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory A", "advisory B"]


def test_triaged_breadcrumb_closes_window():
    events = [
        _ev("2026-07-01T10:00:00Z", note="advisory before"),
        _ev(
            "2026-07-01T10:01:00Z",
            note=(
                "stage-dispatch stage=followup-grid round=1 subagent=x worktree=repo-root "
                "external-markers triaged: none"
            ),
        ),
        _ev("2026-07-01T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_run_launched_bounds_window():
    events = [
        _ev("2026-07-01T10:00:00Z", note="advisory before"),
        _ev("2026-07-01T10:01:00Z", kind="epm:run-launched", note='{"pod": "pod-42"}'),
        _ev("2026-07-01T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_cluster_launched_bounds_window():
    events = [
        _ev("2026-07-01T10:00:00Z", note="advisory before"),
        _ev(
            "2026-07-01T10:01:00Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note='{"attempt_id": "att-1", "backend": "gcp"}',
        ),
        _ev("2026-07-01T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_non_compute_breadcrumb_does_not_close_window():
    # A review-stage dispatch has no triage duty: an advisory posted before it
    # must still surface at the next compute dispatch.
    events = [
        _ev("2026-07-01T10:00:00Z", kind="epm:cluster-launched", by="backends.gcp", note="{}"),
        _ev("2026-07-01T10:01:00Z", note="AUDIT ROUTING NOTE (PM-chat): do not launch as-is"),
        _ev(
            "2026-07-01T10:02:00Z",
            note=(
                "stage-dispatch stage=followup-code-reviewing round=5 "
                "subagent=code-reviewer worktree=repo-root"
            ),
        ),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["AUDIT ROUTING NOTE (PM-chat): do not launch as-is"]


def test_untriaged_compute_breadcrumb_does_not_close_window():
    # Fail-toward-triage: a pre-fix / concurrent-session COMPUTE breadcrumb
    # without the triage line does not close the window either.
    events = [
        _ev("2026-07-01T10:00:00Z", kind="epm:cluster-launched", by="backends.gcp", note="{}"),
        _ev("2026-07-01T10:01:00Z", note="AUDIT: measured 18-20h serial, must not launch as-is"),
        _ev(
            "2026-07-01T10:02:00Z",
            note=(
                "stage-dispatch stage=followup-grid round=1 "
                "subagent=orchestrator-inline worktree=repo-root"
            ),
        ),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["AUDIT: measured 18-20h serial, must not launch as-is"]


def test_machine_by_excluded():
    events = [
        _ev(
            "2026-07-01T10:00:00Z",
            by="poll_pipeline",
            note="[gpu-idle-advisory] all 1 GPUs <= 5% util for 30 min",
        ),
    ]
    assert triage_candidates_since_last_dispatch(events) == []


def test_exempt_kinds_excluded():
    events = [
        _ev("2026-07-01T10:00:00Z", kind="epm:status-changed", note="running -> verifying"),
        _ev("2026-07-01T10:01:00Z", kind="epm:codex-task-completed", note="Codex job done"),
        _ev(
            "2026-07-01T10:02:00Z",
            kind="epm:workflow-fix-task-filed",
            note="filed_task: #885 | target_file: x",
        ),
    ]
    assert triage_candidates_since_last_dispatch(events) == []


def test_breadcrumb_shaped_and_triage_notes_excluded():
    launch = _ev("2026-07-01T10:00:00Z", kind="epm:run-launched", note="launched")
    crumb = _ev(
        "2026-07-01T10:01:00Z",
        note="stage-dispatch stage=verifying round=1 subagent=upload-verifier worktree=repo-root",
    )
    advisory = _ev("2026-07-01T10:02:00Z", note="advisory: mid-window")
    out = triage_candidates_since_last_dispatch([launch, crumb, advisory])
    # The mid-window breadcrumb-shaped note is filtered, not returned.
    assert [e["note"] for e in out] == ["advisory: mid-window"]

    triage_record = _ev(
        "2026-07-01T10:03:00Z",
        note=f"{TRIAGE_LINE_PREFIX} 2 applied / 1 deferred (grid slimmed; baseline deferred)",
    )
    later = _ev("2026-07-01T10:04:00Z", note="advisory: after triage record")
    out2 = triage_candidates_since_last_dispatch([launch, crumb, advisory, triage_record, later])
    # A triage record is never a candidate (it closes the window instead).
    assert [e["note"] for e in out2] == ["advisory: after triage record"]


def test_empty_note_excluded():
    events = [
        {"ts": "2026-07-01T10:00:00Z", "kind": "epm:progress", "by": "unknown"},
        {"ts": "2026-07-01T10:01:00Z", "kind": "epm:progress", "by": "unknown", "note": ""},
        {"ts": "2026-07-01T10:02:00Z", "kind": "epm:progress", "by": "unknown", "note": None},
        {"ts": "2026-07-01T10:03:00Z", "kind": "epm:progress", "by": "unknown", "note": "   \n"},
    ]
    assert triage_candidates_since_last_dispatch(events) == []


def test_by_user_always_candidate():
    events = [
        _ev("2026-07-01T10:00:00Z", by="user", note="user: add the identity baseline first"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["by"] for e in out] == ["user"]


def test_chronological_order_preserved():
    events = [
        _ev("2026-07-01T10:00:00Z", note="advisory 1"),
        _ev("2026-07-01T10:01:00Z", kind="epm:status-changed", note="running -> verifying"),
        _ev("2026-07-01T10:02:00Z", note="advisory 2"),
        _ev("2026-07-01T10:03:00Z", by="poll_pipeline", note="[liveness]"),
        _ev("2026-07-01T10:04:00Z", note="advisory 3"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory 1", "advisory 2", "advisory 3"]


def test_issue779_replay_flags_all_ten_external_markers():
    """Replay the frozen 39-row #779 window VERBATIM (no slicing needed: the
    20:46:04Z incident breadcrumb carries no triage line, so under the
    duty-boundary predicate it does not close the window — the replay answers
    'what should that dispatch have read')."""
    events = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(events) == 39

    # The fixture carries exactly one compute-launch record: the 14:56:21Z
    # GCP epm:cluster-launched row (the duty boundary).
    launch_rows = [e for e in events if e["kind"] == "epm:cluster-launched"]
    assert [e["ts"] for e in launch_rows] == ["2026-07-02T14:56:21Z"]

    out = triage_candidates_since_last_dispatch(events)
    ts = [e["ts"] for e in out]

    # Boundary resolves to the 14:56:21Z epm:cluster-launched row, NOT the
    # 14:27:53Z review-stage breadcrumb: pre-boundary non-breadcrumb progress
    # notes are excluded; the first post-boundary candidate is included; every
    # candidate postdates the boundary.
    assert "2026-07-02T14:41:17Z" not in ts
    assert "2026-07-02T14:54:43Z" not in ts
    assert "2026-07-02T14:56:52Z" in ts
    assert min(ts) > "2026-07-02T14:56:21Z"
    assert ts == sorted(ts)

    # All 10 external audit/directive markers are flagged.
    assert set(ts) >= ISSUE779_EXTERNAL_TS

    # Named exclusions.
    assert "2026-07-02T19:14:19Z" not in ts  # by: poll_pipeline liveness row
    assert "2026-07-02T20:26:21Z" not in ts  # epm:workflow-fix-task-filed (exempt kind)
    assert "2026-07-02T14:39:09Z" not in ts  # epm:code-review review-loop row
    assert "2026-07-02T14:27:53Z" not in ts  # opening review-stage breadcrumb
    assert "2026-07-02T20:46:04Z" not in ts  # the incident grid breadcrumb (no triage line)

    # Exact count: over-approximation is bounded, not unbounded (the 8
    # non-external candidates are self-posted milestones / relaunch notes /
    # free-analysis landings / epm:failure / epm:results rows). A deliberate
    # exempt-set change re-derives this number against the frozen fixture.
    assert len(out) == 18


# ─── audit_dispatch_triage (#967 post-hoc observer predicate) ───────────────
#
# The pure, NON-GATING audit the watcher's triage-observer pass drives.
# BOUNDARY records (launch kinds / triage-line notes) alone bound windows +
# adjacency; the AUDITED set additionally includes line-less stage-dispatch
# breadcrumbs (MF1). Timestamps below are post-epoch
# (TRIAGE_DUTY_EPOCH_TS = 2026-07-03T05:00:00Z) unless a test targets the
# epoch cutoff; mature_before_ts is omitted (no deferral) unless a test
# targets the MF2 maturity gate.

from explore_persona_space.task_workflow import (  # noqa: E402
    TRIAGE_DUTY_EPOCH_TS,
    _normalize_stage,
    audit_dispatch_triage,
)

T0 = "2026-07-10T10:00:00Z"  # post-epoch base for synthetic audit windows


def _launch(ts: str, note: str = "launched") -> dict:
    return _ev(ts, kind="epm:run-launched", by="poll_pipeline", note=note)


def _triage_note(ts: str, disposition: str = "1 applied (read + folded)") -> dict:
    return _ev(ts, note=f"{TRIAGE_LINE_PREFIX} {disposition}")


def _warns(result: dict) -> list[dict]:
    return [v for v in result["violations"] if v["severity"] == "warn"]


def test_audit_launch_missing_line_flags_warn():
    result = audit_dispatch_triage([_ev("2026-07-10T09:00:00Z", note="advisory"), _launch(T0)])
    assert [v["violation"] for v in result["violations"]] == ["launch-missing-line"]
    v = result["violations"][0]
    assert v["severity"] == "warn"
    assert v["record_ts"] == T0
    assert v["record_kind"] == "epm:run-launched"
    assert v["candidate_count"] == 1
    assert result["cursor_ts"] == T0


def test_audit_launch_adjacent_prior_triage_note_within_window_covers():
    events = [_triage_note("2026-07-10T09:50:00Z"), _launch(T0)]  # 600 s prior
    assert audit_dispatch_triage(events)["violations"] == []


def test_audit_launch_adjacent_prior_triage_note_beyond_window_flags():
    events = [_triage_note("2026-07-10T09:00:00Z"), _launch(T0)]  # 3600 s prior
    assert [v["violation"] for v in audit_dispatch_triage(events)["violations"]] == [
        "launch-missing-line"
    ]


def test_audit_launch_adjacent_next_triage_note_covers_despite_breadcrumb_between():
    # MF1: adjacency neighbors are BOUNDARY records only — a line-less
    # COMPUTE breadcrumb sitting between the launch marker and its covering
    # adjacent-next triage note cannot break the launch's coverage (it is
    # not a boundary record); the breadcrumb itself still flags its OWN
    # breadcrumb-missing-line violation.
    events = [
        _launch(T0),
        _ev(
            "2026-07-10T10:05:00Z",
            note="stage-dispatch stage=followup-grid round=1 subagent=x worktree=repo-root",
        ),
        _triage_note("2026-07-10T10:10:00Z"),  # 600 s after the launch
    ]
    result = audit_dispatch_triage(events)
    assert [v["violation"] for v in result["violations"]] == ["breadcrumb-missing-line"]
    assert result["violations"][0]["record_ts"] == "2026-07-10T10:05:00Z"


def test_audit_burst_second_launch_not_covered_by_first_launch_triage():
    # Burst semantics: the coverage check uses the NEAREST boundary neighbor,
    # so launch B's nearest previous boundary is launch A (not the triage
    # note that covered A) — B stays individually duty-bound and flags even
    # though a triage note exists within the +/- adjacency window.
    events = [
        _triage_note("2026-07-10T09:59:00Z"),
        _launch(T0),  # covered by the adjacent-prior note
        _launch("2026-07-10T10:05:00Z"),  # nearest prev boundary = launch A
    ]
    result = audit_dispatch_triage(events)
    assert [(v["violation"], v["record_ts"]) for v in result["violations"]] == [
        ("launch-missing-line", "2026-07-10T10:05:00Z")
    ]


def test_audit_breadcrumb_three_way_classification():
    def crumb(stage_token: str, extra: str = "") -> dict:
        return _ev(
            T0,
            note=f"stage-dispatch stage={stage_token} round=1 subagent=x{extra} worktree=w",
        )

    # Positive compute token -> warn.
    r = audit_dispatch_triage([crumb("followup-grid")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "warn")
    ]
    assert r["violations"][0]["stage"] == "followup-grid"
    # Positive pid= evidence with an UNKNOWN stage token -> warn.
    r = audit_dispatch_triage([crumb("followup-somethingnew", extra=" pid=12345")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "warn")
    ]
    # Exempt via the code-reviewing -> code-review alias -> NO flag.
    assert audit_dispatch_triage([crumb("followup-code-reviewing")])["violations"] == []
    # SUFFIX form passes through _normalize_stage intact (no followup- prefix
    # to strip) -> unknown tier -> info (its 9a-ter duty is content-dependent).
    assert _normalize_stage("free-analysis-followup") == "free-analysis-followup"
    r = audit_dispatch_triage([crumb("free-analysis-followup")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "info")
    ]
    # Unknown token, no positive evidence -> info, never warn.
    r = audit_dispatch_triage([crumb("followup-somethingnew")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "info")
    ]


def test_audit_live_benign_breadcrumb_vocabulary_never_warns():
    """MF4 live-vocabulary regression — the three benign post-epoch
    breadcrumb families observed in the wild produce NO flag at all.

    Notes frozen VERBATIM (truncated) from the live tree on 2026-07-04 via
    ``uv run python scripts/task.py find 810`` / ``find 922`` (never a
    hand-built ``tasks/<status>/...`` path) + jq over each events.jsonl —
    the same extraction recipe as the #779 fixture header; live task folders
    move across status dirs, so the literals are inline, not a live read.
    Sources: #810 events ts 2026-07-03T07:57:34Z (followup-interp-critique),
    #810 ts 2026-07-03T09:29:12Z (followup-clean-result-fix, carrying the
    'compute-character: no fit/battery' prose a substring matcher would
    trip on), #922 ts 2026-07-04T03:27:01Z (followup-value-critique).
    """
    live = [
        _ev(
            "2026-07-03T07:57:34Z",
            note=(
                "stage-dispatch stage=followup-interp-critique round=1 "
                "subagent=interpretation-critic+codex-interpretation-critic "
                "worktree=/home/thomasjiralerspong/explore-persona-space/.claude/"
                "worktrees/issue-810 -- held interpretation "
            ),
        ),
        _ev(
            "2026-07-03T09:29:12Z",
            note=(
                "stage-dispatch stage=followup-clean-result-fix round=2 "
                "subagent=analyzer (clean-result REVISE union: 2 Claude + 5 Codex "
                "items) worktree=/home/thomasjiralerspong/explore-persona-space/"
                ".claude/worktrees/issue-810 -- compute-character: no fit/battery "
                "stages (figure regeneration from persisted per-context JSONs only)"
            ),
        ),
        _ev(
            "2026-07-04T03:27:01Z",
            note=(
                "stage-dispatch stage=followup-value-critique round=1 "
                "subagent=follow-up-critic-ensemble worktree=/home/thomasjiralerspong/"
                "explore-persona-space/.claude/worktrees/issue-922 (single-pass "
                "redundancy screen over epm:follow-u"
            ),
        ),
    ]
    result = audit_dispatch_triage(live)
    assert result["violations"] == []
    assert _warns(result) == []


def test_audit_none_with_candidates_grading():
    # none + empty window -> no flag.
    assert audit_dispatch_triage([_triage_note(T0, "none")])["violations"] == []
    # none + one plain candidate -> info with the correct count.
    events = [
        _ev("2026-07-10T09:30:00Z", note="advisory: self-posted milestone"),
        _triage_note(T0, "none"),
    ]
    result = audit_dispatch_triage(events)
    assert [(v["violation"], v["severity"]) for v in result["violations"]] == [
        ("none-with-candidates", "info")
    ]
    assert result["violations"][0]["candidate_count"] == 1
    # none + a candidate carrying an external signature -> warn + the hit.
    events = [
        _ev("2026-07-10T09:30:00Z", note="# Audit: measured 18-20h serial, do not launch"),
        _triage_note(T0, "none"),
    ]
    result = audit_dispatch_triage(events)
    assert [(v["violation"], v["severity"]) for v in result["violations"]] == [
        ("none-with-candidates", "warn")
    ]
    assert result["violations"][0]["signature_hits"] == ["# Audit"]
    # A candidate inside grace_s of the record is not counted.
    events = [_ev("2026-07-10T09:59:30Z", note="advisory: 30 s before"), _triage_note(T0, "none")]
    assert audit_dispatch_triage(events)["violations"] == []


def test_audit_mf1_untriaged_breadcrumb_never_closes_a_window():
    # MF1 window-interplay pin: [triage record T, advisory A, line-less
    # compute breadcrumb B, advisory C, none-record N] -> B flags
    # breadcrumb-missing-line AND N's re-enumeration window opens at T (the
    # nearest BOUNDARY), so BOTH A and C are counted — the untriaged
    # breadcrumb never closes a window (fail-toward-triage preserved).
    events = [
        _triage_note("2026-07-10T09:00:00Z"),
        _ev("2026-07-10T09:10:00Z", note="advisory A"),
        _ev(
            "2026-07-10T09:20:00Z",
            note="stage-dispatch stage=followup-grid round=1 subagent=x worktree=w",
        ),
        _ev("2026-07-10T09:30:00Z", note="advisory C"),
        _triage_note(T0, "none"),
    ]
    result = audit_dispatch_triage(events)
    by_class = {v["violation"]: v for v in result["violations"]}
    assert set(by_class) == {"breadcrumb-missing-line", "none-with-candidates"}
    assert by_class["breadcrumb-missing-line"]["record_ts"] == "2026-07-10T09:20:00Z"
    assert by_class["none-with-candidates"]["candidate_count"] == 2


def test_audit_mf2_maturity_gate_defers_and_never_consumes():
    launch = _launch(T0)
    note = _triage_note("2026-07-10T10:05:00Z")
    # Evaluation 1: only the launch marker has landed and it is IMMATURE
    # (mature_before_ts predates it) -> zero violations AND the cursor does
    # not consume it (the caller re-sees it next tick).
    r1 = audit_dispatch_triage([launch], mature_before_ts="2026-07-10T09:59:00Z")
    assert r1["violations"] == []
    assert r1["cursor_ts"] is None
    # Evaluation 2: the compliant adjacent-next note has landed and the
    # record is matured -> still zero violations (covered), cursor advances.
    r2 = audit_dispatch_triage([launch, note], mature_before_ts="2026-07-10T11:00:00Z")
    assert r2["violations"] == []
    assert r2["cursor_ts"] == "2026-07-10T10:05:00Z"


def test_audit_mf2_immature_violation_flags_once_matured():
    launch = _launch(T0)
    r1 = audit_dispatch_triage([launch], mature_before_ts="2026-07-10T09:59:00Z")
    assert r1["violations"] == [] and r1["cursor_ts"] is None
    r2 = audit_dispatch_triage([launch], mature_before_ts="2026-07-10T11:00:00Z")
    assert [v["violation"] for v in r2["violations"]] == ["launch-missing-line"]
    assert r2["cursor_ts"] == T0


def test_audit_epoch_and_min_ts_skip_but_consume_cursor():
    pre_epoch = _launch("2026-07-01T10:00:00Z")
    assert pre_epoch["ts"] < TRIAGE_DUTY_EPOCH_TS
    r = audit_dispatch_triage([pre_epoch])
    assert r["violations"] == []
    assert r["cursor_ts"] == "2026-07-01T10:00:00Z"  # skipped, still consumable
    r = audit_dispatch_triage([_launch(T0)], min_ts=T0)
    assert r["violations"] == []
    assert r["cursor_ts"] == T0


def test_audit_unparseable_ts_fail_soft():
    # An audited record with a malformed ts is skipped entirely: no
    # violation, no crash, never consumed by the cursor.
    bad = _launch("not-a-timestamp")
    r = audit_dispatch_triage([bad])
    assert r["violations"] == [] and r["cursor_ts"] is None
    # A malformed-ts NEIGHBOR provides no adjacency coverage but triggers
    # nothing itself: the launch flags.
    events = [_triage_note("garbage-ts"), _launch(T0)]
    r = audit_dispatch_triage(events)
    assert [v["violation"] for v in r["violations"]] == ["launch-missing-line"]


def test_audit_issue779_replay():
    """Acceptance criteria 1-2 (#967 plan §1): the frozen #779 window flags
    EXACTLY the two incident records at warn with the epoch off, and nothing
    with the production epoch on (all fixture rows are legacy pre-fix)."""
    events = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(events) == 39

    result = audit_dispatch_triage(events, epoch_ts=None)
    warns = _warns(result)
    by_ts = {v["record_ts"]: v for v in warns}
    # MF1 pin (b): the returned flag list explicitly contains the incident
    # breadcrumb's record_ts.
    assert "2026-07-02T20:46:04Z" in by_ts
    crumb = by_ts["2026-07-02T20:46:04Z"]
    assert crumb["violation"] == "breadcrumb-missing-line"
    assert crumb["stage"] == "followup-grid"
    assert crumb["candidate_count"] >= 10
    assert crumb["signature_hits"]
    # The machine-posted GCP launch marker with no adjacent triage note.
    assert by_ts["2026-07-02T14:56:21Z"]["violation"] == "launch-missing-line"
    # False-positive bound on real data: NO OTHER warn-class flag (the
    # 14:27:53Z review-stage breadcrumb is exempt via the code-reviewing
    # alias). Residual info rows are surfaced for visibility, not asserted.
    assert set(by_ts) == {"2026-07-02T20:46:04Z", "2026-07-02T14:56:21Z"}
    for v in result["violations"]:
        if v["severity"] != "warn":
            print(f"info residual: {v['record_ts']} {v['violation']}")

    # Production epoch: every fixture row predates TRIAGE_DUTY_EPOCH_TS.
    assert audit_dispatch_triage(events)["violations"] == []


def test_advisory_by_values_are_never_machine_stripped():
    # #966: pm-chat / watcher / spawn-helper identities are ADVISORY emitter
    # identities, not machine bookkeeping — adding one to TRIAGE_MACHINE_BY
    # would strip exactly the advisory content triage exists to surface.
    for by in ("pm-chat", "autonomous_session_watch", "spawn_session-stop", "spawn_session"):
        assert by not in TRIAGE_MACHINE_BY, by
    out = triage_candidates_since_last_dispatch([_ev("2026-07-03T00:00:00Z", by="pm-chat")])
    assert len(out) == 1


# ─── #2105 enumeration-boundary token (enumerate-to-post seam reopen) ────────
#
# A triage-record line may carry a trailing ``(boundary=<ts>)`` token — the
# ts of the LAST event its enumerator actually read. The window then reopens
# from that recorded enumeration point instead of the record's own post
# position, so a marker landing in the enumerate-to-post seam is enumerated
# at the NEXT call (incident #2054: user-directive marker v91 landed 53 s
# before the r11 breadcrumb post and was invisible to rounds r11-r14). The
# ONE-STEP CHAIN covers the pod/backend-launch form (token-bearing
# ``epm:progress`` triage note posted immediately BEFORE the token-less
# launch marker — SKILL.md :4394/:4731 note-then-launch ordering).

from explore_persona_space.task_workflow import (  # noqa: E402
    parse_triage_boundary_ts,
    triage_enumeration_boundary,
)

B0 = "2026-08-05T10:00:00Z"  # the enumeration boundary (last event read at T0)


def test_boundary_token_reopens_enumerate_to_post_seam():
    """#2054 v91 replay: a directive landing between enumeration (T0) and the
    breadcrumb post IS returned by the next call when the record carries the
    ``(boundary=<T0>)`` token."""
    events = [
        _ev(B0, note="advisory: enumerated at T0"),
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive: raise n to 5000"),
        _ev("2026-08-05T10:01:30Z", note=f"{TRIAGE_LINE_PREFIX} none (boundary={B0})"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["user directive: raise n to 5000"]


def test_legacy_triage_line_without_token_keeps_post_boundary():
    # Explicit pin of today's behavior: no token -> post-position boundary.
    events = [
        _ev(B0, note="advisory: enumerated at T0"),
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive in the seam"),
        _ev("2026-08-05T10:01:30Z", note=f"{TRIAGE_LINE_PREFIX} none"),
        _ev("2026-08-05T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_malformed_boundary_ts_falls_back_to_post_boundary():
    events = [
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive in the seam"),
        _ev("2026-08-05T10:01:30Z", note=f"{TRIAGE_LINE_PREFIX} none (boundary=garbage)"),
        _ev("2026-08-05T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_future_boundary_ts_never_shrinks_window():
    # A bogus-future recorded ts degrades to (at worst) today's window: the
    # reopen scan starts at the record's index - 1, so the reopened window is
    # structurally a superset of today's events[idx + 1:].
    events = [
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive in the seam"),
        _ev(
            "2026-08-05T10:01:30Z",
            note=f"{TRIAGE_LINE_PREFIX} none (boundary=2099-01-01T00:00:00Z)",
        ),
        _ev("2026-08-05T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_token_note_followed_by_launch_marker_reopens_seam():
    """The LAUNCH-FORM regression (critic round-1 Must-Fix, mechanized): the
    token-bearing triage ``epm:progress`` note is posted immediately BEFORE
    dispatch, the token-less ``epm:run-launched`` lands AFTER — the one-step
    chain honors the note's token. FAILS without the chain (the launch
    marker would close the window at its own position)."""
    events = [
        _ev(B0, note="advisory: enumerated at T0"),
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive in the seam"),
        _ev("2026-08-05T10:01:30Z", note=f"{TRIAGE_LINE_PREFIX} none (boundary={B0})"),
        _ev("2026-08-05T10:01:40Z", kind="epm:run-launched", note='{"pod": "pod-2105"}'),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["user directive in the seam"]


def test_launch_marker_with_own_token_line_honored():
    # A launch-kind marker whose OWN note carries the line + token reopens
    # directly (no chain needed).
    events = [
        _ev(B0, note="advisory: enumerated at T0"),
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive in the seam"),
        _ev(
            "2026-08-05T10:01:40Z",
            kind="epm:run-launched",
            note=f"launched pod-2105 -- {TRIAGE_LINE_PREFIX} none (boundary={B0})",
        ),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["user directive in the seam"]


def test_chain_stops_at_prior_launch_marker():
    # The chain is EXACTLY ONE step and stops at another launch marker ->
    # today's launch-position boundary (fail-toward-today).
    events = [
        _ev("2026-08-05T10:00:53Z", by="user", note="never-triaged directive"),
        _ev("2026-08-05T10:01:00Z", kind="epm:run-launched", note="launch A"),
        _ev("2026-08-05T10:01:40Z", kind="epm:run-launched", note="launch B, no triage line"),
        _ev("2026-08-05T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_chain_stops_at_legacy_tokenless_note():
    # A token-less (legacy) triage note stops the chain without a token ->
    # today's launch-position boundary.
    events = [
        _ev("2026-08-05T10:00:53Z", by="user", note="user directive in the seam"),
        _ev("2026-08-05T10:01:30Z", note=f"{TRIAGE_LINE_PREFIX} none"),
        _ev("2026-08-05T10:01:40Z", kind="epm:run-launched", note="launched"),
        _ev("2026-08-05T10:02:00Z", note="advisory after"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["advisory after"]


def test_events_at_or_before_recorded_boundary_not_reenumerated():
    # Events at ts <= recorded were read by the prior enumerator run and stay
    # OUT of the reopened window (the `<=` tie semantics).
    events = [
        _ev("2026-08-05T09:59:00Z", note="already triaged: before T0"),
        _ev(B0, note="already triaged: exactly at T0"),
        _ev("2026-08-05T10:00:53Z", by="user", note="seam directive"),
        _ev(
            "2026-08-05T10:01:30Z",
            note=f"{TRIAGE_LINE_PREFIX} 1 applied (folded) (boundary={B0})",
        ),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["seam directive"]


def test_unparseable_ts_events_in_seam_stay_candidates():
    # An event with a malformed ts between the recorded boundary and the
    # record remains enumerated (fail-toward-triage).
    events = [
        _ev(B0, note="already triaged: at T0"),
        _ev("not-a-timestamp", by="user", note="malformed-ts directive"),
        _ev("2026-08-05T10:01:30Z", note=f"{TRIAGE_LINE_PREFIX} none (boundary={B0})"),
    ]
    out = triage_candidates_since_last_dispatch(events)
    assert [e["note"] for e in out] == ["malformed-ts directive"]


def test_triage_enumeration_boundary_helper():
    assert triage_enumeration_boundary([]) == ""
    events = [_ev("2026-08-05T09:00:00Z"), _ev(B0)]
    assert triage_enumeration_boundary(events) == B0
    # A last event with no / empty ts yields "" (composers omit the token).
    assert triage_enumeration_boundary([{"kind": "epm:progress"}]) == ""
    assert triage_enumeration_boundary([{"ts": "", "kind": "epm:progress"}]) == ""


def test_parse_triage_boundary_ts_requires_triage_line():
    # A (boundary=...) token with no triage-line prefix is NOT a triage
    # record -> None.
    assert parse_triage_boundary_ts(f"random note (boundary={B0})") is None


def test_parse_triage_boundary_ts_fail_soft():
    assert parse_triage_boundary_ts(f"{TRIAGE_LINE_PREFIX} none") is None
    assert parse_triage_boundary_ts(f"{TRIAGE_LINE_PREFIX} none (boundary=garbage)") is None
    parsed = parse_triage_boundary_ts(f"{TRIAGE_LINE_PREFIX} none (boundary={B0})")
    assert parsed is not None
    assert parsed.isoformat() == "2026-08-05T10:00:00+00:00"


def test_parse_triage_boundary_ts_anchored_after_line():
    # A note whose BODY quotes a prior triage line with a token BEFORE its
    # own triage line binds its OWN token (rfind anchors at the LAST prefix
    # occurrence; the record's own line is appended last per the format
    # spec) — never the quoted one. The #2054 v98/v108 forensics notes quote
    # triage lines exactly this way.
    quoted = (
        "forensics: the r11 record said "
        f"'{TRIAGE_LINE_PREFIX} none (boundary=2026-08-01T00:00:00Z)'"
    )
    own_with_token = f"{quoted}\n{TRIAGE_LINE_PREFIX} 1 applied (folded) (boundary={B0})"
    parsed = parse_triage_boundary_ts(own_with_token)
    assert parsed is not None
    assert parsed.isoformat() == "2026-08-05T10:00:00+00:00"
    # Own line WITHOUT a token -> None, even though the quoted line has one.
    own_without_token = f"{quoted}\n{TRIAGE_LINE_PREFIX} none"
    assert parse_triage_boundary_ts(own_without_token) is None
