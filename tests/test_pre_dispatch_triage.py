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
    # An advisory between the note and the launch keeps the window non-empty
    # (#1400 empty-window suppression), so ADJACENCY stays the discriminator.
    events = [
        _triage_note("2026-07-10T09:00:00Z"),  # 3600 s prior — beyond adjacency
        _ev("2026-07-10T09:30:00Z", note="advisory in the launch window"),
        _launch(T0),
    ]
    assert [v["violation"] for v in audit_dispatch_triage(events)["violations"]] == [
        "launch-missing-line"
    ]


def test_audit_launch_adjacent_next_triage_note_covers_despite_breadcrumb_between():
    # MF1: adjacency neighbors are BOUNDARY records only — a line-less
    # COMPUTE breadcrumb sitting between the launch marker and its covering
    # adjacent-next triage note cannot break the launch's coverage (it is
    # not a boundary record); the breadcrumb itself still flags its OWN
    # breadcrumb-missing-line violation. The launch's OWN pre-window carries
    # an advisory beyond grace (#1400): its no-flag assert is discriminating
    # for NEXT-side coverage, not satisfied by empty-window suppression; the
    # gap advisory (beyond grace of the crumb) keeps the crumb's window
    # non-empty.
    events = [
        _ev("2026-07-10T09:50:00Z", note="advisory in the launch pre-window"),
        _launch(T0),
        _ev("2026-07-10T10:01:00Z", note="advisory in the crumb window"),
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
    # though a triage note exists within the +/- adjacency window. The 300 s
    # A→B spacing now ALSO pins the no-coalesce boundary (#1400): 300 s >
    # cascade_s default 180 s, so B is a burst attempt, never a cascade
    # sibling of A; the gap advisory (180 s before B, beyond the 120 s grace
    # trim) keeps B's window non-empty so emptiness cannot mask the pin.
    events = [
        _triage_note("2026-07-10T09:59:00Z"),
        _launch(T0),  # covered by the adjacent-prior note
        _ev("2026-07-10T10:02:00Z", note="advisory between the burst launches"),
        _launch("2026-07-10T10:05:00Z"),  # nearest prev boundary = launch A
    ]
    result = audit_dispatch_triage(events)
    assert [(v["violation"], v["record_ts"]) for v in result["violations"]] == [
        ("launch-missing-line", "2026-07-10T10:05:00Z")
    ]


def test_audit_breadcrumb_three_way_classification():
    # Each mini-fixture leads with one advisory beyond grace so the crumb's
    # window is non-empty (#1400 empty-window suppression) — the severity
    # assertions stay byte-identical to the pre-#1400 pins.
    adv = _ev("2026-07-10T09:30:00Z", note="advisory: pending external marker")

    def crumb(stage_token: str, extra: str = "") -> dict:
        return _ev(
            T0,
            note=f"stage-dispatch stage={stage_token} round=1 subagent=x{extra} worktree=w",
        )

    # Positive compute token -> warn.
    r = audit_dispatch_triage([adv, crumb("followup-grid")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "warn")
    ]
    assert r["violations"][0]["stage"] == "followup-grid"
    # Positive pid= evidence with an UNKNOWN stage token -> warn.
    r = audit_dispatch_triage([adv, crumb("followup-somethingnew", extra=" pid=12345")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "warn")
    ]
    # Exempt via the code-reviewing -> code-review alias -> NO flag.
    assert audit_dispatch_triage([adv, crumb("followup-code-reviewing")])["violations"] == []
    # SUFFIX form passes through _normalize_stage intact (no followup- prefix
    # to strip) -> unknown tier -> info (its 9a-ter duty is content-dependent).
    assert _normalize_stage("free-analysis-followup") == "free-analysis-followup"
    r = audit_dispatch_triage([adv, crumb("free-analysis-followup")])
    assert [(v["violation"], v["severity"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", "info")
    ]
    # Unknown token, no positive evidence -> info, never warn.
    r = audit_dispatch_triage([adv, crumb("followup-somethingnew")])
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
    # The advisory 10 min before the launch (beyond the 120 s grace trim)
    # keeps the window non-empty (#1400), so MATURITY stays the only
    # suppressor in the immature leg.
    adv = _ev("2026-07-10T09:50:00Z", note="advisory in the launch window")
    launch = _launch(T0)
    note = _triage_note("2026-07-10T10:05:00Z")
    # Evaluation 1: only the launch marker has landed and it is IMMATURE
    # (mature_before_ts predates it) -> zero violations AND the cursor does
    # not consume it (the caller re-sees it next tick).
    r1 = audit_dispatch_triage([adv, launch], mature_before_ts="2026-07-10T09:59:00Z")
    assert r1["violations"] == []
    assert r1["cursor_ts"] is None
    # Evaluation 2: the compliant adjacent-next note has landed and the
    # record is matured -> still zero violations (covered), cursor advances.
    r2 = audit_dispatch_triage([adv, launch, note], mature_before_ts="2026-07-10T11:00:00Z")
    assert r2["violations"] == []
    assert r2["cursor_ts"] == "2026-07-10T10:05:00Z"


def test_audit_mf2_immature_violation_flags_once_matured():
    # Advisory beyond grace -> non-empty window (#1400): maturity is the
    # discriminating suppressor across the two evaluations.
    adv = _ev("2026-07-10T09:50:00Z", note="advisory in the launch window")
    launch = _launch(T0)
    r1 = audit_dispatch_triage([adv, launch], mature_before_ts="2026-07-10T09:59:00Z")
    assert r1["violations"] == [] and r1["cursor_ts"] is None
    r2 = audit_dispatch_triage([adv, launch], mature_before_ts="2026-07-10T11:00:00Z")
    assert [v["violation"] for v in r2["violations"]] == ["launch-missing-line"]
    assert r2["cursor_ts"] == T0


def test_audit_epoch_and_min_ts_skip_but_consume_cursor():
    # Advisory rows beyond grace keep both windows non-empty (#1400), so
    # epoch / min_ts stay the discriminating suppressors.
    pre_epoch_adv = _ev("2026-07-01T09:50:00Z", note="advisory in the launch window")
    pre_epoch = _launch("2026-07-01T10:00:00Z")
    assert pre_epoch["ts"] < TRIAGE_DUTY_EPOCH_TS
    r = audit_dispatch_triage([pre_epoch_adv, pre_epoch])
    assert r["violations"] == []
    assert r["cursor_ts"] == "2026-07-01T10:00:00Z"  # skipped, still consumable
    r = audit_dispatch_triage(
        [_ev("2026-07-10T09:50:00Z", note="advisory in the launch window"), _launch(T0)],
        min_ts=T0,
    )
    assert r["violations"] == []
    assert r["cursor_ts"] == T0


def test_audit_unparseable_ts_fail_soft():
    # An audited record with a malformed ts is skipped entirely: no
    # violation, no crash, never consumed by the cursor.
    bad = _launch("not-a-timestamp")
    r = audit_dispatch_triage([bad])
    assert r["violations"] == [] and r["cursor_ts"] is None
    # A malformed-ts NEIGHBOR provides no adjacency coverage but triggers
    # nothing itself: the launch (whose window carries a valid-ts advisory
    # beyond grace — non-empty under #1400) flags.
    events = [
        _triage_note("garbage-ts"),
        _ev("2026-07-10T09:50:00Z", note="advisory with a valid ts"),
        _launch(T0),
    ]
    r = audit_dispatch_triage(events)
    assert [v["violation"] for v in r["violations"]] == ["launch-missing-line"]


# ─── #1400: empty-window suppression + launch-cascade coalescing ─────────────
#
# Fix (1): every violation class fires only against a NON-empty
# post-grace-trim candidate window (65 pre-fix zero-candidate sidecar rows —
# 43 launch-missing-line warn / 10 breadcrumb warn / 12 breadcrumb info).
# Fix (2): consecutive line-less launch-kind boundaries within cascade_s
# (180 s default, chained) coalesce as ONE logical dispatch for the
# previous-side triage-note coverage walk.


def test_audit_launch_empty_window_never_flags():
    # #1400 acceptance criterion 1: a lone matured launch has NOTHING to
    # triage (0 post-trim candidates) -> vacuously compliant, no flag — and
    # suppression never blocks cursor consumption (the record is consumed).
    r = audit_dispatch_triage([_launch(T0)])
    assert r["violations"] == []
    assert r["cursor_ts"] == T0


def test_audit_breadcrumb_empty_window_never_flags():
    # Warn class (compute stage token) and info class (unknown stage) are
    # BOTH empty-window-suppressed (#1400); the cursor still consumes.
    crumb_warn = _ev(T0, note="stage-dispatch stage=followup-grid round=1 subagent=x worktree=w")
    r = audit_dispatch_triage([crumb_warn])
    assert r["violations"] == []
    assert r["cursor_ts"] == T0
    crumb_info = _ev(
        T0, note="stage-dispatch stage=followup-somethingnew round=1 subagent=x worktree=w"
    )
    r = audit_dispatch_triage([crumb_info])
    assert r["violations"] == []
    assert r["cursor_ts"] == T0


def test_audit_grace_trim_applies_to_all_classes():
    # #1400: the grace trim (formerly none-with-candidates-only) applies to
    # EVERY class before the emptiness test — a candidate landing within
    # grace_s (120 s) of the record post-dates the session's final
    # enumerator run (the SKILL.md accepted residual).
    # Launch: advisory 30 s before -> trimmed -> empty -> suppressed.
    r = audit_dispatch_triage(
        [_ev("2026-07-10T09:59:30Z", note="advisory 30 s before"), _launch(T0)]
    )
    assert r["violations"] == []
    # Launch: advisory 10 min before -> kept -> flags with count 1.
    r = audit_dispatch_triage(
        [_ev("2026-07-10T09:50:00Z", note="advisory 10 min before"), _launch(T0)]
    )
    assert [(v["violation"], v["candidate_count"]) for v in r["violations"]] == [
        ("launch-missing-line", 1)
    ]
    # Mirrored pair for a breadcrumb.
    crumb = _ev(T0, note="stage-dispatch stage=followup-grid round=1 subagent=x worktree=w")
    r = audit_dispatch_triage([_ev("2026-07-10T09:59:30Z", note="advisory 30 s before"), crumb])
    assert r["violations"] == []
    r = audit_dispatch_triage([_ev("2026-07-10T09:50:00Z", note="advisory 10 min before"), crumb])
    assert [(v["violation"], v["candidate_count"]) for v in r["violations"]] == [
        ("breadcrumb-missing-line", 1)
    ]


def test_audit_cascade_candidate_in_gap_still_covered():
    # Pins fix (2) DISTINCTLY from fix (1): the run-launched's window is
    # NON-empty (the mid-provisioning advisory at T-130 s survives the
    # 120 s grace trim, count would be 1), so ONLY cascade coalescing — the
    # pre-cascade triage note examined through the line-less
    # epm:cluster-launched sibling — suppresses the flag.
    events = [
        _triage_note("2026-07-10T09:57:10Z"),  # T-170 s
        _ev(
            "2026-07-10T09:57:20Z",  # T-160 s: line-less cascade head
            kind="epm:cluster-launched",
            by="backends.gcp",
            note='{"attempt_id": "att-1", "backend": "gcp"}',
        ),
        _ev("2026-07-10T09:57:50Z", note="advisory mid-provisioning"),  # T-130 s > grace
        _launch(T0),  # epm:run-launched, the cascade tail
    ]
    assert audit_dispatch_triage(events)["violations"] == []


def test_audit_cascade_chain_and_disable():
    # (a) CHAINED walk: three line-less launch-kind records 150 s apart
    # (each gap within cascade_s=180; spaced in the 121-179 s band so the
    # gap advisories survive the 120 s grace trim — windows NON-empty, so
    # chained coverage, not emptiness, does the work) after one triage note
    # -> ALL covered through the chain (a single-step walk would leave the
    # third launch anchored at a non-line sibling and flag it).
    events = [
        _triage_note("2026-07-10T10:00:00Z"),
        _ev(
            "2026-07-10T10:02:00Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note="attempt 1",
        ),
        _ev("2026-07-10T10:02:10Z", note="advisory in gap 1"),  # 140 s before L2 > grace
        _ev(
            "2026-07-10T10:04:30Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note="attempt 2",
        ),
        _ev("2026-07-10T10:04:40Z", note="advisory in gap 2"),  # 140 s before L3 > grace
        _launch("2026-07-10T10:07:00Z"),
    ]
    assert audit_dispatch_triage(events)["violations"] == []
    # (b) cascade_s=0 kill switch = EXACT pre-#1400 nearest-boundary
    # semantics: the later launches flag (their windows are non-empty, and
    # each one's nearest previous boundary is a line-less launch).
    r = audit_dispatch_triage(events, cascade_s=0)
    assert [(v["violation"], v["record_ts"]) for v in r["violations"]] == [
        ("launch-missing-line", "2026-07-10T10:04:30Z"),
        ("launch-missing-line", "2026-07-10T10:07:00Z"),
    ]


# The #1005 launch-cascade replay fixture (#1400). Inline literals VERBATIM
# from #1005's live events.jsonl (JSONL has no comments, so the recipe lives
# here — mirroring the #779 fixture header): from the repo root, with the
# task folder resolved via ``uv run python scripts/task.py find 1005`` (never
# a hand-built ``tasks/<status>/...`` path), extracted 2026-07-16 via
#
#     jq -c 'select(<window>) | {ts, kind, by,
#            note: (if .note != null then .note[0:100] else null end)}'
#
# over the four dispatch windows (18:45-19:00Z / 19:20-19:45Z /
# 23:30-23:45Z / 03:45-03:56Z) plus the 18:36:13Z pre-note advisory row.
# Each note is truncated to its first 100 chars (truncation preserves every
# filter input: kind, by, the triage-line prefix, breadcrumb shape). The
# four GCP cascades (cluster->run gaps 46/39/39/32 s) produced the four
# pre-fix zero-candidate launch-missing-line sidecar flags — records
# 2026-07-15T18:53:36Z / 19:42:32Z / 23:39:12Z (three on 07-15) and
# 2026-07-16T03:53:26Z — while the REAL relaunch spacings were 49 min /
# ~3.9 h / ~4.2 h (far above cascade_s=180 s).
_I1005_BACKEND_EMPTY = (
    '{"attempts": [], "chosen_kind": "gcp", "cluster": null, "elapsed_seconds": 0.0, '
    '"extra": {"estimated'
)
_I1005_RUN_LAUNCHED = (
    "pod=eps-issue-1005 pid=n/a-gcp-startup log_abs=/workspace/logs/issue-1005.log "
    "cmd='uv run python scr"
)

ISSUE1005_FLAGGED_LAUNCH_TS = {
    "2026-07-15T18:53:36Z",
    "2026-07-15T19:42:32Z",
    "2026-07-15T23:39:12Z",
    "2026-07-16T03:53:26Z",
}


def _issue1005_cascade_events() -> list[dict]:
    """The four #1005 machine launch cascades, verbatim (see the extraction
    recipe above)."""
    return [
        # Pre-note advisory (keeps the 18:51:16Z none-note's realistic
        # info flag alive under the all-class grace trim).
        _ev(
            "2026-07-15T18:36:13Z",
            kind="epm:compute-deviation",
            note=(
                "component: Phase B capture (+ downstream fits' EPM_FIT_DEVICE)\n"
                "planned_wall_h: 2.5\nprojected_wall_h:"
            ),
        ),
        # Cascade 1 (relaunch #2): note -> cluster (94 s) -> run (+46 s).
        _ev(
            "2026-07-15T18:51:16Z",
            note=(
                "external-markers triaged: none (window since the 14:19Z dispatch "
                "record: only this session's own com"
            ),
        ),
        _ev(
            "2026-07-15T18:51:40Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=_I1005_BACKEND_EMPTY,
        ),
        _ev(
            "2026-07-15T18:52:50Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note=(
                '{"attempt_id": "att-20260715-185141", "backend": "gcp", '
                '"instance_id": "3744396712672389134", "insta'
            ),
        ),
        _ev(
            "2026-07-15T18:52:51Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=(
                '{"attempts": [{"cluster": null, "detail": "gcp rung flexstart_a100_80 '
                "primary-lane attempt #2 of cap"
            ),
        ),
        _ev("2026-07-15T18:53:36Z", kind="epm:run-launched", note=_I1005_RUN_LAUNCHED),
        # Cascade 2 (relaunch #3, the failure+relaunch shape): the watcher's
        # own 19:33:01Z nudge + the failure pair are window candidates for
        # the 19:39:44Z none-note.
        _ev(
            "2026-07-15T19:33:01Z",
            by="autonomous_session_watch",
            note=(
                "[autonomous_session_watch:triage-observer] post-hoc triage-duty "
                "review: the compute dispatch record "
            ),
        ),
        _ev(
            "2026-07-15T19:36:54Z",
            kind="epm:failure",
            note=(
                "failure_class: code\nphase: capture/determinism_check (relaunch #2, "
                "att-20260715-185141, exit 1 at 19"
            ),
        ),
        _ev(
            "2026-07-15T19:36:55Z",
            kind="epm:failure-lesson",
            note=(
                "<!-- epm:failure-lesson v1 -->\nfailure_class: code\nphase: "
                "capture/determinism_check (issue1005_run.p"
            ),
        ),
        _ev(
            "2026-07-15T19:39:30Z",
            kind="epm:workflow-fix-task-filed",
            note=(
                "filed_task: #1362; target_file: .claude/rules/gotchas.md; "
                "fingerprint: c263f7d9ebfa; session_spawned"
            ),
        ),
        _ev(
            "2026-07-15T19:39:44Z",
            note=(
                "external-markers triaged: none (window since the 18:53Z run-launched: "
                "only this session's own failur"
            ),
        ),
        _ev(
            "2026-07-15T19:40:26Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=_I1005_BACKEND_EMPTY,
        ),
        _ev(
            "2026-07-15T19:41:53Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note=(
                '{"attempt_id": "att-20260715-194027", "backend": "gcp", '
                '"instance_id": "7394631250800696507", "insta'
            ),
        ),
        _ev(
            "2026-07-15T19:41:54Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=(
                '{"attempts": [{"cluster": null, "detail": "gcp rung flexstart_a100_80 '
                "primary-lane attempt #3 of cap"
            ),
        ),
        _ev("2026-07-15T19:42:32Z", kind="epm:run-launched", note=_I1005_RUN_LAUNCHED),
        # Cascade 3 (relaunch #4, ~3.9 h later — record 23:39:12Z, the third
        # 07-15 sidecar flag; §14.5 addendum): the failure pair lands within
        # grace of the 23:33:45Z note, so that note stays quiet.
        _ev(
            "2026-07-15T23:32:17Z",
            kind="epm:failure",
            note=(
                "failure_class: code\nphase: F2/F3 prefix-constancy assert "
                "(relaunch #3, att-20260715-194027, exit 1 a"
            ),
        ),
        _ev(
            "2026-07-15T23:32:19Z",
            kind="epm:failure-lesson",
            note=(
                "<!-- epm:failure-lesson v1 -->\nfailure_class: code\nphase: F2/F3 "
                "prefix-constancy assert (issue1005_f"
            ),
        ),
        _ev(
            "2026-07-15T23:33:45Z",
            note=(
                "external-markers triaged: none (window since 18:53Z run-launched: "
                "only this session's own diagnosis/"
            ),
        ),
        _ev(
            "2026-07-15T23:34:23Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=_I1005_BACKEND_EMPTY,
        ),
        _ev(
            "2026-07-15T23:35:56Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=_I1005_BACKEND_EMPTY,
        ),
        _ev(
            "2026-07-15T23:36:36Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=_I1005_BACKEND_EMPTY,
        ),
        _ev(
            "2026-07-15T23:38:33Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note=(
                '{"attempt_id": "att-20260715-233638", "backend": "gcp", '
                '"instance_id": "5330955359776778068", "insta'
            ),
        ),
        _ev(
            "2026-07-15T23:38:35Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=(
                '{"attempts": [{"cluster": null, "detail": "rung flexstart_a100_80: '
                "gcloud create returned 1; matched"
            ),
        ),
        _ev("2026-07-15T23:39:12Z", kind="epm:run-launched", note=_I1005_RUN_LAUNCHED),
        # Cascade 4 (relaunch #5, ~4.2 h later — the 07-16 sidecar flag).
        _ev(
            "2026-07-16T03:47:42Z",
            kind="epm:failure",
            note=(
                "failure_class: code — relaunch #4 (att-20260715-233638, FLEX_START "
                "A100-40 39.49 GiB) died 03:24Z in"
            ),
        ),
        _ev(
            "2026-07-16T03:47:55Z",
            kind="epm:failure-lesson",
            note=(
                "round: 4 (relaunch #4, att-20260715-233638). lesson: after a "
                '"load fresh copy for a determinism chec'
            ),
        ),
        _ev(
            "2026-07-16T03:48:43Z",
            note=(
                "crash-fix round 4 pre-dispatch triage (relaunch #5): external-markers "
                "triaged: none new since round-"
            ),
        ),
        _ev(
            "2026-07-16T03:49:05Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=_I1005_BACKEND_EMPTY,
        ),
        _ev(
            "2026-07-16T03:52:54Z",
            kind="epm:cluster-launched",
            by="backends.gcp",
            note=(
                '{"attempt_id": "att-20260716-034906", "backend": "gcp", '
                '"instance_id": "4001286622270581659", "insta'
            ),
        ),
        _ev(
            "2026-07-16T03:52:55Z",
            kind="epm:backend-selected",
            by="backends.router",
            note=(
                '{"attempts": [{"cluster": null, "detail": "gcp rung flexstart_a100_80 '
                "primary-lane attempt #1 of cap"
            ),
        ),
        _ev("2026-07-16T03:53:26Z", kind="epm:run-launched", note=_I1005_RUN_LAUNCHED),
    ]


def test_audit_issue1005_launch_cascade_replay():
    """#1400 acceptance criterion 2 (the incident fixture): replaying the
    four real #1005 cascades under PRODUCTION defaults (adjacency 1800 /
    grace 120 / epoch on / cascade_s default) yields ZERO launch-class
    violations — in particular none of the four pre-fix zero-candidate
    sidecar records flags. The legitimate ``none-with-candidates`` INFO
    flags on the 18:51:16Z / 19:39:44Z notes themselves remain (they match
    the documented predicate and are deliberately unchanged); the 23:33:45Z
    / 03:48:43Z notes stay quiet — their only window candidates land within
    the 120 s grace trim."""
    events = _issue1005_cascade_events()
    result = audit_dispatch_triage(events)
    violations = result["violations"]
    assert [v for v in violations if v["violation"] != "none-with-candidates"] == []
    assert not {v["record_ts"] for v in violations} & ISSUE1005_FLAGGED_LAUNCH_TS
    # The realistic residual: exactly the two none-notes with post-trim
    # candidates flag, at info (no external signatures in their windows).
    assert [(v["record_ts"], v["severity"]) for v in violations] == [
        ("2026-07-15T18:51:16Z", "info"),
        ("2026-07-15T19:39:44Z", "info"),
    ]


def test_audit_issue779_replay():
    """Acceptance criteria 1-2 (#967 plan §1): the frozen #779 window flags
    EXACTLY the two incident records at warn with the epoch off, and nothing
    with the production epoch on (all fixture rows are legacy pre-fix).

    #1400 regression floor (byte-unchanged): under the all-class grace trim
    the flagged 14:56:21Z launch KEEPS its 14:41:17Z ``epm:progress``
    candidate (904 s before the launch, beyond the 120 s grace; the
    14:54:43Z row at 98 s is trimmed) — post-trim count 1, so the origin
    TRUE POSITIVE survives empty-window suppression; and the 20:46:04Z
    breadcrumb keeps >= 10 post-trim candidates. Cascade coalescing is
    inert here: the fixture's only launch-kind record has no line-less
    launch sibling within 180 s."""
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
