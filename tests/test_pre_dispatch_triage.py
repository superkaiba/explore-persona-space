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


def test_advisory_by_values_are_never_machine_stripped():
    # #966: pm-chat / watcher / spawn-helper identities are ADVISORY emitter
    # identities, not machine bookkeeping — adding one to TRIAGE_MACHINE_BY
    # would strip exactly the advisory content triage exists to surface.
    for by in ("pm-chat", "autonomous_session_watch", "spawn_session-stop", "spawn_session"):
        assert by not in TRIAGE_MACHINE_BY, by
    out = triage_candidates_since_last_dispatch([_ev("2026-07-03T00:00:00Z", by="pm-chat")])
    assert len(out) == 1
