"""Prose-side pin for the Step 6d.2 `phase-done` branch row (#2610).

``scripts/poll_pipeline.py`` returns the TERMINAL-SUCCESS ``phase-done``
when the launch note declared ``done_file=``, every pid probe reads dead,
and the declared completion file is fresh (newer than the pid file) — the
clean completion of a SINGLE-PHASE dispatcher invocation that deliberately
emits no run-level ``[phase=done]``. The orchestrator's Step 6d.2 branch
table is the ONE prose consumer that must route it: exit the completed
phase's wait and dispatch the NEXT planned phase — PER-WORKLOAD success,
never the run-level terminal path (no status:verifying transition, no
Step 8 pod termination, no synthesized run-level done milestone,
``current_phase`` preserved), and the poll chain re-armed for the next
launch. This test pins that row — and the post-tick-duties summary that
mirrors it — so a prose rewording cannot silently re-route the new status
into the run-level terminal path, or drop it back into the unhandled
fall-through the #2610 round-1 review flagged.

Shape copied from ``tests/test_issue_skill_dead_veto_status_pin.py``
(bounded-span extraction over the composed `/issue` spec; stdlib only).
Row assertions run over a WHITESPACE-NORMALIZED slice (the branch table is
a `#`-prefixed comment block wrapped at ~40 columns, so multi-word pins
would otherwise break on any legitimate re-wrap).
"""

from __future__ import annotations

from tests.issue_skill_source import issue_skill_text

_ROW_ANCHOR = 'status == "phase-done"'
# The row lives inside the Step 6d.2 decision table, after the `done` row
# (its terminal-success sibling) and before the `running` row.
_DONE_ROW_ARROW = "-> exit loop; transition to status:verifying"
_TABLE_END = 'status == "running"'


def _normalized_row() -> str:
    """The phase-done row up to the next ``status == "`` sibling row, with
    the comment ``#`` prefixes stripped and all whitespace collapsed
    (raises ValueError if the anchor is gone — itself a pin)."""
    text = issue_skill_text()
    start = text.index(_ROW_ANCHOR)
    nxt = text.find('status == "', start + len(_ROW_ANCHOR))
    raw = text[start:nxt] if nxt != -1 else text[start:]
    return " ".join(raw.replace("#", " ").split())


def test_step_6d2_phase_done_row_sits_between_done_and_running():
    """The row exists INSIDE the Step 6d.2 branch table — after the
    run-level `done` row, before the `running` row — so the composed spec
    routes the status instead of falling through to ad-hoc handling."""
    text = issue_skill_text()
    done_row = text.index(_DONE_ROW_ARROW)
    row = text.index(_ROW_ANCHOR)
    running_row = text.index(_TABLE_END, row)
    assert done_row < row < running_row, (
        "the Step 6d.2 branch table lost its phase-done row (#2610) — the "
        "terminal per-workload success status would fall through unhandled"
    )


def test_phase_done_row_dispatch_next_phase_semantics():
    """The row instructs: exit the completed phase's wait, dispatch the
    NEXT planned phase, re-arm the poll chain, and route the last-phase
    case to the orchestrator's own results-landed handling."""
    row = _normalized_row()
    assert "terminal SUCCESS for the LAUNCHED single-phase workload" in row, (
        "the row lost its terminal-per-workload-success lead"
    )
    assert "Exit the completed phase's wait" in row, "the row lost the exit-phase-wait action"
    assert "dispatches the NEXT planned phase" in row, (
        "the row lost the dispatch-next-phase instruction"
    )
    assert "re-arms the poll chain" in row, "the row lost the poll-chain re-arm instruction"
    assert "results-landed handling" in row, (
        "the row lost the last-phase results-landed disposition"
    )


def test_phase_done_row_excludes_run_level_terminal_path():
    """The row is PER-WORKLOAD success, never run-level `done`: it keeps
    `current_phase` (no synthesized run-level done milestone) and excludes
    the run-level verifying transition + Step 8 pod termination."""
    row = _normalized_row()
    assert "PER-WORKLOAD success, NOT run-level done" in row, (
        "the row lost the not-run-level-done scoping clause"
    )
    assert "preserve the local `current_phase`" in row, (
        "the row lost the current_phase preservation clause"
    )
    assert "never synthesize a run-level done milestone" in row, (
        "the row lost the no-synthesized-done clause"
    )
    assert "do NOT transition to status:verifying" in row, (
        "the row lost the no-run-level-verifying exclusion"
    )
    assert "do NOT enter Step 8 pod-termination" in row, (
        "the row lost the no-Step-8-termination exclusion"
    )


def test_post_tick_duties_summary_names_phase_done():
    """The post-tick-duties summary below the table mirrors the row — the
    line the #2610 round-1 review flagged as claiming `status=done` was
    the only terminal-success action."""
    text = issue_skill_text()
    start = text.index("only post-tick duties are:")
    summary = " ".join(text[start : start + 1500].split())
    assert "dispatch the next planned phase on `status=phase-done`" in summary, (
        "the post-tick-duties summary lost its phase-done dispatch duty"
    )
    assert "never run-level verifying / Step 8 termination" in summary, (
        "the post-tick-duties summary lost the phase-done run-level exclusion"
    )
