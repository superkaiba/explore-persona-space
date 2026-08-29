"""Prose-side pin for the Step 6d.2 `phase-done` branch row (#2610).

``scripts/poll_pipeline.py`` returns the TERMINAL-SUCCESS ``phase-done``
when the launch note declared ``done_file=``, every pid probe reads dead,
and the declared completion file is fresh (newer than the pid file) — the
clean completion of a SINGLE-PHASE dispatcher invocation that deliberately
emits no run-level ``[phase=done]``. The orchestrator's Step 6d.2 branch
table is the ONE prose consumer that must route it, and as of round 3 the
row carries TWO explicitly scoped arms:

- ``(a) NEXT PHASE REMAINS`` — per-workload success: keep
  ``current_phase``, do NOT transition to status:verifying, do NOT enter
  Step 8 pod-termination; dispatch the next planned phase and re-arm the
  poll chain. The run-level exclusions bind THIS arm only.
- ``(b) LAST PLANNED PHASE`` — keep ``current_phase``, then enter Step 7's
  canonical ``epm:results`` check (never Step 8's results-landed
  parallel-spawn block directly); only a landed ``epm:results`` marker
  licenses the verifying transition into Step 8, whose upload-PASS branch
  owns pod termination.

The round-2 single-row shape let UNCONDITIONAL verifying/Step-8 exclusions
coexist with a last-phase "results-landed handling" route — internally
contradictory (the round-2 reconciler blocker
``phase-done-orchestrator-unhandled``), and the round-2 test passed it
because it only grepped for the coexistence of both phrase families. The
arm-span assertions below pin each arm's scope SEPARATELY, so a text that
re-unconditionalizes the exclusions (or routes the last phase straight to
Step 8's parallel-spawn block) is test-breaking, not just reviewable.

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
# Arm anchors (round 3): the two explicitly scoped dispositions.
_ARM_A_ANCHOR = "(a) NEXT PHASE REMAINS"
_ARM_B_ANCHOR = "(b) LAST PLANNED PHASE"
# The run-level exclusions that must bind the NON-final arm only.
_EXCLUSION_PHRASES = (
    "do NOT transition to status:verifying",
    "do NOT enter Step 8 pod-termination",
)


def _normalized_row() -> str:
    """The phase-done row up to the next ``status == "`` sibling row, with
    the comment ``#`` prefixes stripped and all whitespace collapsed
    (raises ValueError if the anchor is gone — itself a pin)."""
    text = issue_skill_text()
    start = text.index(_ROW_ANCHOR)
    nxt = text.find('status == "', start + len(_ROW_ANCHOR))
    raw = text[start:nxt] if nxt != -1 else text[start:]
    return " ".join(raw.replace("#", " ").split())


def _arm_spans() -> tuple[str, str, str]:
    """Split the normalized row into (preamble, next-phase arm, last-phase
    arm) on the two arm anchors (raises ValueError if either anchor is
    gone — itself a pin that the two-arm split survived rewording)."""
    row = _normalized_row()
    a = row.index(_ARM_A_ANCHOR)
    b = row.index(_ARM_B_ANCHOR, a)
    return row[:a], row[a:b], row[b:]


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


def test_phase_done_row_preamble_per_workload_semantics():
    """The shared preamble (before either arm) carries the per-workload
    framing common to BOTH arms: terminal per-workload success, exit the
    completed phase's wait, preserve `current_phase`, and never a
    synthesized run-level done milestone."""
    preamble, _, _ = _arm_spans()
    assert "terminal SUCCESS for the LAUNCHED single-phase workload" in preamble, (
        "the row lost its terminal-per-workload-success lead"
    )
    assert "Exit the completed phase's wait" in preamble, "the row lost the exit-phase-wait action"
    assert "PER-WORKLOAD success, NOT run-level done" in preamble, (
        "the row lost the not-run-level-done scoping clause"
    )
    assert "preserve the local `current_phase`" in preamble, (
        "the row lost the current_phase preservation clause"
    )
    assert "never synthesize a run-level done milestone" in preamble, (
        "the row lost the no-synthesized-done clause"
    )


def test_next_phase_arm_owns_dispatch_and_exclusions():
    """Arm (a) — next phase remains — carries the next-phase dispatch, the
    poll-chain re-arm, BOTH run-level exclusions, and the explicit
    statement that the exclusions bind the non-final arm only."""
    _, arm_a, _ = _arm_spans()
    assert "dispatches the NEXT planned phase" in arm_a, (
        "arm (a) lost the dispatch-next-phase instruction"
    )
    assert "re-arms the poll chain" in arm_a, "arm (a) lost the poll-chain re-arm instruction"
    for phrase in _EXCLUSION_PHRASES:
        assert phrase in arm_a, (
            f"arm (a) lost the run-level exclusion {phrase!r} — the "
            "non-final arm is where the verifying/Step-8 exclusions live"
        )
    assert "non-final arm only" in arm_a, (
        "arm (a) lost the explicit exclusions-bind-the-non-final-arm-only scoping statement"
    )


def test_exclusions_scoped_to_next_phase_arm_only():
    """The verifying/Step-8 exclusions appear ONLY inside arm (a) — not in
    the shared preamble (where they would bind both arms, the round-2
    contradiction) and not in arm (b) (where they would forbid the only
    terminal route). This is the assertion the round-2
    coexistence-of-phrases grep lacked."""
    preamble, _, arm_b = _arm_spans()
    for phrase in _EXCLUSION_PHRASES:
        assert phrase not in preamble, (
            f"run-level exclusion {phrase!r} moved into the shared preamble — "
            "unconditional again, it forbids the last-phase terminal route "
            "(the #2610 round-2 reconciler blocker)"
        )
        assert phrase not in arm_b, (
            f"run-level exclusion {phrase!r} appears in the LAST-planned-phase "
            "arm — it would forbid the very verifying -> Step 8 route that arm "
            "exists to take"
        )


def test_last_phase_arm_routes_via_step7_results_check():
    """Arm (b) — last planned phase — preserves `current_phase` and routes
    through Step 7's canonical `epm:results` check; only a landed
    `epm:results` licenses the verifying transition into Step 8. It never
    routes directly to Step 8's results-landed parallel-spawn block, and
    the round-2 affirmative results-landed-handling route is gone."""
    _, _, arm_b = _arm_spans()
    assert "preserve `current_phase`" in arm_b, "arm (b) lost the current_phase preservation clause"
    assert "Step 7's canonical `epm:results` check" in arm_b, (
        "arm (b) lost the Step 7 epm:results check route"
    )
    assert "never Step 8's results-landed parallel-spawn block directly" in arm_b, (
        "arm (b) lost the explicit never-direct-to-Step-8-parallel-spawn negation"
    )
    assert "Only a landed `epm:results` marker licenses the verifying transition" in arm_b, (
        "arm (b) lost the landed-epm:results-licenses-verifying clause"
    )
    assert "upload-PASS branch owns pod termination" in arm_b, (
        "arm (b) lost the Step-8-upload-PASS-owns-termination attribution"
    )
    # The round-2 shape routed the last phase to the orchestrator's own
    # "results-landed handling" — Step 8's parallel-spawn block by name —
    # while the exclusions forbade it. That affirmative route must not
    # reappear anywhere in the row.
    row = _normalized_row()
    assert "results-landed handling" not in row, (
        "the row re-grew the round-2 affirmative results-landed-handling "
        "route — the last phase goes through Step 7's epm:results check, "
        "never straight to Step 8's parallel-spawn block"
    )


def test_post_tick_duties_summary_scopes_both_arms():
    """The post-tick-duties summary below the table mirrors the two-arm
    row: next-phase dispatch with the never-verifying/Step-8 clause scoped
    to the non-final arm, plus the last-phase -> Step 7 `epm:results`
    route. Round 2's summary carried an unconditional \"never\" that
    forbade the only terminal path (the reconciler's blocker 2)."""
    text = issue_skill_text()
    start = text.index("only post-tick duties are:")
    summary = " ".join(text[start : start + 1500].split())
    assert "exit the completed phase's wait on `status=phase-done`" in summary, (
        "the post-tick-duties summary lost its phase-done duty"
    )
    assert "with a next planned phase remaining, dispatch it" in summary, (
        "the post-tick-duties summary lost the next-phase dispatch clause"
    )
    assert "on the LAST planned phase, enter Step 7's `epm:results` check" in summary, (
        "the post-tick-duties summary lost the last-phase Step 7 epm:results route"
    )
    # Every occurrence of the exclusion clause must carry its non-final-arm
    # scope inline — an unconditional "never" is the round-2 shape.
    token = "never run-level verifying / Step 8 termination"
    scoped = token + " on that non-final arm"
    idx = 0
    occurrences = 0
    while (i := summary.find(token, idx)) != -1:
        occurrences += 1
        assert summary[i:].startswith(scoped), (
            "the post-tick-duties summary carries an UNSCOPED "
            "never-run-level-verifying/Step-8 clause — round 2's "
            "unconditional form forbade the last-phase terminal route"
        )
        idx = i + len(token)
    assert occurrences >= 1, (
        "the post-tick-duties summary lost the phase-done run-level exclusion clause"
    )
