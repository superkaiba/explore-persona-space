"""Prose-side pin for the Step 6d.2 `pid-stale-workload-live` branch row (#2265).

``scripts/poll_pipeline.py`` refuses a ``dead`` verdict its own same-tick
evidence contradicts (the #2223 false-dead class: ``status="dead"`` beside
``gpu_util="97,100,100,100"`` and a 294s-fresh log) and reports the
non-terminal ``pid-stale-workload-live`` instead. The orchestrator's Step
6d.2 branch table is the ONE prose consumer that must route it: NOT a
failure trigger (the ``stalled``/``dead`` row posts ``epm:failure v1`` +
CRON-TEARDOWN + ``status:blocked`` — exactly the damage the veto exists to
prevent), loop on the short interval, and resolve the contradiction on the
FIRST such tick (probe → repair the pid file + re-post ``epm:run-launched``,
or let workload-scoped evidence decay to ``dead`` — with the explicit
conclude/post-failure arm when POD-WIDE evidence persists and decay
structurally cannot arrive). This test pins that row so a prose rewording
cannot silently re-route the new status into the failure path.

Shape copied from ``tests/test_issue_skill_tick_parse_preservation.py``
(bounded-span extraction over the SKILL.md text; no imports beyond stdlib).
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"

_ROW_ANCHOR = 'status == "pid-stale-workload-live"'
# The row lives inside the Step 6d.2 decision table, between the
# stalled|dead failure row and the running row.
_TABLE_START = 'status == "stalled" | "dead"'
_TABLE_END = 'status == "running"'


def _branch_table_span() -> str:
    """The branch-table slice from the stalled|dead row to the running row
    (raises ValueError if either anchor is gone — itself a pin)."""
    text = issue_skill_text()
    start = text.index(_TABLE_START)
    end = text.index(_TABLE_END, start)
    return text[start:end]


def test_step_6d2_pid_stale_workload_live_row_present():
    """The row exists in the 6d.2 region and carries the never-post-
    epm:failure clause: the token, the explicit do-NOT-post instruction, and
    the stalled|dead row's not-a-failure-trigger cross-clause."""
    span = _branch_table_span()
    assert _ROW_ANCHOR in span, (
        "the Step 6d.2 branch table lost its pid-stale-workload-live row "
        "(#2265) — the orchestrator would fall through to ad-hoc handling"
    )
    row = span[span.index(_ROW_ANCHOR) :]
    assert "NOT a failure trigger" in row, "the row lost its not-a-failure-trigger lead"
    assert "do NOT" in row and "epm:failure" in row, (
        "the row lost the explicit never-post-epm:failure clause"
    )
    # The stalled|dead row's preamble clause: the failure path is scoped to
    # those two statuses ONLY.
    preamble = span[: span.index(_ROW_ANCHOR)]
    assert "never a" in preamble and "failure trigger" in preamble, (
        "the stalled|dead row lost its pid-stale-workload-live exclusion clause"
    )


def test_probe_then_repair_or_decay_instruction_present():
    """The row instructs the first-tick contradiction probe → pid-file
    repair + epm:run-launched re-post, the decay-to-dead arm, AND the v4
    conclude/post-failure arm for persisting pod-wide evidence."""
    span = _branch_table_span()
    row = span[span.index(_ROW_ANCHOR) :]
    # Probe-then-repair: bracketed pgrep per the pid-file launch contract,
    # rewrite the pid file, re-post epm:run-launched.
    assert "FIRST such tick" in row, "the row lost the first-tick probe timing"
    assert "bracketed pgrep" in row and "Pid-file launch" in row, (
        "the row lost the pod-side-reporting.md probe recipe pointer"
    )
    assert "rewrite" in row and "pid file" in row, "the row lost the pid-file repair step"
    assert "re-post epm:run-launched" in row, "the row lost the marker re-post step"
    # Decay arm: workload-scoped evidence ages out and a later tick reads dead.
    assert "decays within ~stall_sec" in row, "the row lost the evidence-decay arm"
    # v4 conclude/post-failure arm: pod-wide evidence never decays — the
    # orchestrator concludes the leg dead itself.
    assert "POD-WIDE" in row, "the row lost the pod-wide-evidence fork"
    assert "CONCLUDE the leg dead" in row, (
        "the row lost the conclude/post-failure arm for persisting pod-wide evidence"
    )
