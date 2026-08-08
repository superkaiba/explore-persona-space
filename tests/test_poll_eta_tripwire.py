"""Tests for the phase-ETA tripwire in scripts/poll_pipeline.py (#873).

The tripwire posts ``epm:compute-deviation`` (``source: poller``,
``basis: elapsed-vs-plan``) when elapsed wall-time — per current phase or
for the whole run — exceeds ``ETA_DEVIATION_MULT`` x the plan §9
``planned_wall_h`` TOTAL. These tests pin:

* ``_parse_plan_wall_budget`` — the AC #2 parser contract: markdown pipe
  tables AND HTML tables, ALL planned_wall_h tables summed, header-derived
  table-scoped value columns, cosmetic-prefix suffixed cells, ANY
  unparseable located data row -> None (never a partial sum). Since #2172
  the parser is a thin delegation to the SHARED
  ``explore_persona_space.plan_wall_budget`` module (also c47's parser);
  the contract stays pinned HERE through the wrapper, and the float rule's
  own accept/reject fixtures live in ``tests/test_plan_wall_budget.py``;
* ``_eta_deviation_update`` — the pure decision core (strict ``>``
  boundary, per-phase + ``__run_total__`` dedup keys, fail-safe OFF on
  missing budget / disabled mult / non-running status / unknown clocks);
* ``_maybe_post_eta_deviation`` — the wiring (marker body shape, one-shot
  missing-budget log, post-failure retry, run-scope relaunch reset via
  ``_tripwire_run_scope`` — AC #6; since #2172 also the durable
  ``eta-tripwire-disabled`` ``epm:progress`` note an UNPARSEABLE-cell
  budget posts once per run — #2172 AC #5);
* the ``PollResult.eta_deviation_posted`` field + its ``main()`` JSON
  enumeration (parity with ``gpu_idle_advisory_posted``).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_eta_tripwire_under_test")

# Imported AFTER the pp load: poll_pipeline's src/ shim has then pinned
# ``explore_persona_space`` resolution to THIS checkout's src/, so the
# budgets the tests construct come from the same module copy pp itself uses.
from explore_persona_space.plan_wall_budget import (  # noqa: E402
    PlanWallBudget,
    UnparseableWallCell,
)

# ── _parse_plan_wall_budget (AC #2 parser contract) ──────────────────────────

# The planner-section-reference §9 exemplar table shape (markdown pipe form).
MD_TABLE = """\
## 9. Compute

| component | planned_wall_h | planned_gpu_h | parallelism | basis |
|---|---|---|---|---|
| smoke-phase per-cell train | 0.5 | 0.5 | TP=1 | "matched to #382 round-2" |
| sweep all-cells train | 16 | 64 | 4x H100 ZeRO-3 across 8 cells | "16h x 8 cells / 4 GPU" |
| eval all-cells generation | 2 | 2 | TP=1 | "vLLM batched" |
"""

# The #779-style HTML form (single <tr><th> header row, suffixed numeric cell).
HTML_TABLE = """\
<h3>Per-component compute-projection table</h3>
<table>
<tr><th>component</th><th>planned_wall_h</th><th>planned_gpu_h</th><th>basis</th></tr>
<tr><td>Arm B/C corpus gen</td><td>4.5</td><td>4.5</td><td>72k short rollouts</td></tr>
<tr><td>Batch-API judge</td><td>3 (async, off-GPU)</td><td>0</td><td>batch precedent</td></tr>
</table>
"""


def test_parse_plan_wall_budget_markdown_table() -> None:
    assert pp._parse_plan_wall_budget(MD_TABLE) == pytest.approx(18.5)


def test_parse_plan_wall_budget_html_table() -> None:
    """HTML <th>-derived column; the "3 (async, off-GPU)" cell parses via
    leading-float to 3.0 (suffixed-but-numeric is NOT unparseable)."""
    assert pp._parse_plan_wall_budget(HTML_TABLE) == pytest.approx(7.5)


def test_parse_plan_wall_budget_missing_table_returns_none() -> None:
    assert pp._parse_plan_wall_budget("## 9. Compute\n\nno table here\n") is None
    assert pp._parse_plan_wall_budget("") is None
    # A table WITHOUT a planned_wall_h header is not a located table.
    assert (
        pp._parse_plan_wall_budget(
            "| component | est. wall_h | basis |\n|---|---|---|\n| a | 5 | b |\n"
        )
        is None
    )


def test_parse_plan_wall_budget_any_unparseable_row_returns_none() -> None:
    """ONE non-numeric planned_wall_h cell -> None for the WHOLE budget —
    never a partial sum (an under-parsed budget is the one false-positive
    path, AC #2)."""
    bad_md = MD_TABLE.replace("| sweep all-cells train | 16 |", "| sweep all-cells train | TBD |")
    assert pp._parse_plan_wall_budget(bad_md) is None
    bad_html = HTML_TABLE.replace("<td>4.5</td>", "<td>see prose</td>", 1)
    assert pp._parse_plan_wall_budget(bad_html) is None


def test_parse_plan_wall_budget_two_tables_summed() -> None:
    """#479-style Stage 1 + Stage 2: rows from BOTH planned_wall_h tables
    contribute (a single-header parse under-counts and false-fires)."""
    two_stage = (
        "### Stage 1\n\n"
        "| component | planned_wall_h | basis |\n|---|---|---|\n| s1 train | 4 | b |\n"
        "\n### Stage 2\n\n"
        "| component | planned_wall_h | basis |\n|---|---|---|\n| s2 eval | 1.5 | b |\n"
    )
    assert pp._parse_plan_wall_budget(two_stage) == pytest.approx(5.5)
    # Mixed formats sum too (markdown + HTML in one document).
    assert pp._parse_plan_wall_budget(MD_TABLE + "\n" + HTML_TABLE) == pytest.approx(26.0)


def test_parse_plan_wall_budget_html_reordered_columns() -> None:
    """planned_wall_h NOT the 2nd column -> the header-DERIVED index sums the
    right column (never a hardcoded ordinal)."""
    reordered = """\
<table>
<tr><th>component</th><th>planned_gpu_h</th><th>planned_wall_h</th></tr>
<tr><td>train</td><td>64</td><td>16</td></tr>
<tr><td>eval</td><td>2</td><td>2.5</td></tr>
</table>
"""
    assert pp._parse_plan_wall_budget(reordered) == pytest.approx(18.5)


def test_parse_plan_wall_budget_html_scoped_to_owning_table() -> None:
    """An UNRELATED HTML table in the same document (numeric 2nd column, no
    planned_wall_h header) contributes NOTHING — the row scan is scoped to
    the owning table, never a document-wide <td> scan."""
    doc = (
        HTML_TABLE
        + """
<table>
<tr><th>condition</th><th>n_seeds</th></tr>
<tr><td>c1</td><td>500</td></tr>
<tr><td>c2</td><td>1000</td></tr>
</table>
"""
    )
    assert pp._parse_plan_wall_budget(doc) == pytest.approx(7.5)


_2163_ROW = "| P6-GPU conditional cell (1× H100 fp64 eigh + solve + score) | (1.5) | (1.5 realized; 4 booked) | single GPU cell | pilot-gated on the GPU cell itself: eigh(8,192) fp64 timed on the H100 FIRST, ×(d_B/8192)³ cubic extrapolation before committing the full eigh, abort >2×; cross-check: cusolver RAM arithmetic (≤49,152 fp64 = 19.3 GB matrix + vectors + workspace < 80 GB) and the batched-eigh cuSOLVER caveat (one-cell benchmark on BOTH devices before committing, per vectorize-many-cell-fits GPU caveat) |"  # noqa: E501, RUF001


def test_parse_plan_wall_budget_2163_verbatim_row_regression() -> None:
    """AC #1 on the real artifact (#2163 plans/plan.md:303, row VERBATIM):
    pre-#2172 the leading ``(`` of ``(1.5)`` yielded no leading float, so
    this ONE cell discarded every sibling row and disarmed the whole ~6h
    run; under the shared cosmetic-prefix rule it contributes 1.5 and the
    budget stays armed."""
    table = (
        "| component | planned_wall_h | planned_gpu_h | parallelism | basis |\n"
        "|---|---|---|---|---|\n"
        "| P0 staging | 0.5 | 0 | single VM session | measured |\n" + _2163_ROW + "\n"
    )
    assert pp._parse_plan_wall_budget(table) == pytest.approx(2.0)


def test_plan_total_wall_h_for_issue_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing task / raising resolver yields None, never a crash."""
    monkeypatch.setattr(pp, "find_task_path", lambda issue: (_ for _ in ()).throw(KeyError(issue)))
    assert pp._plan_total_wall_h_for_issue(999_999) is None


# ── _eta_deviation_update (pure decision core) ───────────────────────────────

# Budget = 2.0 x 1.0h = 7200s; BASE phase elapsed = 7201s (just over).
ETA_KW: dict[str, Any] = {
    "status": "running",
    "current_phase": "extract",
    "phase_started_epoch": 1000,
    "run_age_sec": None,
    "total_planned_wall_h": 1.0,
    "posted_keys": set(),
    "now_epoch": 1000 + 7200 + 1,
    "mult": 2.0,
}


def test_eta_no_budget_never_fires() -> None:
    kw = {**ETA_KW, "total_planned_wall_h": None, "now_epoch": 10**9}
    assert pp._eta_deviation_update(**kw).posts == ()
    kw = {**ETA_KW, "total_planned_wall_h": 0.0, "now_epoch": 10**9}
    assert pp._eta_deviation_update(**kw).posts == ()


def test_eta_disabled_when_mult_non_positive() -> None:
    assert pp._eta_deviation_update(**{**ETA_KW, "mult": 0.0}).posts == ()
    assert pp._eta_deviation_update(**{**ETA_KW, "mult": -1.0}).posts == ()


def test_eta_phase_over_threshold_fires_once_and_dedups() -> None:
    update = pp._eta_deviation_update(**ETA_KW)
    (post,) = update.posts
    assert post.scope == "phase" and post.dedup_key == "extract"
    assert post.planned_wall_h == pytest.approx(1.0)
    assert post.ratio == pytest.approx(7201 / 3600, rel=1e-3)
    # Same tick with the key already posted -> no repost.
    deduped = pp._eta_deviation_update(**{**ETA_KW, "posted_keys": {"extract"}})
    assert deduped.posts == ()


def test_eta_phase_under_threshold_does_not_fire() -> None:
    kw = {**ETA_KW, "now_epoch": 1000 + 7200 - 1}
    assert pp._eta_deviation_update(**kw).posts == ()


def test_eta_exactly_at_threshold_does_not_fire() -> None:
    """The boundary is STRICT ``>`` (asymmetric with the width advisory's
    ``>=`` — both pinned)."""
    kw = {**ETA_KW, "now_epoch": 1000 + 7200}
    assert pp._eta_deviation_update(**kw).posts == ()


def test_eta_run_total_fires_with_run_total_dedup_key() -> None:
    kw = {
        **ETA_KW,
        "current_phase": "unknown",
        "phase_started_epoch": 0,
        "run_age_sec": 7201.0,
    }
    (post,) = pp._eta_deviation_update(**kw).posts
    assert post.scope == "run" and post.dedup_key == pp.ETA_RUN_TOTAL_KEY
    # Run key already posted -> no repost.
    deduped = pp._eta_deviation_update(**{**kw, "posted_keys": {pp.ETA_RUN_TOTAL_KEY}})
    assert deduped.posts == ()
    # Exactly at the run budget -> strict > does not fire.
    assert pp._eta_deviation_update(**{**kw, "run_age_sec": 7200.0}).posts == ()


def test_eta_both_checks_fire_two_posts() -> None:
    kw = {**ETA_KW, "run_age_sec": 7201.0}
    posts = pp._eta_deviation_update(**kw).posts
    assert {p.dedup_key for p in posts} == {"extract", pp.ETA_RUN_TOTAL_KEY}


def test_eta_non_running_status_does_not_fire() -> None:
    for status in ("stalled", "dead", "done", "gate"):
        kw = {**ETA_KW, "status": status, "run_age_sec": 10**9}
        assert pp._eta_deviation_update(**kw).posts == ()


def test_eta_unknown_phase_skips_phase_check_run_check_still_fires() -> None:
    for phase in ("", "unknown", "done"):
        kw = {**ETA_KW, "current_phase": phase, "run_age_sec": 7201.0}
        posts = pp._eta_deviation_update(**kw).posts
        assert [p.scope for p in posts] == ["run"]


def test_eta_unknown_phase_start_and_run_age_does_not_fire() -> None:
    """The full fail-safe: no phase-start clock AND no run-launch clock ->
    nothing can fire, no matter how large now_epoch is."""
    kw = {**ETA_KW, "phase_started_epoch": 0, "run_age_sec": None, "now_epoch": 10**9}
    assert pp._eta_deviation_update(**kw).posts == ()


# ── _maybe_post_eta_deviation (wiring) ───────────────────────────────────────


def _budget_for(total: float | None) -> PlanWallBudget:
    """Map the legacy ``total`` seam value onto a :class:`PlanWallBudget`.

    ``None`` models the #873 shape — a REAL plan with no wall table
    (``reason == "no_table"``, the quiet log-only disable); a float models
    a fully-parseable one-row table.
    """
    if total is None:
        return PlanWallBudget(total_h=None, rows=(), unparseable=(), reason="no_table")
    return PlanWallBudget(total_h=total, rows=(total,), unparseable=(), reason="")


def _wire(monkeypatch: pytest.MonkeyPatch, *, total: float | None, posted: list[dict]):
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_plan_wall_budget_for_issue", lambda issue: _budget_for(total))


def test_maybe_post_eta_deviation_marker_body_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict] = []
    _wire(monkeypatch, total=1.0, posted=posted)
    now = 100_000
    keys, posted_flag, warned = pp._maybe_post_eta_deviation(
        issue=873,
        pod="pod-873",
        status="running",
        current_phase="extract",
        last_phase_change_epoch=now - 3 * 3600,
        run_age_sec=None,
        prev_state={},
        now_epoch=now,
    )
    assert posted_flag is True and "extract" in keys and warned is False
    (p,) = posted
    assert p["key"] == "epm:compute-deviation"
    assert p["source"] == "poller" and p["basis"] == "elapsed-vs-plan"
    assert p["phase"] == "extract" and p["pod"] == "pod-873"
    note = p["note"]
    for token in ("component:", "planned_wall_h:", "projected_wall_h:", "ratio:", "basis:"):
        assert token in note, f"missing {token!r} in note: {note}"
    assert "source: poller" in note and "nothing stopped" in note


def test_maybe_post_eta_deviation_logs_once_on_missing_budget(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    posted: list[dict] = []
    _wire(monkeypatch, total=None, posted=posted)
    now = 100_000

    def _tick(prev_state: dict[str, str]) -> tuple[set[str], bool, bool]:
        return pp._maybe_post_eta_deviation(
            issue=873,
            pod="pod-873",
            status="running",
            current_phase="extract",
            last_phase_change_epoch=now - 10 * 3600,
            run_age_sec=10 * 3600.0,
            prev_state=prev_state,
            now_epoch=now,
        )

    with caplog.at_level(logging.INFO, logger="poll_pipeline"):
        _keys, posted_flag, warned = _tick({})
    assert posted_flag is False and posted == [] and warned is True
    matches = [r for r in caplog.records if "phase-ETA tripwire disabled" in r.getMessage()]
    assert len(matches) == 1
    # Second tick with the persisted flag -> no second log line.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="poll_pipeline"):
        _keys, _posted, warned2 = _tick({"eta_budget_warned": "1"})
    assert warned2 is True
    assert [r for r in caplog.records if "phase-ETA tripwire disabled" in r.getMessage()] == []


def test_eta_post_failure_key_not_recorded_retries_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}
    posted: list[dict] = []

    def _flaky(issue, key, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("marker post failed")
        posted.append({"key": key, **kw})

    monkeypatch.setattr(pp, "post_event", _flaky)
    monkeypatch.setattr(pp, "_plan_wall_budget_for_issue", lambda issue: _budget_for(1.0))
    now = 100_000
    kw: dict[str, Any] = {
        "issue": 873,
        "pod": "pod-873",
        "status": "running",
        "current_phase": "extract",
        "last_phase_change_epoch": now - 3 * 3600,
        "run_age_sec": None,
        "prev_state": {},
        "now_epoch": now,
    }
    keys1, posted1, _ = pp._maybe_post_eta_deviation(**kw)
    assert posted1 is False and "extract" not in keys1  # failure -> key NOT recorded
    keys2, posted2, _ = pp._maybe_post_eta_deviation(**kw)  # next tick retries
    assert posted2 is True and "extract" in keys2 and len(posted) == 1


# ── AC #5 (#2172): the durable eta-tripwire-disabled note ────────────────────

# An unparseable-cell budget: one TBD-style offender discarding two
# parseable rows (the residual class AFTER the #2172 cosmetic-prefix
# widening — a cell carrying no number the rule will trust).
_UNPARSEABLE_BUDGET = PlanWallBudget(
    total_h=None,
    rows=(0.5, 16.0),
    unparseable=(
        UnparseableWallCell(
            row_text="| sweep all-cells train | TBD | basis |", reason="no_float", fmt="markdown"
        ),
    ),
    reason="unparseable_cell",
)

_DISABLE_KW: dict[str, Any] = {
    "issue": 2172,
    "pod": "pod-2172",
    "status": "running",
    "current_phase": "extract",
    "last_phase_change_epoch": 100_000 - 3 * 3600,
    "run_age_sec": 3 * 3600.0,
    "prev_state": {},
    "now_epoch": 100_000,
}


def test_eta_disable_posts_progress_naming_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #5: an unparseable-cell disable posts ONE durable ``epm:progress``
    — FIRST token the fixed, greppable ``eta-tripwire-disabled``, the
    offending row named verbatim, the discarded-parseable count carried,
    and the remedy stated. ``posted_this_tick`` stays False (it counts
    ``epm:compute-deviation`` posts only) and no dedup key is recorded."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_plan_wall_budget_for_issue", lambda issue: _UNPARSEABLE_BUDGET)
    keys, posted_flag, warned = pp._maybe_post_eta_deviation(**_DISABLE_KW)
    assert warned is True and posted_flag is False and keys == set()
    (p,) = posted
    assert p["key"] == "epm:progress"
    assert p["phase"] == "extract" and p["pod"] == "pod-2172"
    note = p["note"]
    assert note.startswith("eta-tripwire-disabled")
    assert "| sweep all-cells train | TBD | basis |" in note
    assert "2 parseable row(s) discarded" in note
    assert "bare float" in note and "`basis` cell" in note


def test_eta_disable_posts_once_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #5: the post is gated on the run-scoped ``eta_budget_warned``
    flag — a second tick with the persisted flag posts NOTHING (and the
    run-scope reset re-arms it on a fresh ``epm:run-launched``, exactly as
    the INFO line always was)."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_plan_wall_budget_for_issue", lambda issue: _UNPARSEABLE_BUDGET)
    _keys, _flag, warned = pp._maybe_post_eta_deviation(**_DISABLE_KW)
    assert warned is True and len(posted) == 1
    _keys2, _flag2, warned2 = pp._maybe_post_eta_deviation(
        **{**_DISABLE_KW, "prev_state": {"eta_budget_warned": "1"}}
    )
    assert warned2 is True and len(posted) == 1  # no second post


def test_eta_disable_no_marker_on_no_table(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AC #5 boundary: a plan with NO wall table posts NO marker — log-only,
    with ``budget_warned`` still SET so the INFO line stays once-per-run
    (an infra/analysis plan without a §9 compute table is the normal case,
    not a degradation worth a durable note)."""
    posted: list[dict] = []
    _wire(monkeypatch, total=None, posted=posted)  # _budget_for(None) => no_table
    with caplog.at_level(logging.INFO, logger="poll_pipeline"):
        _keys, _flag, warned = pp._maybe_post_eta_deviation(**_DISABLE_KW)
    assert warned is True and posted == []
    assert [r for r in caplog.records if "phase-ETA tripwire disabled" in r.getMessage()]


def test_eta_disable_post_failure_retries_next_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #5: a failed progress post leaves ``budget_warned`` UNSET so the
    next tick retries (the ``epm:compute-deviation`` retry contract)."""
    calls = {"n": 0}
    posted: list[dict] = []

    def _flaky(issue, key, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("marker post failed")
        posted.append({"key": key, **kw})

    monkeypatch.setattr(pp, "post_event", _flaky)
    monkeypatch.setattr(pp, "_plan_wall_budget_for_issue", lambda issue: _UNPARSEABLE_BUDGET)
    _keys1, _flag1, warned1 = pp._maybe_post_eta_deviation(**_DISABLE_KW)
    assert warned1 is False and posted == []  # failure -> flag NOT set
    _keys2, _flag2, warned2 = pp._maybe_post_eta_deviation(**_DISABLE_KW)  # next tick retries
    assert warned2 is True and len(posted) == 1 and posted[0]["key"] == "epm:progress"


def test_eta_relaunch_resets_posted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #6: a fresh epm:run-launched epoch (newer than the stored
    tripwire_run_epoch by >60s) clears the dedup keys so the SECOND run's
    own overrun still posts — __run_total__ and common phase names no
    longer collide across runs."""
    posted: list[dict] = []
    _wire(monkeypatch, total=1.0, posted=posted)
    now = 1_000_000
    prev = {
        "eta_deviation_posted_keys": f"{pp.ETA_RUN_TOTAL_KEY},extract",
        "eta_budget_warned": "0",
        "tripwire_run_epoch": "1000",  # the PREVIOUS run's launch epoch
    }
    state, epoch = pp._tripwire_run_scope(prev, run_age_sec=3 * 3600.0, now_epoch=now)
    assert epoch == now - 3 * 3600
    assert "eta_deviation_posted_keys" not in state  # keys cleared
    keys, posted_flag, _ = pp._maybe_post_eta_deviation(
        issue=873,
        pod="pod-873",
        status="running",
        current_phase="extract",
        last_phase_change_epoch=now - 3 * 3600,
        run_age_sec=3 * 3600.0,
        prev_state=state,
        now_epoch=now,
    )
    assert posted_flag is True
    assert {"extract", pp.ETA_RUN_TOTAL_KEY} <= keys  # both re-armed AND re-fired
    assert len(posted) == 2


def test_run_scope_clears_idle_keys_on_new_run() -> None:
    """#1033: the run-scope clear set is ``_RUN_SCOPED_STATE_KEYS`` — the #873
    tripwire dedup keys PLUS the three GPU-idle advisory/escalation keys. A
    fresh ``epm:run-launched`` epoch clears ALL of them (the pre-#1033
    "idle keys untouched by the reset" contract was the bug: #763 printed a
    543-min idle advisory on a ~17-min-old fresh instance). Non-run-scoped
    keys (``phase``) are kept."""
    now = 1_000_000
    prev = {
        "phase": "workload",
        "gpu_idle_since_epoch": str(now - 543 * 60),
        "gpu_idle_advised_phases": "startup,workload",
        "gpu_idle_escalated_phases": "workload",
        "eta_deviation_posted_keys": "extract",
        "tripwire_run_epoch": "1000",  # the PREVIOUS run's launch epoch
    }
    state, epoch = pp._tripwire_run_scope(prev, run_age_sec=120.0, now_epoch=now)
    assert epoch == now - 120
    idle_keys = (
        "gpu_idle_since_epoch",
        "gpu_idle_advised_phases",
        "gpu_idle_escalated_phases",
    )
    for key in idle_keys:
        assert key not in state
    assert "eta_deviation_posted_keys" not in state  # #873 keys still cleared
    assert state["phase"] == "workload"  # non-run-scoped keys survive
    # The clear-set invariant: tripwire keys ⊂ run-scoped keys, idle keys in.
    assert set(pp._TRIPWIRE_STATE_KEYS) < set(pp._RUN_SCOPED_STATE_KEYS)
    assert set(idle_keys) <= set(pp._RUN_SCOPED_STATE_KEYS)


def test_run_scope_keeps_escalation_counts_key() -> None:
    """#1752: ``gpu_idle_escalation_counts`` is deliberately NOT in
    ``_RUN_SCOPED_STATE_KEYS`` — the per-phase escalation count must SURVIVE
    the fresh-run reset (the repeat pathology it detects only manifests
    ACROSS run epochs, #1689) while the idle span/dedup keys clear."""
    assert "gpu_idle_escalation_counts" not in pp._RUN_SCOPED_STATE_KEYS
    now = 1_000_000
    prev = {
        "gpu_idle_escalated_phases": "workload",
        "gpu_idle_escalation_counts": "workload:2",
        "tripwire_run_epoch": "1000",
    }
    state, _epoch = pp._tripwire_run_scope(prev, run_age_sec=120.0, now_epoch=now)
    assert "gpu_idle_escalated_phases" not in state  # dedup re-armed
    assert state["gpu_idle_escalation_counts"] == "workload:2"  # count survives


def test_run_scope_keeps_idle_keys_same_run() -> None:
    """#1033 fail-safes pinned: a same-run anchor (within the 60s jitter
    tolerance) AND an unknown run age (missing/unreadable marker) BOTH keep
    the idle keys — the anchor semantics + fail-safe branches are
    byte-unchanged from #873; only the clear SET widened."""
    now = 1_000_000
    prev = {
        "gpu_idle_since_epoch": str(now - 40 * 60),
        "gpu_idle_advised_phases": "scoring",
        "gpu_idle_escalated_phases": "",
        "tripwire_run_epoch": str(now - 3 * 3600),
    }
    # Same run: 30s jitter, within tolerance -> kept.
    state, epoch = pp._tripwire_run_scope(prev, run_age_sec=3 * 3600.0 - 30, now_epoch=now)
    assert epoch == now - 3 * 3600
    assert state["gpu_idle_since_epoch"] == str(now - 40 * 60)
    assert state["gpu_idle_advised_phases"] == "scoring"
    # Unknown run age -> kept verbatim.
    state2, _epoch2 = pp._tripwire_run_scope(prev, run_age_sec=None, now_epoch=now)
    assert state2 is prev


def test_eta_same_run_does_not_reset_posted_keys() -> None:
    """The no-reset control: a run-launched epoch within the 60s jitter
    tolerance of the stored anchor preserves the dedup keys."""
    now = 1_000_000
    stored_epoch = now - 3 * 3600
    prev = {
        "eta_deviation_posted_keys": "extract",
        "tripwire_run_epoch": str(stored_epoch),
    }
    state, epoch = pp._tripwire_run_scope(
        prev, run_age_sec=3 * 3600.0 - 30, now_epoch=now
    )  # 30s jitter, within tolerance
    assert epoch == stored_epoch
    assert state["eta_deviation_posted_keys"] == "extract"
    # Unknown run age (missing marker) also never resets.
    state2, epoch2 = pp._tripwire_run_scope(prev, run_age_sec=None, now_epoch=now)
    assert epoch2 == stored_epoch and state2 is prev
    # A MALFORMED stored anchor with a known run age fails toward RE-ARMING:
    # clear every tripwire key and adopt the current epoch (a duplicate
    # advisory is cheaper than a suppressed one — reconciler CONCERN
    # tripwire-corrupt-anchor-preserves-dedup, round 1).
    state3, epoch3 = pp._tripwire_run_scope(
        {**prev, "tripwire_run_epoch": "garbage"}, run_age_sec=100.0, now_epoch=now
    )
    assert epoch3 == now - 100
    assert all(k not in state3 for k in pp._TRIPWIRE_STATE_KEYS)
    # Malformed anchor + UNKNOWN run age still keeps everything (cannot decide).
    state4, epoch4 = pp._tripwire_run_scope(
        {**prev, "tripwire_run_epoch": "garbage"}, run_age_sec=None, now_epoch=now
    )
    assert epoch4 == 0 and state4["eta_deviation_posted_keys"] == "extract"


# ── PollResult field + main() JSON enumeration ───────────────────────────────


def test_eta_flag_in_pollresult_and_main_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """``eta_deviation_posted`` (and its width sibling) are DEFAULTED on
    PollResult (constructor back-compat) AND enumerated in ``main()``'s
    explicitly-built JSON dict — parity with ``gpu_idle_advisory_posted``,
    so the LIVE polling orchestrator sees the tripwire in-session, not only
    the dashboard."""
    fields = {f.name: f for f in dataclasses.fields(pp.PollResult)}
    assert fields["eta_deviation_posted"].default is False
    assert fields["gpu_width_advisory_posted"].default is False

    result = pp.PollResult(
        status="running",
        current_phase="extract",
        new_milestone=False,
        last_log_mtime_sec_ago=1,
        pid_alive=True,
        pid_file_missing=False,
        log_tail_excerpt="",
        eta_deviation_posted=True,
        gpu_width_advisory_posted=True,
    )
    monkeypatch.setattr(pp, "poll_once", lambda **kw: result)
    rc = pp.main(
        [
            "--issue",
            "873",
            "--pod",
            "pod-873",
            "--log",
            "/tmp/l.log",
            "--pid-file",
            "/tmp/p.pid",
            "--state-file",
            str(tmp_path / "state.json"),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["eta_deviation_posted"] is True
    assert out["gpu_width_advisory_posted"] is True
