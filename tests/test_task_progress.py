"""Unit tests for the task-progress estimator (task #587).

Pins, per the approved plan §5:

1.  Stage stats from a synthetic fixture tree — exact quantiles; blocked-exit,
    backward-exit and archived-exit spans excluded; null→null rows skipped;
    zero-duration spans epsilon-floored.
2.  Recency windowing (last 60 by span-end ts) + bucket→pooled→all-history
    fallback with per-cell ``basis`` tagging.
3.  Monotonicity: floors strictly increase; ``floor_i + 0.95*span_i`` stays
    below the next floor; reviewing floor+span reaches 1.0; the parked
    plan_pending floor case.
4.  Human-wait exclusion: plan_pending frac=0 / ``human_wait``; machine ETA
    excludes plan_pending in current AND future positions; plan_review_ahead.
5.  Blocked suspension (eta=null, pct frozen) incl. the no-prior-machine-stage
    crash guard.
6.  Overdue: band suppressed past stage p75; boundary ``elapsed == p75`` is
    NOT overdue; the title suffix drops the hour band.
7.  GPU-hours refinement: note-regex / intent-map / assumed-1gpu recovery
    chain; clamp ≥ historical p25; ratio-scaled band; ``gpu_hours_total=0``
    and missing-token skip; anchored regex ignores prose "2x consideration".
7b. Expected-total machine time: sum of per-stage quantiles minus the
    plan_pending human wait, current stage at its EFFECTIVE (GPU-refined)
    quantiles.
7c. followups_running renders its own 0→1 track: floor 0 / span 1, paced by
    the round's own historical spans, totals = the round itself, no GPU
    refinement, no title suffix; blocked exits excluded from its stats cell.
8.  Read-only invariant: the fixture tree is byte-unchanged after a full
    stats + snapshot run; the snapshot write is atomic.
9.  Shared vectors: ``interpolate`` + ``format_eta_band`` +
    ``format_duration`` + ``format_title_suffix`` reproduce every row of
    ``tests/fixtures/task_progress_vectors.json`` (the same fixture the tsx
    mirror test replays through dashboard/lib/progress.ts).
10. Snapshot tick: in-scope statuses only (7 machine stages +
    followups_running + blocked); stats TTL reuse + --force-stats.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from explore_persona_space import task_progress as tp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "task_progress_vectors.json"

T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ev(ts: datetime, frm, to, kind="epm:status-changed", **extra) -> dict:
    return {
        "ts": _iso(ts),
        "kind": kind,
        "version": 1,
        "by": "task.py",
        "from": frm,
        "to": to,
        **extra,
    }


def _note(ts: datetime, kind: str, note: str) -> dict:
    return {"ts": _iso(ts), "kind": kind, "version": 1, "by": "test", "note": note}


def _write_task(root: Path, status: str, tid: int, kind: str, events: list[dict]) -> Path:
    d = root / status / str(tid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "body.md").write_text(f"---\ntitle: task {tid}\nkind: {kind}\n---\nbody\n")
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return d


def _forward_chain(start: datetime, durs_h: dict[str, float]) -> list[dict]:
    """A clean forward pass: enter every machine stage in order, spending
    ``durs_h[stage]`` (default 0) in each, ending at awaiting_promotion."""
    events = []
    t = start
    prev = "proposed"
    for stage in tp.MACHINE_STAGES:
        events.append(_ev(t, prev, stage))
        t += timedelta(hours=durs_h.get(stage, 0.0))
        prev = stage
    events.append(_ev(t, prev, "awaiting_promotion"))
    return events


def _install_tree(monkeypatch, root: Path) -> None:
    """Point the estimator's task_workflow surface at a synthetic tree."""
    monkeypatch.setattr(tp, "tasks_dir", lambda: root)

    def _get_task(issue: int) -> dict:
        for sd in root.iterdir():
            d = sd / str(issue)
            if d.is_dir():
                kind = None
                for line in (d / "body.md").read_text().splitlines():
                    if line.startswith("kind:"):
                        kind = line.split(":", 1)[1].strip()
                return {
                    "id": issue,
                    "status": sd.name,
                    "frontmatter": {"kind": kind, "title": f"task {issue}"},
                    "body": "",
                }
        raise FileNotFoundError(f"task #{issue} not found")

    def _list_events(issue: int) -> list[dict]:
        for sd in root.iterdir():
            p = sd / str(issue) / "events.jsonl"
            if p.is_file():
                return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return []

    monkeypatch.setattr(tp, "get_task", _get_task)
    monkeypatch.setattr(tp, "list_events", _list_events)


def _make_stats(cells_by_stage: dict[str, tuple[float, float, float]]) -> dict:
    """Synthetic stats dict in the snapshot's pinned shape (one shared bucket).

    Floors cover the 7 machine stages (+ the flat 0.0 followups_running
    entry, mirroring build_stage_stats); pass a "followups_running" key in
    ``cells_by_stage`` when the test needs the follow-up cell.
    """
    cells = {
        s: {
            "n": 30,
            "p25_h": p25,
            "median_h": max(med, tp.EPS_H),
            "p75_h": p75,
            "basis": "bucket",
        }
        for s, (p25, med, p75) in cells_by_stage.items()
    }
    total = sum(cells[s]["median_h"] for s in tp.MACHINE_STAGES)
    floors, acc = {}, 0.0
    for s in tp.MACHINE_STAGES:
        floors[s] = acc / total
        acc += cells[s]["median_h"]
    floors[tp.FOLLOWUP_STAGE] = 0.0
    return {
        "window_rule": "test",
        "stats_generated_at": _iso(T0),
        "buckets": {"experiment": cells, "code": cells, "pooled": cells},
        "pct_floor_by_stage": {"experiment": floors, "code": floors, "pooled": floors},
    }


DEFAULT_CELLS = {
    "planning": (0.3, 0.6, 0.9),
    "plan_pending": (0.1, 0.5, 2.5),
    "approved": (0.01, 0.02, 0.05),
    "running": (2.0, 4.0, 10.0),
    "verifying": (0.05, 0.1, 0.2),
    "interpreting": (0.5, 0.9, 1.5),
    "reviewing": (0.01, 0.05, 0.15),
}


# ── 1. stats from fixtures ─────────────────────────────────────────────────


def test_stats_from_fixture_tree(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    # Four clean forward passes; planning durations 1,2,3,4 h, all else 0.
    for i, dur in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        _write_task(
            root,
            "completed",
            i,
            "experiment",
            _forward_chain(T0 + timedelta(days=i), {"planning": dur}),
        )
    # Task 5: blocked detour — the 5 h planning→blocked span is EXCLUDED, the
    # 2 h re-entry span (planning→approved, a forward skip) is INCLUDED.
    t5 = T0 + timedelta(days=10)
    ev5 = [
        _ev(t5, "proposed", "planning"),
        _ev(t5 + timedelta(hours=5), "planning", "blocked"),
        _ev(t5 + timedelta(hours=5, minutes=30), None, None),  # null→null noise: skipped
        _ev(t5 + timedelta(hours=6), "blocked", "planning"),
        _ev(t5 + timedelta(hours=8), "planning", "approved"),
        _ev(t5 + timedelta(hours=8), "approved", "running"),
        _ev(t5 + timedelta(hours=8), "running", "awaiting_promotion"),
    ]
    _write_task(root, "completed", 5, "experiment", ev5)
    # Task 6: backward re-plan — plan_pending→planning span is EXCLUDED.
    t6 = T0 + timedelta(days=11)
    ev6 = [
        _ev(t6, "proposed", "plan_pending"),
        _ev(t6 + timedelta(hours=3), "plan_pending", "planning"),
    ]
    _write_task(root, "awaiting_promotion", 6, "experiment", ev6)
    # Task 7: running→archived exit is EXCLUDED (abandonment, not stage cost).
    t7 = T0 + timedelta(days=12)
    _write_task(
        root,
        "archived",
        7,
        "experiment",
        [_ev(t7, "proposed", "running"), _ev(t7 + timedelta(hours=9), "running", "archived")],
    )
    # Task 8: a 3 h follow-up round re-parking at awaiting_promotion is a
    # clean followups_running span; task 9's followups_running→blocked exit
    # is EXCLUDED like any other blocked exit.
    t8 = T0 + timedelta(days=13)
    _write_task(
        root,
        "completed",
        8,
        "experiment",
        [
            _ev(t8, "reviewing", "awaiting_promotion"),
            _ev(t8 + timedelta(hours=1), "awaiting_promotion", "followups_running"),
            _ev(t8 + timedelta(hours=4), "followups_running", "awaiting_promotion"),
            _ev(t8 + timedelta(hours=5), "awaiting_promotion", "completed"),
        ],
    )
    t9 = T0 + timedelta(days=14)
    _write_task(
        root,
        "archived",
        9,
        "experiment",
        [
            _ev(t9, "awaiting_promotion", "followups_running"),
            _ev(t9 + timedelta(hours=9), "followups_running", "blocked"),
        ],
    )

    stats = tp.build_stage_stats(now=T0 + timedelta(days=30))
    cell = stats["buckets"]["experiment"]["planning"]
    # Samples [1,2,3,4,2] → inclusive quartiles 2.0 / 2.0 / 3.0.
    assert cell["n"] == 5
    assert cell["p25_h"] == pytest.approx(2.0)
    assert cell["median_h"] == pytest.approx(2.0)
    assert cell["p75_h"] == pytest.approx(3.0)
    # n=5 < MIN_N in every bucket → all-history pooled basis.
    assert cell["basis"] == "all-history"
    # Zero-duration reviewing spans are floored at EPS_H (no ZeroDivision,
    # floors stay strictly increasing).
    rev = stats["buckets"]["experiment"]["reviewing"]
    assert rev["median_h"] == tp.EPS_H
    # task 6's plan_pending sample never landed (backward exit): only the four
    # zero-duration chain spans from tasks 1-4 + task... (task 5 entered
    # plan_pending never) — all excluded-or-zero, so median is the EPS floor.
    pp = stats["buckets"]["experiment"]["plan_pending"]
    assert pp["median_h"] == tp.EPS_H
    assert pp["n"] == 4  # the four zero spans from the clean chains only
    # Follow-up rounds get their own cell (task 8's clean 3 h span only —
    # task 9's blocked exit is excluded) and a flat 0.0 floor in every bucket.
    fu = stats["buckets"]["experiment"][tp.FOLLOWUP_STAGE]
    assert fu["n"] == 1
    assert fu["median_h"] == pytest.approx(3.0)
    assert fu["basis"] == "all-history"
    for bucket in ("experiment", "code", "pooled"):
        assert stats["pct_floor_by_stage"][bucket][tp.FOLLOWUP_STAGE] == 0.0


# ── 2. windowing + fallback bases ──────────────────────────────────────────


def test_windowing_keeps_last_60_by_end_ts(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    # 80 running spans: the 20 OLDEST last 100 h, the 60 newest last 1 h.
    for i in range(80):
        dur = 100.0 if i < 20 else 1.0
        start = T0 + timedelta(days=i)
        _write_task(
            root,
            "completed",
            100 + i,
            "experiment",
            [
                _ev(start, "approved", "running"),
                _ev(start + timedelta(hours=dur), "running", "verifying"),
            ],
        )
    stats = tp.build_stage_stats(now=T0 + timedelta(days=100))
    cell = stats["buckets"]["experiment"]["running"]
    assert cell["n"] == tp.WINDOW_K == 60
    assert cell["basis"] == "bucket"
    # All-history p75 would be ~25.75 h (20 spans of 100 h in the tail); the window
    # drops every 100 h span.
    assert cell["p75_h"] == pytest.approx(1.0)


def test_fallback_bucket_to_pooled_to_all_history(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    # 12 code running spans + 5 experiment running spans:
    #   code bucket n=12 ≥ MIN_N → basis "bucket"
    #   experiment bucket n=5 < MIN_N → pooled (17 ≥ MIN_N) → basis "pooled"
    for i in range(12):
        start = T0 + timedelta(days=i)
        _write_task(
            root,
            "completed",
            200 + i,
            "infra",
            [
                _ev(start, "approved", "running"),
                _ev(start + timedelta(hours=2.0), "running", "verifying"),
            ],
        )
    for i in range(5):
        start = T0 + timedelta(days=20 + i)
        _write_task(
            root,
            "completed",
            300 + i,
            "experiment",
            [
                _ev(start, "approved", "running"),
                _ev(start + timedelta(hours=8.0), "running", "verifying"),
            ],
        )
    # 3 verifying spans only → pooled windowed n=3 < MIN_N → all-history.
    for i in range(3):
        start = T0 + timedelta(days=30 + i)
        _write_task(
            root,
            "completed",
            400 + i,
            "experiment",
            [
                _ev(start, "running", "verifying"),
                _ev(start + timedelta(hours=0.5), "verifying", "interpreting"),
            ],
        )
    stats = tp.build_stage_stats(now=T0 + timedelta(days=100))
    assert stats["buckets"]["code"]["running"]["basis"] == "bucket"
    assert stats["buckets"]["code"]["running"]["median_h"] == pytest.approx(2.0)
    exp_cell = stats["buckets"]["experiment"]["running"]
    assert exp_cell["basis"] == "pooled"
    assert exp_cell["n"] == 17
    assert stats["buckets"]["experiment"]["verifying"]["basis"] == "all-history"
    assert stats["buckets"]["experiment"]["verifying"]["n"] == 3


# ── 3. monotonicity ────────────────────────────────────────────────────────


def test_floors_monotone_and_frac_cap_stays_below_next_floor():
    stats = _make_stats(DEFAULT_CELLS)
    floors = stats["pct_floor_by_stage"]["experiment"]
    ordered = [floors[s] for s in tp.MACHINE_STAGES] + [1.0]
    for a, b in itertools.pairwise(ordered):
        assert a < b  # strictly increasing (epsilon floor guarantees this)
    for i in range(len(tp.MACHINE_STAGES)):
        span = ordered[i + 1] - ordered[i]
        assert ordered[i] + tp.FRAC_CAP * span < ordered[i + 1]
    # pct at reviewing exit reaches exactly 1.0.
    assert floors["reviewing"] + (1.0 - floors["reviewing"]) == pytest.approx(1.0)


def test_zero_median_stage_keeps_floors_strictly_increasing():
    cells = dict(DEFAULT_CELLS)
    cells["reviewing"] = (0.0, 0.0, 0.0)  # epsilon floor kicks in
    stats = _make_stats(cells)
    floors = stats["pct_floor_by_stage"]["experiment"]
    assert floors["reviewing"] < 1.0


def test_parked_plan_pending_floor_below_next_stage(monkeypatch):
    stats = _make_stats(DEFAULT_CELLS)
    floors = stats["pct_floor_by_stage"]["experiment"]
    _install_tree(monkeypatch, _noop_tree(monkeypatch))
    parked = {
        **_row_template(stats, "plan_pending"),
        "human_wait": True,
        "stage_entered_at": _iso(T0),
    }
    # Even after 1000 h parked, the bar holds EXACTLY the plan_pending floor…
    pct, _eta, overdue = tp.interpolate(parked, T0 + timedelta(hours=1000))
    assert pct == pytest.approx(floors["plan_pending"])
    assert not overdue  # human-wait never goes overdue
    # …while a task one stage later (approved, elapsed 0) always exceeds it.
    approved = {**_row_template(stats, "approved"), "stage_entered_at": _iso(T0)}
    pct2, _eta2, _ = tp.interpolate(approved, T0)
    assert pct2 > pct


def _noop_tree(monkeypatch):
    """Tiny placeholder tree (some helpers only need tasks_dir to exist)."""
    import tempfile

    return Path(tempfile.mkdtemp()) / "tasks"


def _row_template(stats: dict, stage: str, bucket: str = "experiment") -> dict:
    cells = stats["buckets"][bucket]
    floors = stats["pct_floor_by_stage"][bucket]
    idx = tp.MACHINE_STAGES.index(stage)
    nxt = floors[tp.MACHINE_STAGES[idx + 1]] if idx + 1 < len(tp.MACHINE_STAGES) else 1.0
    remaining = {"p25_h": 0.0, "median_h": 0.0, "p75_h": 0.0}
    for s in tp.MACHINE_STAGES[idx + 1 :]:
        if s == "plan_pending":
            continue
        for q in remaining:
            remaining[q] += cells[s][q]
    total = {"p25_h": 0.0, "median_h": 0.0, "p75_h": 0.0}
    for s in tp.MACHINE_STAGES:
        if s == "plan_pending":
            continue
        for q in total:
            total[q] += cells[s][q]
    return {
        "issue": 1,
        "status": stage,
        "stage": stage,
        "kind_bucket": bucket,
        "stats_basis": "bucket",
        "stage_entered_at": _iso(T0),
        "pct_floor": floors[stage],
        "pct_span": nxt - floors[stage],
        "frac_median_h": max(cells[stage]["median_h"], tp.EPS_H),
        "stage_p25_h": cells[stage]["p25_h"],
        "stage_median_h": cells[stage]["median_h"],
        "stage_p75_h": cells[stage]["p75_h"],
        "remaining_after_p25_h": remaining["p25_h"],
        "remaining_after_median_h": remaining["median_h"],
        "remaining_after_p75_h": remaining["p75_h"],
        "total_p25_h": total["p25_h"],
        "total_median_h": total["median_h"],
        "total_p75_h": total["p75_h"],
        "human_wait": False,
        "blocked": False,
        "plan_review_ahead": stage == "planning",
        "gpu_hours_total": None,
        "gpu_count": None,
        "gpu_conversion": None,
        "eta_basis": "historical",
    }


# ── 4. human-wait exclusion ────────────────────────────────────────────────


def test_plan_pending_is_human_wait_and_eta_excludes_it(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(
        root,
        "plan_pending",
        50,
        "experiment",
        [_ev(T0, "proposed", "planning"), _ev(T0 + timedelta(hours=1), "planning", "plan_pending")],
    )
    # plan_pending median is HUGE — it must not contaminate any machine ETA.
    cells = dict(DEFAULT_CELLS)
    cells["plan_pending"] = (50.0, 100.0, 200.0)
    stats = _make_stats(cells)
    row = tp.estimate_task_progress(50, stats, now=T0 + timedelta(hours=2))
    assert row is not None
    assert row["human_wait"] is True
    assert row["blocked"] is False
    # Remaining-after sums cover approved..reviewing only (plan_pending is
    # the CURRENT stage here; its own term is skipped by interpolate()).
    expected_median = sum(
        cells[s][1] for s in ("approved", "running", "verifying", "interpreting", "reviewing")
    )
    assert row["remaining_after_median_h"] == pytest.approx(expected_median)
    pct, eta, overdue = tp.interpolate(row, T0 + timedelta(hours=500))
    assert pct == pytest.approx(row["pct_floor"])  # frac parked at 0
    assert eta is not None and eta["median_h"] == pytest.approx(expected_median)
    assert overdue is False
    assert tp.format_title_suffix(row, T0 + timedelta(hours=2)) is None


def test_planning_task_eta_excludes_plan_pending_ahead(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(root, "planning", 51, "experiment", [_ev(T0, "proposed", "planning")])
    cells = dict(DEFAULT_CELLS)
    cells["plan_pending"] = (50.0, 100.0, 200.0)
    stats = _make_stats(cells)
    row = tp.estimate_task_progress(51, stats, now=T0 + timedelta(minutes=10))
    assert row is not None
    assert row["plan_review_ahead"] is True
    # plan_pending's 100 h median must NOT appear in the machine ETA.
    expected_median = sum(
        cells[s][1] for s in ("approved", "running", "verifying", "interpreting", "reviewing")
    )
    assert row["remaining_after_median_h"] == pytest.approx(expected_median)


# ── 5. blocked suspension ──────────────────────────────────────────────────


def test_blocked_task_returns_suspended_eta(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(
        root,
        "blocked",
        60,
        "experiment",
        [
            _ev(T0, "proposed", "planning"),
            _ev(T0 + timedelta(hours=1), "planning", "running"),
            _ev(T0 + timedelta(hours=2), "running", "blocked"),
        ],
    )
    stats = _make_stats(DEFAULT_CELLS)
    row = tp.estimate_task_progress(60, stats, now=T0 + timedelta(hours=3))
    assert row is not None
    assert row["blocked"] is True
    assert row["stage"] == "running"  # last machine stage before blocking
    pct, eta, overdue = tp.interpolate(row, T0 + timedelta(hours=300))
    # Snapshot rows round floats to 6 decimals — compare at that tolerance.
    assert pct == pytest.approx(stats["pct_floor_by_stage"]["experiment"]["running"], abs=1e-6)
    assert eta is None
    assert overdue is False
    assert tp.format_title_suffix(row, T0 + timedelta(hours=3)) is None


def test_blocked_with_no_prior_machine_stage_floor_zero(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(root, "blocked", 61, "experiment", [_ev(T0, "proposed", "blocked")])
    stats = _make_stats(DEFAULT_CELLS)
    row = tp.estimate_task_progress(61, stats, now=T0 + timedelta(hours=1))
    assert row is not None
    assert row["blocked"] is True
    assert row["stage"] == "planning"
    assert row["pct_floor"] == pytest.approx(0.0)


# ── 6. overdue ─────────────────────────────────────────────────────────────


def test_overdue_suppresses_band_and_boundary_is_not_overdue():
    stats = _make_stats(DEFAULT_CELLS)
    row = _row_template(stats, "running")  # p75 = 10 h
    pct_at, eta_at, overdue_at = tp.interpolate(row, T0 + timedelta(hours=10))
    assert overdue_at is False  # elapsed == p75 → NOT overdue (strict >)
    assert eta_at is not None
    pct_over, eta_over, overdue_over = tp.interpolate(row, T0 + timedelta(hours=10, seconds=1))
    assert overdue_over is True
    assert eta_over is None  # band suppressed
    assert pct_over >= pct_at  # bar parks at the frac cap, never regresses
    suffix = tp.format_title_suffix(row, T0 + timedelta(hours=30))
    assert suffix is not None and suffix.endswith("overdue")
    assert "h" not in suffix.split()[-1]  # no hour band on the overdue title


# ── 7. GPU-hours refinement ────────────────────────────────────────────────


def _gpu_task(root, tid, plan_note, extra_notes=()):
    events = [
        _ev(T0, "proposed", "planning"),
        _note(T0 + timedelta(minutes=5), "epm:plan", plan_note),
        _ev(T0 + timedelta(hours=1), "planning", "running"),
        *extra_notes,
    ]
    return _write_task(root, "running", tid, "experiment", events)


def test_gpu_refinement_note_regex_clamped_and_ratio_scaled(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _gpu_task(
        root,
        70,
        "Plan v1 (gpu_hours_total=19.0)",
        [_note(T0 + timedelta(hours=2), "epm:progress", "provisioned 4× H100 SXM")],  # noqa: RUF001
    )
    stats = _make_stats(DEFAULT_CELLS)  # running 2/4/10
    row = tp.estimate_task_progress(70, stats, now=T0 + timedelta(hours=3))
    assert row is not None
    assert row["gpu_hours_total"] == pytest.approx(19.0)
    assert row["gpu_count"] == 4
    assert row["gpu_conversion"] == "note-regex"
    assert row["eta_basis"] == "gpu-refined"
    refined = max(19.0 / 4, 2.0)  # clamp ≥ historical p25
    assert row["stage_median_h"] == pytest.approx(refined)
    assert row["stage_p25_h"] == pytest.approx(refined * (2.0 / 4.0))
    assert row["stage_p75_h"] == pytest.approx(refined * (10.0 / 4.0))
    # frac pace stays HISTORICAL (§3.3): the bar moves on the 4 h median.
    assert row["frac_median_h"] == pytest.approx(4.0)


def test_gpu_refinement_clamp_binds_at_historical_p25(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _gpu_task(
        root,
        71,
        "gpu_hours_total=1.0",
        [_note(T0 + timedelta(hours=2), "epm:progress", "8x H200 ready")],
    )
    stats = _make_stats(DEFAULT_CELLS)
    row = tp.estimate_task_progress(71, stats, now=T0 + timedelta(hours=3))
    assert row is not None
    assert row["stage_median_h"] == pytest.approx(2.0)  # clamped to p25, not 0.125


def test_gpu_refinement_intent_map_and_unknown_intent(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _gpu_task(root, 72, "gpu_hours_total=8.0")
    _gpu_task(root, 73, "gpu_hours_total=8.0")
    pods = {
        "pods": {
            "pod-72": {"issue": 72, "gpu_intent": "ft-7b"},
            "pod-73": {"issue": 73, "gpu_intent": "custom"},  # unknown → fall through
        }
    }
    pods_path = tmp_path / "pods_ephemeral.json"
    pods_path.write_text(json.dumps(pods))
    monkeypatch.setattr(tp, "_pods_ephemeral_path", lambda: pods_path)
    stats = _make_stats(DEFAULT_CELLS)
    row72 = tp.estimate_task_progress(72, stats, now=T0 + timedelta(hours=3))
    assert row72 is not None
    assert (row72["gpu_count"], row72["gpu_conversion"]) == (4, "intent-map")
    assert row72["eta_basis"] == "gpu-refined"
    row73 = tp.estimate_task_progress(73, stats, now=T0 + timedelta(hours=3))
    assert row73 is not None
    assert (row73["gpu_count"], row73["gpu_conversion"]) == (1, "assumed-1gpu")
    assert row73["eta_basis"] == "gpu-assumed"


def test_pods_ephemeral_path_prefers_live_when_exists(tmp_path, monkeypatch):
    """Task #1183: the intent-map fallback reads the LIVE relocated sidecar
    at ``<git-common-dir>/eps/`` when it exists, else the tracked seed —
    read-only, no migration."""
    import explore_persona_space.task_workflow as task_workflow

    monkeypatch.setattr(task_workflow, "repo_root", lambda: tmp_path)

    # No live copy yet → the tracked seed path.
    assert tp._pods_ephemeral_path() == tmp_path / "scripts" / "pods_ephemeral.json"

    # Live copy present → preferred.
    live = tmp_path / ".git" / "eps" / "pods_ephemeral.json"
    live.parent.mkdir(parents=True)
    live.write_text('{"version": 2, "pods": {}}\n')
    assert tp._pods_ephemeral_path() == live


def test_gpu_refinement_skipped_on_zero_or_missing_token(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _gpu_task(root, 74, "Plan written (gpu_hours_total=0)")  # infra-style zero
    _gpu_task(root, 75, "Plan written, no token here")
    stats = _make_stats(DEFAULT_CELLS)
    for tid in (74, 75):
        row = tp.estimate_task_progress(tid, stats, now=T0 + timedelta(hours=3))
        assert row is not None
        assert row["eta_basis"] == "historical"
        assert row["gpu_hours_total"] is None
        assert row["stage_median_h"] == pytest.approx(4.0)


def test_gpu_count_regex_is_anchored_to_gpu_type_token():
    assert tp._GPU_COUNT_RE.search("a 2x consideration of the design") is None
    assert tp._GPU_COUNT_RE.search("provisioned 4× H100 SXM").group(1) == "4"  # noqa: RUF001
    assert tp._GPU_COUNT_RE.search("4xH100").group(1) == "4"
    assert tp._GPU_COUNT_RE.search("1x H100 pod up").group(1) == "1"
    assert tp._GPU_COUNT_RE.search("8x H200 ready").group(1) == "8"


# ── 7b. expected-total machine time ────────────────────────────────────────


def test_total_expected_machine_time_sums_stages_minus_plan_pending(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(
        root,
        "running",
        80,
        "experiment",
        [_ev(T0, "proposed", "planning"), _ev(T0 + timedelta(hours=1), "planning", "running")],
    )
    stats = _make_stats(DEFAULT_CELLS)
    row = tp.estimate_task_progress(80, stats, now=T0 + timedelta(hours=2))
    assert row is not None
    # planning + approved + running + verifying + interpreting + reviewing —
    # plan_pending (human wait) excluded.
    expected = sum(
        DEFAULT_CELLS[s][1]
        for s in ("planning", "approved", "running", "verifying", "interpreting", "reviewing")
    )
    assert row["total_median_h"] == pytest.approx(expected)
    assert row["total_p25_h"] == pytest.approx(
        sum(
            DEFAULT_CELLS[s][0]
            for s in ("planning", "approved", "running", "verifying", "interpreting", "reviewing")
        )
    )


def test_total_uses_effective_gpu_refined_running_cell(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _gpu_task(
        root,
        81,
        "Plan v1 (gpu_hours_total=19.0)",
        [_note(T0 + timedelta(hours=2), "epm:progress", "provisioned 4× H100 SXM")],  # noqa: RUF001
    )
    stats = _make_stats(DEFAULT_CELLS)
    row = tp.estimate_task_progress(81, stats, now=T0 + timedelta(hours=3))
    assert row is not None
    refined = max(19.0 / 4, 2.0)  # 4.75, replaces the 4.0 historical median
    expected = (
        sum(
            DEFAULT_CELLS[s][1]
            for s in ("planning", "approved", "verifying", "interpreting", "reviewing")
        )
        + refined
    )
    assert row["total_median_h"] == pytest.approx(expected)


# ── 7c. followups_running own 0→1 track ────────────────────────────────────


FOLLOWUP_CELLS = {**DEFAULT_CELLS, "followups_running": (1.0, 2.0, 4.5)}


def test_followup_round_renders_own_track(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(
        root,
        "followups_running",
        85,
        "experiment",
        [
            _ev(T0, "reviewing", "awaiting_promotion"),
            _note(T0 + timedelta(minutes=5), "epm:plan", "gpu_hours_total=19.0"),
            _ev(T0 + timedelta(hours=1), "awaiting_promotion", "followups_running"),
        ],
    )
    stats = _make_stats(FOLLOWUP_CELLS)
    row = tp.estimate_task_progress(85, stats, now=T0 + timedelta(hours=2))
    assert row is not None
    assert row["stage"] == tp.FOLLOWUP_STAGE
    assert row["pct_floor"] == 0.0 and row["pct_span"] == 1.0  # own track
    assert row["frac_median_h"] == pytest.approx(2.0)
    assert row["stage_p25_h"] == pytest.approx(1.0)
    assert row["stage_p75_h"] == pytest.approx(4.5)
    # Nothing comes after the round — it re-parks at awaiting_promotion.
    assert row["remaining_after_median_h"] == 0.0
    # The round's total IS its own expected duration (main pass is behind it).
    assert row["total_median_h"] == pytest.approx(2.0)
    assert row["human_wait"] is False and row["plan_review_ahead"] is False
    # The main-pass GPU plan token must NOT refine a follow-up round.
    assert row["eta_basis"] == "historical" and row["gpu_hours_total"] is None
    # stage_entered_at = the followups_running entry, not the first event.
    assert row["stage_entered_at"] == _iso(T0 + timedelta(hours=1))
    # Halfway through the round's 2 h median → bar at 50%.
    pct, eta, overdue = tp.interpolate(row, T0 + timedelta(hours=2))
    assert pct == pytest.approx(0.5)
    assert eta is not None and eta["median_h"] == pytest.approx(1.0)
    assert overdue is False
    # Session titles stay clean for follow-up rounds (dashboard-only track).
    assert tp.format_title_suffix(row, T0 + timedelta(hours=2)) is None


# ── 8. read-only invariant + atomic snapshot ───────────────────────────────


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_full_stats_and_snapshot_run_is_read_only_over_tasks(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    for i, dur in enumerate([1.0, 2.0], start=1):
        _write_task(
            root,
            "completed",
            i,
            "experiment",
            _forward_chain(T0 + timedelta(days=i), {"running": dur}),
        )
    _write_task(
        root,
        "running",
        90,
        "experiment",
        [_ev(T0, "proposed", "planning"), _ev(T0 + timedelta(hours=1), "planning", "running")],
    )
    _write_task(root, "plan_pending", 91, "infra", [_ev(T0, "proposed", "plan_pending")])
    _write_task(
        root,
        "blocked",
        92,
        "experiment",
        [_ev(T0, "proposed", "planning"), _ev(T0 + timedelta(hours=1), "planning", "blocked")],
    )
    _write_task(root, "completed", 93, "experiment", _forward_chain(T0 + timedelta(days=5), {}))
    _write_task(
        root,
        "followups_running",
        94,
        "experiment",
        [
            _ev(T0, "reviewing", "awaiting_promotion"),
            _ev(T0 + timedelta(hours=1), "awaiting_promotion", "followups_running"),
        ],
    )
    snap_path = tmp_path / "cache" / "task_progress.json"
    monkeypatch.setattr(tp, "SNAPSHOT_PATH", snap_path)

    before = _tree_digest(root)
    tp.build_stage_stats(now=T0 + timedelta(days=30))
    out = tp.write_snapshot(now=T0 + timedelta(days=30))
    after = _tree_digest(root)
    assert before == after, "stats+snapshot run mutated the tasks/ tree"
    assert out == snap_path and snap_path.is_file()
    assert not list(snap_path.parent.glob("*.tmp")), "snapshot write not atomic"

    snap = json.loads(snap_path.read_text())
    # In-scope rows only: running, plan_pending, blocked, followups_running.
    # Completed → absent.
    assert set(snap["tasks"]) == {"90", "91", "92", "94"}
    assert snap["tasks"]["92"]["blocked"] is True
    assert snap["tasks"]["91"]["human_wait"] is True
    assert snap["tasks"]["94"]["stage"] == tp.FOLLOWUP_STAGE
    assert snap["tasks"]["94"]["pct_floor"] == 0.0
    assert snap["version"] == 1 and "generated_at" in snap


def test_snapshot_reuses_fresh_stats_and_force_rebuilds(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    _write_task(root, "completed", 1, "experiment", _forward_chain(T0, {"running": 2.0}))
    snap_path = tmp_path / "cache" / "task_progress.json"
    monkeypatch.setattr(tp, "SNAPSHOT_PATH", snap_path)
    t1 = T0 + timedelta(days=1)
    tp.write_snapshot(now=t1)
    first = json.loads(snap_path.read_text())["stats"]["stats_generated_at"]
    # One hour later (within TTL) the stats section is REUSED…
    tp.write_snapshot(now=t1 + timedelta(hours=1))
    second = json.loads(snap_path.read_text())["stats"]["stats_generated_at"]
    assert second == first
    # …unless forced, or the TTL lapses.
    tp.write_snapshot(now=t1 + timedelta(hours=2), force_stats=True)
    third = json.loads(snap_path.read_text())["stats"]["stats_generated_at"]
    assert third != first
    tp.write_snapshot(now=t1 + timedelta(days=2))
    fourth = json.loads(snap_path.read_text())["stats"]["stats_generated_at"]
    assert fourth != third


def test_load_stats_readonly_never_rebuilds(tmp_path, monkeypatch):
    snap_path = tmp_path / "task_progress.json"
    monkeypatch.setattr(tp, "SNAPSHOT_PATH", snap_path)

    def _boom(*a, **kw):  # any rebuild attempt from the read path is a bug
        raise AssertionError("load_stats_readonly must NEVER rebuild stats")

    monkeypatch.setattr(tp, "build_stage_stats", _boom)
    assert tp.load_stats_readonly() is None  # missing file → None, no rebuild
    stats = _make_stats(DEFAULT_CELLS)
    stats["stats_generated_at"] = _iso(datetime.now(tz=UTC))
    snap_path.write_text(json.dumps({"version": 1, "stats": stats, "tasks": {}}))
    got = tp.load_stats_readonly()
    assert got is not None and got["buckets"]["experiment"]["running"]["median_h"] == 4.0
    # Stale stats (older than the read max-age) → None.
    stats["stats_generated_at"] = _iso(datetime.now(tz=UTC) - timedelta(days=3))
    snap_path.write_text(json.dumps({"version": 1, "stats": stats, "tasks": {}}))
    assert tp.load_stats_readonly() is None


def test_out_of_scope_statuses_get_no_row(tmp_path, monkeypatch):
    # followups_running is NO LONGER out of scope — it renders its own track
    # (test_followup_round_renders_own_track).
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    stats = _make_stats(DEFAULT_CELLS)
    for status in ("proposed", "awaiting_promotion", "completed", "archived"):
        _write_task(root, status, 700, "experiment", [_ev(T0, None, status)])
        assert tp.estimate_task_progress(700, stats, now=T0) is None
        import shutil

        shutil.rmtree(root / status)


# ── 9. shared vectors (the Python↔TS lockstep pin) ─────────────────────────


def test_shared_vectors_interpolate_and_labels():
    fixture = json.loads(FIXTURE.read_text())
    vectors = fixture["interpolate_vectors"]
    assert len(vectors) >= 10
    for v in vectors:
        row, now = v["row"], tp._parse_iso(v["now"])
        exp = v["expected"]
        pct, eta, overdue = tp.interpolate(row, now)
        assert pct == pytest.approx(exp["pct"], abs=1e-9), v["name"]
        assert overdue == exp["overdue"], v["name"]
        if exp["eta_p75_h"] is None:
            assert eta is None, v["name"]
        else:
            assert eta is not None, v["name"]
            assert eta["p25_h"] == pytest.approx(exp["eta_p25_h"], abs=1e-9), v["name"]
            assert eta["median_h"] == pytest.approx(exp["eta_median_h"], abs=1e-9), v["name"]
            assert eta["p75_h"] == pytest.approx(exp["eta_p75_h"], abs=1e-9), v["name"]
            label = tp.format_eta_band(
                eta["p25_h"], eta["p75_h"], row.get("eta_basis", "historical")
            )
            assert label == exp["eta_label"], v["name"]
        # Median remaining/total labels (format_duration ↔ TS formatDuration).
        basis = row.get("eta_basis", "historical")
        if eta is None:
            assert exp["remaining_label"] is None, v["name"]
        else:
            assert tp.format_duration(eta["median_h"], basis) == exp["remaining_label"], v["name"]
        assert tp.format_duration(row["total_median_h"], basis) == exp["total_label"], v["name"]
        # The fixture pins the FULL title format (band included) so the band
        # path stays mirrored in TS for re-enablement; production currently
        # ships band-less (ETA_BAND_ENABLED=False, §7 kill criterion).
        full_title = tp.format_title_suffix(row, now, include_band=True)
        assert full_title == exp["title_suffix"], v["name"]


def test_eta_band_kill_switch_drops_band_from_title_by_default():
    # §7 kill criterion fired at implementation (backtest coverage
    # 0.368/0.404 < 0.50): the default title carries the bar + pct ONLY.
    assert tp.ETA_BAND_ENABLED is False
    stats = _make_stats(DEFAULT_CELLS)
    row = _row_template(stats, "running")
    now = T0 + timedelta(hours=2)
    suffix = tp.format_title_suffix(row, now)  # production default
    assert suffix is not None
    assert suffix.endswith("%")  # no hour band
    assert "~" not in suffix and "≈" not in suffix
    # The band path stays intact behind the explicit override.
    full = tp.format_title_suffix(row, now, include_band=True)
    assert full is not None and full.startswith(suffix) and "h" in full
    # Overdue keeps its state word either way (honesty state, not countdown).
    over = tp.format_title_suffix(row, T0 + timedelta(hours=30))
    assert over is not None and over.endswith("overdue")


def test_shared_vectors_cover_required_cases():
    fixture = json.loads(FIXTURE.read_text())
    names = {v["name"] for v in fixture["interpolate_vectors"]}
    for needle in ("overdue", "blocked", "human-wait", "gpu-refined"):
        assert any(needle in n for n in names), f"missing required vector class: {needle}"
    gate_names = {g["name"] for g in fixture["gating_vectors"]}
    for needle in ("awaiting_promotion", "completed", "followups_running", "blocked", "stale"):
        assert any(needle in n for n in gate_names), f"missing gating vector: {needle}"


# ── 12. #950 regression: splitlines() Unicode-boundary shred ───────────────


def test_unicode_boundary_note_does_not_drop_whole_task(tmp_path, monkeypatch):
    """#950 regression (incident #825): one raw-U+2028-bearing event note must
    not drop the WHOLE task's events from collect_stage_spans. The pre-fix
    shape read `ev_path.read_text().splitlines()` inside a whole-file
    `except (OSError, json.JSONDecodeError): continue` — splitlines() shredded
    the note line (raw U+2028/U+2029/NEL are Unicode line boundaries left by
    the `ensure_ascii=False` writer), the JSONDecodeError fired, and EVERY
    event of the task was silently dropped. The fix routes through the
    canonical tolerant `_iter_jsonl` (split("\\n") + per-line skip)."""
    root = tmp_path / "tasks"
    _install_tree(monkeypatch, root)
    d = _write_task(root, "completed", 42, "experiment", [_ev(T0, "approved", "running")])
    # Rewrite events.jsonl exactly as post_event does: ensure_ascii=False,
    # one \n-terminated record per line, one note carrying a raw U+2028.
    events = [
        _ev(T0, "approved", "running", note="line one\u2028line two"),
        _ev(T0 + timedelta(hours=2.0), "running", "verifying"),
    ]
    (d / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )
    spans = tp.collect_stage_spans()
    running = [s for s in spans if s["task_id"] == 42 and s["stage"] == "running"]
    assert len(running) == 1, spans  # the task's events were NOT dropped
    assert running[0]["dur_h"] == 2.0
