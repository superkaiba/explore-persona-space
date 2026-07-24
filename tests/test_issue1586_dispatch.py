"""#1586 dispatcher wave-scheduler pins (crash-fix r5, epm:failure v5).

The linear p2->p3 phase ordering trained ALL cells before any ladder ran,
accumulating ~2.5 TB of rung checkpoints on a 750 GB volume (safetensors
ENOSPC at p2_train, cell 3 of 11). run_waves pins the bounded-wave flow:
train -> ladder -> persist -> reap, W cells at a time, in STRICT alternation.

FAILS-PRE-FIX CHARACTER: pre-fix there was no run_waves and main() called
phase_train(cfg) -> phase_ladder(cfg) -> phase_persist(cfg, selections)
linearly — the ordering test's interleaving assert and the main() source pin
both fail against that shape.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import typing
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1586_dispatch as d  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────────

CELLS5 = ("c1", "c2", "c3", "c4", "c5")


def _cfg(tmp_path, cells=CELLS5, **kw):
    kw.setdefault("smoke", False)
    kw.setdefault("ladder_disk_mode", "keep-cell")
    return d.Cfg(cells=tuple(cells), out_root=Path(tmp_path), **kw)


def _eight_gpus(monkeypatch):
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: [str(i) for i in range(8)])


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _mk_rungs(train_dir: Path, steps) -> None:
    """Rung fixtures satisfy the r7 production completeness predicate
    (config.json + non-empty single-file weights = complete under
    d._ckpt_incomplete_reason), so resolver-consuming tests never route to a
    real overflow restage (review r6 Minor: fixtures below the bar mask the
    partial-rung class)."""
    for s in steps:
        ck = train_dir / f"checkpoint-{s}"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "model.safetensors").write_text("x")
        (ck / "config.json").write_text("{}")


# ── the wave scheduler (ordering; the headline pin) ──────────────────────────


def test_wave_scheduler_strict_alternation(monkeypatch, tmp_path):
    """n_cells=5, W=2: expect [train(c1,c2), ladder(c1,c2), persist(c1,c2),
    reap(c1,c2), train(c3,c4), ...] + the terminal residual persist. The
    pre-fix ordering (train ALL 5 before any ladder) fails the interleaving
    assert below."""
    _eight_gpus(monkeypatch)
    cfg = _cfg(tmp_path)
    calls: list[tuple] = []

    def fake_train(cfg, cells=None):
        calls.append(("train", tuple(cells)))
        return {}

    def fake_ladder(cfg, cells=None):
        calls.append(("ladder", tuple(cells)))
        return {c: {"step": 2} for c in cells}

    def fake_persist(cfg, selections, cells=None):
        calls.append(("persist", tuple(cells) if cells is not None else None))
        return {"uploaded": {}}

    def fake_reap(cfg, cells):
        calls.append(("reap", tuple(cells)))

    monkeypatch.setattr(d, "phase_train", fake_train)
    monkeypatch.setattr(d, "phase_ladder", fake_ladder)
    monkeypatch.setattr(d, "phase_persist", fake_persist)
    monkeypatch.setattr(d, "_wave_reap", fake_reap)
    monkeypatch.setattr(d, "_wave_headroom", lambda cfg, k, wave: None)

    sel = d.run_waves(cfg, do_train=True, do_ladder=True, do_persist=True)

    assert calls == [
        ("train", ("c1", "c2")),
        ("ladder", ("c1", "c2")),
        ("persist", ("c1", "c2")),
        ("reap", ("c1", "c2")),
        ("train", ("c3", "c4")),
        ("ladder", ("c3", "c4")),
        ("persist", ("c3", "c4")),
        ("reap", ("c3", "c4")),
        ("train", ("c5",)),
        ("ladder", ("c5",)),
        ("persist", ("c5",)),
        ("reap", ("c5",)),
        ("persist", None),  # terminal residual pass (writes persist_done.json)
    ]
    assert set(sel) == set(CELLS5)
    # Interleaving invariant (fails pre-fix): a ladder call arrives BEFORE
    # the last train call — the linear ordering trains all 5 first.
    train_idx = [i for i, c in enumerate(calls) if c[0] == "train"]
    first_ladder = next(i for i, c in enumerate(calls) if c[0] == "ladder")
    assert first_ladder < train_idx[-1]


def test_wave_scheduler_phase_subsets(monkeypatch, tmp_path):
    """--phases subsets run the SAME wave loop with un-named stages skipped;
    reap runs whenever ladder or persist ran (selection records on disk)."""
    _eight_gpus(monkeypatch)
    cfg = _cfg(tmp_path, cells=("c1", "c2"))
    calls: list[str] = []
    monkeypatch.setattr(d, "phase_train", lambda cfg, cells=None: calls.append("train"))
    monkeypatch.setattr(d, "phase_ladder", lambda cfg, cells=None: calls.append("ladder") or {})
    monkeypatch.setattr(
        d, "phase_persist", lambda cfg, selections, cells=None: calls.append("persist") or {}
    )
    monkeypatch.setattr(d, "_wave_reap", lambda cfg, cells: calls.append("reap"))
    monkeypatch.setattr(d, "_wave_headroom", lambda cfg, k, wave: None)

    d.run_waves(cfg, do_train=True, do_ladder=False, do_persist=False)
    assert calls == ["train"]  # train-only: no ladder/persist/reap
    calls.clear()
    d.run_waves(cfg, do_train=False, do_ladder=True, do_persist=True)
    assert calls == ["ladder", "persist", "reap", "persist"]


def test_run_waves_terminal_writes_reused_selection(monkeypatch, tmp_path):
    """The reused #1112 cell never enters a wave; run_waves' terminal pass
    synthesizes its selection record (+ the §6.5 deliverable mirror)."""
    _eight_gpus(monkeypatch)
    cfg = _cfg(tmp_path, cells=(d.G.REUSED_FT_CELL, "c1"))
    monkeypatch.setattr(d, "phase_train", lambda cfg, cells=None: {})
    monkeypatch.setattr(d, "phase_ladder", lambda cfg, cells=None: {c: {"step": 2} for c in cells})
    monkeypatch.setattr(d, "phase_persist", lambda cfg, selections, cells=None: {"uploaded": {}})
    monkeypatch.setattr(d, "_wave_reap", lambda cfg, cells: None)
    monkeypatch.setattr(d, "_wave_headroom", lambda cfg, k, wave: None)
    sel = d.run_waves(cfg, do_train=True, do_ladder=True, do_persist=True)
    assert sel[d.G.REUSED_FT_CELL]["reused"] is True
    assert (tmp_path / d.G.REUSED_FT_CELL / "selection.json").exists()
    assert (tmp_path / "selection" / d.G.REUSED_FT_CELL / "selection.json").exists()


def test_wave_partition_widths(monkeypatch, tmp_path):
    _eight_gpus(monkeypatch)
    cfg = _cfg(tmp_path)
    assert d._wave_partition(cfg) == [["c1", "c2"], ["c3", "c4"], ["c5"]]
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0", "1", "2", "3"])
    assert d._wave_partition(cfg) == [["c1"], ["c2"], ["c3"], ["c4"], ["c5"]]
    # The reused #1112 cell never enters a wave (it never trains).
    cfg2 = _cfg(tmp_path, cells=(d.G.REUSED_FT_CELL, "c1"))
    assert d._wave_partition(cfg2) == [["c1"]]


def test_main_routes_p2_p4_through_wave_loop():
    """Source pin (fails pre-fix): main() routes train/ladder/persist through
    run_waves — the linear un-scoped phase_train(cfg)/phase_ladder(cfg) call
    shape is the epm:failure v5 ENOSPC ordering."""
    src = inspect.getsource(d.main)
    assert "run_waves(" in src
    assert "phase_train(cfg)" not in src
    assert "phase_ladder(cfg)" not in src


# ── phase_train wave scoping + pilot-gate idempotence ────────────────────────


class _FakeProc:
    def wait(self):  # the only Popen surface _await_train consumes
        return 0


def test_phase_train_wave_scope_and_gate_idempotence(monkeypatch, tmp_path):
    _eight_gpus(monkeypatch)
    cells = ("c1", "c2", "c3")
    cfg = _cfg(tmp_path, cells=cells)
    monkeypatch.setattr(d, "_headroom", lambda cfg, phase: None)
    launched: list[tuple[str, int]] = []

    def fake_train_one_cell(cfg, cell, lane):
        launched.append((cell, lane))
        return _FakeProc()

    def fake_await(cfg, cell, proc):
        proc.wait()
        _write_json(cfg.out_root / cell / "build_result.json", {"cell": cell})

    monkeypatch.setattr(d, "_train_one_cell", fake_train_one_cell)
    monkeypatch.setattr(d, "_await_train", fake_await)

    out = d.phase_train(cfg, cells=cells[:2])
    assert [c for c, _ in launched] == ["c1", "c2"]  # wave scope only
    assert set(out) == {"c1", "c2"}
    rep = tmp_path / "pilot_gate_report_p2_train.json"
    assert rep.exists()  # p2 pilot gate fired on the first fresh batch
    stamp = rep.read_bytes()

    launched.clear()
    d.phase_train(cfg, cells=(cells[2],))
    assert [c for c, _ in launched] == ["c3"]
    assert rep.read_bytes() == stamp  # gate did NOT re-fire on wave 2

    # resume: completed cells skip (no relaunch)
    launched.clear()
    d.phase_train(cfg, cells=cells[:2])
    assert launched == []


# ── phase_persist per-cell resume + terminal residual pass ───────────────────


def test_phase_persist_per_cell_resume_and_terminal_pass(monkeypatch, tmp_path):
    cells = ("cA", "cB")
    cfg = _cfg(tmp_path, cells=cells)
    for c, step in (("cA", 4), ("cB", 6)):
        _write_json(tmp_path / c / "selection.json", {"cell": c, "step": step})
        _write_json(
            tmp_path / c / "build_result.json", {"adapter_root": str(tmp_path / c / "train")}
        )
        _mk_rungs(tmp_path / c / "train", [step])
        _write_json(tmp_path / c / "ladder.json", {"reads_by_step": {str(step): {}}})
    monkeypatch.setattr(
        d, "_ensure_dir_tokenizer", create_autospec(d._ensure_dir_tokenizer, return_value=True)
    )
    fake_upload = create_autospec(d.hub._upload, return_value="https://hf.co/x")
    monkeypatch.setattr(d.hub, "_upload", fake_upload)
    fake_records = create_autospec(d._upload_with_transport_retry, return_value="https://hf.co/r")
    monkeypatch.setattr(d, "_upload_with_transport_retry", fake_records)

    # wave 1: persist cA only (selections carries cA)
    out1 = d.phase_persist(cfg, {"cA": {"step": 4}}, cells=["cA"])
    assert out1["uploaded"]["cA"] == "https://hf.co/x"
    assert (tmp_path / "cA" / "persist.json").exists()
    assert fake_upload.call_count == 1

    # wave 1 re-run (crash resume): per-cell record short-circuits, no re-upload
    d.phase_persist(cfg, {}, cells=["cA"])
    assert fake_upload.call_count == 1

    # terminal pass: cB persists via its ON-DISK selection.json (empty
    # selections dict), persist_done.json lands with both cells
    out3 = d.phase_persist(cfg, {}, cells=None)
    assert fake_upload.call_count == 2
    assert set(out3["uploaded"]) >= {"cA", "cB"}
    done = json.loads((tmp_path / "persist_done.json").read_text())
    assert set(done["uploaded"]) >= {"cA", "cB"}

    # legacy all-done marker still short-circuits everything
    d.phase_persist(cfg, {}, cells=["cA"])
    assert fake_upload.call_count == 2


# ── _wave_reap: keeps ONLY the selected rung; asserts the reap took ──────────


def test_wave_reap_keeps_selected_only_and_is_idempotent(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("cA",))
    train = tmp_path / "cA" / "train"
    _mk_rungs(train, [2, 4, 6])
    _write_json(tmp_path / "cA" / "selection.json", {"step": 4})
    _write_json(tmp_path / "cA" / "build_result.json", {"adapter_root": str(train)})

    d._wave_reap(cfg, ["cA"])
    assert sorted(p.name for p in train.glob("checkpoint-*")) == ["checkpoint-4"]
    d._wave_reap(cfg, ["cA"])  # idempotent re-run (crash resume)
    assert sorted(p.name for p in train.glob("checkpoint-*")) == ["checkpoint-4"]


def test_wave_reap_degenerate_missing_selected_rung_raises(tmp_path):
    """Data-dependent gate probe: the selected rung absent from the adapter
    root trips the reap assert (a silent no-op here re-creates the ENOSPC
    class)."""
    cfg = _cfg(tmp_path, cells=("cB",))
    train = tmp_path / "cB" / "train"
    _mk_rungs(train, [2])
    _write_json(tmp_path / "cB" / "selection.json", {"step": 4})
    _write_json(tmp_path / "cB" / "build_result.json", {"adapter_root": str(train)})
    with pytest.raises(RuntimeError, match="wave reap failed for cB"):
        d._wave_reap(cfg, ["cB"])


def test_wave_reap_skips_cells_without_records(tmp_path, caplog):
    cfg = _cfg(tmp_path, cells=("cC",))
    (tmp_path / "cC").mkdir(parents=True)
    d._wave_reap(cfg, ["cC"])  # no selection/build record: warn + skip, no raise


def test_wave_reap_retrained_reselect_empties_original_train_dir(tmp_path):
    """stream-reap + retrain case: adapter_root re-pointed at train_reselect;
    the ORIGINAL train dir's leftover latest rung is reaped too."""
    cfg = _cfg(tmp_path, cells=("cD",))
    train = tmp_path / "cD" / "train"
    resel = tmp_path / "cD" / "train_reselect"
    _mk_rungs(train, [30])  # stream-reap kept only the latest
    _mk_rungs(resel, [8])  # deterministic retrain to the selected step
    _write_json(tmp_path / "cD" / "selection.json", {"step": 8})
    _write_json(tmp_path / "cD" / "build_result.json", {"adapter_root": str(resel)})
    d._wave_reap(cfg, ["cD"])
    assert sorted(p.name for p in resel.glob("checkpoint-*")) == ["checkpoint-8"]
    assert list(train.glob("checkpoint-*")) == []


# ── disk arithmetic (KEEPCELL re-derivation, crash-fix r5) ───────────────────


def test_keepcell_demand_arithmetic():
    content = [c for c in d.G.ALL_FT_CELLS if not d._is_marker(c)]
    marker = [c for c in d.G.ALL_FT_CELLS if d._is_marker(c)]
    # pod A (11 trainable content cells, 8 GPUs): 2x15x15.2 + 11x15.2 + 60
    assert d.keepcell_demand_gb(content, 8) == pytest.approx(
        2 * 15 * d.RUNG_GB + 11 * d.RUNG_GB + d.KEEPCELL_FIXED_OVERHEAD_GB
    )
    # pod B (4 marker cells, 8 GPUs): 2x6x15.2 + 4x15.2 + 60
    assert d.keepcell_demand_gb(marker, 8) == pytest.approx(
        2 * 6 * d.RUNG_GB + 4 * d.RUNG_GB + d.KEEPCELL_FIXED_OVERHEAD_GB
    )
    # 4-GPU landing halves the wave demand (W=1)
    assert d.keepcell_demand_gb(content, 4) == pytest.approx(
        15 * d.RUNG_GB + 11 * d.RUNG_GB + d.KEEPCELL_FIXED_OVERHEAD_GB
    )
    # smoke trains one rung per cell
    assert d.keepcell_demand_gb((d.G.SMOKE_CELL,), 8, smoke=True) == pytest.approx(
        2 * d.RUNG_GB + d.KEEPCELL_FIXED_OVERHEAD_GB
    )
    assert d.wave_width(8) == 2 and d.wave_width(4) == 1


def test_probe_disk_mode_grid_aware_need(monkeypatch, tmp_path):
    """~400 GB free: keep-cell for the marker grid (need ~303) but
    stream-reap for the content grid (need ~683) — the flat 300 GB floor
    alone would have kept keep-cell for BOTH (the v5 ENOSPC)."""

    class _SV:
        f_bavail = int(400e9 / 4096)
        f_frsize = 4096

    monkeypatch.setattr(d, "_runpod_workspace_quota_lane", lambda: False)
    monkeypatch.setattr(d.os, "statvfs", lambda p: _SV)
    assert d.probe_disk_mode(Path(tmp_path), need_gb=303.2) == "keep-cell"
    assert d.probe_disk_mode(Path(tmp_path), need_gb=683.2) == "stream-reap"


def test_wave_headroom_asserts_wave_demand(monkeypatch, tmp_path):
    """The per-wave assert demands the WAVE's rung bytes (not the 60 GB
    phase canary): a content wave of 2 needs ~481 GB."""
    _eight_gpus(monkeypatch)
    seen = {}

    def fake_assert(out_root, *, need_gb, phase):
        seen[phase] = need_gb

    monkeypatch.setattr(d, "assert_out_root_headroom", fake_assert)
    content2 = [c for c in d.G.ALL_FT_CELLS if not d._is_marker(c)][1:3]
    cfg = _cfg(tmp_path, cells=tuple(content2))
    d._wave_headroom(cfg, 1, content2)
    assert seen["p2_train_wave1"] == pytest.approx(2 * 15 * d.RUNG_GB + d.WAVE_MARGIN_GB)


def test_wave_headroom_clears_pending_stale_partial_before_assert(monkeypatch, tmp_path):
    """Pin for CONCERN wave-headroom-stale-partial-not-credited (r2): a
    PENDING cell's stale partial train dir (the pod-A crash-resume shape:
    sibling done + stale partial P held on disk) is cleared BEFORE the wave
    need assert — the _maybe_extend reap-before-assert shape — crediting its
    bytes to free. The fake assert below raises exactly while the stale dir
    still exists (free < need with the doomed partial held), so the pre-fix
    ordering (assert first; _train_one_cell clears later) raises RuntimeError
    and fails this test. A completed sibling's train dir is never touched."""
    _eight_gpus(monkeypatch)
    content2 = [c for c in d.G.ALL_FT_CELLS if not d._is_marker(c)][1:3]
    done, pending_cell = content2
    cfg = _cfg(tmp_path, cells=tuple(content2))
    _write_json(tmp_path / done / "build_result.json", {"cell": done})
    done_train = tmp_path / done / "train"
    _mk_rungs(done_train, [30])  # completed sibling's retained selected rung
    stale = tmp_path / pending_cell / "train"
    _mk_rungs(stale, range(2, 32, 2))  # full 15-rung stale partial (last-rungs crash)
    asserted: list[float] = []

    def fake_assert(out_root, *, need_gb, phase):
        asserted.append(need_gb)
        if stale.exists():
            raise RuntimeError(f"insufficient headroom for {phase}: stale partial still held")

    monkeypatch.setattr(d, "assert_out_root_headroom", fake_assert)
    d._wave_headroom(cfg, 1, content2)  # must NOT raise
    assert not stale.exists()
    assert done_train.exists()
    assert asserted == [pytest.approx(d._cell_rung_demand_gb(pending_cell) + d.WAVE_MARGIN_GB)]


# ── resume-aware wave headroom (code-review v5 Critical pin) ─────────────────


def test_run_waves_resume_headroom_pending_only(monkeypatch, tmp_path):
    """Critical pin (code-review v5 BLOCKER wave-headroom-resume-deadlock):
    on a standard relaunch, _wave_headroom SKIPS a fully-completed wave and
    sizes ``need`` over PENDING cells only (build_result.json — the same
    per-cell resume predicate phase_train no-ops on). The headroom stub below
    FAILS any full-wave demand (simulating the post-crash disk state: wave-1
    rungs still on disk), so the pre-fix unconditional full-wave assert
    (be0020c050, run_waves L1615-17) raises at wave 1 and this test fails
    against that shape — verified by stashing the dispatch fix."""
    _eight_gpus(monkeypatch)
    content4 = tuple(
        c for c in d.G.ALL_FT_CELLS if not d._is_marker(c) and c != d.G.REUSED_FT_CELL
    )[:4]
    cfg = _cfg(tmp_path, cells=content4)
    # wave 1 (cells 0,1) fully trained pre-crash; wave 2 cell 2 trained,
    # cell 3 pending — the mid-train-crash shape.
    for c in content4[:3]:
        _write_json(tmp_path / c / "build_result.json", {"cell": c})
    pending_need = d._cell_rung_demand_gb(content4[3]) + d.WAVE_MARGIN_GB
    asserted: list[tuple[str, float]] = []

    def fake_assert(out_root, *, need_gb, phase):
        asserted.append((phase, need_gb))
        if need_gb > pending_need + 1e-6:
            raise RuntimeError(f"insufficient headroom for {phase}: need {need_gb:.1f} GB")

    monkeypatch.setattr(d, "assert_out_root_headroom", fake_assert)
    trained: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        d, "phase_train", lambda cfg, cells=None: trained.append(tuple(cells)) or {}
    )

    d.run_waves(cfg, do_train=True, do_ladder=False, do_persist=False)  # no RuntimeError

    # wave 1 (both cells done): the assert is skipped entirely; wave 2
    # (1 done / 1 pending): need sized to the PENDING cell only.
    assert asserted == [("p2_train_wave2", pytest.approx(pending_need))]
    # training semantics unchanged: phase_train still receives the FULL wave
    # (its own per-cell build_result.json resume no-ops the completed cells).
    assert trained == [content4[:2], content4[2:]]


# ── resume-aware phase-entry headroom (crash-fix r8, epm:failure v13) ────────


def _record_headroom(monkeypatch):
    seen: list[tuple[str, float]] = []

    def fake_assert(out_root, *, need_gb, phase):
        seen.append((phase, need_gb))

    monkeypatch.setattr(d, "assert_out_root_headroom", fake_assert)
    return seen


def test_headroom_p2_all_done_resume_skips_gate(monkeypatch, tmp_path):
    """The epm:failure v13 shape: free (50.8 GB) < the blanket 60 GB p2 floor
    but EVERY cell is resume-done (build_result.json — the same predicate
    phase_train no-ops on) ⇒ the phase-entry gate is SKIPPED, never asserted.
    The fake below raises on ANY call, so the pre-fix unconditional blanket
    assert (the relaunch-4 crash ~0.1 s into p2) fails this test."""
    cells = ("cA", "cB", "cC")
    cfg = _cfg(tmp_path, cells=cells)
    for c in cells:
        _write_json(tmp_path / c / "build_result.json", {"cell": c})

    def fake_assert(out_root, *, need_gb, phase):
        raise RuntimeError(f"insufficient headroom for {phase}: free 50.8 < {need_gb} GB")

    monkeypatch.setattr(d, "assert_out_root_headroom", fake_assert)
    d._headroom(cfg, "p2_train")  # must NOT raise


def test_headroom_p2_p3_partial_pending_scales_need(monkeypatch, tmp_path):
    """Partial-pending resume: need = PHASE_HEADROOM_GB floor x pending
    fraction (constants untouched), keyed on each phase's OWN resume record
    (p2 build_result.json, p3 ladder_done.json)."""
    cells = ("cA", "cB", "cC", "cD")
    cfg = _cfg(tmp_path, cells=cells)
    for c in cells[:3]:
        _write_json(tmp_path / c / "build_result.json", {"cell": c})
    for c in cells[:2]:
        _write_json(tmp_path / c / "ladder_done.json", {"ts": 0})
    seen = _record_headroom(monkeypatch)
    d._headroom(cfg, "p2_train")
    d._headroom(cfg, "p3_ladder")
    assert seen == [
        ("p2_train", pytest.approx(d.PHASE_HEADROOM_GB["p2_train"] * 1 / 4)),
        ("p3_ladder", pytest.approx(d.PHASE_HEADROOM_GB["p3_ladder"] * 2 / 4)),
    ]


def test_headroom_p8_capture_resume_scaling(monkeypatch, tmp_path):
    """p8 pending is keyed on capture/<arm>/pooled.pt over the REAL
    capture_passes enumeration: all-done skips; one-done scales."""
    cell = next(c for c in d.G.ALL_FT_CELLS if not d._is_marker(c) and c != d.G.REUSED_FT_CELL)
    cfg = _cfg(tmp_path, cells=(cell,))
    passes = d.capture_passes(cfg)
    assert len(passes) == 3  # ft arm + lora pair + base pass
    seen = _record_headroom(monkeypatch)
    for a, _k in passes:
        (tmp_path / "capture" / a).mkdir(parents=True, exist_ok=True)
        (tmp_path / "capture" / a / "pooled.pt").write_bytes(b"x")
    d._headroom(cfg, "p8_capture")
    assert seen == []  # all done ⇒ skipped
    (tmp_path / "capture" / passes[0][0] / "pooled.pt").unlink()
    d._headroom(cfg, "p8_capture")
    assert seen == [("p8_capture", pytest.approx(d.PHASE_HEADROOM_GB["p8_capture"] * 1 / 3))]


def test_headroom_fresh_run_need_identical(monkeypatch, tmp_path):
    """Fresh run (0 resume-done): the asserted need is the UNCHANGED
    PHASE_HEADROOM_GB float for every per-cell-gated phase — the scale
    branch is entered only when n_pending < n_total, so the plan-§9
    fresh-run floor semantics are byte-identical to the pre-fix code."""
    cell = next(c for c in d.G.ALL_FT_CELLS if not d._is_marker(c) and c != d.G.REUSED_FT_CELL)
    cfg = _cfg(tmp_path, cells=(cell,))
    seen = _record_headroom(monkeypatch)
    for phase in ("p2_train", "p3_ladder", "p8_capture"):
        d._headroom(cfg, phase)
    assert seen == [(p, d.PHASE_HEADROOM_GB[p]) for p in ("p2_train", "p3_ladder", "p8_capture")]
    for _phase_name, need in seen:
        assert need == d.PHASE_HEADROOM_GB[_phase_name]  # identity, not approx


def test_headroom_blanket_phases_unchanged(monkeypatch, tmp_path):
    """p0_stage / p9_upload keep the blanket floor (fixed staging / commit
    margins, not per-cell demand) — asserted at the unchanged constant even
    on a fully resume-done out_root."""
    cells = ("cA",)
    cfg = _cfg(tmp_path, cells=cells)
    _write_json(tmp_path / "stage_done.json", {"ts": 0})
    _write_json(tmp_path / "cA" / "build_result.json", {"cell": "cA"})
    seen = _record_headroom(monkeypatch)
    d._headroom(cfg, "p0_stage")
    d._headroom(cfg, "p9_upload")
    assert seen == [
        ("p0_stage", d.PHASE_HEADROOM_GB["p0_stage"]),
        ("p9_upload", d.PHASE_HEADROOM_GB["p9_upload"]),
    ]


# ── registered-extension disk demand (code-review v5 Minor 1 / CONCERN) ──────


def test_maybe_extend_marker_prereaps_and_asserts_headroom(monkeypatch, tmp_path):
    """CONCERN marker-extension-disk-unmodeled: the marker extension pre-reaps
    run-A rungs to the top read (content-branch mirror) and asserts headroom
    for the EXT grid's rung bytes BEFORE the retrain subprocess — un-modeled,
    both marker cells extending held 12 co-resident rungs each (~455 GB >
    pod B's 400 GB disk). Grid + selection logic untouched (a reaped selected
    rung re-derives via _retrain_to_step)."""
    _eight_gpus(monkeypatch)
    mk = next(c for c in d.G.ALL_FT_CELLS if d._is_marker(c))
    cfg = _cfg(tmp_path, cells=(mk,))
    train = tmp_path / mk / "train"
    _mk_rungs(train, d.G.MARKER_FT_GRID)
    _write_json(tmp_path / mk / "build_result.json", {"adapter_root": str(train)})
    reads = {str(s): {"delta_logp_mean": 1.0} for s in d.G.MARKER_FT_GRID}  # ΔG@6 < 5 nat
    _write_json(tmp_path / mk / "ladder.json", {"reads_by_step": reads})
    seen: dict[str, float] = {}
    monkeypatch.setattr(
        d,
        "assert_out_root_headroom",
        lambda out_root, *, need_gb, phase: seen.setdefault(phase, need_gb),
    )
    fake_run = create_autospec(d._run_subprocess)
    monkeypatch.setattr(d, "_run_subprocess", fake_run)
    fake_ladder_unit = create_autospec(d.run_ladder_unit, return_value={})
    monkeypatch.setattr(d, "run_ladder_unit", fake_ladder_unit)

    d._maybe_extend(cfg, mk)

    # pre-reap took: only the top run-A rung survives at retrain launch
    top = max(d.G.MARKER_FT_GRID)
    assert sorted(p.name for p in train.glob("checkpoint-*")) == [f"checkpoint-{top}"]
    assert seen == {
        f"p3_extend_{mk}": pytest.approx(len(d.G.MARKER_FT_EXT_GRID) * d.RUNG_GB + d.WAVE_MARGIN_GB)
    }
    assert fake_run.call_count == 1
    assert fake_ladder_unit.call_count == 1
    assert (tmp_path / mk / "extended.json").exists()


def test_maybe_extend_content_asserts_ext_headroom(monkeypatch, tmp_path):
    """Content branch: the same modeled demand — 15 ext rungs (32..60 step 2)
    asserted after the existing latest-rung pre-reap."""
    _eight_gpus(monkeypatch)
    cc = next(c for c in d.G.ALL_FT_CELLS if not d._is_marker(c))
    cfg = _cfg(tmp_path, cells=(cc,))
    train = tmp_path / cc / "train"
    _mk_rungs(train, [30])
    _write_json(tmp_path / cc / "build_result.json", {"adapter_root": str(train)})
    reads = {str(s): {"rate": 0.1} for s in (10, 20, 30)}  # no rate in band -> extends
    _write_json(tmp_path / cc / "ladder.json", {"reads_by_step": reads})
    seen: dict[str, float] = {}
    monkeypatch.setattr(
        d,
        "assert_out_root_headroom",
        lambda out_root, *, need_gb, phase: seen.setdefault(phase, need_gb),
    )
    monkeypatch.setattr(d, "_run_subprocess", create_autospec(d._run_subprocess))
    monkeypatch.setattr(d, "run_ladder_unit", create_autospec(d.run_ladder_unit, return_value={}))

    d._maybe_extend(cfg, cc)

    n_ext = len(range(32, d.G.CONTENT_EXT_CEILING + 1, 2))
    assert seen == {f"p3_extend_{cc}": pytest.approx(n_ext * d.RUNG_GB + d.WAVE_MARGIN_GB)}


# ── coverage-aware persist_done.json (code-review v5 Minor 2) ────────────────


def test_phase_persist_terminal_pass_is_coverage_aware(monkeypatch, tmp_path):
    """Minor 2 (code-review v5): a terminal pass with a cell still missing its
    selection does NOT write persist_done.json (whose existence permanently
    short-circuits later persist passes, orphaning later-laddered cells'
    uploads); once every non-reused cell has records the marker lands."""
    cells = ("cA", "cB")
    cfg = _cfg(tmp_path, cells=cells)

    def _seed_cell(c, step):
        _write_json(tmp_path / c / "selection.json", {"cell": c, "step": step})
        _write_json(
            tmp_path / c / "build_result.json", {"adapter_root": str(tmp_path / c / "train")}
        )
        _mk_rungs(tmp_path / c / "train", [step])

    _seed_cell("cA", 4)  # cB not yet laddered (--phases persist on a partial root)
    monkeypatch.setattr(
        d, "_ensure_dir_tokenizer", create_autospec(d._ensure_dir_tokenizer, return_value=True)
    )
    monkeypatch.setattr(
        d.hub, "_upload", create_autospec(d.hub._upload, return_value="https://hf.co/x")
    )
    monkeypatch.setattr(
        d,
        "_upload_with_transport_retry",
        create_autospec(d._upload_with_transport_retry, return_value="https://hf.co/r"),
    )

    out1 = d.phase_persist(cfg, {}, cells=None)
    assert "cA" in out1["uploaded"] and "cB" not in out1["uploaded"]
    # pre-fix: persist_done.json was written here, masking cB forever
    assert not (tmp_path / "persist_done.json").exists()

    _seed_cell("cB", 6)  # cB ladders later; the next persist pass must still fire
    out2 = d.phase_persist(cfg, {}, cells=None)
    assert set(out2["uploaded"]) >= {"cA", "cB"}
    done = json.loads((tmp_path / "persist_done.json").read_text())
    assert set(done["uploaded"]) >= {"cA", "cB"}


# ── crash-fix r6: the _generate_responses_vllm reuse seam (token-id rows) ────
#
# The reused helper emits TOKEN-ID rows with NO "response" text key; both
# marker read sites consumed r["response"] -> KeyError at p1_parity (live
# crash) + the p5-8 marker H2-lattice twin. These tests pin the seam
# statically (schema extracted from the helper SOURCE) and execute the
# real consumption bodies on that exact schema.

_SEAM_SCHEMA = {
    "persona",
    "question_idx",
    "prompt_token_ids",
    "response_token_ids",
    "finish_reason",
}


class _FakeTok:
    """Signature-conformant decode-only tokenizer fake (external model
    boundary). Vocab pins the marker token id 83399 -> " ※"."""

    VOCAB: typing.ClassVar[dict[int, str]] = {
        11: "<|im_start|>user\nQ0<|im_end|>\n<|im_start|>assistant\n",
        12: "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n",
        21: "The answer is 42.",
        22: " Thanks.",
        23: "No marker in this one.",
        31: "好的",
        83399: " ※",
    }

    def decode(self, ids):
        return "".join(self.VOCAB[i] for i in ids)


def _seam_rows():
    """Two rows in EXACTLY the helper's emitted schema (pinned below);
    row 0 emits the marker (id 83399 in response_token_ids), row 1 not."""
    return [
        {
            "persona": "persona_software_engineer",
            "question_idx": 0,
            "prompt_token_ids": [11],
            "response_token_ids": [21, 83399, 22],
            "finish_reason": "stop",
        },
        {
            "persona": "persona_software_engineer",
            "question_idx": 1,
            "prompt_token_ids": [12],
            "response_token_ids": [23],
            "finish_reason": "length",
        },
    ]


def test_generate_responses_vllm_row_schema_pin():
    """Extract the REAL emitted row schema from the reused helper's source:
    the rows.append block carries exactly the five token-id keys and no
    "response" text key. Helper schema drift fails HERE, loudly, instead of
    as a pod-side KeyError."""
    from explore_persona_space.analysis import representation_shift as rs

    src = inspect.getsource(rs._generate_responses_vllm)
    start = src.index("rows.append(")
    block = src[start : src.index("}", start)]
    keys = set(re.findall(r'"([a-z_]+)":', block))
    assert keys == _SEAM_SCHEMA, keys


def test_marker_source_read_decodes_token_id_rows(monkeypatch, tmp_path):
    """Crash-fix r6 body pin (p1_parity crash): _marker_source_read consumes
    the REAL token-id row schema — pre-fix this body raised KeyError:
    'response' at the strip loop. Executes the real body; fakes only the
    vLLM / HF-model / tokenizer-load boundaries."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis import representation_shift as rs

    rows = _seam_rows()
    fake_gen = create_autospec(rs._generate_responses_vllm, return_value=rows)
    monkeypatch.setattr(rs, "_generate_responses_vllm", fake_gen)
    fake_tok = _FakeTok()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tok)

    slot_calls: list[tuple[str, list[str]]] = []

    def fake_slot_read(model_path, contexts, device="cuda:0"):
        # def mirrors d1112._marker_slot_read's signature (boundary fake)
        slot_calls.append((model_path, list(contexts)))
        return [
            {"logp": -2.0, "z_marker": 1.0, "z_eos": 3.0, "argmax_id": 151645} for _ in contexts
        ]

    monkeypatch.setattr(d, "_marker_slot_read", fake_slot_read)

    cfg = _cfg(tmp_path, cells=("mk-pers-ft-con-s42",))
    out_dir = tmp_path / "slotread"
    rec = d._marker_source_read(cfg, "trained/model", out_dir)

    # contexts = decode(prompt_token_ids) + strip-at-marker(decode(response))
    # — the d1333 TEXT-concat recipe, trained AND base reads alike.
    exp = [
        _FakeTok.VOCAB[11] + "The answer is 42.",
        _FakeTok.VOCAB[12] + "No marker in this one.",
    ]
    assert [mp for mp, _ in slot_calls] == ["trained/model", d.DEFAULT_BASE_MODEL]
    assert all(ctx == exp for _, ctx in slot_calls)
    assert rec["gen_emission_rate"] == 0.5  # row 0 emitted, row 1 not
    assert rec["n"] == 2

    # rollouts.json persists rollout TEXT (#779) alongside the token ids
    saved = json.loads((out_dir / "rollouts.json").read_text())
    assert [r["response_text"] for r in saved["rows"]] == [
        "The answer is 42. ※ Thanks.",
        "No marker in this one.",
    ]
    assert set(saved["rows"][0]) == _SEAM_SCHEMA | {"response_text"}


def test_decode_marker_rows_body():
    """Direct body test of the shared consumption helper (the ONE function
    both marker read sites route rows through): contexts, emission flags,
    response_text write-back."""
    rows = _seam_rows()
    contexts, emitted = d._decode_marker_rows(_FakeTok(), rows)
    assert contexts == [
        _FakeTok.VOCAB[11] + "The answer is 42.",
        _FakeTok.VOCAB[12] + "No marker in this one.",
    ]
    assert emitted == [True, False]
    assert [r["response_text"] for r in rows] == [
        "The answer is 42. ※ Thanks.",
        "No marker in this one.",
    ]


def test_marker_read_sites_share_decode_helper():
    """Both marker read sites (p1_parity source read + p5-8 H2-lattice panel
    read) consume rows through the shared helper; neither retains the
    crash-class r["response"] read."""
    for fn in (d._marker_source_read, d._marker_panel_read):
        src = inspect.getsource(fn)
        assert "_decode_marker_rows(" in src, fn.__name__
        assert 'r["response"]' not in src, fn.__name__


def test_cjk_scan_reads_response_text_and_decodes_token_ids(monkeypatch, tmp_path):
    """Latent-consumer fix (same seam class): _cjk_scan read
    r.get("response", "") from token-id rows — "" for EVERY row, so the
    audit silently counted 0 intruded. It now prefers response_text and
    lazily decodes token-id-only rows (capture raw_rows.json)."""
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTok())
    cfg = _cfg(tmp_path, cells=())
    _write_json(
        tmp_path / "marker_panel" / "armX" / "rollouts.json",
        {"rows": [{"response_text": "clean text"}, {"response_text": "好的 intruded"}]},
    )
    _write_json(
        tmp_path / "capture" / "armY" / "raw_rows.json",
        {"rows": [dict(_seam_rows()[0], response_token_ids=[31])]},
    )
    pools = d._cjk_scan(cfg)["pools"]
    assert pools["marker_panel/armX/rollouts.json"] == {"n": 2, "intruded": 1}
    assert pools["capture/armY/raw_rows.json"] == {"n": 1, "intruded": 1}


# ── billing-403 deferral (crash-fix r7, pod-1586 crash #6) ───────────────────

BILLING_MSG = (
    "403 Forbidden: You need to setup automatic credit recharge in order to upload more data"
)


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _billing_err(code=403, msg=BILLING_MSG):
    e = RuntimeError(msg)
    if code is not None:
        e.response = _Resp(code)
    return e


def test_is_billing_403_predicate():
    """(a) ONLY 403 + 'credit recharge' matches; 403-other-text and 5xx (even
    phrase-bearing) never do — those keep the fail-loud path."""
    assert d._is_billing_403(_billing_err())
    # response-less xet-boundary shape: message carries BOTH markers
    assert d._is_billing_403(_billing_err(code=None))
    # 403 with other text (e.g. the storage-quota 403) -> NOT billing
    assert not d._is_billing_403(
        _billing_err(msg="403 Forbidden: You have exceeded your public storage space")
    )
    # 5xx carrying the phrase -> response code wins -> NOT billing
    assert not d._is_billing_403(_billing_err(code=500))
    # response-less phrase WITHOUT the 403 marker -> NOT billing
    assert not d._is_billing_403(RuntimeError("please setup credit recharge"))


def _persist_fixture(monkeypatch, tmp_path, cells=(("cA", 4), ("cB", 6))):
    for c, step in cells:
        _write_json(tmp_path / c / "selection.json", {"cell": c, "step": step})
        _write_json(
            tmp_path / c / "build_result.json", {"adapter_root": str(tmp_path / c / "train")}
        )
        _mk_rungs(tmp_path / c / "train", [step])
    monkeypatch.setattr(
        d, "_ensure_dir_tokenizer", create_autospec(d._ensure_dir_tokenizer, return_value=True)
    )
    fake_records = create_autospec(d._upload_with_transport_retry, return_value="https://hf.co/r")
    monkeypatch.setattr(d, "_upload_with_transport_retry", fake_records)
    return fake_records


def test_phase_persist_defers_billing_403_and_continues_wave(monkeypatch, tmp_path):
    """(b) FAILS-PRE-FIX: pre-r7 hub._upload swallowed the billing-403
    (logged 'Keeping local path.', returned '') and phase_persist raised
    'selected-rung upload returned no path' at the FIRST cell — no durable
    record, wave dead (pod-1586 crash #6, wave-1 p4_persist)."""
    cfg = _cfg(tmp_path, cells=("cA", "cB"))
    _persist_fixture(monkeypatch, tmp_path)
    fake_upload = create_autospec(d.hub._upload, side_effect=_billing_err())
    monkeypatch.setattr(d.hub, "_upload", fake_upload)

    out = d.phase_persist(cfg, {"cA": {"step": 4}, "cB": {"step": 6}}, cells=["cA", "cB"])
    # wave CONTINUED past cA's deferral: BOTH cells attempted, no raise
    assert fake_upload.call_count == 2
    assert "cA" not in out["uploaded"] and "cB" not in out["uploaded"]
    for c, step in (("cA", 4), ("cB", 6)):
        rec = json.loads((tmp_path / c / "persist_deferred.json").read_text())
        assert rec["cell"] == c and rec["step"] == step
        assert rec["repo_id"] == d.G.OVERFLOW_REPO
        assert rec["path_in_repo"] == f"issue1586/{c}/checkpoint-{step}"
        assert "credit recharge" in rec["error"]
        assert Path(rec["local_path"]).exists()  # checkpoint RETAINED
        assert not (tmp_path / c / "persist.json").exists()  # no success record


@pytest.mark.parametrize(
    "effect",
    [
        _billing_err(msg="403 Forbidden: some other reason"),  # 403, other text
        _billing_err(code=500),  # 5xx (even phrase-bearing)
    ],
)
def test_phase_persist_non_billing_exceptions_still_raise(monkeypatch, tmp_path, effect):
    cfg = _cfg(tmp_path, cells=("cA",))
    _persist_fixture(monkeypatch, tmp_path, cells=(("cA", 4),))
    monkeypatch.setattr(d.hub, "_upload", create_autospec(d.hub._upload, side_effect=effect))
    with pytest.raises(RuntimeError):
        d.phase_persist(cfg, {"cA": {"step": 4}}, cells=["cA"])
    assert not (tmp_path / "cA" / "persist_deferred.json").exists()


def test_phase_persist_no_path_return_still_fail_loud(monkeypatch, tmp_path):
    """The legacy '' no-path return (missing token / 0-files verify) keeps
    the L1770-class RuntimeError — deferral is exception-classified only."""
    cfg = _cfg(tmp_path, cells=("cA",))
    _persist_fixture(monkeypatch, tmp_path, cells=(("cA", 4),))
    monkeypatch.setattr(d.hub, "_upload", create_autospec(d.hub._upload, return_value=""))
    with pytest.raises(RuntimeError, match="selected-rung upload returned no path"):
        d.phase_persist(cfg, {"cA": {"step": 4}}, cells=["cA"])


def test_phase_persist_resume_retries_deferred_and_clears(monkeypatch, tmp_path):
    """(e) p4-resume on a deferred cell RETRIES (no persist.json to skip on);
    on success it writes persist.json and clears the deferral record."""
    cfg = _cfg(tmp_path, cells=("cA",))
    _persist_fixture(monkeypatch, tmp_path, cells=(("cA", 4),))
    _write_json(
        tmp_path / "cA" / "persist_deferred.json",
        {"cell": "cA", "step": 4, "local_path": str(tmp_path / "cA" / "train" / "checkpoint-4")},
    )
    fake_upload = create_autospec(d.hub._upload, return_value="ovf/issue1586/cA/checkpoint-4")
    monkeypatch.setattr(d.hub, "_upload", fake_upload)
    out = d.phase_persist(cfg, {"cA": {"step": 4}}, cells=["cA"])
    assert out["uploaded"]["cA"] == "ovf/issue1586/cA/checkpoint-4"
    assert not (tmp_path / "cA" / "persist_deferred.json").exists()
    rec = json.loads((tmp_path / "cA" / "persist.json").read_text())
    assert rec["url"].endswith("checkpoint-4")


def test_wave_reap_retains_deferred_selected_rung_both_modes(tmp_path):
    """(c) The reap keep-set is {sel_step} from selection.json — persist-
    independent — so a deferred-unuploaded selected rung survives in BOTH
    shapes: keep-cell (rungs in train/) and stream-reap's retrained-reselect
    (adapter_root re-pointed at train_reselect). Absence would trip the
    post-reap left == {sel_step} assert."""
    cfg = _cfg(tmp_path, cells=("cA", "cD"))
    # keep-cell shape
    train = tmp_path / "cA" / "train"
    _mk_rungs(train, [2, 4, 6])
    _write_json(tmp_path / "cA" / "selection.json", {"step": 4})
    _write_json(tmp_path / "cA" / "build_result.json", {"adapter_root": str(train)})
    _write_json(tmp_path / "cA" / "persist_deferred.json", {"cell": "cA", "step": 4})
    # stream-reap retrained-reselect shape
    resel = tmp_path / "cD" / "train_reselect"
    _mk_rungs(tmp_path / "cD" / "train", [30])
    _mk_rungs(resel, [8])
    _write_json(tmp_path / "cD" / "selection.json", {"step": 8})
    _write_json(tmp_path / "cD" / "build_result.json", {"adapter_root": str(resel)})
    _write_json(tmp_path / "cD" / "persist_deferred.json", {"cell": "cD", "step": 8})
    d._wave_reap(cfg, ["cA", "cD"])
    assert (train / "checkpoint-4" / "model.safetensors").exists()
    assert (resel / "checkpoint-8" / "model.safetensors").exists()
    assert sorted(p.name for p in train.glob("checkpoint-*")) == ["checkpoint-4"]
    assert sorted(p.name for p in resel.glob("checkpoint-*")) == ["checkpoint-8"]


def _p9_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "_headroom", create_autospec(d._headroom, return_value=None))
    fake_records = create_autospec(d._upload_with_transport_retry, return_value="https://hf.co/r")
    monkeypatch.setattr(d, "_upload_with_transport_retry", fake_records)
    _write_json(
        tmp_path / "cA" / "persist_deferred.json",
        {
            "cell": "cA",
            "step": 4,
            "local_path": str(tmp_path / "cA" / "train" / "checkpoint-4"),
            "repo_id": d.G.OVERFLOW_REPO,
            "repo_type": "model",
            "path_in_repo": "issue1586/cA/checkpoint-4",
        },
    )
    _mk_rungs(tmp_path / "cA" / "train", [4])
    return fake_records


def test_phase_upload_replays_deferred_success_clears_record(monkeypatch, tmp_path):
    """(d) p9 replay success path: deferral record replayed FIRST, cleared,
    normal persist.json written; phase completes without raising."""
    cfg = _cfg(tmp_path, cells=())
    _p9_fixture(monkeypatch, tmp_path)
    fake_upload = create_autospec(d.hub._upload, return_value="ovf/issue1586/cA/checkpoint-4")
    monkeypatch.setattr(d.hub, "_upload", fake_upload)
    d.phase_upload(cfg, {})
    assert fake_upload.call_count == 1
    assert not (tmp_path / "cA" / "persist_deferred.json").exists()
    rec = json.loads((tmp_path / "cA" / "persist.json").read_text())
    assert rec == {"cell": "cA", "step": 4, "url": "ovf/issue1586/cA/checkpoint-4"}


def test_phase_upload_still_blocked_exits_fail_loud_naming_cells(monkeypatch, tmp_path):
    """(d) p9 still-blocked path: residual (non-LFS) commits still land, then
    the phase raises naming every deferred cell + retained local path; the
    durable record + summary JSON survive for the next relaunch."""
    cfg = _cfg(tmp_path, cells=())
    fake_records = _p9_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        d.hub, "_upload", create_autospec(d.hub._upload, side_effect=_billing_err())
    )
    monkeypatch.setattr(d, "_cjk_scan", create_autospec(d._cjk_scan, return_value={"pools": {}}))
    with pytest.raises(RuntimeError, match=r"billing-403.*cA \(retained at .*checkpoint-4\)"):
        d.phase_upload(cfg, {})
    assert (tmp_path / "cA" / "persist_deferred.json").exists()
    summary = json.loads((tmp_path / "persist_deferred_summary.json").read_text())
    assert [r["cell"] for r in summary["deferred"]] == ["cA"]
    # residual staged commits ran BEFORE the terminal raise (misc stage:
    # cjk_audit.json exists), so nothing small was held hostage
    assert fake_records.call_count >= 1


def test_phase_upload_replay_non_billing_failure_raises_immediately(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=())
    _p9_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        d.hub,
        "_upload",
        create_autospec(d.hub._upload, side_effect=_billing_err(code=500)),
    )
    with pytest.raises(RuntimeError):
        d.phase_upload(cfg, {})
    assert (tmp_path / "cA" / "persist_deferred.json").exists()  # record survives


def test_hub_upload_raise_on_error_real_body(monkeypatch, tmp_path):
    """Production-body test for the r7-modified hub._upload (the other tests
    in this section stub it): executes the REAL body — token check, dir
    branch, filecount guard, create_repo, upload_folder via _retry_upload,
    except chain — faking ONLY the HfApi network boundary with a
    def-mirroring fake. The billing-403 re-raises under raise_on_error=True
    (un-retried: 403 is non-transient) and keeps the legacy '' swallow under
    the default, so every existing caller is behavior-identical."""
    import huggingface_hub

    from explore_persona_space.orchestrate import hub

    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "model.safetensors").write_text("x")
    err = _billing_err()
    calls = {"n": 0}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=True):
            return None

        def upload_folder(
            self,
            folder_path=None,
            repo_id=None,
            path_in_repo=None,
            repo_type=None,
            ignore_patterns=None,
        ):
            calls["n"] += 1
            raise err

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    # default: legacy log-and-return-"" swallow (every existing caller)
    assert hub._upload(src, "org/ovf", "model", "issue1586/cA/checkpoint-4") == ""
    assert calls["n"] == 1  # non-transient 403: _retry_upload did NOT retry
    with pytest.raises(RuntimeError, match="credit recharge"):
        hub._upload(src, "org/ovf", "model", "issue1586/cA/checkpoint-4", raise_on_error=True)
    assert calls["n"] == 2
    assert d._is_billing_403(err)
