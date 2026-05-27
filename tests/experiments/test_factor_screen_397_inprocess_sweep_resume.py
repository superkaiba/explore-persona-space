"""Round 11 in-process sweep ↔ resume integration (task #397).

Companion to ``test_factor_screen_397_inprocess_sweep.py``: that file
pins the per-cell pipeline ORDER. This file pins that the sweep-level
resume filter (``filter_jobs_for_resume``) runs BEFORE the per-cell
helper is invoked — completed cells must be SKIPPED, never re-launched.

Round 11 deleted the subprocess pool; before the rewrite, this contract
was covered by tests asserting ``_launch_cell_subprocess`` was called N-
or-fewer times. With the in-process serial design,
``_run_one_cell_inprocess`` is the analogous call site — and the resume
filter MUST exclude completed cells before this helper is invoked,
otherwise the dispatcher re-trains cells that already have an HF Hub
adapter, wasting GPU + blowing the MooseFS quota with redundant
checkpoints.

CPU-only; no GPU, no model load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

# Load the dispatcher (lives under scripts/, not a package).
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


def _build_args(slab_root: Path, *, no_resume: bool, resume_source: str) -> argparse.Namespace:
    return argparse.Namespace(
        issue=397,
        mode="sweep",
        pool_dir=slab_root / "pools",
        slab_root=slab_root,
        smoke_cell="10010",
        smoke_source="librarian",
        smoke_seed=42,
        sources="librarian",
        seeds="42",
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        require_smoke_pass=True,
        skip_smoke_pass_check=False,
        smoke_pass_confirmed=True,
        dry_run=False,
        no_resume=no_resume,
        resume_source=resume_source,
        log_level="INFO",
    )


def _write_metrics_json(slab_root: Path, cell_key: str, source: str, seed: int) -> None:
    cell_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "metrics.json").write_text(
        json.dumps(
            {
                "marker": "※",
                "cell_key": cell_key,
                "source": source,
                "seed": seed,
                "personas": {source: {"substring_rate": 0.5, "total": 100}},
            }
        ),
        encoding="utf-8",
    )


def test_resume_filter_skips_completed_cells_before_inprocess_call(monkeypatch) -> None:
    """Resume filter MUST skip completed cells BEFORE
    ``_run_one_cell_inprocess`` is invoked.

    Without this, the dispatcher re-trains a cell that already has an HF
    Hub adapter — wasting ~15 min of GPU per cell and blowing the
    MooseFS ~130 GB per-pod quota with redundant intermediate
    checkpoints.

    Stages two cells: cell 00000 has metrics.json on disk (complete);
    cell 00001 does not. With ``--resume-source=local``, only cell
    00001 should reach ``_run_one_cell_inprocess``.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # Cell 00000 is already complete locally.
        _write_metrics_json(slab, "00000", "librarian", 42)
        # Cell 00001 is NOT complete → should be the only one launched.

        args = _build_args(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        inprocess_calls: list[tuple[str, str, int]] = []

        def _fake_inprocess(**kwargs):
            inprocess_calls.append((kwargs["cell"].key, kwargs["source"], kwargs["seed"]))
            return 0

        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0, f"Sweep should return 0; got {rc}"

        # Only cell 00001 reached the in-process helper.
        assert len(inprocess_calls) == 1, (
            f"Resume filter failed: expected 1 in-process call (00001 only); "
            f"got {len(inprocess_calls)} → {inprocess_calls}"
        )
        assert inprocess_calls[0][0] == "00001", (
            f"Expected cell 00001 to be the survivor; got {inprocess_calls[0]}"
        )


def test_no_resume_flag_forces_completed_cells_through_inprocess(monkeypatch) -> None:
    """``--no-resume`` bypasses the resume filter — every cell reaches
    ``_run_one_cell_inprocess``, even ones that already have metrics.json.

    Used when the user explicitly wants to regenerate everything (e.g.
    after discovering a training-side bug that invalidated prior results).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        _write_metrics_json(slab, "00001", "librarian", 42)

        args = _build_args(slab, no_resume=True, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        inprocess_calls: list[tuple[str, str, int]] = []

        def _fake_inprocess(**kwargs):
            inprocess_calls.append((kwargs["cell"].key, kwargs["source"], kwargs["seed"]))
            return 0

        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        # Both cells reached the in-process helper (--no-resume overrode the skip).
        assert len(inprocess_calls) == 2, (
            f"--no-resume must reach all cells; got {len(inprocess_calls)} calls"
        )


def test_resume_filter_runs_before_first_inprocess_call(monkeypatch) -> None:
    """The resume filter MUST run BEFORE ANY ``_run_one_cell_inprocess``
    call. If filter_jobs_for_resume is called AFTER the first cell, then
    on a partial sweep crash + restart, the dispatcher would
    unnecessarily re-train the cell it had just finished (because the
    skip-list wasn't checked yet for that cell).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)

        args = _build_args(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        # Record interleaved order: filter_jobs_for_resume vs in-process calls.
        order: list[str] = []
        real_filter = _dispatch.filter_jobs_for_resume

        def _spy_filter(*args, **kwargs):
            order.append("filter")
            return real_filter(*args, **kwargs)

        monkeypatch.setattr(_dispatch, "filter_jobs_for_resume", _spy_filter)

        def _fake_inprocess(**kwargs):
            order.append(f"inprocess_{kwargs['cell'].key}")
            return 0

        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        # Order MUST be: filter THEN any inprocess calls.
        assert order[0] == "filter", (
            f"filter_jobs_for_resume must run BEFORE any _run_one_cell_inprocess "
            f"call; got order: {order}"
        )
        # And only cell 00001 ran (00000 was filtered out as complete).
        assert order[1:] == ["inprocess_00001"], (
            f"Expected only 00001 to run after filter; got: {order[1:]}"
        )
