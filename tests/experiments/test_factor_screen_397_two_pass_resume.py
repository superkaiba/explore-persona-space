"""Round 12 two-pass sweep resume contract (task #397).

Companion to ``test_factor_screen_397_two_pass_sweep.py``: that file
pins the per-pass pipeline order. This file pins that the
two-pass resume scan correctly partitions cells across Pass 1 and
Pass 2 based on which artifacts already exist on disk + HF Hub.

The Round 11 single-phase resume model ("complete = metrics.json on
disk") doesn't fit the two-pass design — a cell can be Pass-1-complete
(logprob_panel.json + Hub adapter) but Pass-2-pending (no metrics.json),
in which case ONLY Pass 2 should run for that cell.

Round 12 resume rules (per the brief):

  - Cell fully complete (skip both passes) = logprob_panel.json AND
    metrics.json AND adapter-on-Hub all present.
  - Cell needs Pass 1 only = no logprob_panel.json OR no Hub adapter.
  - Cell needs Pass 2 only = logprob_panel.json + Hub adapter present
    but no metrics.json.
  - Cell needs both passes = neither logprob_panel.json nor metrics.json
    present.

CPU-only; no GPU, no model load. Heavy entry points monkeypatched.
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


def _stage_pass1_complete(slab_root: Path, cell_key: str, source: str, seed: int) -> None:
    """Pre-stage Pass-1 outputs (logprob_panel.json + adapter dir)."""
    cell_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "adapter").mkdir()
    (cell_dir / "logprob_panel.json").write_text(json.dumps({"ckpt-25": {"※": [-1.0]}}))


def _stage_pass2_complete(slab_root: Path, cell_key: str, source: str, seed: int) -> None:
    """Pre-stage Pass-2 output (metrics.json)."""
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
        )
    )


def test_pass1_only_cells_skip_pass1_and_run_pass2(monkeypatch) -> None:
    """Cells with logprob_panel.json AND adapter on disk are Pass-1-complete
    but Pass-2-pending — they must skip Pass 1 and reach Pass 2.

    This is the canonical resume case: pod crashed between Pass 1 and
    Pass 2 (or Pass 2 crashed mid-cell); the next run should not redo
    Pass 1 training for cells already done.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab = Path(tmpdir)
        _stage_pass1_complete(slab, "00000", "librarian", 42)

        args = _build_args(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        pass1_jobs_seen: list[tuple[str, str, int]] = []
        pass2_jobs_seen: list[tuple[str, str, int]] = []

        def _spy_pass1(cells_to_run, *, args):
            pass1_jobs_seen.extend((c.key, s, sd) for (c, s, sd) in cells_to_run)
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        def _spy_pass2(cells_to_run, *, args):
            pass2_jobs_seen.extend((c.key, s, sd) for (c, s, sd) in cells_to_run)
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        monkeypatch.setattr(_dispatch, "_run_pass1_hf", _spy_pass1)
        monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", lambda *a, **k: None)
        monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _spy_pass2)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        # Cell 00000 already Pass-1-complete → only cell 00001 needs Pass 1.
        assert pass1_jobs_seen == [("00001", "librarian", 42)], (
            f"Pass 1 wrong queue: {pass1_jobs_seen}"
        )
        # Both cells need Pass 2 (neither has metrics.json).
        assert sorted(pass2_jobs_seen) == sorted(
            [("00000", "librarian", 42), ("00001", "librarian", 42)]
        ), f"Pass 2 wrong queue: {pass2_jobs_seen}"


def test_fully_complete_cells_skip_both_passes(monkeypatch) -> None:
    """Cells with logprob_panel.json AND adapter AND metrics.json are
    FULLY complete — skip both passes entirely.

    Without this, a fresh launch on a recovered pod would re-run every
    cell from scratch, wasting GPU + blowing the MooseFS quota.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab = Path(tmpdir)
        _stage_pass1_complete(slab, "00000", "librarian", 42)
        _stage_pass2_complete(slab, "00000", "librarian", 42)

        args = _build_args(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        pass1_calls = {"n": 0}
        pass2_calls = {"n": 0}
        teardown_calls = {"n": 0}

        def _spy_pass1(cells_to_run, *, args):
            pass1_calls["n"] += 1
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        def _spy_pass2(cells_to_run, *, args):
            pass2_calls["n"] += 1
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        def _spy_teardown(*a, **k):
            teardown_calls["n"] += 1

        monkeypatch.setattr(_dispatch, "_run_pass1_hf", _spy_pass1)
        monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", _spy_teardown)
        monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _spy_pass2)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        assert pass1_calls["n"] == 0, "Pass 1 must not run for fully complete cells"
        assert pass2_calls["n"] == 0, "Pass 2 must not run for fully complete cells"
        assert teardown_calls["n"] == 0, (
            "Teardown must not fire when there's no Pass 2 work — the teardown's "
            "only job is preparing for vLLM load."
        )


def test_no_resume_flag_forces_both_passes_for_every_cell(monkeypatch) -> None:
    """``--no-resume`` bypasses the resume scan — every cell ends up in
    both Pass 1 and Pass 2 queues, even ones that have all artifacts on
    disk.

    Used when the user explicitly wants to regenerate everything (e.g.
    after discovering a training-side bug that invalidated prior results).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab = Path(tmpdir)
        _stage_pass1_complete(slab, "00000", "librarian", 42)
        _stage_pass2_complete(slab, "00000", "librarian", 42)

        args = _build_args(slab, no_resume=True, resume_source="local")
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        pass1_jobs: list = []
        pass2_jobs: list = []

        def _spy_pass1(cells_to_run, *, args):
            pass1_jobs.extend(cells_to_run)
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        def _spy_pass2(cells_to_run, *, args):
            pass2_jobs.extend(cells_to_run)
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        monkeypatch.setattr(_dispatch, "_run_pass1_hf", _spy_pass1)
        monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", lambda *a, **k: None)
        monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _spy_pass2)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        # With --no-resume, the fully-complete cell still ends up in BOTH queues.
        assert len(pass1_jobs) == 1
        assert len(pass2_jobs) == 1


def test_pass2_only_jobs_skip_pass1_call_entirely(monkeypatch) -> None:
    """When every queued cell only needs Pass 2 (all Pass-1-complete),
    _run_pass1_hf must NOT be invoked. Invoking it with an empty queue
    is wasteful but more importantly: a no-op pass-1 call could re-create
    cell dirs / re-load tokenizer / do other side-effecty setup that
    isn't needed.

    The orchestrator-visible signal is also cleaner: "Pass 1: nothing
    to run" is what the orchestrator parses.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab = Path(tmpdir)
        # Both cells Pass-1-complete; neither Pass-2-complete.
        _stage_pass1_complete(slab, "00000", "librarian", 42)
        _stage_pass1_complete(slab, "00001", "librarian", 42)

        args = _build_args(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        pass1_calls = {"n": 0}
        pass2_calls = {"n": 0}

        def _spy_pass1(cells_to_run, *, args):
            pass1_calls["n"] += 1
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        def _spy_pass2(cells_to_run, *, args):
            pass2_calls["n"] += 1
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        monkeypatch.setattr(_dispatch, "_run_pass1_hf", _spy_pass1)
        monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", lambda *a, **k: None)
        monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _spy_pass2)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        assert pass1_calls["n"] == 0, (
            "Pass 1 must NOT be invoked when every cell is Pass-1-complete"
        )
        assert pass2_calls["n"] == 1, "Pass 2 must run for both cells (one combined call)"


def test_is_cell_pass1_complete_true_for_well_formed_logprob_json() -> None:
    """The Pass-1 resume probe ``is_cell_pass1_complete`` returns True
    when logprob_panel.json parses cleanly to a non-empty dict.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "c"
        cell_dir.mkdir()
        (cell_dir / "logprob_panel.json").write_text(
            json.dumps({"ckpt-25": {"※": [-1.0, -1.2]}, "ckpt-50": {"※": [-0.9]}})
        )
        assert _dispatch.is_cell_pass1_complete(cell_dir) is True


def test_is_cell_pass1_complete_false_for_missing_or_malformed() -> None:
    """Missing logprob_panel.json OR malformed JSON OR empty dict →
    treat as Pass 1 NOT complete (re-run is safe; idempotent).
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Missing file.
        cell_dir_missing = Path(tmp) / "missing"
        cell_dir_missing.mkdir()
        assert _dispatch.is_cell_pass1_complete(cell_dir_missing) is False

        # Malformed JSON.
        cell_dir_bad = Path(tmp) / "bad"
        cell_dir_bad.mkdir()
        (cell_dir_bad / "logprob_panel.json").write_text("{not-json")
        assert _dispatch.is_cell_pass1_complete(cell_dir_bad) is False

        # Empty dict (parses cleanly but scored zero checkpoints).
        cell_dir_empty = Path(tmp) / "empty"
        cell_dir_empty.mkdir()
        (cell_dir_empty / "logprob_panel.json").write_text(json.dumps({}))
        assert _dispatch.is_cell_pass1_complete(cell_dir_empty) is False


def test_resume_writes_sweep_resume_verdict_file_when_skipping(monkeypatch) -> None:
    """When ≥1 cell is skipped (fully complete) OR ≥1 cell is Pass-1-
    skipped, the dispatcher writes SWEEP_RESUME.json under slab_root.
    Orchestrator reads it and posts the epm:sweep-resume marker.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # Cell 00000 fully complete (Pass-1 + Pass-2).
        _stage_pass1_complete(slab, "00000", "librarian", 42)
        _stage_pass2_complete(slab, "00000", "librarian", 42)
        # Cell 00001 fresh — needs both passes.

        args = _build_args(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(
            _dispatch,
            "_run_pass1_hf",
            lambda cells_to_run, *, args: {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run},
        )
        monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", lambda *a, **k: None)
        monkeypatch.setattr(
            _dispatch,
            "_run_pass2_vllm",
            lambda cells_to_run, *, args: {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run},
        )

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        resume_path = slab / "SWEEP_RESUME.json"
        assert resume_path.exists(), (
            f"Round 12: dispatcher must write SWEEP_RESUME.json on resume; missing {resume_path}"
        )
        payload = json.loads(resume_path.read_text())
        assert payload["kind"] == "epm:sweep-resume"
        assert payload["fully_complete"] == 1
        assert payload["pass1_queue"] == 1
        assert payload["pass2_queue"] == 1
        assert "skip_summary" in payload
