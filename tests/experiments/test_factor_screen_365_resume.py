"""Tests for the round-5 ``--resume`` short-circuit (task #365).

After the silent dispatcher death at hour 25 of the round-4 run (10 of
32 cells trained), the relaunched dispatcher must skip cells whose
artifacts already exist on disk so it can pick up where round 4 left off.

Two layers are exercised:

  1. ``cell_complete_on_disk`` in ``scripts.dispatch_factor_screen_365``
     — the dispatcher's pre-subprocess gate. Tests synthesise fake
     ``cell_<key>/source_<src>/seed_<N>/`` trees with various combinations
     of present / missing / empty metrics.json + adapter/ artifacts and
     assert the predicate matches the documented contract.
  2. ``_cell_complete_on_disk`` in
     ``explore_persona_space.experiments.factor_screen_365.__main__``
     — the in-process defense-in-depth gate. Asserted against the same
     synthesised directories so the two predicates can't drift.

Dispatcher resume probe of the HF Hub is exercised by injecting a
``hub_files_cache`` so the test doesn't hit the real Hub.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from explore_persona_space.experiments.factor_screen_365.__main__ import (
    _cell_complete_on_disk as cellmode_complete_on_disk,
)


def _load_dispatch_module():
    """Load ``scripts/dispatch_factor_screen_365`` as a module without requiring sys.path tweaks.

    The script is hyphen-friendly under ``scripts/``, not on the package
    path, so we import it via importlib. Keeping this in-test rather than
    in conftest keeps the suite self-contained.
    """
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "dispatch_factor_screen_365.py"
    spec = importlib.util.spec_from_file_location("dispatch_factor_screen_365", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_complete_cell(slab_root: Path, cell_key: str, source: str, seed: int) -> Path:
    """Synthesise a ``slab_root/cell_X/source_Y/seed_Z/`` tree that looks complete."""
    out = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text('{"cell_key": "00000", "failed": false}')
    adapter = out / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 32}')
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 128)
    return out


# ---- cell_complete_on_disk (dispatcher) -------------------------------------


def test_cell_complete_on_disk_true_when_metrics_and_adapter_present(tmp_path: Path) -> None:
    """Happy path: a fully written cell directory is reported complete."""
    mod = _load_dispatch_module()
    _make_complete_cell(tmp_path, "00000", "librarian", 42)
    assert mod.cell_complete_on_disk(tmp_path, "00000", "librarian", 42)


def test_cell_complete_on_disk_false_when_metrics_missing(tmp_path: Path) -> None:
    """metrics.json missing -> not complete (training crashed before eval)."""
    mod = _load_dispatch_module()
    out = tmp_path / "cell_00000" / "source_librarian" / "seed_42"
    out.mkdir(parents=True)
    adapter = out / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 32)
    # No metrics.json.
    assert not mod.cell_complete_on_disk(tmp_path, "00000", "librarian", 42)


def test_cell_complete_on_disk_false_when_adapter_missing(tmp_path: Path) -> None:
    """adapter/ missing -> not complete (an artifact of a partial run)."""
    mod = _load_dispatch_module()
    out = tmp_path / "cell_00000" / "source_librarian" / "seed_42"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    # No adapter directory.
    assert not mod.cell_complete_on_disk(tmp_path, "00000", "librarian", 42)


def test_cell_complete_on_disk_false_when_metrics_empty(tmp_path: Path) -> None:
    """An empty metrics.json is treated as missing (no successful training)."""
    mod = _load_dispatch_module()
    out = tmp_path / "cell_00000" / "source_librarian" / "seed_42"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("")
    (out / "adapter").mkdir()
    (out / "adapter" / "adapter_model.safetensors").write_bytes(b"\x00" * 32)
    assert not mod.cell_complete_on_disk(tmp_path, "00000", "librarian", 42)


def test_cell_complete_on_disk_false_when_adapter_empty(tmp_path: Path) -> None:
    """An empty adapter/ dir is treated as incomplete."""
    mod = _load_dispatch_module()
    out = tmp_path / "cell_00000" / "source_librarian" / "seed_42"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    (out / "adapter").mkdir()
    # Empty adapter dir.
    assert not mod.cell_complete_on_disk(tmp_path, "00000", "librarian", 42)


# ---- _cell_complete_on_disk (cell-mode defense-in-depth) -------------------


def test_cellmode_predicate_matches_dispatcher(tmp_path: Path) -> None:
    """The cell-mode and dispatcher predicates must agree on the same input tree.

    Regression: if the two ever drift, ``--resume`` could double-launch a
    completed cell (or skip an incomplete one).
    """
    mod = _load_dispatch_module()
    out = _make_complete_cell(tmp_path, "01010", "surgeon", 137)
    assert mod.cell_complete_on_disk(tmp_path, "01010", "surgeon", 137)
    assert cellmode_complete_on_disk(out)


def test_cellmode_predicate_false_on_partial_dir(tmp_path: Path) -> None:
    """Partial dir (metrics-only) -> cell-mode reports incomplete."""
    out = tmp_path / "cell_01010" / "source_surgeon" / "seed_137"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    # No adapter.
    assert not cellmode_complete_on_disk(out)


# ---- HF Hub resume probe ----------------------------------------------------


def test_cell_complete_on_hub_uses_cached_index(tmp_path: Path) -> None:
    """When an adapter index is supplied, the probe doesn't hit the Hub.

    Asserts the run-name -> Hub-prefix mapping matches the training-script
    contract (``adapters/issue_365/i365_cell_<key>_source_<src>_seed<N>/``).
    """
    mod = _load_dispatch_module()
    expected_run_name = mod.hf_hub_adapter_run_name("01010", "surgeon", 137)
    assert expected_run_name == "i365_cell_01010_source_surgeon_seed137"

    hub_index = [
        f"adapters/issue_365/{expected_run_name}/adapter_config.json",
        f"adapters/issue_365/{expected_run_name}/adapter_model.safetensors",
        "adapters/issue_365/i365_cell_11111_source_librarian_seed42/adapter_config.json",
    ]
    assert mod.cell_complete_on_hub("01010", "surgeon", 137, hub_files_cache=hub_index)
    # A cell that ISN'T in the index returns False.
    assert not mod.cell_complete_on_hub("00000", "programmer", 42, hub_files_cache=hub_index)


def test_cell_complete_on_hub_false_on_empty_index() -> None:
    """An empty Hub index (no matching adapter prefix) returns False."""
    mod = _load_dispatch_module()
    assert not mod.cell_complete_on_hub("00000", "librarian", 42, hub_files_cache=[])


# ---- Job-queue integration: resume actually skips ---------------------------


def test_resume_queue_skips_complete_and_queues_incomplete(tmp_path: Path) -> None:
    """End-to-end: on a slab tree where 1 cell is complete and 1 is not,
    iterating the dispatcher's resume probe correctly returns
    ``[skipped, queued]``.

    Mirrors the dispatcher's training-stage loop without actually spawning
    subprocesses. This is the behaviour the relaunch on pod-365 depends on
    (10 round-4 cells skipped, remaining 22 queued).
    """
    mod = _load_dispatch_module()
    # Two synthetic cells: 00000 = complete, 00001 = not started.
    _make_complete_cell(tmp_path, "00000", "librarian", 42)
    # 00001: no directory at all.

    jobs = [("00000", "librarian", 42), ("00001", "librarian", 42)]
    skipped, queued = [], []
    for cell_key, source, seed in jobs:
        if mod.cell_complete_on_disk(tmp_path, cell_key, source, seed):
            skipped.append((cell_key, source, seed))
        else:
            queued.append((cell_key, source, seed))
    assert skipped == [("00000", "librarian", 42)]
    assert queued == [("00001", "librarian", 42)]
