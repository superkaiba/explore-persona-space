"""Unit tests for the #570 round-6 fix (alignment-grid auto-discovery).

2026-06-11: the run completed everything up to the final alignment phase,
then ``eval_issue570_alignment.build_default_grid()`` crashed — it hardcoded
the UNLABELED cell layout (``eval_results/issue_570/org_benign/seed42/...``)
while the r4 rescue-label threading realized the LABELED layout
(``org_benign_rescue_lr2e6/seed{42,137,256}`` + ``phase1_rescue_lr2e6``
picks). Covered here, against ``discover_default_grid(root)``:

- LABELED layout (the realized ``_rescue_lr2e6`` pod tree) discovered
  correctly, including the variant-matched seed-42 picked-install spot-check.
- UNLABELED layout (plain ``org_benign/``, the 5e-6 install path) discovered
  correctly with the legacy slugs.
- Wrong cell-count fails loud, LISTING what was found.
- Mixed install variants fail loud.
- Missing / pick-less ``phase1_pick_record.json`` raises.

All CPU-only and fast (no model loads, no network, no GPU pins — module
import side effects are env/logging only).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


align = _load("i570_alignment_under_test", "eval_issue570_alignment.py")

SEEDS = (42, 137, 256)


def _write_phase2_cell(root: Path, cell_name: str, seed: int) -> str:
    """Write ``<root>/<cell_name>/seed<S>/phase2_result.json``; return adapter path."""
    cell = root / cell_name / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    adapter = f"/workspace/outputs/{cell_name}_seed{seed}/adapter"
    (cell / "phase2_result.json").write_text(json.dumps({"final_adapter_path": adapter}))
    return adapter


def _write_pick_record(root: Path, p1_leaf: str, seed: int, record: dict) -> Path:
    cell = root / p1_leaf / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    path = cell / "phase1_pick_record.json"
    path.write_text(json.dumps(record))
    return path


def _labeled_tree(root: Path) -> dict[str, str]:
    """The pod's exact realized layout (6 labeled cells + labeled picks)."""
    adapters = {}
    for arm in ("org_benign", "org_em"):
        for seed in SEEDS:
            adapters[f"{arm}_rescue_lr2e6_seed{seed}"] = _write_phase2_cell(
                root, f"{arm}_rescue_lr2e6", seed
            )
    _write_pick_record(
        root,
        "phase1_rescue_lr2e6",
        42,
        {"pick_step": 96, "picked_local_dir": "/workspace/ckpts/rescue_seed42/checkpoint-96"},
    )
    return adapters


# ── Happy paths: both realized layouts discovered correctly ──────────────────


def test_labeled_rescue_layout_discovered(tmp_path):
    adapters = _labeled_tree(tmp_path)
    grid = align.discover_default_grid(tmp_path)
    assert [m["slug"] for m in grid] == [
        "org_benign_rescue_lr2e6_seed42",
        "org_benign_rescue_lr2e6_seed137",
        "org_benign_rescue_lr2e6_seed256",
        "org_em_rescue_lr2e6_seed42",
        "org_em_rescue_lr2e6_seed137",
        "org_em_rescue_lr2e6_seed256",
        "picked_install_rescue_lr2e6_seed42",
    ]
    for m in grid[:6]:
        assert m["kind"] == "post" and m["adapter_path"] == adapters[m["slug"]]
    spot = grid[6]
    assert spot["kind"] == "pre_spot_check"
    assert spot["adapter_path"] == "/workspace/ckpts/rescue_seed42/checkpoint-96"


def test_unlabeled_layout_discovered(tmp_path):
    for arm in ("org_benign", "org_em"):
        for seed in SEEDS:
            _write_phase2_cell(tmp_path, arm, seed)
    _write_pick_record(
        tmp_path, "phase1", 42, {"pick_step": 80, "picked_local_dir": "/tmp/ckpt-80"}
    )
    grid = align.discover_default_grid(tmp_path)
    assert [m["slug"] for m in grid] == [
        "org_benign_seed42",
        "org_benign_seed137",
        "org_benign_seed256",
        "org_em_seed42",
        "org_em_seed137",
        "org_em_seed256",
        "picked_install_seed42",
    ]
    assert grid[6]["adapter_path"] == "/tmp/ckpt-80"


# ── Fail-loud paths ──────────────────────────────────────────────────────────


def test_wrong_cell_count_fails_loud_listing_found(tmp_path):
    _labeled_tree(tmp_path)
    # Remove one cell -> 5 of 6; the error must LIST what was found.
    (tmp_path / "org_em_rescue_lr2e6" / "seed256" / "phase2_result.json").unlink()
    with pytest.raises(RuntimeError, match="org_benign/seed42"):
        align.discover_default_grid(tmp_path)


def test_empty_tree_fails_loud(tmp_path):
    with pytest.raises(RuntimeError, match=r"\(none\)"):
        align.discover_default_grid(tmp_path)


def test_mixed_install_variants_fail_loud(tmp_path):
    _labeled_tree(tmp_path)
    # A stray UNLABELED cell from an earlier 5e-6 attempt alongside the
    # labeled grid -> ambiguous install variant; must not silently pick one.
    _write_phase2_cell(tmp_path, "org_benign", 42)
    with pytest.raises(RuntimeError, match="ONE consistent install variant"):
        align.discover_default_grid(tmp_path)


def test_missing_pick_record_raises(tmp_path):
    _labeled_tree(tmp_path)
    rec = tmp_path / "phase1_rescue_lr2e6" / "seed42" / "phase1_pick_record.json"
    rec.unlink()
    with pytest.raises(FileNotFoundError, match="phase1_rescue_lr2e6"):
        align.discover_default_grid(tmp_path)


def test_pickless_record_raises(tmp_path):
    _labeled_tree(tmp_path)
    _write_pick_record(tmp_path, "phase1_rescue_lr2e6", 42, {"pick_step": None})
    with pytest.raises(RuntimeError, match="no picked_local_dir"):
        align.discover_default_grid(tmp_path)
