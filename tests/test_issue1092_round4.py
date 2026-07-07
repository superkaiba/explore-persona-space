"""Issue #1092 round-4 regression pins for pca48 r_B, shard order, and layer nulls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue1092_fit_grid as fit_grid  # noqa: E402
import issue1092_gpu_phase as gpu_phase  # noqa: E402


def test_pca48_projects_rb_for_hidden_dim_gt_48(tmp_path):
    rng = np.random.default_rng(1092)
    hidden_dim = 64
    Y = rng.normal(size=(72, hidden_dim))
    Yb, basis_info = fit_grid._basis_targets_with_info(
        Y,
        "pca48",
        hidden_dim=hidden_dim,
        targets=["t1"],
        projection_target="t1",
    )
    rb = rng.normal(size=hidden_dim)

    rb_b = fit_grid._project_rb_to_basis(rb, basis_info, expected_dim=Yb.shape[1])
    assert Yb.shape == (72, 48)
    assert rb_b.shape == (48,)
    projection = Yb @ rb_b
    assert projection.shape == (72,)

    stacked = np.concatenate([Y, 2.0 * Y, -Y], axis=1)
    Ys, ambient_info = fit_grid._basis_targets_with_info(
        stacked,
        "ambient",
        hidden_dim=hidden_dim,
        targets=["t1", "t2", "t3"],
        projection_target="t1",
    )
    rb_s = fit_grid._project_rb_to_basis(rb, ambient_info, expected_dim=Ys.shape[1])
    assert rb_s.shape == (3 * hidden_dim,)
    assert np.count_nonzero(rb_s[hidden_dim:]) == 0
    assert (Ys @ rb_s).shape == (72,)


def test_numeric_12_shard_order_for_consolidation_and_fit_grid_loaders(tmp_path):
    cell = "cell_x"
    summary_dir = tmp_path / "summaries" / cell
    summary_dir.mkdir(parents=True)
    pool_dir = tmp_path / "summaries" / "b0_rB_pool"
    pool_dir.mkdir(parents=True)
    for shard in range(12):
        np.save(summary_dir / f"prefix_end_L00_shard{shard}.npy", np.array([[shard]]))
        np.save(pool_dir / f"{cell}_shard{shard}.npy", np.array([[[[shard]]]], dtype=np.float32))

    loaded, _paths = fit_grid._load_summary(tmp_path / "summaries", cell, "prefix_end", 0)
    assert loaded[:, 0].tolist() == list(range(12))
    b0_loaded = fit_grid._load_b0_pool(tmp_path / "summaries", cell)
    assert b0_loaded[:, 0, 0, 0].tolist() == list(range(12))

    root = tmp_path / "summaries" / "dynamics_instruct"
    root.mkdir()
    for shard in range(12):
        (root / f"row_index_u1_shard{shard}.jsonl").write_text(
            json.dumps({"conv_id": f"c{shard}", "turn_index": shard}) + "\n"
        )
    rows = fit_grid._read_index_files(root, "row_index_u1")
    assert [row["turn_index"] for row in rows] == list(range(12))

    gpu_phase.consolidate_cell_shards(tmp_path, cell, n_layers=1)
    consolidated = np.load(summary_dir / "prefix_end_L00.npy")
    assert consolidated[:, 0].tolist() == list(range(12))
    consolidated_b0 = np.load(pool_dir / f"{cell}.npy")
    assert consolidated_b0[:, 0, 0, 0].tolist() == list(range(12))


def test_layer_max_null_uses_shared_draw_seed_across_layers(tmp_path):
    rng = np.random.default_rng(123)
    factors = {
        "f": rng.normal(size=(6, 4)),
        "g": rng.normal(size=(6, 4)),
        "i": rng.normal(size=(6, 4)),
        "basis": "dense_core",
    }
    rb = rng.normal(size=(28, 2, 4))
    basis_info = {
        "basis": "ambient",
        "ambient_dim": 4,
        "hidden_dim": 4,
        "targets": ["t1"],
        "projection_target": "t1",
        "projection_block_index": 0,
        "v_basis": None,
    }
    result = fit_grid._selection_symmetric_projection_null(
        unit_key="unit",
        factors=factors,
        rb_directions=rb,
        trait_names=["evil", "syc"],
        layer=3,
        basis_info=basis_info,
        n_draws=5,
        seed=77,
        out_dir=tmp_path,
    )
    draws = np.load(result["persist_path"])
    assert draws.shape == (5, 28, 3, 2)
    assert result["persist_shape"] == [5, 28, 3, 2]

    manual_rng = np.random.default_rng(77)
    expected = np.empty_like(draws, dtype=np.float64)
    arrays = [np.asarray(factors[name], dtype=np.float64) for name in ("f", "g", "i")]
    for draw in range(5):
        signs = manual_rng.choice(np.array([-1.0, 1.0]), size=6)
        for factor_i, arr in enumerate(arrays):
            signed = arr * signs[:, None]
            expected[draw, :, factor_i, :] = np.abs(
                np.einsum("nd,ltd->lt", signed, rb, optimize=True) / signed.shape[0]
            )
    assert np.allclose(draws, expected.astype(np.float32))
