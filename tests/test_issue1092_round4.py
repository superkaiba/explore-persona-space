"""Issue #1092 regression pins for pca48 r_B, shard order, dynamics, and nulls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue1092_fit_grid as fit_grid  # noqa: E402
import issue1092_gpu_phase as gpu_phase  # noqa: E402


class SeamMergingTokenizer:
    """Tiny tokenizer stub with BPE-like role/content and content/suffix seams."""

    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False, **kwargs):
        assert add_special_tokens is False
        ids, offsets = self._tokenize(text)
        out = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = offsets
        return out

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids, _offsets = self._tokenize(text)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(self._pieces[int(i)] for i in ids)

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        pieces: list[str] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text.startswith(": ", i):
                j = i + 2
                while j < len(text) and not text[j].isspace():
                    j += 1
                pieces.append(text[i:j])
                offsets.append((i, j))
                i = j
                continue
            if text.startswith(".\n\n", i):
                pieces.append(text[i : i + 3])
                offsets.append((i, i + 3))
                i += 3
                continue
            if text[i].isspace():
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
            else:
                j = i + 1
                while j < len(text) and not text[j].isspace() and not text.startswith(": ", j):
                    if text.startswith(".\n\n", j):
                        break
                    j += 1
            pieces.append(text[i:j])
            offsets.append((i, j))
            i = j
        self._pieces = pieces
        return list(range(len(pieces))), offsets


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


def test_mixed_padded_unpadded_duplicate_shards_fail_loud(tmp_path):
    paths = [
        tmp_path / "prefix_end_L00_shard3.npy",
        tmp_path / "prefix_end_L00_shard00003.npy",
    ]
    for path in paths:
        path.touch()

    with pytest.raises(ValueError, match="duplicate shard index 3"):
        fit_grid._sorted_shards(paths)
    with pytest.raises(ValueError, match="duplicate shard index 3"):
        gpu_phase._sorted_shards(paths)


def test_pretrained_dynamics_cut_plan_uses_full_render_offsets_for_bpe_seams():
    tokenizer = SeamMergingTokenizer()
    turns = [
        {"role": "user", "content": "Explain photosynthesis briefly."},
        {"role": "assistant", "content": "Plants turn light into sugar."},
        {"role": "user", "content": "Name one input."},
        {"role": "assistant", "content": "Carbon dioxide."},
    ]
    full_render = gpu_phase._render_full_conversation(turns, "pretrained")
    encoded = tokenizer(full_render, add_special_tokens=False, return_offsets_mapping=True)

    cuts = gpu_phase._dynamics_cut_plan(
        turns,
        tokenizer,
        "pretrained",
        len(encoded["input_ids"]),
        full_token_ids=encoded["input_ids"],
    )

    first_user = cuts["u1"][0]
    second_user = cuts["u1"][1]
    assert first_user[2] == 0
    assert second_user[2] == 2
    first_text = tokenizer.decode(encoded["input_ids"][first_user[0] : first_user[1]])
    assert first_text.startswith(": Explain")
    assert first_text.endswith(".\n\n")
    assert "Explain photosynthesis briefly." in first_text
    assert "Name one input." in tokenizer.decode(
        encoded["input_ids"][second_user[0] : second_user[1]]
    )


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
    assert result["implementation"] == "sign_matrix_gemm"
    assert result["wall_s"] < 1.0

    manual_rng = np.random.default_rng(77)
    signs = manual_rng.choice(np.array([-1.0, 1.0]), size=(5, 6))
    arrays = [np.asarray(factors[name], dtype=np.float64) for name in ("f", "g", "i")]
    expected = np.empty_like(draws, dtype=np.float64)
    rb_flat = rb.reshape(28 * 2, 4)
    for factor_i, arr in enumerate(arrays):
        projected = (signs @ arr / arr.shape[0]) @ rb_flat.T
        expected[:, :, factor_i, :] = np.abs(projected.reshape(5, 28, 2))
    assert np.allclose(draws, expected.astype(np.float32))


def test_d5_iterated_d2_chaining_companion_is_emitted(tmp_path):
    summaries = tmp_path / "summaries"
    root = summaries / "dynamics_instruct"
    root.mkdir(parents=True)
    kinds = (
        "context_k",
        "s_k",
        "answer_k_t1",
        "answer_k_t2",
        "answer_k_t3",
        "u1",
        "u2",
        "u3",
    )
    rows = []
    arrays = []
    for conv_i in range(4):
        for turn in (0, 2, 4, 6):
            rows.append(
                {
                    "conv_id": f"c{conv_i}",
                    "turn_index": turn,
                    "token_start": turn,
                    "token_end": turn + 1,
                }
            )
            arrays.append([conv_i + 0.1 * turn, 1.0 + turn])
    base = np.asarray(arrays, dtype=np.float32)
    for kind_i, kind in enumerate(kinds):
        np.save(root / f"{kind}_L00.npy", base + kind_i * 0.01)
        with open(root / f"row_index_{kind}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    args = argparse.Namespace(n_folds=2)
    out = fit_grid._compute_dynamics_reads(
        summaries,
        "cell_inst_own",
        0,
        args,
        judge_rows=[],
    )

    companion = out["D5_iterated_D2_chaining_companion"]
    assert companion["status"] == "computed"
    assert companion["horizons"] == [0, 2, 4, 6]
    entry = companion["profile"]["context_k"]["4"]["context_k"]
    assert entry["status"] == "computed"
    assert isinstance(entry["direct_cv_r2"], float)
    assert isinstance(entry["iterated_d2_chain_r2"], float)
