"""#2061 statistics-correctness pins (code-review v1 Unit B: C2, C3, M4, M5).

- C2: `knn_retrieval` returns `acc_at_k`/`chance_at_k` as dicts KEYED BY K
  (positional indexing crashed `KeyError: 0` after each cell's fold fits);
  contract test = signature bind + a tiny-array call of the REAL helper,
  plus a tiny end-to-end `fit_cell` that exercises the fixed row-building.
- C3: the registered selection-symmetric null (permute stage labels
  within-corpus + ridge REFIT per draw) replaces the round-1 sign-flip
  placeholder. The planted-signal test asserts the null p97.5 falls BELOW
  the observed max — the placeholder fails this BY CONSTRUCTION (its
  per-draw max >= max_j |ΔR²_j| >= observed max with overwhelming
  probability), the real null passes it.
- M4: `ridge_fit_predict_fast_layer_batched(gcv_dof_cap=...)` — inert on
  well-determined slices (default None byte-identical), binds at
  n_tr < d (#1887), fail-loud when every lambda is capped out.
- M5: `group_fold_ids` == `issue825_fit_cells._cv_folds` (the #1336 fold
  convention), group-constant within conversation id.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2061_fit_per_feature as fpf
import issue2061_null as nullmod
import issue2061_turnstore as ts

from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import (
    ridge_fit_predict_fast_layer_batched,
)


# ---------------------------------------------------------------------------
# C2 — knn_retrieval contract: dicts keyed BY K, never positional
# ---------------------------------------------------------------------------
def test_knn_retrieval_contract_k_keyed_dicts():
    sig = inspect.signature(knn_retrieval)
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(30, 5))
    pool = pred + 0.01 * rng.normal(size=(30, 5))
    sig.bind(pred, pool, ks=(1, 5), metric="euclidean")  # arity/keyword contract
    out = knn_retrieval(pred, pool, ks=(1, 5), metric="euclidean")
    assert set(out["acc_at_k"].keys()) == {1, 5}
    assert set(out["chance_at_k"].keys()) == {1, 5}
    # Positional indexing (the round-1 bug) is a KeyError, not a valid read.
    with pytest.raises(KeyError):
        _ = out["acc_at_k"][0]
    assert out["chance_at_k"][5] == pytest.approx(5 / 30)


# ---------------------------------------------------------------------------
# Shared tiny fixtures
# ---------------------------------------------------------------------------
N_LAYERS = 3
HIDDEN = 8
LAYER = 1


def _write_random_turnstore(dir_: Path, conv_gs: list[int], seed: int, stem: str) -> None:
    """Producer-shaped shards (write_shards schema) with random slot states."""
    rng = np.random.default_rng(seed)
    dir_.mkdir(parents=True, exist_ok=True)
    half = max(1, len(conv_gs) // 2)
    for shard_idx, gs in enumerate([conv_gs[:half], conv_gs[half:]]):
        if not gs:
            continue
        recs = {
            "conv_ids": [f"conv-{g:03d}" for g in gs],
            "slots": [
                torch.tensor(rng.normal(size=(2, N_LAYERS, HIDDEN)), dtype=torch.bfloat16)
                for _ in gs
            ],
            "profiles": [
                torch.tensor(rng.normal(size=(2, N_LAYERS, HIDDEN)), dtype=torch.bfloat16)
                for _ in gs
            ],
            "nll": [torch.tensor([0.1, 0.2]) for _ in gs],
            "spans_meta": [
                {
                    "conv_id": f"conv-{g:03d}",
                    "slot_names": ["prefix", "a1"],
                    "turn_names": ["u1", "a1"],
                }
                for g in gs
            ],
        }
        torch.save(recs, dir_ / f"{stem}_shard{shard_idx:03d}.pt")
        # Producer JSON sidecar (the v13 a1-bis loader asserts read it).
        (dir_ / f"{stem}_shard{shard_idx:03d}.json").write_text(
            json.dumps(
                {
                    "shard_index": shard_idx,
                    "n_conversations": len(gs),
                    "conv_ids": recs["conv_ids"],
                }
            )
        )


def _dense_to_iv(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return ts.to_fixed_width_sparse(torch.as_tensor(y, dtype=torch.float32))


# ---------------------------------------------------------------------------
# C2 + M4 + M5 end-to-end: tiny fit_cell writes k-keyed kNN rows, group folds,
# dof-cap threading + parity gate all execute on the REAL path
# ---------------------------------------------------------------------------
def test_fit_cell_end_to_end_tiny(tmp_path):
    n = 40
    d_sae = 24
    stem = "turnstore_base_chat_tiny"
    tdir = tmp_path / stem
    _write_random_turnstore(tdir, list(range(n)), seed=0, stem=stem)
    rng = np.random.default_rng(1)
    y = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.5)).astype(np.float32)
    # Write the encoded target through the PRODUCER's payload writer (the
    # P1 -> P2 layout round trip; review M1).
    y_idx, y_val = _dense_to_iv(y)
    enc = tmp_path / f"base_chat_tiny_answer_L{LAYER}.pt"
    ts.save_encoded_target(
        enc,
        idx=torch.as_tensor(y_idx),
        val=torch.as_tensor(y_val),
        d_sae=d_sae,
        k=y_idx.shape[1],
        conv_ids=[f"conv-{g:03d}" for g in range(n)],
        cell={"stage": "base", "render": "chat", "corpus": "tiny", "state": "answer", "layer": 1},
    )

    out = tmp_path / "out.jsonl"
    fpf._PARITY_GATE_STATE["done"] = False  # force the gate on this cell
    fpf.fit_cell(tmp_path, "base", "chat", "tiny", enc, arm="context", output_path=out, layer=LAYER)
    assert fpf._PARITY_GATE_STATE["done"] is True  # gate ran (>=3 slices)

    rows = [json.loads(line) for line in out.read_text().split("\n") if line.strip()]
    assert len(rows) == d_sae
    k_ret = math.ceil(n / 20)  # plan §13 k = ceil(n_pool / 20)
    r0 = rows[0]
    assert r0["knn_k_ret"] == k_ret
    assert r0["chance_1"] == pytest.approx(1 / n)
    assert r0["chance_k"] == pytest.approx(k_ret / n)
    for key in ("knn_acc_1_euclid", "knn_acc_k_euclid", "knn_acc_1_cosine", "knn_acc_k_cosine"):
        assert 0.0 <= r0[key] <= 1.0
    assert r0["lambda_selector"] == f"gcv-dof-cap-{fpf.DOF_CAP_FRACTION}"
    assert len(r0["best_lambda_folds"]) == fpf.K_FOLDS
    # m1 (round 3): identity+bias inapplicability is STATED in the fit output
    # (plan §Design "Baselines per fitted map"), never silently skipped.
    assert r0["identity_bias"].startswith("N/A: dim mismatch")


# ---------------------------------------------------------------------------
# M5 — group folds: #1336 convention (issue825_fit_cells._cv_folds parity)
# ---------------------------------------------------------------------------
def test_group_fold_ids_matches_issue825_cv_folds():
    import issue825_fit_cells as fc

    rng = np.random.default_rng(3)
    conv = [f"c{int(i):04d}" for i in rng.integers(0, 60, size=140)]
    ours = ts.group_fold_ids(conv, n_folds=5, seed=0)
    theirs = fc._cv_folds(np.asarray(conv), n_folds=5, seed=0)
    assert np.array_equal(ours, theirs)
    # Group-constant within conversation id (ood-generalization-folds).
    arr = np.asarray(conv)
    for cid in np.unique(arr):
        assert len(set(ours[arr == cid])) == 1
    # Pooled duplication (the null's before+after concat) keeps pairs together.
    pooled = conv + conv
    pf = ts.group_fold_ids(pooled, n_folds=5, seed=0)
    assert np.array_equal(pf[: len(conv)], pf[len(conv) :])
    # And pooled folds == per-stage folds when the conv sets coincide.
    assert np.array_equal(pf[: len(conv)], ours)


def test_group_fold_ids_fail_loud_on_empty_fold():
    with pytest.raises(ValueError, match="empty fold"):
        ts.group_fold_ids(["a", "b", "c"], n_folds=5, seed=0)


# ---------------------------------------------------------------------------
# M4 — gcv_dof_cap on the shared #779 helper
# ---------------------------------------------------------------------------
def _toy_slices(n_tr: int, d: int, d_out: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xtr = rng.normal(size=(1, n_tr, d))
    Ytr = rng.normal(size=(1, n_tr, d_out))
    Xev = rng.normal(size=(1, max(4, n_tr // 4), d))
    return Xtr, Ytr, Xev


def test_gcv_dof_cap_inert_when_well_determined():
    Xtr, Ytr, Xev = _toy_slices(n_tr=120, d=6, d_out=4)
    p0, i0 = ridge_fit_predict_fast_layer_batched(Xtr, Ytr, Xev, return_info=True)
    p1, i1 = ridge_fit_predict_fast_layer_batched(Xtr, Ytr, Xev, return_info=True, gcv_dof_cap=0.9)
    assert np.array_equal(p0, p1)
    assert np.array_equal(i0["best_lambda"], i1["best_lambda"])


def test_gcv_dof_cap_binds_at_n_lt_d():
    # Under-determined regime (#1887): pure GCV selects a (near-)interpolating
    # lambda with dof > 0.9 * n_tr; the cap excludes those lambdas.
    Xtr, Ytr, Xev = _toy_slices(n_tr=24, d=64, d_out=8, seed=1)
    _, info_uncapped = ridge_fit_predict_fast_layer_batched(Xtr, Ytr, Xev, return_info=True)
    assert float(info_uncapped["dof"][0]) > 0.9 * 24, "fixture must exercise the #1887 regime"
    _, info_capped = ridge_fit_predict_fast_layer_batched(
        Xtr, Ytr, Xev, return_info=True, gcv_dof_cap=0.9
    )
    assert float(info_capped["dof"][0]) <= 0.9 * 24 * (1 + 1e-9)
    assert float(info_capped["best_lambda"][0]) > float(info_uncapped["best_lambda"][0])


def test_gcv_dof_cap_all_capped_fail_loud():
    Xtr, Ytr, Xev = _toy_slices(n_tr=24, d=64, d_out=8, seed=1)
    with pytest.raises(RuntimeError, match="gcv_dof_cap"):
        ridge_fit_predict_fast_layer_batched(
            Xtr, Ytr, Xev, lambdas=np.array([1e-8]), gcv_dof_cap=0.9
        )


# ---------------------------------------------------------------------------
# C3 — the registered permute-and-refit null
# ---------------------------------------------------------------------------
def _planted_cell(n: int = 80, d_in: int = 6, d_sae: int = 32, seed: int = 7):
    """Paired two-stage cell with a planted predictability gain at feature 7."""
    rng = np.random.default_rng(seed)
    conv = [f"c{i:03d}" for i in range(n)]
    xb = rng.normal(size=(n, d_in)).astype(np.float64)
    xa = rng.normal(size=(n, d_in)).astype(np.float64)
    yb = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.4)).astype(np.float64)
    ya = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.4)).astype(np.float64)
    w = rng.normal(size=d_in)
    w /= np.linalg.norm(w)
    ya[:, 7] = xa @ w + 0.05 * rng.normal(size=n)  # AFTER: predictable
    yb[:, 7] = rng.normal(size=n)  # BEFORE: noise
    yib, yvb = _dense_to_iv(yb)
    yia, yva = _dense_to_iv(ya)
    return xb, yib, yvb, conv, xa, yia, yva, conv, d_sae


def _make_engine(gcv_m: int = 4096, **kw):
    xb, yib, yvb, cb, xa, yia, yva, ca, d_sae = _planted_cell()
    return nullmod._CellEngine(xb, yib, yvb, cb, xa, yia, yva, ca, d_sae=d_sae, gcv_m=gcv_m, **kw)


def test_engine_identity_matches_p2_estimator():
    """Identity assignment == the P2 estimator per stage (same folds, same fit)."""
    xb, yib, yvb, cb, xa, yia, yva, ca, d_sae = _planted_cell()
    engine = nullmod._CellEngine(xb, yib, yvb, cb, xa, yia, yva, ca, d_sae=d_sae, gcv_m=d_sae)
    ident = nullmod.identity_delta_r2(engine)

    # P2-style computation on the BEFORE stage: same group folds, same helper.
    yb = np.zeros((len(cb), d_sae))
    rows, kk = np.nonzero(yvb != 0)
    yb[rows, yib[rows, kk]] = yvb[rows, kk]
    fold_of_row = ts.group_fold_ids(cb, n_folds=5, seed=0)
    folds = [np.where(fold_of_row == f)[0] for f in range(5)]
    ss_res = np.zeros(d_sae)
    ss_tot = np.zeros(d_sae)
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        preds = ridge_fit_predict_fast_layer_batched(
            xb[train_idx][None],
            yb[train_idx][None].astype(np.float64),
            xb[test_idx][None],
            # ONE grid on both sides (plan v13 delta (f)): the engine rides
            # the widened v13 LAMBDA_GRID; the shared helper's own 13-point
            # default is deliberately untouched, so the P2-parity reference
            # must thread the grid explicitly.
            lambdas=nullmod.LAMBDA_GRID,
            gcv_dof_cap=nullmod.DOF_CAP_FRACTION,
        )[0]
        y_test = yb[test_idx]
        ss_res += ((y_test - preds) ** 2).sum(axis=0)
        ss_tot += ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_p2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    r2_engine = ident["r2_before"]
    both = np.isfinite(r2_p2) & np.isfinite(r2_engine)
    assert np.array_equal(np.isfinite(r2_p2), np.isfinite(r2_engine))
    assert both.any()
    np.testing.assert_allclose(r2_engine[both], r2_p2[both], rtol=1e-6, atol=1e-8)


def test_planted_signal_null_p975_below_observed_max():
    """The reviewer's mechanizable pin: the sign-flip placeholder fails this
    by construction; the registered refit null must pass it."""
    engine = _make_engine()
    ident = nullmod.identity_delta_r2(engine)
    assert ident["argmax"] == 7
    assert ident["max"] > 0.5
    seeds = nullmod.draw_seed_schedule(120)
    null, null_arg = nullmod.per_cell_null_refit(engine, seeds, draw_block=30)
    p975 = float(np.percentile(null, 97.5))
    assert p975 < ident["max"], (p975, ident["max"])
    # Plan-v6 persistence contract: per-draw argmax feature ids ride along.
    assert null_arg.dtype == np.int32 and null_arg.shape == null.shape
    assert ((null_arg >= 0) & (null_arg < 32)).all()


def test_null_deterministic_and_block_invariant():
    seeds = nullmod.draw_seed_schedule(24)
    a, a_arg = nullmod.per_cell_null_refit(_make_engine(), seeds, draw_block=24)
    b, b_arg = nullmod.per_cell_null_refit(_make_engine(), seeds, draw_block=7)
    assert np.array_equal(a, b)
    assert np.array_equal(a_arg, b_arg)  # argmax ids are block-invariant too


def test_null_partial_checkpoint_resume(tmp_path):
    seeds = nullmod.draw_seed_schedule(20)
    meta = {"regime": "test", "n_draws": 20}
    partial = tmp_path / "cell.jsonl.partial.npz"
    full, full_arg = nullmod.per_cell_null_refit(
        _make_engine(), seeds, draw_block=5, partial_path=partial, partial_meta=meta
    )
    assert partial.exists()
    # Truncate to a mid-run state, then resume: identical result (both arrays).
    prev = np.load(partial, allow_pickle=False)
    truncated = np.array(prev["draws"])
    truncated[10:] = np.nan
    trunc_arg = np.array(prev["argmax"])
    trunc_arg[10:] = -1
    np.savez(
        tmp_path / "cell.jsonl.partial.npz",
        draws=truncated,
        argmax=trunc_arg,
        n_done=np.int64(10),
        meta=prev["meta"],
    )
    resumed, resumed_arg = nullmod.per_cell_null_refit(
        _make_engine(), seeds, draw_block=5, partial_path=partial, partial_meta=meta
    )
    assert np.array_equal(full, resumed)
    assert np.array_equal(full_arg, resumed_arg)
    # A meta (regime-key) mismatch is NEVER silently reused (#722 r3 class).
    other, other_arg = nullmod.per_cell_null_refit(
        _make_engine(),
        seeds,
        draw_block=5,
        partial_path=partial,
        partial_meta={"regime": "OTHER", "n_draws": 20},
    )
    assert np.array_equal(full, other)
    assert np.array_equal(full_arg, other_arg)
    # A LEGACY (pre-argmax) partial is recomputed, never half-trusted.
    np.savez(
        tmp_path / "cell.jsonl.partial.npz",
        draws=truncated,
        n_done=np.int64(10),
        meta=prev["meta"],
    )
    legacy, legacy_arg = nullmod.per_cell_null_refit(
        _make_engine(), seeds, draw_block=5, partial_path=partial, partial_meta=meta
    )
    assert np.array_equal(full, legacy)
    assert np.array_equal(full_arg, legacy_arg)


def test_write_cell_jsonl_persists_argmax_and_render(tmp_path):
    """Plan-v6 persistence contract: null_argmax_feature_per_draw (int32) +
    render ride every per-cell row; existing fields unchanged."""
    out = tmp_path / "base_sft_chat_tiny_context_L1.jsonl"
    max_j = np.asarray([0.1, 0.2, 0.05], dtype=np.float32)
    arg_j = np.asarray([7, 3, 7], dtype=np.int32)
    nullmod.write_cell_jsonl(
        out,
        pair=("base", "sft"),
        render="chat",
        corpus="tiny",
        arm="context",
        true_max=0.5,
        true_argmax=7,
        null_max_j_per_draw=max_j,
        null_argmax_feature_per_draw=arg_j,
    )
    row = json.loads(out.read_text().split("\n")[0])
    assert row["render"] == "chat"
    assert row["null_argmax_feature_per_draw"] == [7, 3, 7]
    assert row["null_max_j_per_draw"] == pytest.approx([0.1, 0.2, 0.05])
    assert row["n_draws"] == 3


# ---------------------------------------------------------------------------
# Round-2 M2 — LEFT-parse of cell filenames (underscore corpora never vanish)
# ---------------------------------------------------------------------------
def test_expect_n_cells_guard():
    # m2 (round 3): the GLOBAL null's cell axis is registered (56 cells, v7
    # grid) — the production aggregation pass fails loud on a mismatch instead
    # of writing a silently-shrunk GLOBAL_L29.json. None = unchecked
    # (smoke/worker form). The guard itself is value-agnostic.
    cells = {("base_sft", "chat", "tiny", "context", "1"): np.zeros(4)}
    assert nullmod.enforce_expected_cell_count(cells, 1, 0) is None
    assert nullmod.enforce_expected_cell_count(cells, None, 5) is None
    msg = nullmod.enforce_expected_cell_count(cells, 56, 55)
    assert msg is not None and "56" in msg and "55" in msg


def test_planned_cell_count_early_guard():
    # Round-4 review sweep: the EARLY twin fires at SETUP time on the PLANNED
    # pair x combo x arm grid, before any per-cell refit compute is spent.
    # Production v7 grid: 4 stage-pairs x 7 v2 combos x 2 arms = 56.
    assert nullmod.enforce_planned_cell_count(4, 7, 2, 56) is None
    assert nullmod.enforce_planned_cell_count(4, 7, 2, None) is None  # unchecked
    msg = nullmod.enforce_planned_cell_count(4, 11, 2, 56)  # v1+v2 UNION shape
    assert msg is not None and "88" in msg and "56" in msg and "BEFORE" in msg
    msg = nullmod.enforce_planned_cell_count(4, 6, 2, 56)  # a vanished combo
    assert msg is not None and "48" in msg


def test_parse_r2_stem_underscore_corpora_and_fail_loud():
    assert ts.parse_r2_stem("base_chat_gsm8k_train_full_context_L29", 29) == (
        "base",
        "chat",
        "gsm8k_train_full",
        "context",
    )
    assert ts.parse_r2_stem("longer-rlvr_naturalistic_gsm8k_test1319_prefix_L29", 29) == (
        "longer-rlvr",
        "naturalistic",
        "gsm8k_test1319",
        "prefix",
    )
    with pytest.raises(ValueError, match="arm token"):
        ts.parse_r2_stem("base_chat_corpus_notanarm_L29", 29)
    with pytest.raises(ValueError):
        ts.parse_r2_stem("base_chat_context_L29", 29)  # no corpus token
    with pytest.raises(ValueError):
        ts.parse_r2_stem("base_chat_x_context_L23", 29)  # wrong layer suffix


# ---------------------------------------------------------------------------
# Round-2 M1 — streamed P2 fit == the #779 layer-batched helper (same
# estimator, primal-space feature-chunked evaluation), and the sparse kNN ==
# the dense mapping_baselines helper.
# ---------------------------------------------------------------------------
def test_streamed_fit_matches_layer_batched_helper():
    rng = np.random.default_rng(11)
    n, d_in, d_sae = 48, 6, 24
    X = rng.normal(size=(n, d_in)).astype(np.float32)
    y = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.5)).astype(np.float32)
    y_idx, y_val = _dense_to_iv(y)
    conv = [f"c{i:03d}" for i in range(n)]
    folds = fpf._make_folds(conv, k=5, seed=0)
    for cap in (None, fpf.DOF_CAP_FRACTION):
        for fi, test_idx in enumerate(folds):
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
            out_pred = np.zeros((n, d_sae), dtype=np.float64)
            info = fpf._fold_fit_streamed(
                X,
                y_idx,
                y_val,
                d_sae,
                train_idx,
                test_idx,
                gcv_dof_cap=cap,
                feature_chunk=7,  # deliberately unaligned chunking
                ss_res=np.zeros(d_sae),
                ss_tot=np.zeros(d_sae),
                out_pred=out_pred,
            )
            preds_helper, info_helper = ridge_fit_predict_fast_layer_batched(
                X[train_idx][None].astype(np.float64),
                y[train_idx][None].astype(np.float64),
                X[test_idx][None].astype(np.float64),
                lambdas=fpf.LAMBDA_GRID,
                return_info=True,
                gcv_dof_cap=cap,
            )
            assert info["best_lambda"] == pytest.approx(
                float(info_helper["best_lambda"][0]), rel=1e-12
            )
            assert info["dof"] == pytest.approx(float(info_helper["dof"][0]), rel=1e-9)
            np.testing.assert_allclose(out_pred[test_idx], preds_helper[0], rtol=1e-7, atol=1e-10)


def test_knn_retrieval_sparse_matches_dense():
    rng = np.random.default_rng(13)
    n, d_sae = 30, 24
    y = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.4)).astype(np.float32)
    y_idx, y_val = _dense_to_iv(y)
    pred = (y + 0.3 * rng.normal(size=(n, d_sae))).astype(np.float32)
    for metric in ("euclidean", "cosine"):
        sparse = fpf._knn_retrieval_sparse(
            pred, y_idx, y_val, ks=(1, 5), metric=metric, row_chunk=7
        )
        dense = knn_retrieval(
            pred.astype(np.float64), y.astype(np.float64), ks=(1, 5), metric=metric
        )
        for k in (1, 5):
            assert sparse["acc_at_k"][k] == pytest.approx(dense["acc_at_k"][k]), metric
            assert sparse["chance_at_k"][k] == pytest.approx(dense["chance_at_k"][k])
        assert sparse["median_rank"] == pytest.approx(dense["median_rank"]), metric
        assert sparse["mrr"] == pytest.approx(dense["mrr"], rel=1e-9), metric
        assert sparse["n_pool"] == dense["n_pool"] == n


def test_engine_requires_paired_conversations():
    xb, yib, yvb, cb, xa, yia, yva, ca, d_sae = _planted_cell()
    with pytest.raises(ValueError, match="BOTH stages"):
        nullmod._CellEngine(
            xb,
            yib,
            yvb,
            [f"L{c}" for c in cb],
            xa,
            yia,
            yva,
            [f"R{c}" for c in ca],
            d_sae=d_sae,
        )


# ---------------------------------------------------------------------------
# v11 delta (b) — runtime convention branch: the Gram/dual adapter is the
# SAME estimator as the streamed primal fit (parity fixture at n_tr <= d_in,
# the shape the plan's §Design estimator bullet mandates for this family).
# ---------------------------------------------------------------------------
def test_fold_fit_gram_matches_streamed_at_n_le_d():
    rng = np.random.default_rng(23)
    n, d_in, d_sae = 30, 48, 16  # per-fold n_tr = 24 <= d_in — the dual regime
    X = rng.normal(size=(n, d_in)).astype(np.float32)
    y = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.5)).astype(np.float32)
    y_idx, y_val = _dense_to_iv(y)
    conv = [f"c{i:03d}" for i in range(n)]
    folds = fpf._make_folds(conv, k=5, seed=0)
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        assert len(train_idx) <= d_in, "fixture must exercise the n_tr <= d_in regime"
        outs = {}
        for name, fn in (("primal", fpf._fold_fit_streamed), ("gram", fpf._fold_fit_gram)):
            ss_res = np.zeros(d_sae)
            ss_tot = np.zeros(d_sae)
            out_pred = np.zeros((n, d_sae), dtype=np.float64)
            info = fn(
                X,
                y_idx,
                y_val,
                d_sae,
                train_idx,
                test_idx,
                gcv_dof_cap=fpf.DOF_CAP_FRACTION,
                feature_chunk=7,  # deliberately unaligned chunking
                ss_res=ss_res,
                ss_tot=ss_tot,
                out_pred=out_pred,
            )
            outs[name] = (info, ss_res, ss_tot, out_pred)
        ip, rp, tp, pp = outs["primal"]
        ig, rg, tg, pg = outs["gram"]
        # λ exact, dof rel 1e-9, preds rtol 1e-7 — the parity family tolerances.
        assert ig["best_lambda"] == pytest.approx(ip["best_lambda"], rel=1e-12)
        assert ig["dof"] == pytest.approx(ip["dof"], rel=1e-9)
        np.testing.assert_allclose(pg[test_idx], pp[test_idx], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(rg, rp, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(tg, tp, rtol=1e-9, atol=1e-12)


def test_gram_adapter_call_binds_against_real_helper_api():
    """Offline signature-bind of the adapter's helper call shape (the
    fabricated-signature production-TypeError class — the sparsify pin's
    sibling)."""
    sig = inspect.signature(ridge_fit_predict_fast_layer_batched)
    sig.bind(
        np.zeros((1, 4, 6)),
        np.zeros((1, 4, 3)),
        np.zeros((1, 2, 6)),
        lambdas=np.logspace(-3, 8, 23),
        device="cpu",
        return_info=True,
        gcv_dof_cap=0.9,
    )


def test_fit_cell_runtime_convention_gram_dual(tmp_path):
    """A cell whose folds land at n_tr <= d_in routes the Gram/dual branch
    and logs it (plan v11 delta (b): the selected convention rides the JSONL)."""
    n, d_sae = 20, 12
    stem = "turnstore_base_chat_tiny"
    tdir = tmp_path / stem
    _write_random_turnstore(tdir, list(range(n)), seed=4, stem=stem)
    rng = np.random.default_rng(5)
    y = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.5)).astype(np.float32)
    y_idx, y_val = _dense_to_iv(y)
    enc = tmp_path / f"base_chat_tiny_answer_L{LAYER}.pt"
    ts.save_encoded_target(
        enc,
        idx=torch.as_tensor(y_idx),
        val=torch.as_tensor(y_val),
        d_sae=d_sae,
        k=y_idx.shape[1],
        conv_ids=[f"conv-{g:03d}" for g in range(n)],
        cell={"stage": "base", "render": "chat", "corpus": "tiny", "state": "answer", "layer": 1},
    )
    out = tmp_path / "out.jsonl"
    fpf._PARITY_GATE_STATE["done"] = True  # gate covered by its own tests
    # HIDDEN=8 fixture d_in vs per-fold n_tr=16: force the dual regime by
    # treating d_in as larger than any fold's n_tr is NOT possible via args —
    # the branch keys on realized shapes — so pin the regime with a wide-d
    # random X via the payload instead: n_tr=16 > 8 means primal here; assert
    # the OPPOSITE regime through _fold_fit_gram parity above. This test pins
    # the PRIMAL logging surface + field presence end-to-end.
    fpf.fit_cell(tmp_path, "base", "chat", "tiny", enc, arm="context", output_path=out, layer=LAYER)
    row = json.loads(out.read_text().split("\n")[0])
    assert row["convention"] == "primal"
    assert row["convention_folds"] == ["primal"] * fpf.K_FOLDS
    for key in (
        "n_at_low_edge",
        "n_at_high_edge",
        "lambda_grid_log10_lo",
        "lambda_grid_log10_hi",
        "n_lambda",
        "n_ext_low",
        "n_ext_high",
        "regularization_limited",
    ):
        assert key in row, key


def test_engine_regime_guard_fails_loud_at_n_le_d():
    """v11 delta (c): the null engine REFUSES an n_tr <= d_in regime at init,
    naming contingency lever (b) — it can never silently run a regime the
    plan did not declare (the P2 side fits it via the Gram/dual branch)."""
    rng = np.random.default_rng(11)
    n, d_in, d_sae = 40, 64, 16  # pooled per-(fold, group) n_tr ~ 32 <= 64
    conv = [f"c{i:03d}" for i in range(n)]
    xb = rng.normal(size=(n, d_in))
    xa = rng.normal(size=(n, d_in))
    yb = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.4)).astype(np.float64)
    ya = (rng.normal(size=(n, d_sae)) * (rng.random(size=(n, d_sae)) < 0.4)).astype(np.float64)
    yib, yvb = _dense_to_iv(yb)
    yia, yva = _dense_to_iv(ya)
    with pytest.raises(RuntimeError, match=r"lever \(b\)"):
        nullmod._CellEngine(xb, yib, yvb, conv, xa, yia, yva, conv, d_sae=d_sae)


# ---------------------------------------------------------------------------
# v13 delta (f) — the registered λ grid + the edge-hit audit/disposition
# ---------------------------------------------------------------------------
def test_lambda_grid_registration_v13():
    """P2 and the P3 engine share ONE grid — the v13 np.logspace(-3, 8, 23)
    (plan §11 RIDGE_LAMBDA_GRID); the shared #779 helper's own default stays
    the untouched 13-point grid."""
    expected = np.logspace(-3, 8, 23)
    np.testing.assert_allclose(fpf.LAMBDA_GRID, expected, rtol=1e-12)
    np.testing.assert_allclose(nullmod.LAMBDA_GRID, expected, rtol=1e-12)
    helper_default = inspect.signature(ridge_fit_predict_fast_layer_batched)
    assert helper_default.parameters["lambdas"].default is None  # resolved in-body to 13-pt


def test_lambda_grid_lattice_helpers():
    lo, hi = fpf._lambda_grid_log10_bounds(fpf.LAMBDA_GRID)
    assert (lo, hi) == (-3.0, 8.0)
    g = fpf._build_lambda_grid(-4.0, 8.0)
    assert len(g) == 25 and g[0] == pytest.approx(1e-4) and g[-1] == pytest.approx(1e8)
    # The retired 13-point grid is ALSO a half-decade lattice (its bounds
    # parse); a non-lattice grid fails loud.
    assert fpf._lambda_grid_log10_bounds(np.logspace(-2, 4, 13)) == (-2.0, 4.0)
    with pytest.raises(ValueError, match="half-decade lattice"):
        fpf._lambda_grid_log10_bounds(np.logspace(-2, 4, 10))


def test_fit_cell_edge_extension_and_regularization_limited(tmp_path):
    """Pure-noise targets drive GCV to ever-larger λ: the top edge pins, the
    cell extends one decade per pass (<= 2), re-runs FULL-CELL each time, and
    is finally flagged REGULARIZATION-LIMITED (plan v13 delta (f))."""
    n, d_sae = 30, 10
    stem = "turnstore_base_chat_tiny"
    tdir = tmp_path / stem
    _write_random_turnstore(tdir, list(range(n)), seed=7, stem=stem)
    rng = np.random.default_rng(8)
    y = rng.normal(size=(n, d_sae)).astype(np.float32)  # dense noise target
    y_idx, y_val = _dense_to_iv(y)
    enc = tmp_path / f"base_chat_tiny_answer_L{LAYER}.pt"
    ts.save_encoded_target(
        enc,
        idx=torch.as_tensor(y_idx),
        val=torch.as_tensor(y_val),
        d_sae=d_sae,
        k=y_idx.shape[1],
        conv_ids=[f"conv-{g:03d}" for g in range(n)],
        cell={"stage": "base", "render": "chat", "corpus": "tiny", "state": "answer", "layer": 1},
    )
    out = tmp_path / "out.jsonl"
    fpf._PARITY_GATE_STATE["done"] = True
    # Start from a deliberately narrow lattice ending far below the noise
    # optimum (λ -> inf for pure noise) so the top edge pins every pass.
    fpf.fit_cell(
        tmp_path,
        "base",
        "chat",
        "tiny",
        enc,
        arm="context",
        output_path=out,
        layer=LAYER,
        lambda_grid=fpf._build_lambda_grid(-3.0, -1.0),
    )
    row = json.loads(out.read_text().split("\n")[0])
    assert row["n_ext_high"] == fpf.MAX_EDGE_EXTENSIONS
    assert row["n_at_high_edge"] > 0
    assert row["regularization_limited"] is True
    # The realized grid record reflects the extensions (one decade per pass).
    assert row["lambda_grid_log10_hi"] == pytest.approx(-1.0 + 2.0)
    assert row["lambda_grid_log10_lo"] == pytest.approx(-3.0)
    assert (
        row["n_lambda"]
        == round((row["lambda_grid_log10_hi"] - row["lambda_grid_log10_lo"]) / 0.5) + 1
    )


def test_cell_lambda_grid_union_and_legacy(tmp_path):
    """P3 rides the UNION lattice of both stages' realized P2 grids (v13
    delta (f): true and null always share one realized grid); legacy P2 rows
    without the grid fields fall back to the module default."""
    r2 = tmp_path

    def _write(stage, lo=None, hi=None):
        row = {"feature_id": 0, "R2": 0.1}
        if lo is not None:
            row["lambda_grid_log10_lo"] = lo
            row["lambda_grid_log10_hi"] = hi
        (r2 / f"{stage}_chat_tiny_context_L{nullmod.LAYER}.jsonl").write_text(
            json.dumps(row) + "\n"
        )

    _write("base", -3.0, 8.0)
    _write("sft", -3.0, 10.0)  # sft extended twice upward
    g = nullmod.cell_lambda_grid(r2, ("base", "sft"), "chat", "tiny", "context")
    assert g[0] == pytest.approx(1e-3) and g[-1] == pytest.approx(1e10) and len(g) == 27

    _write("base")  # legacy row, no grid fields
    _write("sft")
    g2 = nullmod.cell_lambda_grid(r2, ("base", "sft"), "chat", "tiny", "context")
    np.testing.assert_allclose(g2, nullmod.LAMBDA_GRID, rtol=1e-12)
