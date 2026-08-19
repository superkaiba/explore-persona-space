"""Committed port of the issue-2222 P3/P4 math verification smoke (round-2 C5).

Ports the load-bearing assertions of the round-1 ``/tmp/issue2222_smoke_p3.py``
(31 checks) into pytest: map-projection identity, perm-null GEMM vs per-draw
loop equivalence, flat + clustered bootstrap, selection-inherited delta, LOFO
sweep, spearman/AUC, LOGO bias algebra, dof-capped GCV ridge incl. the n<=d
refusal (#1887), the judge probe stand-in identity, merge/censored accounting,
the verdict lattice, and the ``preds_frozen``/``collect_cosines`` coupling in
the #825 core. Round-2 additions: the ``_load_percell`` sidecar-key refusal
(C3). Everything is synthetic-array offline CPU work (no GPU, no network).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2222_analysis as ana
import issue2222_judge as jdg
import issue2222_lib as lib
import issue2222_reduce as red


def _fmap(rng: np.random.Generator, L: int, D: int) -> dict[str, np.ndarray]:
    """Synthetic frozen-map dict in the plan-A7 npz schema."""
    return {
        "w": rng.standard_normal((L, D, D)).astype(np.float32),
        "x_mu": rng.standard_normal((L, 1, D)).astype(np.float32),
        "x_sd": (0.5 + rng.random((L, 1, D))).astype(np.float32),
        "y_mu": rng.standard_normal((L, 1, D)).astype(np.float32),
    }


def test_map_project_via_u_matches_explicit_projection(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n, L, D, T = 7, 3, 5, 2
    v = rng.standard_normal((n, L, D)).astype(np.float16)
    fmap = _fmap(rng, L, D)
    vhat = ana.unit_normalize_rows(rng.standard_normal((T, L, D)))
    proj = ana.map_project_via_u(v, fmap, vhat)
    expl = np.empty((n, T, L))
    for layer in range(L):
        z = (v[:, layer, :].astype(np.float64) - fmap["x_mu"][layer]) / fmap["x_sd"][layer]
        m = z @ fmap["w"][layer].astype(np.float64) + fmap["y_mu"][layer]
        expl[:, :, layer] = m @ vhat[:, layer, :].T
    assert np.allclose(proj, expl, atol=2e-2)
    # load_frozen_map round-trip (shape asserts + key set)
    p = tmp_path / "m.npz"
    np.savez(p, **fmap)
    assert set(ana.load_frozen_map(p)) == {"w", "x_mu", "x_sd", "y_mu"}


def test_pearson_r_cols_matches_corrcoef() -> None:
    rng = np.random.default_rng(1)
    vals = rng.standard_normal((24, 4))
    y = rng.standard_normal(24)
    r = ana.pearson_r_cols(vals, y)
    ref = np.array([np.corrcoef(vals[:, j], y)[0, 1] for j in range(4)])
    assert np.allclose(r, ref, atol=1e-12)


def test_perm_null_shape_and_range() -> None:
    rng = np.random.default_rng(2)
    vals = rng.standard_normal((24, 4))
    y = rng.standard_normal(24)
    perm = ana.perm_null_abs_r(vals, y, n_perms=200, seed=1)
    assert perm.shape == (200, 4)
    assert (perm >= 0).all() and (perm <= 1 + 1e-9).all()


def test_boot_r_matrix_matches_per_draw_corrcoef_loop() -> None:
    """The batched GEMM bootstrap equals the per-draw corrcoef loop."""
    rng = np.random.default_rng(3)
    vals = rng.standard_normal((24, 4))
    y = rng.standard_normal(24)
    idx = ana.boot_indices_flat(24, 50, seed=2)
    rb = ana.boot_r_matrix(vals, y, idx)
    ref = [[np.corrcoef(vals[idx[b], j], y[idx[b]])[0, 1] for j in range(4)] for b in range(5)]
    assert np.allclose(rb[:5], np.array(ref), atol=1e-10, equal_nan=True)


def test_boot_indices_clustered_carries_whole_equal_groups() -> None:
    groups = np.repeat(np.arange(8), 3)  # 8 families x 3 versions
    idx_c = ana.boot_indices_clustered(groups, 50, seed=3)
    assert idx_c.shape == (50, 24)
    for row in idx_c[:5]:
        assert len(row) == 24
        assert set(groups[row]) <= set(groups)
    with pytest.raises(ValueError):
        ana.boot_indices_clustered(np.array([0, 0, 1]), 5, seed=0)


def test_selection_inherited_delta_picks_per_arm_argmax() -> None:
    ra = np.array([[0.1, 0.9], [0.5, -0.7]])
    rb2 = np.array([[0.2, 0.1], [-0.8, 0.3]])
    # row1: |-0.7| > 0.5 -> arm-a argmax layer 1, SIGNED value -0.7; arm-b -> layer 0.
    d, sa, sb = ana.selection_inherited_delta(ra, rb2)
    assert np.allclose(d, [0.9 - 0.2, -0.7 - (-0.8)])
    assert sa.tolist() == [1, 1] and sb.tolist() == [0, 0]


def test_lofo_layer_sweep_selects_informative_layer() -> None:
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(8), 3)
    sig = rng.standard_normal(24)
    vals = np.column_stack([sig + 0.1 * rng.standard_normal(24), rng.standard_normal(24)])
    sw = ana.lofo_layer_sweep(vals, sig, groups)
    assert all(v == 0 for v in sw["selected_layer_by_fold"].values())
    assert sw["lofo_r"] > 0.9


def test_spearman_and_auc_hand_values() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(ana.spearman(a, a**3) - 1.0) < 1e-12
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0, 0], dtype=bool)
    assert abs(ana.auc_mann_whitney(scores, labels) - 1.0) < 1e-12
    # ties take mid-ranks -> chance
    assert abs(ana.auc_mann_whitney(np.ones(4), np.array([1, 0, 1, 0], bool)) - 0.5) < 1e-12


def test_leave_one_group_out_bias_holds_out_own_group() -> None:
    sr = np.array([[2.0, 0.0], [0.0, 4.0]])  # F=2 groups, D=2
    cnt = np.array([2.0, 2.0])
    assert np.allclose(ana.leave_one_group_out_bias(sr, cnt), [[0.0, 2.0], [1.0, 0.0]])


def test_dof_capped_ridge_multi_y_and_n_le_d_refusal() -> None:
    rng = np.random.default_rng(5)
    n, d, t = 80, 6, 2
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal((d, t)) + 0.05 * rng.standard_normal((n, t))
    fold_ids = np.repeat(np.arange(4), 20)
    res = ana.dof_capped_ridge_multi_y(x, y, fold_ids, lambdas=np.logspace(-3, 3, 9), dof_cap=0.9)
    assert (res["heldout_r2"] > 0.9).all()
    assert res["gcv_lambda"].shape == (4, t)
    with pytest.raises(ValueError, match="under-determined"):
        ana.dof_capped_ridge_multi_y(
            rng.standard_normal((6, 10)),
            rng.standard_normal(6),
            np.array([0, 0, 0, 1, 1, 1]),
            lambdas=np.logspace(-2, 2, 5),
        )


def test_ridge_fit_all_predict_and_pca_basis() -> None:
    rng = np.random.default_rng(6)
    n, d, t = 80, 6, 2
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal((d, t)) + 0.05 * rng.standard_normal((n, t))
    fit = ana.dof_capped_ridge_fit_all(x, y, lambdas=np.logspace(-3, 3, 9), dof_cap=0.9)
    pred = ana.ridge_predict(fit, x)
    assert 1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum() > 0.95
    _mu, basis = ana.pca_train_basis(x, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-10)


def test_probe_mapped_standin_identity() -> None:
    rng = np.random.default_rng(7)
    n, L, D, t = 7, 3, 5, 2
    fmap = _fmap(rng, L, D)
    w = rng.standard_normal((D, t))
    b0 = rng.standard_normal(t)
    src = rng.standard_normal((n, D)).astype(np.float32)
    layer = 1
    got = jdg.probe_mapped_standin(src, fmap, layer, w, b0)
    z = (src.astype(np.float64) - fmap["x_mu"][layer]) / fmap["x_sd"][layer]
    m_v = z @ fmap["w"][layer].astype(np.float64) + fmap["y_mu"][layer]
    assert np.allclose(got, m_v @ w + b0, atol=1e-8)


def test_merge_censored_accounting_and_item_id_grammar() -> None:
    rec = {"per_item_api_refusals": {"a": 2, "b": 0}, "per_item_transport_losses": {"a": 1, "c": 3}}
    assert jdg.censored_counts(rec) == {"a": 3, "c": 3}
    merged = jdg.merge_judge_draws({"a": [10, 20]}, {"a": [30], "b": [40]}, ["a", "b", "z"])
    assert merged["a"]["mean"] == 20.0
    assert merged["a"]["n_batch"] == 2 and merged["a"]["n_sync"] == 1
    assert merged["b"]["rate_gt_50"] == 0.0
    assert merged["z"]["mean"] is None  # zero-kept: None, never coerced
    iid = jdg.item_id_for("evil_normal", 123)
    assert jdg.split_item_id(iid) == ("evil_normal", 123)
    with pytest.raises(ValueError):
        jdg.item_id_for("x" * 60, 1)  # over-long id violates the custom_id grammar


def test_verdict_lattice_disjoint_exhaustive() -> None:
    assert red._verdict(0.2, 0.05, 0.4) == "Confirmed"
    assert red._verdict(-0.2, -0.4, -0.05) == "Falsified"
    assert red._verdict(0.1, -0.05, 0.3) == "Inconclusive"


def test_preds_frozen_populated_only_under_collect_cosines() -> None:
    """The #825 core writes preds_frozen ONLY under collect_cosines=True — the
    coupling the reduce call site documents (a False call silently degenerates
    the mapped_tuned arm to the raw arm)."""
    import issue825_fit_cells as core

    rng = np.random.default_rng(8)
    n, L, d = 40, 2, 6
    x = rng.standard_normal((n, L, d)).astype(np.float32)
    wm = rng.standard_normal((d, d))
    y = np.stack([x[:, li, :] @ wm for li in range(L)], axis=1).astype(np.float32)
    conv = np.repeat(np.arange(4), 10)
    kw = dict(
        n_folds=4,
        seed=0,
        null_draws=0,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        frozen_layers=(0, 1),
        reduced_basis_companion=False,
    )
    sw = core.heldout_r2_sweep(x, y, conv, collect_cosines=True, **kw)
    assert all(np.abs(sw["preds_frozen"][li]).sum() > 0 for li in (0, 1))
    sw0 = core.heldout_r2_sweep(x, y, conv, collect_cosines=False, **kw)
    assert all(np.abs(sw0["preds_frozen"][li]).sum() == 0 for li in (0, 1))


def test_load_percell_refuses_stale_sidecar_key(tmp_path: Path) -> None:
    """Round-2 C3: a standalone --stage aggregate/form_b must not silently reduce
    stale projections — _load_percell recomputes the key and refuses a mismatch."""
    ds = "evil_normal"
    meta = {"rb_source": "rb_v2"}
    knn_layers = (15, 19)
    # Capture manifest the key derives from.
    cap_dir = lib.capture_dir(tmp_path, ds)
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / "manifest.json").write_text(json.dumps({"resume_fingerprint": "fp-test"}))
    manifest = red.load_capture_manifest(tmp_path, ds)
    good_key = red._percell_key(manifest, meta, knn_layers)
    # Percell npz + sidecar.
    p = red.percell_path(tmp_path, ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, row_ids=np.arange(3, dtype=np.int64))
    sidecar = p.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({"key": "stale-key"}))
    with pytest.raises(RuntimeError, match="key mismatch"):
        red._load_percell(tmp_path, [ds], meta=meta, knn_layers=knn_layers)
    sidecar.write_text(json.dumps({"key": good_key}))
    cells = red._load_percell(tmp_path, [ds], meta=meta, knn_layers=knn_layers)
    assert list(cells[ds]["row_ids"]) == [0, 1, 2]
    sidecar.unlink()  # missing sidecar is a refusal too (never silently trusted)
    with pytest.raises(FileNotFoundError):
        red._load_percell(tmp_path, [ds], meta=meta, knn_layers=knn_layers)
