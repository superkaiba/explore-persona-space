"""Unit tests for scripts/issue2661_flat_ctx_sae.py (task #2661 brief roster:
nMSE formula, flat-SAE one-tier construction, zero-variance B reindexing, edge
gate on planted edges, CSR/dense encode round-trip) plus the standardized
closed-form Gram/XtY algebra and the blocked shuffle-null parity."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "vendored_2476"))

import issue2661_flat_ctx_sae as D  # noqa: E402


def _tiny_sae(act_dim=8, dict_size=16, k=4, seed=7, threshold=0.05):
    sae = D.T.MatryoshkaBatchTopKSAE(
        act_dim=act_dim, dict_size=dict_size, k=k, tier_bounds=(dict_size,), seed=seed
    )
    with torch.no_grad():
        sae.threshold.fill_(threshold)
    return sae.eval()


def test_nmse_formula_on_toy_tensor():
    """_der_metrics_pass raw nMSE == E||x-xhat||^2 / E||x||^2 computed densely;
    the mean-centered variant == 1 - variance-FVE; realized L0 matches."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 8)).astype(np.float16)
    sae = _tiny_sae()
    got = D._der_metrics_pass(sae, x, np.arange(32), chunk=8)
    xf = torch.as_tensor(x.astype(np.float32))
    f = sae.encode(xf)
    xhat = sae.decode(f).numpy()
    xd = x.astype(np.float64)
    nmse_raw = float(((xd - xhat) ** 2).sum() / (xd**2).sum())
    var_x = xd.var(axis=0, ddof=1).sum()
    var_r = (xd - xhat).var(axis=0, ddof=1).sum()
    assert np.isclose(got["nmse_raw"], nmse_raw, rtol=1e-5)
    assert got["nmse_raw_is_ders_metric"] is True
    assert np.isclose(got["nmse_mean_centered"], var_r / var_x, rtol=1e-5)
    assert np.isclose(got["variance_fve"], 1.0 - var_r / var_x, rtol=1e-5)
    assert np.isclose(got["realized_l0"], float((f > 0).sum()) / 32, rtol=1e-6)


def test_flat_ctx_sae_construction_one_tier_32768():
    """The production construction is EXACTLY one tier at 32,768 (the #2552
    plan §4 P1.2 trap: never through _sae_tier_bounds)."""
    args = Namespace(ctx_width=0, ctx_k=0)
    sae = D._build_ctx_sae(args, "cpu")
    assert sae.tier_bounds == (32_768,)
    assert sae.dict_size == 32_768 and sae.k == 128
    assert sae.seed == 2661
    # smoke narrowing keeps the 1-tier shape
    sae_s = D._build_ctx_sae(Namespace(ctx_width=64, ctx_k=8), "cpu")
    assert sae_s.tier_bounds == (64,) and sae_s.k == 8


def test_csr_encode_roundtrips_dense_on_64_rows():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 8)).astype(np.float16)
    sae = _tiny_sae(seed=3)
    csr = D._encode_csr(sae, x, np.arange(64), chunk=16)
    dense = sae.encode(torch.as_tensor(x.astype(np.float32))).numpy()
    assert csr.shape == (64, 16)
    np.testing.assert_allclose(csr.toarray(), dense, rtol=1e-6, atol=1e-7)
    assert csr.nnz == int((dense > 0).sum())


def test_standardized_gram_xty_closed_forms_match_dense():
    """_standardized_gram/_standardized_xty (raw sparse products + closed-form
    corrections) == the direct dense computation, INCLUDING the split-half case
    where xmu/xsd come from a different (full-train) row set."""
    rng = np.random.default_rng(2)
    X = sp.csr_matrix(np.maximum(rng.standard_normal((60, 7)), 0).astype(np.float32))
    Y = sp.csr_matrix(np.maximum(rng.standard_normal((60, 5)), 0).astype(np.float32))
    full_rows = np.arange(60)
    xmu, xvar = D._col_moments_csr(X, full_rows)
    xsd = np.sqrt(np.maximum(xvar, 1e-12))
    half = np.arange(0, 60, 2)  # a HALF standardized by the FULL-train stats
    acc = D._accumulate_raw_products(X, Y, half, torch.device("cpu"), chunk=16)
    gs = D._standardized_gram(acc, xmu, xsd, torch.device("cpu")).numpy()
    ymu_h = np.asarray(Y[half].mean(axis=0), np.float64).ravel()
    xty = D._standardized_xty(acc, xmu, xsd, ymu_h, torch.device("cpu")).numpy()
    xs = (X[half].toarray().astype(np.float64) - xmu) / xsd
    np.testing.assert_allclose(gs, xs.T @ xs, rtol=1e-4, atol=1e-4)  # fp32 chunk GEMM
    yc = Y[half].toarray().astype(np.float64) - ymu_h
    np.testing.assert_allclose(xty, xs.T @ yc, rtol=1e-4, atol=1e-4)  # fp32 chunk GEMM


def test_zero_variance_drop_reindexes_B_to_full_width(tmp_path):
    """_eigh_ridge_fit drops zero-variance columns mechanically and writes B
    reindexed to the FULL input width (zero rows at dropped ids), matching a
    direct dense ridge solve at the selected lambda."""
    rng = np.random.default_rng(3)
    n, d_full, d_y = 200, 24, 10
    xd = np.maximum(rng.standard_normal((n, d_full)), 0).astype(np.float32) + 0.05
    dead = [3, 17]
    xd[:, dead] = 0.0
    X = sp.csr_matrix(xd)
    yd = np.maximum(rng.standard_normal((n, d_y)), 0).astype(np.float32)
    Y = sp.csr_matrix(yd)
    tr, va, te = np.arange(160), np.arange(160, 180), np.arange(180, 200)
    _mu, var = D._col_moments_csr(X, tr)
    live = np.flatnonzero(var > D.ZERO_VAR_EPS)
    assert set(dead).isdisjoint(live.tolist()) and len(live) == d_full - len(dead)
    args = Namespace(device="cpu")
    doc = D._eigh_ridge_fit(
        args,
        X,
        Y,
        tr,
        va,
        te,
        live,
        D.LAMBDA_GRID_27,
        tag="t",
        out_dir=tmp_path,
        b_full_rows=d_full,
        col_block=4,
    )
    B = np.load(tmp_path / "B_t.fp16.npy")
    assert B.shape == (d_full, d_y)
    assert (B[dead] == 0).all(), "dropped zero-variance rows must be zero"
    assert np.abs(B[live]).sum() > 0
    # parity vs a direct dense solve at the selected lambda
    xmu = xd[tr][:, live].mean(0).astype(np.float64)
    xsd = xd[tr][:, live].std(0).astype(np.float64)
    ymu = yd[tr].mean(0).astype(np.float64)
    xs = (xd[tr][:, live].astype(np.float64) - xmu) / xsd
    lam = float(doc["selected_lambda"])
    b_direct = np.linalg.solve(
        xs.T @ xs + lam * np.eye(len(live)), xs.T @ (yd[tr].astype(np.float64) - ymu)
    )
    np.testing.assert_allclose(B[live].astype(np.float64), b_direct, rtol=5e-2, atol=5e-3)
    xs_te = (xd[te][:, live].astype(np.float64) - xmu) / xsd
    pred_direct = xs_te @ b_direct + ymu
    pred = np.load(tmp_path / "pred_te_t.fp32.npy")
    np.testing.assert_allclose(pred, pred_direct, rtol=5e-2, atol=5e-3)
    assert doc["lambda_grid_edge_hit"] in (True, False)


def test_edge_survival_gate_on_planted_edges():
    """Planted large sign-consistent edges survive; a sign-flipped strong edge
    and sub-threshold noise do not (#1482 recipe semantics)."""
    rng = np.random.default_rng(4)
    d, d_y = 30, 40
    noise = 0.01
    B = (rng.standard_normal((d, d_y)) * noise).astype(np.float32)
    ba = (rng.standard_normal((d, d_y)) * noise).astype(np.float32)
    bb = (rng.standard_normal((d, d_y)) * noise).astype(np.float32)
    planted = [(2, 5, 5.0), (7, 11, -4.0), (13, 23, 3.0), (21, 31, 6.0), (29, 39, -5.5)]
    for i, j, v in planted:
        B[i, j] = v
        ba[i, j] = v * (1 + 0.1)
        bb[i, j] = v * (1 - 0.1)
    flip = (9, 9, 4.5)
    B[flip[0], flip[1]] = flip[2]
    ba[flip[0], flip[1]] = flip[2]
    bb[flip[0], flip[1]] = -flip[2]  # sign flips in half b -> must NOT survive
    null_sd = np.full(d_y, noise, np.float64)
    tau = 10.0 * null_sd  # ~10 null SDs — noise stays below, plants clear it
    gate = D._edge_survival(B, ba, bb, null_sd, tau, topm=50)
    surv = {
        (int(i), int(j))
        for i, j, ok in zip(gate["ci"], gate["cj"], gate["surviving"], strict=True)
        if ok
    }
    for i, j, _v in planted:
        assert (i, j) in surv, f"planted edge ({i},{j}) must survive"
    assert (flip[0], flip[1]) not in surv, "sign-flipped edge must not survive"
    # nothing at noise scale clears a 10-SD threshold
    for i, j in surv:
        assert abs(B[i, j]) > 10 * noise


def test_shuffle_null_blocked_matches_vendored_kernel():
    rng = np.random.default_rng(5)
    pred = rng.standard_normal((50, 12)).astype(np.float32)
    true = np.maximum(rng.standard_normal((50, 12)), 0).astype(np.float32)
    seeds = (11, 22, 33)
    blocked = D._shuffle_null_r2_blocked(pred, sp.csc_matrix(true), seeds, col_block=5)
    vend = D.T._shuffle_null_r2(pred, true, seeds)
    np.testing.assert_allclose(
        blocked.astype(np.float32), vend.astype(np.float32), rtol=2e-3, atol=2e-3
    )


def test_topm_flat_blocked_matches_global_argpartition():
    rng = np.random.default_rng(6)
    B = rng.standard_normal((17, 23)).astype(np.float16)
    got = set(D._topm_flat_blocked(B, 40, col_block=7).tolist())
    absB = np.abs(B.astype(np.float32)).ravel()
    want = set(np.argsort(-absB)[:40].tolist())
    assert got == want


def test_resolve_repo_revision_ported_and_reachable_from_judge():
    """Review r1 Major 1: the judge's prep path calls the driver's
    _resolve_repo_revision — the attribute must exist on the driver module and
    resolve through the judge's lazy import (no network: attribute checks only)."""
    assert callable(D._resolve_repo_revision)
    import inspect

    sig = inspect.signature(D._resolve_repo_revision)
    sig.bind(None, "what")  # the judge call shape: (revision=None, what=str)
    import issue2661_judge_waves as J

    assert J._u2661()._resolve_repo_revision is D._resolve_repo_revision


def test_half_reusable_rejects_partial_memmap(tmp_path):
    """Review r1 Major 2: a pre-allocated (partially-zero) half WITHOUT its
    done-marker is never reused; marker + matching shape is; a shape-mismatched
    marker is not."""
    d, d_y = 6, 8
    hp = tmp_path / "B_half_a.fp16.npy"
    bh = np.lib.format.open_memmap(str(hp), mode="w+", dtype=np.float16, shape=(d, d_y))
    del bh  # simulated crash mid-fill: full-size, all-zero, NO done marker
    assert hp.exists()
    assert D._half_reusable(hp, d, d_y) is False
    D.T._write_json(D._half_done_path(hp), {"half": "a", "shape": [d, d_y]}, phase="edges")
    assert D._half_reusable(hp, d, d_y) is True
    assert D._half_reusable(hp, d + 1, d_y) is False  # shape drift -> recompute


def test_csr_moments_accumulate_fp64_exactly():
    """r2 pod-smoke regression pin: scipy sparse sum/mean accumulate at the
    MATRIX dtype (fp32) and sat 6.6e-7 off the fp64 canonical helper on the pod;
    _col_moments_csr/_csr_colsum64 must match dense fp64 to summation-order
    precision on a magnitude-hostile fixture (fails pre-fix at ~1e-7 relative)."""
    rng = np.random.default_rng(8)
    dense = (rng.standard_normal((2_000, 4)) * 1e4).astype(np.float32)
    dense[dense < 0] = 0.0  # gated-activation shape (nonnegative sparse values)
    X = sp.csr_matrix(dense)
    want_sum = dense.astype(np.float64).sum(axis=0)
    got_sum = D._csr_colsum64(X)
    np.testing.assert_allclose(got_sum, want_sum, rtol=1e-12)
    mu, var = D._col_moments_csr(X, np.arange(2_000))
    d64 = dense.astype(np.float64)
    np.testing.assert_allclose(mu, d64.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(
        var, d64.var(axis=0), rtol=1e-9, atol=1e-6
    )  # population var; cancellation-limited
    # the scipy fp32 path is measurably WRONG on this fixture (the pre-fix bug)
    scipy_fp32 = np.asarray(X.mean(axis=0), np.float64).ravel()
    assert np.abs(scipy_fp32 - d64.mean(axis=0)).max() > 1e-4


def test_mlp_reaches_ridge_r2_on_linear_problem():
    """r4 divergence-fix guard: on a synthetic LINEAR problem the MLP trainer
    (production lr/clip/target-transform defaults) must reach >= 0.9x the
    closed-form ridge's val pooled R^2, start AT the train-mean baseline
    (zero-init head => epoch-1 val R^2 never off-scale), and return raw-unit
    predictions via pred = model(x_std) * sy + ymu."""
    rng = np.random.default_rng(2661)
    n, d, d_y = 2_400, 24, 12
    dense = rng.standard_normal((n, d)).astype(np.float32)
    dense[dense < 0.3] = 0.0  # sparse nonneg inputs, SAE-code-like
    w = rng.standard_normal((d, d_y)) * 0.5
    b = rng.standard_normal(d_y) * 2.0
    y_dense = dense.astype(np.float64) @ w + b + 0.05 * rng.standard_normal((n, d_y))
    X = sp.csr_matrix(dense)
    Y = sp.csr_matrix(y_dense.astype(np.float32))
    tr = np.arange(1_600)
    va = 1_600 + np.arange(400)
    xmu, xvar = D._col_moments_csr(X, tr)
    xsd = np.sqrt(xvar)
    assert (xsd > 0).all()
    ymu, _ = D._col_moments_csr(Y, tr)
    # closed-form ridge on the SAME standardized inputs / centered targets
    xs_tr = (X[tr].toarray().astype(np.float64) - xmu) / xsd
    xs_va = (X[va].toarray().astype(np.float64) - xmu) / xsd
    y_tr = Y[tr].toarray().astype(np.float64)
    y_va = Y[va].toarray().astype(np.float64)
    a = xs_tr.T @ xs_tr + 1e-3 * np.eye(d)
    w_r = np.linalg.solve(a, xs_tr.T @ (y_tr - ymu))
    ridge_r2 = D._pooled_r2(xs_va @ w_r + ymu, y_va)
    assert ridge_r2 > 0.95  # the problem IS linear; anchor sanity
    model, sy, (best_r2, _best_epoch), epochs_log = D._fit_mlp(
        X,
        Y,
        tr,
        va,
        xmu,
        xsd,
        ymu,
        torch.device("cpu"),
        hidden=64,
        batch=256,
        max_epochs=200,
        patience=200,
    )
    # stability: zero-init head starts at the train-mean baseline, never off-scale
    assert epochs_log[0]["val_pooled_r2"] > -0.5, epochs_log[0]
    assert best_r2 >= 0.9 * ridge_r2, (best_r2, ridge_r2)
    # raw-unit prediction path matches the phase's un-transform
    with torch.no_grad():
        xv = torch.as_tensor(xs_va, dtype=torch.float32)
        pv = (model(xv) * sy + torch.as_tensor(ymu, dtype=torch.float32)).numpy()
    assert abs(D._pooled_r2(pv.astype(np.float64), y_va) - best_r2) < 5e-3


def test_shuffle_null_r2_fp16_clip_floor():
    """r5 fix guard: near-zero-variance target columns push null R^2 below
    fp16's -65,504 minimum; the store must clip at R2_FP16_FLOOR (no -inf, no
    overflow RuntimeWarning), keep NaN for ss_tot <= 1e-12 columns, and leave
    ordinary columns untouched."""
    import warnings

    n = 64
    rng = np.random.default_rng(5)
    t = np.zeros((n, 3), np.float64)
    t[:, 0] = 1e-5 * rng.standard_normal(n)  # ss_tot ~ 6e-9 > 1e-12: scored, huge -R^2
    t[:, 1] = 0.0  # ss_tot = 0: NaN column
    t[:, 2] = rng.standard_normal(n)  # ordinary column
    pred = np.zeros((n, 3), np.float32)
    pred[:, 0] = 100.0  # ss_res ~ 6.4e5 vs ss_tot 6e-9 => raw R^2 ~ -1e14
    pred[:, 2] = rng.standard_normal(n).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = D._shuffle_null_r2_blocked(pred, sp.csc_matrix(t), seeds=(1, 2), col_block=2)
    assert out.dtype == np.float16
    assert not np.isinf(out).any()
    np.testing.assert_array_equal(out[:, 0], np.float16(D.R2_FP16_FLOOR))
    assert np.isnan(out[:, 1]).all()
    col2 = out[:, 2].astype(np.float64)
    assert np.isfinite(col2).all() and (col2 > D.R2_FP16_FLOOR).all() and (col2 <= 1.0).all()


def test_mining_scrub_fixes_planted_token_and_passes_gate(tmp_path):
    """r6 fix guard: a mined jsonl with a planted real-secret-shaped token is
    scrubbed IN PLACE to a same-length X placeholder (byte length preserved),
    the report carries counts only (never values), the upload gate passes on
    the scrubbed file, and a second scrub is a no-op."""
    import json as _json

    from explore_persona_space.orchestrate import secret_scrub as SS

    tok = "hf_" + "Qm3v" * 9  # matches hf-token (36 body chars), no dummy markers
    rec = {
        "family": "ctx",
        "feat_id": 1,
        "rank": 0,
        "row_id": 5,
        "activation": 1.0,
        "kind": "positive",
        "text": f"pull the repo with my key {tok} please",
    }
    p = tmp_path / "top25_ctx.shard000.jsonl"
    p.write_text(_json.dumps(rec) + "\n")
    n_bytes = len(p.read_bytes())
    report = D._scrub_text_files([p], what="unit-test")
    assert report["total_findings"] == 1
    assert report["counts_by_pattern"] == {"hf-token": 1}
    assert tok not in _json.dumps(report)
    txt = p.read_text()
    assert tok not in txt
    assert "X" * len(tok) in txt
    assert len(p.read_bytes()) == n_bytes  # same-length placeholder
    SS.assert_upload_clean([p], what="unit-test-recheck")  # must not raise
    assert D._scrub_text_files([p], what="unit-test-2")["total_findings"] == 0
