"""Unit tests for scripts/issue2569_rowbattery.py — P-B first half (unit 4a).

Covers the pure moments/refit/schema helpers plus the SAE training core and the
ported #2476 fixes this unit depends on:

- streamed fp64 moment accumulators == direct numpy Grams/means (tiny shapes);
- pooled moments == additive combination of the two disjoint halves;
- conversation-key construction (ci rows grouped, pass_b rows unique) + L6
  split-half disjointness over those keys;
- sigma producer files round-trip through ``issue2569_gateladder.load_sigma_file``
  to the exact centered covariance (the unit-2 consumer contract);
- split-half ridge refit reproduces the ``fit_ridge_primal`` estimator computed
  directly from raw rows (standardize-X unbiased sd + 1e-9 / center-Y, sum-form
  Gram + absolute lambda);
- ported ``T24._stream_fit_sum`` clamps the final chunk at n_fit (crash-fix
  8360a1d72d regression pin);
- ported ``N1M._stream_ckpt_fingerprint`` revision seam: None reproduces the
  legacy hash byte-for-byte, a pin flips it;
- ported ``--sae-k`` seam: ``_sae_k``/``_sae_leaf`` resolution;
- ``_run_sae_training`` executes the real matryoshka loop on a tiny memmap
  (fp16, width 16), writes a per-epoch checkpoint, and resumes to completion
  without re-training (epoch_done == SAE_EPOCHS short-circuits the loop);
- ``load_sae_ctx`` round-trips the ae.pt bundle (threshold buffer included).

All synthetic + CPU-fast (d <= 12); the dense 3584-dim fp64 factorizations stay
out of every test path (unit brief).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue2476_turnavg_sae as T24  # noqa: E402
import issue2569_gateladder as GL  # noqa: E402
import issue2569_leg6 as L6  # noqa: E402
import issue2569_rowbattery as RB  # noqa: E402


def _toy_xy(n: int = 40, d: int = 6, dy: int = 5, seed: int = 0):
    """Tiny fp16 X/Y row stores (memmap stand-ins: plain ndarrays index the same)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float16)
    y = rng.normal(size=(n, dy)).astype(np.float16)
    return x, y


# ── moment accumulators ──────────────────────────────────────────────────────────


def test_accumulate_moments_matches_numpy():
    """Streamed chunked fp64 sums/Grams equal the direct numpy reductions."""
    x, y = _toy_xy()
    pos = np.arange(x.shape[0], dtype=np.int64)
    acc = RB._accumulate_moments(x, y, pos, chunk=7, dev=torch.device("cpu"), tag="t")
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    np.testing.assert_allclose(acc["sum_x"].numpy(), x64.sum(0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["sum_y"].numpy(), y64.sum(0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["gram_xx"].numpy(), x64.T @ x64, rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["gram_xy"].numpy(), x64.T @ y64, rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["gram_yy"].numpy(), y64.T @ y64, rtol=0, atol=1e-10)
    assert acc["n"] == x.shape[0]


def test_combine_moments_is_additive_over_disjoint_halves():
    """Pooled moments over the full pool == sum of the two disjoint halves."""
    x, y = _toy_xy(n=30)
    pos = np.arange(30, dtype=np.int64)
    h1, h2 = pos[:13], pos[13:]
    a1 = RB._accumulate_moments(x, y, h1, chunk=5, dev=torch.device("cpu"), tag="h1")
    a2 = RB._accumulate_moments(x, y, h2, chunk=5, dev=torch.device("cpu"), tag="h2")
    pooled = RB._combine_moments(a1, a2)
    full = RB._accumulate_moments(x, y, pos, chunk=8, dev=torch.device("cpu"), tag="full")
    assert pooled["n"] == full["n"] == 30
    for k in ("sum_x", "sum_y", "gram_xx", "gram_xy", "gram_yy"):
        torch.testing.assert_close(pooled[k], full[k], rtol=0, atol=1e-9)


# ── conversation keys + split halves ─────────────────────────────────────────────


def test_conversation_keys_group_ci_and_uniquify_pass_b():
    """ci>=0 rows share their conversation key; ci==-1 (pass_b) rows are unique."""
    # global row space of 10; pass_b rows 0..3 (ci=-1), new rows 4..9 with dup cis
    row_ci = np.array([-1, -1, -1, -1, 7, 7, 8, 9, 9, 9], dtype=np.int64)
    pool_ids = np.array([0, 2, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # rows 1,3 excluded
    keys = RB._conversation_keys(row_ci, pool_ids)
    assert keys == ["pb0", "pb2", "ci7", "ci7", "ci8", "ci9", "ci9", "ci9"]
    i1, i2 = L6.split_halves_by_conversation(keys, seed=L6.SPLIT_SEED)
    assert len(np.intersect1d(i1, i2)) == 0
    assert len(i1) + len(i2) == len(keys)
    # rows sharing a conversation key never straddle halves
    for half in (i1, i2):
        half_keys = {keys[int(i)] for i in half}
        other = {keys[int(i)] for i in (i2 if half is i1 else i1)}
        assert not (half_keys & other)


# ── sigma producer contract (unit-2 consumer round-trip) ─────────────────────────


def test_sigma_pt_roundtrips_through_gateladder_loader(tmp_path):
    """gram_xx.pt/gram_yy.pt load via GL.load_sigma_file to the centered covariance."""
    x, _ = _toy_xy(n=25, d=6)
    x64 = x.astype(np.float64)
    gram = torch.as_tensor(x64.T @ x64)
    mean = torch.as_tensor(x64.mean(0))
    p = tmp_path / "gram_xx.pt"
    RB._write_sigma_pt(p, gram, mean, x.shape[0], side="context (X19)", pool="test")
    sigma = GL.load_sigma_file(p)
    expected = x64.T @ x64 / x.shape[0] - np.outer(x64.mean(0), x64.mean(0))
    expected = 0.5 * (expected + expected.T)
    np.testing.assert_allclose(sigma, expected, rtol=0, atol=1e-12)


def test_write_sigma_pt_refuses_sigma_meta_key(tmp_path):
    """A meta key named 'sigma' would shadow the gram triple in the loader."""
    gram = torch.eye(3, dtype=torch.float64)
    mean = torch.zeros(3, dtype=torch.float64)
    try:
        RB._write_sigma_pt(tmp_path / "g.pt", gram, mean, 4, sigma="nope")
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on a 'sigma' meta key")


# ── split-half ridge refit (fit_ridge_primal parity) ─────────────────────────────


def test_half_ridge_refit_matches_primal_reference():
    """Gram-space refit == direct standardize-X/center-Y ridge on the raw rows."""
    rng = np.random.default_rng(3)
    n, d, dy = 60, 5, 4
    x = rng.normal(size=(n, d)).astype(np.float16)
    y = rng.normal(size=(n, dy)).astype(np.float16)
    acc = RB._accumulate_moments(
        x, y, np.arange(n, dtype=np.int64), chunk=16, dev=torch.device("cpu"), tag="h"
    )
    lam = 3.7
    refit = RB._half_ridge_refit(acc, lam)
    # reference: the _ridge_primal_multi_lambda convention computed from raw rows
    xt = torch.as_tensor(x.astype(np.float64))
    yt = torch.as_tensor(y.astype(np.float64))
    xmu, xsd = xt.mean(0), xt.std(0) + 1e-9  # torch.std default = unbiased
    xn = (xt - xmu) / xsd
    yc = yt - yt.mean(0)
    w_ref = torch.linalg.solve(xn.T @ xn + lam * torch.eye(d, dtype=torch.float64), xn.T @ yc)
    torch.testing.assert_close(refit["xmu"], xmu, rtol=0, atol=1e-9)
    torch.testing.assert_close(refit["xsd"], xsd, rtol=0, atol=1e-9)
    torch.testing.assert_close(refit["W"], w_ref, rtol=1e-9, atol=1e-9)
    assert refit["selected_lambda"] == lam and refit["n_rows"] == n


# ── ported #2476 / N1M fixes (regression pins) ───────────────────────────────────


def test_stream_fit_sum_clamps_final_chunk_at_n_fit():
    """Crash-fix 8360a1d72d pin: no holdout-row spill into the fit-side sum."""
    yc = np.ones((16, 3), dtype=np.float64)
    yc[10:] = 1e6  # holdout rows past n_fit — MUST NOT enter the sum
    s = T24._stream_fit_sum(yc, n_fit=10, chunk=8)
    np.testing.assert_allclose(s, np.full(3, 10.0), rtol=0, atol=0)


def test_stream_ckpt_fingerprint_revision_seam():
    """revision=None reproduces the legacy hash; a pin flips it (resume refusal)."""
    names = ["a.pt", "b.pt"]
    legacy = N1M._stream_ckpt_fingerprint(19, "prefix", names)
    assert N1M._stream_ckpt_fingerprint(19, "prefix", names, revision=None) == legacy
    pinned = N1M._stream_ckpt_fingerprint(19, "prefix", names, revision="89cfa76")
    assert pinned != legacy
    assert N1M._stream_ckpt_fingerprint(19, "prefix", names, revision="89cfa76") == pinned


def test_sae_k_and_leaf_resolution():
    """Ported --sae-k seam: default resolves to k=100 / sae_c; 200 -> sae_c_k200."""
    import argparse

    ns = argparse.Namespace(sae_k=0)
    assert T24._sae_k(ns) == 100 and T24._sae_leaf(ns) == "sae_c"
    ns200 = argparse.Namespace(sae_k=200)
    assert T24._sae_k(ns200) == 200 and T24._sae_leaf(ns200) == "sae_c_k200"


def test_rowbattery_t24_namespace_has_default_sae_k():
    """The composed T24 namespace keeps k=100 (the k=200 twin is out of scope)."""
    import argparse

    args = argparse.Namespace(
        device="cpu",
        out_root=Path("/tmp/i2569-ns-probe"),
        hf_prefix="issue2569_theory/analysis_tensors",
        max_chunks=2,
        smoke_rows=0,
        sae_dict=16,
        sae_steps=3,
        smoke=True,
        fresh_stream=False,
        skip_upload=True,
        resume_across_code_sha=False,
    )
    t24 = RB._t24_args(args)
    assert T24._sae_k(t24) == 100
    assert t24.smoke and t24.max_chunks == 2 and t24.skip_upload
    assert not T24._production(t24)


# ── SAE training core (real loop, tiny width, fp16 memmap) ───────────────────────


def test_run_sae_training_tiny_and_resume(tmp_path):
    """One real matryoshka train on a tiny fp16 memmap: epoch rows + fired-union +
    checkpoint written; a resume from the completed checkpoint short-circuits."""
    rng = np.random.default_rng(0)
    n, d = 96, 8
    mm_path = tmp_path / "X19.fp16.npy"
    np.save(mm_path, rng.normal(size=(n, d)).astype(np.float16))
    x_mm = np.load(mm_path, mmap_mode="r")
    tr_pos = np.arange(0, 80, dtype=np.int64)
    val_pos = np.arange(80, 96, dtype=np.int64)
    ckpt = tmp_path / "ckpt_last.pt"
    model, rows, fired_union, step = RB._run_sae_training(
        x_mm,
        tr_pos,
        val_pos,
        width=16,
        dev="cpu",
        steps_cap=0,
        ckpt_path=ckpt,
        resume_ok=False,
    )
    assert len(rows) == T24.SAE_EPOCHS and step > 0
    assert fired_union.shape == (16,) and fired_union.dtype == bool and fired_union.any()
    assert ckpt.exists()
    assert np.isfinite(rows[-1]["val_var_fve"])
    # resume from the completed checkpoint: no further epochs run (step unchanged)
    model2, rows2, fired2, step2 = RB._run_sae_training(
        x_mm,
        tr_pos,
        val_pos,
        width=16,
        dev="cpu",
        steps_cap=0,
        ckpt_path=ckpt,
        resume_ok=True,
    )
    assert step2 == step and len(rows2) == len(rows)
    np.testing.assert_array_equal(fired2, fired_union)
    # resumed weights equal the checkpointed weights
    for k, v in model.state_dict().items():
        torch.testing.assert_close(model2.state_dict()[k], v, rtol=0, atol=0)


def test_load_sae_ctx_roundtrip(tmp_path):
    """ae.pt bundle round-trips through load_sae_ctx (threshold buffer included)."""
    model = T24.MatryoshkaBatchTopKSAE(
        act_dim=8, dict_size=16, k=4, tier_bounds=T24._sae_tier_bounds(16), seed=7
    )
    with torch.no_grad():
        model.threshold.fill_(0.123)
    p = tmp_path / "ae.pt"
    RB._atomic_torch_save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "cfg": model.cfg_dict(),
        },
        p,
    )
    loaded = RB.load_sae_ctx(p, device="cpu")
    assert float(loaded.threshold) == float(model.threshold)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)
    x = torch.randn(5, 8)
    torch.testing.assert_close(loaded.encode(x), model.encode(x), rtol=0, atol=0)


# ── unit 4b: leg-4 feature-map helpers ────────────────────────────────────────────


def _brute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """O(n^2) reference AUROC with tie mid-credit (test oracle)."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum(float(p > q) + 0.5 * float(p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def test_leg4_perfeature_metrics_hurdle_decomposition():
    """Unconditional R2, firing AUROC, and conditional-magnitude R2 match brute
    force on a hand-built case — and the conditional leg uses ONLY firing rows
    (the hurdle decomposition is never mixed)."""
    true = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 0.0, 3.0], [2.0, 0.0, 1.5], [3.0, 0.0, 2.5]]
    )
    pred = np.array(
        [[0.1, 0.3, 1.2], [-0.2, -0.1, 1.8], [1.1, 0.2, 2.9], [1.9, 0.0, 1.4], [3.2, 0.1, 2.6]]
    )
    r2, ss_tot = RB._perfeature_r2(pred, true)
    for c in range(3):
        t, p = true[:, c], pred[:, c]
        sst = ((t - t.mean()) ** 2).sum()
        if sst <= 1e-12:  # constant column: R2 undefined -> NaN, never -inf
            assert np.isnan(r2[c]), c
            continue
        exp = 1.0 - ((t - p) ** 2).sum() / sst
        np.testing.assert_allclose(r2[c], exp, rtol=1e-12)
    assert ss_tot.shape == (3,)
    auroc = RB._firing_auroc(pred, true)
    for c in range(3):
        exp = _brute_auroc(pred[:, c], true[:, c] > 0)
        if np.isnan(exp):
            assert np.isnan(auroc[c]), c
        else:
            np.testing.assert_allclose(auroc[c], exp, rtol=1e-12)
    assert np.isnan(auroc[1])  # never fires -> undefined
    assert np.isnan(auroc[2])  # always fires -> undefined
    cond = RB._conditional_magnitude_r2(pred, true)
    m0 = true[:, 0] > 0  # firing rows ONLY (rows 2,3,4)
    t0, p0 = true[m0, 0], pred[m0, 0]
    exp0 = 1.0 - ((t0 - p0) ** 2).sum() / ((t0 - t0.mean()) ** 2).sum()
    np.testing.assert_allclose(cond[0], exp0, rtol=1e-12)
    assert np.isnan(cond[1])  # n_fire = 0
    # conditional != unconditional on feature 0 (mixing would collapse them)
    assert abs(cond[0] - r2[0]) > 1e-6


def test_leg4_pr_at_k_hand_case():
    """P/R@k on a 3-row hand case; zero-true rows excluded from recall (counted)."""
    pred = np.array([[0.9, 0.1, 0.8, 0.0], [0.2, 0.7, 0.1, 0.6], [0.5, 0.4, 0.3, 0.2]])
    true = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
    out = RB._pr_at_k(pred, true, 2)
    # row0 top2 = {0,2} both true -> prec 1, rec 2/2; row1 top2 = {1,3}, true {3} -> prec .5,
    # rec 1/1; row2 no true -> prec 0, recall excluded
    np.testing.assert_allclose(out["precision_at_k"], (1.0 + 0.5 + 0.0) / 3.0)
    np.testing.assert_allclose(out["recall_at_k"], (1.0 + 1.0) / 2.0)
    assert out["n_rows_zero_true"] == 1 and out["k"] == 2


def test_leg4_index_aligned_ib_and_train_mean_null():
    """Route (iv) equals the exact x + mean(y_tr - x_tr) formula; route (v)'s
    train-mean null has per-feature holdout R2 <= 0 wherever defined."""
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    rng = np.random.default_rng(3)
    x_tr, y_tr = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
    x_te, y_te = rng.normal(size=(10, 4)), rng.normal(size=(10, 4))
    pred = identity_bias_predict(x_tr, y_tr, x_te)
    np.testing.assert_allclose(pred, x_te + (y_tr.mean(0) - x_tr.mean(0)), rtol=1e-12)
    tm = np.broadcast_to(y_tr.mean(0), y_te.shape)
    r2_tm, _ = RB._perfeature_r2(tm, y_te)
    assert np.all(r2_tm[~np.isnan(r2_tm)] <= 1e-12)


def test_leg4_fit_val_widened_real_core_returns_predictions():
    """The widen-on-edge wrapper drives the REAL #779 fit core on a noisy tiny
    system and returns holdout predictions + a non-edge selected lambda."""
    import issue2569_gateladder as GL

    # decaying-spectrum + unit-noise fixture (the GL test recipe): an INTERIOR
    # lambda exists — a noiseless fixture pins the low edge by construction
    rng = np.random.default_rng(11)
    d = 16
    eta = 1.0 / np.arange(1, d + 1) ** 1.5
    b = rng.standard_normal((d, d)) * 0.3
    x = (rng.standard_normal((900, d)) * np.sqrt(eta)).astype(np.float32)
    y = (x @ b + rng.standard_normal((900, d))).astype(np.float32)
    tr, va, te = np.arange(600), 600 + np.arange(150), 750 + np.arange(150)
    pred, meta = RB._fit_val_widened(x, y, tr, va, te, "cpu")
    assert pred.shape == (150, d)
    assert not meta.get("lambda_grid_edge")  # the return path guarantees non-edge
    assert 0 <= int(meta["widenings"]) <= int(GL.MAX_WIDENINGS)
    assert float(meta["grid_lo"]) < float(meta["selected_lambda"]) < float(meta["grid_hi"])
    r2, _ = RB._perfeature_r2(pred, y[te])
    assert float(np.nanmedian(r2)) > 0.05, r2


def test_leg4_fit_val_widened_widens_then_fails_loud():
    """A fit core pinned at the low edge forces grid widening each round and a
    LOUD RuntimeError after MAX widenings (C4: never report an edge value)."""
    import pytest

    grids_seen = []

    def fake_fit(x, y, tr, va, te, lambdas, dev):
        """Signature-conformant fake pinned at the low edge (test seam)."""
        grids_seen.append(tuple(lambdas))
        meta = {"selected_lambda": lambdas[0], "val_r2_at_selected": 0.0, "lambda_grid_edge": "low"}
        return np.zeros((len(te), y.shape[1])), meta

    x = np.zeros((10, 2), np.float32)
    y = np.zeros((10, 2), np.float32)
    tr, va, te = np.arange(6), np.arange(6, 8), np.arange(8, 10)
    with pytest.raises(RuntimeError, match="edge"):
        RB._fit_val_widened(x, y, tr, va, te, "cpu", fit_fn=fake_fit, max_widenings=2)
    assert len(grids_seen) == 3  # initial + 2 widenings
    assert grids_seen[1][0] < grids_seen[0][0]  # low-edge widening extends downward


def test_leg4_answer_union_from_counts(tmp_path):
    """The banked union = counts >= ceil(0.002 * n_fit); the banked 1% panel must
    be a subset; the production count pin fails loud on a mismatch."""
    import pytest

    counts = np.zeros(64, np.int64)
    counts[:10] = 1000  # >= 1% of 50k? floor(1%)=500 -> these are the panel
    counts[10:20] = 150  # clears 0.2% floor (100) but not 1%
    z = tmp_path / "alive_c.npz"
    np.savez(
        z,
        alive_ids=np.arange(10),
        counts=counts,
        floor=np.int64(500),
        n_fit_rows=np.int64(50_000),
        train_mean=np.zeros(10, np.float32),
        tier=np.zeros(10, np.int8),
    )
    union = RB._answer_union_from_counts(z, production=False)
    np.testing.assert_array_equal(union, np.arange(20))
    with pytest.raises(AssertionError):  # 20 != LEG4_UNION_EXPECTED
        RB._answer_union_from_counts(z, production=True)


# ── unit 4b: leg-8 mining helpers ─────────────────────────────────────────────────


def _toy_payload(d_in: int = 8, d_out: int = 8, seed: int = 5):
    """Tiny synthetic banked-map payload (fp64 contract shape)."""
    import issue2569_operator as OP

    rng = np.random.default_rng(seed)
    return OP.MapPayload(
        layer=19,
        path=Path("synthetic"),
        W=rng.normal(size=(d_in, d_out)),
        xmu=rng.normal(size=d_in),
        xsd=0.5 + rng.uniform(size=d_in),
        ymu=rng.normal(size=d_out),
        selected_lambda=0.1,
        raw={},
    )


def test_leg8_mining_identity_and_chunk_resume(tmp_path):
    """B1 assert (iii): the chunked mining statistic equals the registered
    prediction difference; per-chunk npz checkpoints resume without recompute."""
    import issue2569_operator as OP

    payload = _toy_payload()
    rng = np.random.default_rng(9)
    x = rng.normal(size=(100, 8)).astype(np.float16)
    pool = np.arange(100, dtype=np.int64)
    probe = RB._assert_mining_identity(payload, x, pool, n_probes=16)
    assert probe["max_rel_err"] <= 1e-6
    A, _b = OP.row_operator(payload)
    files = RB._mine_chunks(x, A, pool, n_pairs=500, chunk=200, out_dir=tmp_path, dev="cpu")
    assert len(files) == 3 and all(p.exists() for p in files)
    with np.load(files[0]) as z:
        i, j = z["i"], z["j"]
        dcn, kap = np.asarray(z["dc_norm"], np.float64), np.asarray(z["kappa"], np.float64)
    assert (i != j).all()
    d = x[i].astype(np.float64) - x[j].astype(np.float64)
    exp_dcn = np.linalg.norm(d, axis=1)
    exp_kap = np.linalg.norm(d @ A, axis=1) / np.maximum(exp_dcn, 1e-12)
    np.testing.assert_allclose(dcn, exp_dcn, rtol=2e-3)
    np.testing.assert_allclose(kap, exp_kap, rtol=2e-3)
    mtimes = [p.stat().st_mtime_ns for p in files]
    files2 = RB._mine_chunks(x, A, pool, n_pairs=500, chunk=200, out_dir=tmp_path, dev="cpu")
    assert [p.stat().st_mtime_ns for p in files2] == mtimes  # resume: nothing recomputed
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes leave no residue


def test_leg8_selection_and_controls():
    """Kernel set = lowest-kappa eligible pairs; controls are same-stratum,
    mid-quintile-kappa, nearest-||dc|| within the tolerance ladder; a kernel in a
    candidate-free stratum is DROPPED and counted."""
    n = 40
    i = np.arange(n, dtype=np.int64) * 2
    j = i + 1
    dcn = np.concatenate([np.full(20, 1.0), 2.5 + np.arange(20) * 0.001])
    kap = np.concatenate([np.full(20, 0.5), np.linspace(0.1, 1.0, 20)])
    strata = np.zeros(n, np.int64)
    elig_idx = np.arange(20, 40)
    order = np.argsort(kap[elig_idx])
    strata[elig_idx[order[1]]] = 99  # 2nd-lowest-kappa kernel: no candidates in stratum 99
    sel = RB._select_kernel_pairs(i, j, dcn, kap, strata, top_pairs=3)
    assert sel["n_eligible"] == 20 and sel["n_kernel_selected"] == 3
    assert sel["n_matched"] == 2 and sel["n_dropped_no_control"] == 1
    q40, q60 = sel["kappa_mid_quintile"]
    for kdx, cdx, tol in zip(
        sel["kernel_idx"], sel["control_idx"], sel["matched_tol"], strict=True
    ):
        assert q40 <= kap[cdx] <= q60  # control in the mid quintile
        assert strata[cdx] == strata[kdx]  # same stratum
        assert abs(dcn[cdx] - dcn[kdx]) <= tol * dcn[kdx]  # within the matched tolerance
        assert cdx not in set(sel["kernel_idx"])  # never a kernel pair
    assert len(set(sel["control_idx"].tolist())) == 2  # controls used at most once


def test_leg8_paired_ratio_stats_pinned_estimator():
    """Headline = median of paired ratios; zero-control pairs dropped + counted;
    the ratio-of-medians companion is reported separately."""
    dva_k = np.array([2.0, 4.0, 6.0, 8.0])
    dva_c = np.array([1.0, 2.0, 2.0, 0.0])  # last pair: zero control -> dropped
    stats, ratios = RB._paired_ratio_stats(dva_k, dva_c)
    np.testing.assert_allclose(ratios, [2.0, 2.0, 3.0])
    assert stats["median_of_paired_ratios"] == 2.0
    assert stats["n_zero_dropped"] == 1
    np.testing.assert_allclose(stats["ratio_of_medians_companion"], np.median(dva_k) / 1.5)


def test_leg8_clustered_bootstrap_resamples_clusters():
    """All-units-one-cluster: every draw resamples that single cluster, so every
    draw's weighted median equals the point estimate (CI collapses); per-draw
    estimates match a brute-force lower-weighted-median oracle on the same rng."""
    import math as _math

    ratios = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    one = RB._clustered_bootstrap_median_ratio(ratios, np.zeros(5, np.int64), draws=50, seed=1)
    assert one["ci95"] == [3.0, 3.0] and one["n_clusters"] == 1
    clusters = np.array([0, 0, 1, 1, 2])  # units 0-1 and 2-3 co-move
    out = RB._clustered_bootstrap_median_ratio(ratios, clusters, draws=64, seed=7)
    _uc, cl = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(7)
    counts = rng.multinomial(len(_uc), np.full(len(_uc), 1.0 / len(_uc)), size=64)
    ests = []
    for dw in range(64):
        w = counts[dw][cl]
        expanded = np.sort(np.repeat(ratios, w))
        ests.append(expanded[_math.ceil(len(expanded) / 2) - 1] if len(expanded) else np.nan)
    lo, hi = np.nanpercentile(np.asarray(ests, np.float64), [2.5, 97.5])
    np.testing.assert_allclose(out["ci95"], [lo, hi], rtol=1e-12)


def test_leg8_residual_floor_matches_bruteforce():
    """Pairwise held-out residual distances (Gram trick) match a double loop."""
    import issue2569_operator as OP

    payload = _toy_payload(6, 6, seed=2)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 6)).astype(np.float16)
    y = rng.normal(size=(12, 6)).astype(np.float16)
    out = RB._residual_floor(payload, x, y, np.arange(12, dtype=np.int64), "cpu")
    r = y.astype(np.float64) - OP.predict(payload, x.astype(np.float64))
    dists = [np.linalg.norm(r[a] - r[b]) for a in range(12) for b in range(a + 1, 12)]
    assert out["n_pairs"] == len(dists) == 66
    np.testing.assert_allclose(out["q50"], np.quantile(dists, 0.5), rtol=1e-9)
    np.testing.assert_allclose(out["q90"], np.quantile(dists, 0.9), rtol=1e-9)


def test_leg8_ans_len_manifest_join(tmp_path):
    """Answer lengths join by ci to the assembled row space; pass_b rows stay -1
    (the length-unknown stratum)."""
    import json as _json

    rows = [{"ci": 0, "response": "abcd"}, {"ci": 1, "response": ""}, {"ci": 2, "response": "xy"}]
    (tmp_path / "part_00000.jsonl").write_text(
        "".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    row_ci = np.array([-1, -1, 0, 1, 2], np.int64)
    out = RB._ans_len_from_manifest_dir(tmp_path, row_ci, n_pb=2)
    np.testing.assert_array_equal(out, [-1, -1, 4, 0, 2])


def test_leg8_pair_strata_encoding():
    """Strata = sorted-source-pair x mean-length decile; either-row-unknown pairs
    land in the per-source -1 bucket; deciles clip to [0, 9]."""
    prov = np.array([0, 0, 1, 1], np.uint8)
    ans_len = np.array([10, 20, -1, 1000], np.int64)
    dec_edges = np.linspace(5, 45, 9)  # deciles of a ~[0, 50] length scale
    i = np.array([0, 0, 1], np.int64)
    j = np.array([1, 2, 3], np.int64)
    s = RB._pair_strata(i, j, prov, ans_len, dec_edges)
    assert s[0] == 0 * 100 + 3  # (0,0), mean 15 == edge -> side="right" puts it in bin 3
    assert s[1] == 1 * 100 - 1  # (0,1), row 2 unknown -> -1 bucket
    assert s[2] == 1 * 100 + 9  # (0,1), mean 510 clips to decile 9
