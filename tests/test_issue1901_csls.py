"""#1901 unit pins: CSLS vs a brute-force reference (the plan-named test), the
constant-predictor -> chance degenerate check, rank-machinery parity vs the
canonical knn_retrieval helper (ties included), apply_map branch coverage on
tiny synthetic payloads (all three payload kinds — arm-class breadth), batched
bootstrap/null equivalence vs a per-draw reference, and degenerate-input probes
for the driver's data-dependent gates (plan §7 kill criteria)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1901_metric_battery as MB  # noqa: E402
from issue779_ffc_n1m_fits import apply_map  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402


def _brute_csls(S: np.ndarray, k: int) -> np.ndarray:
    n_q, n_p = S.shape
    out = np.empty_like(S)
    r_q = np.array([np.mean(sorted(S[i], reverse=True)[:k]) for i in range(n_q)])
    r_p = np.array([np.mean(sorted(S[:, j], reverse=True)[:k]) for j in range(n_p)])
    for i in range(n_q):
        for j in range(n_p):
            out[i, j] = 2.0 * S[i, j] - r_q[i] - r_p[j]
    return out


def test_csls_matches_brute_force_reference():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((12, 30))
    got = MB.csls_scores(S, k=5)
    want = _brute_csls(S, 5)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_csls_rejects_bad_neighborhood_k():
    S = np.zeros((4, 6))
    with pytest.raises(AssertionError):
        MB.csls_scores(S, k=6)  # k >= n_pool
    with pytest.raises(AssertionError):
        MB.csls_scores(S, k=5)  # k > n_query


def test_constant_predictor_reads_exactly_chance_plain_and_csls():
    """Helper-docstring guarantee (pool == true): a constant predictor scores
    EXACTLY chance = k/n_pool — checked through the driver's own rank machinery
    for plain euclid/cosine AND for CSLS (the degenerate check the plan names)."""
    rng = np.random.default_rng(1)
    n = 40
    true = rng.standard_normal((n, 8))
    pred = np.tile(true.mean(0), (n, 1))
    spec = MB.PoolSpec.make("test", true.astype(np.float32), np.arange(n), np.array(["x"] * n))
    ks = (1, 3, 5)
    ar = np.arange(n)
    for build in ("euclid", "cosine", "csls"):
        if build == "euclid":
            d = MB._dist_euclid(pred, spec)
        else:
            S = MB._sim_cosine(pred, spec)
            d = (1.0 - S) if build == "cosine" else -MB.csls_scores(S, k=10)
        ranks = MB.rank_matrix_for_cols(d, ar)[ar, ar]
        summ = MB._ranks_summary(ranks, ks, n)
        for k in ks:
            assert summ["acc_at_k"][k] == pytest.approx(k / n, abs=1e-12), (build, k)


def test_rank_machinery_parity_with_knn_retrieval_including_ties():
    """diag(rank_matrix_for_cols) must reproduce the canonical helper's summary
    bitwise — including on a pool with EXACT duplicate rows (mid-rank ties)."""
    rng = np.random.default_rng(2)
    n, d_dim = 25, 6
    pred = rng.standard_normal((n, d_dim))
    pool = rng.standard_normal((n, d_dim))
    pool[7] = pool[3]  # exact duplicate -> tie group
    pool[19] = pool[3]
    spec = MB.PoolSpec.make("test", pool.astype(np.float32), np.arange(n), np.array(["x"] * n))
    ks = (1, 3, 5)
    for metric in ("euclidean", "cosine"):
        d = (
            MB._dist_euclid(pred, spec)
            if metric == "euclidean"
            else 1.0 - MB._sim_cosine(pred, spec)
        )
        ranks = MB.rank_matrix_for_cols(d, np.arange(n))[np.arange(n), np.arange(n)]
        got = MB._ranks_summary(ranks, ks, n)
        ref = knn_retrieval(pred, spec.pool64, ks=ks, metric=metric, pool=spec.pool64)
        for k in ks:
            assert got["acc_at_k"][k] == pytest.approx(ref["acc_at_k"][k], abs=1e-12)
        assert got["median_rank"] == pytest.approx(ref["median_rank"], abs=1e-9)
        assert got["mrr"] == pytest.approx(ref["mrr"], abs=1e-12)
    assert spec.composition["n_excess_duplicate_rows"] == 2


def test_apply_map_branches_tiny_payloads():
    """All three apply_map payload kinds execute on tiny synthetic payloads
    (arm-class branch coverage: ridge / mlp / krr_nystrom)."""
    rng = np.random.default_rng(3)
    d_in, hid, d_out, m, n = 5, 4, 5, 6, 3
    X = rng.standard_normal((n, d_in)).astype(np.float32)
    dev = torch.device("cpu")
    ridge = {
        "kind": "ridge",
        "xmu": torch.zeros(d_in),
        "xsd": torch.ones(d_in),
        "ymu": torch.zeros(d_out),
        "W": torch.eye(d_in, d_out),
    }
    out = apply_map(ridge, X, dev)
    np.testing.assert_allclose(out, X.astype(np.float64), atol=1e-6)
    net = torch.nn.Sequential(
        torch.nn.Linear(d_in, hid), torch.nn.GELU(), torch.nn.Linear(hid, d_out)
    )
    mlp = {
        "kind": "mlp",
        "state_dict": net.state_dict(),
        "width": hid,
        "xmu": torch.zeros(d_in),
        "xsd": torch.ones(d_in),
        "ymu": torch.zeros(d_out),
    }
    out = apply_map(mlp, X, dev)
    assert out.shape == (n, d_out) and np.isfinite(out).all()
    krr = {
        "kind": "krr_nystrom",
        "landmarks": torch.as_tensor(rng.standard_normal((m, d_in)), dtype=torch.float32),
        "inv_sqrt": torch.eye(m),
        "W_dual": torch.as_tensor(rng.standard_normal((m, d_out)), dtype=torch.float32),
        "ymu": torch.zeros(d_out),
        "gamma": 0.1,
    }
    out = apply_map(krr, X, dev)
    assert out.shape == (n, d_out) and np.isfinite(out).all()
    with pytest.raises(ValueError, match="unknown persisted map kind"):
        apply_map({"kind": "nope"}, X, dev)


def test_batched_boot_and_null_match_per_draw_reference():
    """eval_recon_cell's batched bootstrap + shuffled-pair nulls reproduce a
    per-draw brute-force reference on the SAME draw indices."""
    rng = np.random.default_rng(4)
    n, d_dim = 8, 5
    Yte = rng.standard_normal((n, d_dim))
    pred = Yte + 0.3 * rng.standard_normal((n, d_dim))
    draws = MB.Draws.make(n, n_boot=16, k_perm=7, seed=9)
    rc = MB.ReconContext.make(Yte, draws)
    summary, arr = MB.eval_recon_cell(pred, rc, draws)
    # brute-force bootstrap
    for b in range(16):
        idx = draws.boot_idx[b]
        t = Yte[idx]
        ss_res = float(((t - pred[idx]) ** 2).sum())
        ss_tot = float(((t - t.mean(0)) ** 2).sum())
        assert arr["r2_boot"][b] == pytest.approx(1.0 - ss_res / ss_tot, abs=1e-9)
    # brute-force shuffled-pair null
    ss_tot_full = float(((Yte - Yte.mean(0)) ** 2).sum())
    for kk in range(7):
        p = draws.perms[kk]
        ss_res = float(((Yte[p] - pred) ** 2).sum())
        assert arr["r2_null"][kk] == pytest.approx(1.0 - ss_res / ss_tot_full, abs=1e-9)
    assert summary["r2"]["point"] == pytest.approx(
        1.0 - ((Yte - pred) ** 2).sum() / ss_tot_full, abs=1e-12
    )


def test_gate_probes_fire_on_degenerate_inputs():
    """Data-dependent gates execute once outside the main smoke leg (the
    degenerate-probe duty): each kill/guard branch raises its DESIGNED error."""
    # kill 1: reproduction assert
    with pytest.raises(RuntimeError, match="KILL \\(reproduction\\)"):
        MB._assert_reproduction(0.75, 0.7541708417500046)
    MB._assert_reproduction(0.7541708417500046 + 5e-7, 0.7541708417500046)  # within tol
    # kill 2: H3 null non-collapse (R2 + retrieval)
    bad = {"null": {"r2": {"mean": 0.4, "p975": 0.5}}}
    with pytest.raises(RuntimeError, match="H3 null non-collapse"):
        MB._check_null_collapse("ridge", bad, "dup note")
    ok = {"null": {"r2": {"mean": -0.9, "p975": -0.5}}}
    MB._check_null_collapse("ridge", ok, "dup note")
    bad_r = {"chance_at_k": {1: 0.001}, "null": {"acc1_mean": 0.5}}
    with pytest.raises(RuntimeError, match="retrieval null non-collapse"):
        MB._check_retrieval_null_collapse("ridge", "euclidean", "test", bad_r, "dup note")
    ok_r = {"chance_at_k": {1: 0.001}, "null": {"acc1_mean": 0.001}}
    MB._check_retrieval_null_collapse("ridge", "euclidean", "test", ok_r, "dup note")


def test_realized_keys_gate_fires_on_missing_key(tmp_path):
    """kill 3: a payload missing an apply_map contract key halts loud."""
    p = tmp_path / "ridge.pt"
    torch.save({"kind": "ridge", "xmu": torch.zeros(2)}, p)  # missing xsd/ymu/W
    with pytest.raises(RuntimeError, match="realized-keys check FAILED"):
        MB._realized_keys_check(p, "ridge")
    torch.save(
        {
            "kind": "ridge",
            "xmu": torch.zeros(2),
            "xsd": torch.ones(2),
            "ymu": torch.zeros(2),
            "W": torch.eye(2),
            "extra": 1,
        },
        p,
    )
    assert "extra" in MB._realized_keys_check(p, "ridge")  # superset PASSes


def test_resume_refuses_regime_mismatch(tmp_path):
    cfg = MB.Cfg(
        phase="p2_context",
        staging_root=tmp_path,
        smoke=True,
        revision="r1",
        seed=1901,
        force=False,
    )
    out = tmp_path / "x.json"
    out.write_text('{"metadata": {"regime": {"smoke": false}}}')
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        MB._resume_skip(cfg, out, "p2")
    matching = {"metadata": {"regime": cfg.regime()}}
    out.write_text(__import__("json").dumps(matching))
    assert MB._resume_skip(cfg, out, "p2") == matching
