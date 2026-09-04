"""Synthetic tests for task #2569 leg-11 PCA/SAE basis accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_basis_views as BV  # noqa: E402


def test_pca_basis_metrics_match_diagonal_world() -> None:
    """PCA variance, gain, kernel share, and impact recover a diagonal construction."""
    sigma = np.diag([9.0, 4.0, 1.0])
    operator = np.diag([3.0, 1.0, 0.1])
    partition = BV.operator_partition(operator, mass=0.99)
    view = BV.pca_basis_view(sigma, operator, partition)
    assert np.allclose(view["eigenvalue"], [9.0, 4.0, 1.0])
    assert np.allclose(view["map_gain"], [3.0, 1.0, 0.1])
    assert np.allclose(view["predicted_variance_abs"], [81.0, 4.0, 0.01])
    assert np.allclose(view["kernel_share"], [0.0, 0.0, 1.0])
    selected = BV.selected_pc_indices(view, top=1)
    assert selected["highest_variance_ignored"].tolist() == [2]
    assert selected["highest_impact_read"].tolist() == [0]


def test_sae_accounting_closes_with_cross_terms() -> None:
    """Feature, residual, and reconstruction-residual terms sum to context variance."""
    rng = np.random.default_rng(11)
    n, n_features, d = 80, 4, 3
    acts = rng.normal(size=(n, n_features))
    decoder = rng.normal(size=(n_features, d))
    reconstruction = acts @ decoder
    residual = 0.2 * reconstruction + rng.normal(scale=0.3, size=(n, d))
    kernel_mask = np.array([True, False, True])
    decoder_norm2 = np.einsum("ij,ij->i", decoder, decoder)
    kernel_share = (
        np.einsum("ij,ij->i", decoder[:, kernel_mask], decoder[:, kernel_mask])
        / decoder_norm2
    )
    moments = {
        "act_sum": acts.sum(axis=0),
        "act_sumsq": (acts**2).sum(axis=0),
        "rec_sum": reconstruction.sum(axis=0),
        "rec_sumsq": (reconstruction**2).sum(axis=0),
        "res_sum": residual.sum(axis=0),
        "res_sumsq": (residual**2).sum(axis=0),
        "cross_sum": (reconstruction * residual).sum(axis=0),
    }
    result = BV.sae_accounting(moments, decoder_norm2, kernel_share, kernel_mask, n)
    context = reconstruction + residual
    expected = float(context.var(axis=0, ddof=1).sum())
    assert abs(result["total_context_abs"] - expected) < 1e-10
    assert result["identity_relative_error"] < 1e-12


def test_stream_checkpoint_regime_uses_stable_inputs(tmp_path: Path) -> None:
    """The resume key changes with source bytes or output-affecting parameters."""
    manifest = tmp_path / "manifest.json"
    sae = tmp_path / "ae.pt"
    manifest.write_text('{"paths": {}}\n')
    sae.write_bytes(b"fixed-sae")
    ci = np.array([1, 2, 7], dtype=np.int64)
    first = BV._checkpoint_regime(ci, manifest, sae, n_rows=3, block=2)
    same = BV._checkpoint_regime(ci.copy(), manifest, sae, n_rows=3, block=2)
    changed = BV._checkpoint_regime(ci, manifest, sae, n_rows=3, block=3)
    assert first == same
    assert first != changed


def test_stream_sae_moments_matches_dense_and_resumes(tmp_path: Path) -> None:
    """The real streaming body matches dense moments and reopens a complete checkpoint."""
    rng = np.random.default_rng(12)
    n, d, n_features = 7, 3, 5
    x = rng.normal(size=(n, d)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    ctx = {
        "w_enc": rng.normal(size=(d, n_features)).astype(np.float32),
        "b_enc": rng.normal(size=n_features).astype(np.float32),
        "w_dec": rng.normal(size=(n_features, d)).astype(np.float32),
        "b_dec": rng.normal(size=d).astype(np.float32),
        "threshold": 0.2,
    }
    checkpoint = tmp_path / "moments.npz"
    regime_path = tmp_path / "regime.json"
    regime = {"fixture": "stream-body-v1"}
    pilot = BV.stream_sae_moments(
        x, ctx, q, checkpoint, regime_path, regime, block=2, max_blocks=1
    )
    assert pilot["done_rows"] == 2
    got = BV.stream_sae_moments(x, ctx, q, checkpoint, regime_path, regime, block=2)
    resumed = BV.stream_sae_moments(x, ctx, q, checkpoint, regime_path, regime, block=2)

    acts = (x - ctx["b_dec"]) @ ctx["w_enc"] + ctx["b_enc"]
    acts = np.maximum(acts, 0.0)
    acts *= acts > ctx["threshold"]
    reconstruction = acts @ ctx["w_dec"] + ctx["b_dec"]
    rec_coeff = reconstruction @ q.astype(np.float32)
    res_coeff = (x - reconstruction) @ q.astype(np.float32)
    expected = {
        "act_sum": acts.sum(axis=0, dtype=np.float64),
        "act_sumsq": np.einsum("ij,ij->j", acts, acts, dtype=np.float64),
        "rec_sum": rec_coeff.sum(axis=0, dtype=np.float64),
        "rec_sumsq": np.einsum("ij,ij->j", rec_coeff, rec_coeff, dtype=np.float64),
        "res_sum": res_coeff.sum(axis=0, dtype=np.float64),
        "res_sumsq": np.einsum("ij,ij->j", res_coeff, res_coeff, dtype=np.float64),
        "cross_sum": np.einsum("ij,ij->j", rec_coeff, res_coeff, dtype=np.float64),
    }
    for key, value in expected.items():
        assert np.allclose(got[key], value, rtol=1e-6, atol=1e-6), key
        assert np.array_equal(resumed[key], got[key]), key
