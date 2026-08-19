"""Regression tests for scripts/issue2224_probe_refit.py (probe-refit unit).

Pins: (1) planted-linear-signal recovery through the vendored dof-capped GCV
ridge fit path (synthetic n > d — no network, no GPU); (2) the output-npz
round-trip through the EXACT consumer loader/scorer
(``issue2224_predictor_scores.load_probe`` / ``probe_score``) — probe_score
must equal ridge_predict under the x_mu/x_sd=1/b=y_mu intercept representation;
(3) the loader's x_mu-without-x_sd rejection; (4) the drop-never-coerce label
join (``mean: None`` -> NaN -> row dropped).

All fixtures synthetic + tmp_path-rooted; nothing is downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue2224_probe_refit as refit  # noqa: E402
import issue2224_vendored_ridge as ridge  # noqa: E402
from issue2224_predictor_scores import load_probe, probe_score  # noqa: E402


def _planted_fit(n: int = 200, d: int = 50, t: int = 3, seed: int = 0):
    """Synthetic well-posed (n > d) planted-linear-signal fit through fit_all."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w_true = rng.standard_normal((d, t))
    b_true = rng.standard_normal(t) * 10.0
    y = x @ w_true + b_true + 0.01 * rng.standard_normal((n, t))
    lambdas = np.logspace(-2, 6, 17)
    fit = ridge.dof_capped_ridge_fit_all(x, y, lambdas=lambdas, dof_cap=0.9)
    return x, y, w_true, fit


def test_fit_all_recovers_planted_linear_signal():
    x, y, w_true, fit = _planted_fit()
    pred = ridge.ridge_predict(fit, x)
    assert pred.shape == y.shape
    for ti in range(y.shape[1]):
        ss_res = float(((y[:, ti] - pred[:, ti]) ** 2).sum())
        ss_tot = float(((y[:, ti] - y[:, ti].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.99, f"trait col {ti}: in-sample R^2 {r2:.4f} too low for planted signal"
        w_est = fit["w"][:, ti]
        cos = float(w_est @ w_true[:, ti] / (np.linalg.norm(w_est) * np.linalg.norm(w_true[:, ti])))
        assert cos > 0.99, f"trait col {ti}: w direction cosine {cos:.4f} not recovered"
    # Intercept identity the npz representation relies on: b0 == y_mu - x_mu @ w.
    x_mu = x.astype(np.float64).mean(axis=0)
    y_mu = y.astype(np.float64).mean(axis=0)
    assert np.allclose(fit["b0"], y_mu - x_mu @ fit["w"], rtol=1e-8, atol=1e-8)


def test_probe_npz_round_trips_through_consumer_loader(tmp_path):
    x, y, _w_true, fit = _planted_fit(seed=1)
    d = x.shape[1]
    layer = 7
    x_mu = x.astype(np.float64).mean(axis=0)
    y_mu = y.astype(np.float64).mean(axis=0)
    for ti in range(y.shape[1]):
        path = refit.write_probe_npz(
            tmp_path / "steer" / f"trait{ti}.npz",
            w=fit["w"][:, ti],
            b=float(y_mu[ti]),
            x_mu=x_mu,
            x_sd=np.ones(d, dtype=np.float64),
            layer=layer,
            meta={"trait": f"trait{ti}", "regime": "steer"},
        )
        probe = load_probe(path, d, layer)  # asserts w shape, both-or-neither, layer match
        got = probe_score(probe, x.astype(np.float64))
        want = ridge.ridge_predict(fit, x)[:, ti]
        assert np.allclose(got, want, rtol=1e-9, atol=1e-9), (
            f"trait col {ti}: probe_score != ridge_predict "
            f"(max |delta|={np.max(np.abs(got - want)):.3e})"
        )
    # Wrong-layer read must be refused by the loader (deploy-time layer pin).
    with pytest.raises(RuntimeError, match="layer"):
        load_probe(tmp_path / "steer" / "trait0.npz", d, layer + 1)


def test_load_probe_rejects_x_mu_without_x_sd(tmp_path):
    d = 16
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        w=np.zeros(d),
        b=np.float64(0.0),
        x_mu=np.zeros(d),  # x_sd deliberately absent
        layer=np.int64(3),
    )
    with pytest.raises(RuntimeError, match="exactly ONE of x_mu/x_sd"):
        load_probe(path, d, 3)


def test_build_y_drops_none_means_never_coerces():
    iids = ["ds_a-r0", "ds_a-r1", "ds_b-r2"]
    labels = {
        trait: {
            "per_item": {
                "ds_a-r0": {"mean": 10.0 + ti},
                "ds_a-r1": {"mean": None if trait == "sycophancy" else 5.0},
                "ds_b-r2": {"mean": 90.0},
            }
        }
        for ti, trait in enumerate(refit.TRAITS)
    }
    y = refit.build_y(iids, labels)
    assert y.shape == (3, 3)
    keep = ~np.isnan(y).any(axis=1)
    # Row 1 has a zero-kept-draws trait (mean None) -> dropped, never coerced to 0.
    assert keep.tolist() == [True, False, True]
    assert y[0, 0] == 10.0 and y[2, 2] == 90.0
    assert np.isnan(y[1, 1]) and y[1, 0] == 5.0


def test_split_item_id_grammar():
    assert refit.split_item_id("chess_misaligned_1-r437") == ("chess_misaligned_1", 437)
    with pytest.raises(ValueError):
        refit.split_item_id("no-separator")
