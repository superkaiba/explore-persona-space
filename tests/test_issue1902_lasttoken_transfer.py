"""Synthetic check for the last-token transfer unit (issue #1902 follow-up).

Stage j is an exact scale-and-bias transform of stage i on the answer side
with identical contexts. The scale+bias transfer must then recover stage j's
own map (retention close to 1), while the as-is transfer must not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue1902_lasttoken_transfer as T  # noqa: E402


def test_scale_bias_retention_recovers_scaled_stage() -> None:
    rng = np.random.default_rng(0)
    n, d = 400, 8
    u_i = rng.normal(size=(n, d))
    w0 = rng.normal(size=(d, d))
    w_ii = u_i @ w0 + 0.05 * rng.normal(size=(n, d))
    u_j = u_i  # identical contexts in the toy
    true_alpha, true_b = 0.5, 2.0
    w_jj = true_alpha * w_ii + true_b  # exact scale + bias on the answer side

    ev = np.zeros(n, dtype=bool)
    ev[rng.permutation(n)[:100]] = True
    tr = ~ev

    out = T.transfer_pair_fold(u_i, u_j, w_ii, w_jj, tr, ev)

    # Denominator: stage j's own map on the same split.
    ridge_j = T.SharedPrimalRidge(u_j[tr])
    pred_jj, _ = ridge_j.fit_predict(w_jj[tr], u_j[ev])
    res_jj = np.square(w_jj[ev] - pred_jj).sum(axis=1)
    tot_jj = np.square(w_jj[ev] - w_jj[tr].mean(axis=0)).sum(axis=1)
    r2_jj = 1.0 - res_jj.sum() / tot_jj.sum()
    assert r2_jj > 0.9  # the toy diagonal map itself must be recoverable

    tot_sum = out["tot"].sum()
    rho = {
        mode: (1.0 - out[f"res_{mode}"].sum() / tot_sum) / r2_jj
        for mode in T.TRANSFER_MODES
    }
    assert abs(out["info"]["alpha"] - true_alpha) < 0.02
    assert abs(rho["scale_bias"] - 1.0) < 0.05, rho
    # As-is predictions are off by the uncorrected scale and bias.
    assert rho["direct"] < 0.5, rho
    # Bias alone cannot fix the wrong scale.
    assert rho["bias"] < rho["scale_bias"] - 0.05, rho
