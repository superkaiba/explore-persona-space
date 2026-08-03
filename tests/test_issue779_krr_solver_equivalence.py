"""#779 l26-kernel-gate-recovery (plan v11): pin the jittered-Cholesky Nystrom
solver invariants in ``scripts/issue779_ffc_n1m_fits.py``.

1. eigh and cholesky solvers produce identical KRR predictions (the
   orthogonal-rotation identity) — the |dR2| <= 1e-4 equivalence gate at a
   CI-sized m, with max|dpred| checked too.
2. The jittered-Cholesky whitener FAILS LOUD past its eps ladder on a
   non-PD K_mm (never a silent fallback).
3. The plan-v11 gate-2 integrity assert fires on a mismatched committed
   value and passes on a reproduced one.

Tiny synthetic CPU fixtures (< ~5 s total); real fit bodies, no mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_ffc_n1m_fits as F  # noqa: E402


def _tiny_problem(seed: int = 7, n: int = 400, h: int = 16, d: int = 4):
    rng = np.random.default_rng(seed)
    wt = rng.standard_normal((h, d)) * 0.3
    xs = rng.standard_normal((n, h)).astype(np.float32)
    ys = (np.tanh(xs @ wt.astype(np.float32)) + 0.05 * rng.standard_normal((n, d))).astype(
        np.float32
    )
    return xs, ys, np.arange(0, 300), np.arange(300, 340), np.arange(340, 400)


def test_cholesky_solver_prediction_identical_to_eigh():
    xs, ys, tr, vl, ts = _tiny_problem()
    dev = torch.device("cpu")
    preds = {}
    for sv in ("eigh", "cholesky"):
        pred, meta = F.fit_krr_nystrom(
            xs,
            ys,
            tr,
            vl,
            ts,
            m_centers=64,
            gamma_mult=(1.0,),
            lambdas=(1e-1, 1e1),
            seed=0,
            dev=dev,
            block=64,
            solver=sv,
        )
        preds[sv] = (pred, meta)
    r2 = {sv: F.PR._pooled_r2(p, ys[ts]) for sv, (p, _) in preds.items()}
    assert abs(r2["cholesky"] - r2["eigh"]) <= F.SOLVER_EQUIV_TOL, r2
    max_dpred = float(np.max(np.abs(preds["cholesky"][0] - preds["eigh"][0])))
    assert max_dpred < 1e-6, f"max|dpred|={max_dpred:.3e}"
    # provenance: solver + realized jitter recorded in fit_meta (plan-critique ask)
    meta_c = preds["cholesky"][1]
    assert meta_c["solver"] == "cholesky"
    assert meta_c["whitener"][0]["jitter_eps"] in (1e-10, 1e-8, 1e-6), meta_c["whitener"]
    assert preds["eigh"][1]["solver"] == "eigh"


def test_cholesky_whitener_fails_loud_past_jitter_ladder():
    with pytest.raises(RuntimeError, match="eps ladder"):
        F._cholesky_whitener(-torch.eye(8, dtype=torch.float64))


def test_integrity_assert_fires_on_mismatch_and_passes_on_match():
    with pytest.raises(SystemExit, match="INTEGRITY ASSERT FAILED"):
        F._assert_integrity("probe", 1.0, 2.0, 1e-4)
    with pytest.raises(SystemExit, match="INTEGRITY ASSERT FAILED"):
        F._assert_integrity("probe-rel", 1.0 + 1e-6, 1.0, 1e-9, relative=True)
    F._assert_integrity("probe-ok", 1.00005, 1.0, 1e-4)  # within tol: no raise
    F._assert_integrity("probe-rel-ok", 1.0 + 1e-12, 1.0, 1e-9, relative=True)
