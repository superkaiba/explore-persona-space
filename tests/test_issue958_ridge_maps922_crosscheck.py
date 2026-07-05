"""#958 plan §12.6: cross-check the NEW batched dual-GCV ridge against maps922.

``issue958_fit_maps.fit_rows_batched`` (stacked-eigh DUAL/Gram path, new #958
code) is reproduced against the validated #922 implementation
``maps922.ridge_gcv_from_grams`` (primal Gram/eigh path) on a small synthetic
fit. This pins the shared conventions permanently — std ddof=0 (+1e-9), the
+1-intercept df, the GCV formula, the λ grid — against an INDEPENDENT
implementation (the in-run equivalence gate's serial reference shares its
derivation with the batched code; this cross-check does not).

Identities exercised: the dual (N-space) and primal (d-space) GCV curves are
exactly equal (nonzero eigenvalues of XnXnᵀ and XnᵀXn coincide; zeros
contribute 0 to df and SSE), and predictions at a COMMON λ agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue958_fit_maps as FM  # noqa: E402

from explore_persona_space.experiments.issue_922.maps922 import (  # noqa: E402
    RIDGE_LAMBDAS_922,
    GramStats,
    ridge_gcv_from_grams,
    ridge_predict,
)


def test_batched_dual_gcv_matches_maps922():
    """GCV curves ≤1e-8 rel + common-λ predictions vs the maps922 fit."""
    rng = np.random.default_rng(958)
    r, n, d, p = 2, 48, 12, 5
    Xf = torch.from_numpy(rng.standard_normal((r, n, d)))
    W = torch.from_numpy(rng.standard_normal((r, d, p)))
    Yf = Xf @ W * 0.1 + torch.from_numpy(rng.standard_normal((r, n, p))) * 0.05
    X_tf = torch.from_numpy(rng.standard_normal((r, 16, d)))
    # BOTH paths must consume byte-identical values: cast ONCE to the fp32 the
    # production store feeds the batched path, then lift to fp64 for maps922.
    X, Y, X_t = Xf.to(torch.float32), Yf.to(torch.float32), X_tf.to(torch.float32)

    fit = FM.fit_rows_batched(X, Y, lambdas=RIDGE_LAMBDAS_922, device="cpu")
    pred_b = FM.predict_from_fit(fit, X_t, device="cpu")

    for i in range(r):
        stats = GramStats.zeros(d, p, "cpu")
        # chunked accumulation — the production maps922 code path
        for lo in range(0, n, 16):
            stats.add_chunk(
                X[i, lo : lo + 16].to(torch.float64), Y[i, lo : lo + 16].to(torch.float64)
            )

        # (a) full-grid GCV curve identity (dual N-space vs primal d-space)
        _rmap, diag = ridge_gcv_from_grams(stats, lambdas=RIDGE_LAMBDAS_922)
        gb = fit["gcv_curve"][i].numpy()
        gs = np.asarray(diag["gcv_curve"], dtype=np.float64)
        assert gb.shape == gs.shape
        assert np.all(np.isinf(gb) == np.isinf(gs)), "GCV inf-mask mismatch vs maps922"
        fin = ~np.isinf(gb)
        rel = (
            float(np.max(np.abs(gb[fin] - gs[fin]) / (np.abs(gs[fin]) + 1e-300)))
            if fin.any()
            else 0.0
        )
        assert rel <= 1e-8, f"row {i}: GCV curve drift vs maps922 {rel:.3e} > 1e-8"

        # (b) predictions at the batched-selected λ match the maps922 map.
        # λ is PINNED to a common value (argmin over a flat GCV minimum is
        # ill-posed under fp jitter — the same design as the in-run gate).
        lam_i = float(fit["best_lam"][i])
        rmap_pin, _ = ridge_gcv_from_grams(stats, lambdas=[lam_i])
        pred_s = ridge_predict(rmap_pin, X_t[i].to(torch.float32))
        # RidgeMap stores fp32 weights → parity bounded by fp32 storage, not
        # the fp64 math (which the ≤1e-8 curve check + the in-run ≤1e-6
        # batched-vs-serial gate already pin).
        max_abs = float((pred_b[i].to(torch.float32) - pred_s).abs().max())
        assert max_abs <= 1e-4, f"row {i}: prediction drift vs maps922 {max_abs:.3e} > 1e-4"
