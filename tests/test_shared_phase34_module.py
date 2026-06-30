# ruff: noqa: RUF002, RUF003
# Intentional Unicode (σ, Δ, ×, ≤, ᵀ, ‖) in scientific docstrings + asserts.
"""Shared Phase-3/4 self-check module exports for issue-666 (plan §4c, §4j, Must-Fix 3).

The A3.8 rank-one residual and the A3.9 key/metric ablation are computed by ONE
frozen code path that BOTH Phase 4 (#666) and Phase 3 (#665) import from the
shared net-new module ``src/explore_persona_space/analysis/leakage_predictor.py``
(consistency-checker Must-Fix 3 — freeze the inline-self-check recipe in one
place so the phases cannot diverge). These tests confirm the exports are
importable from the SHARED module with the documented signatures + shapes, so
Phase 3 can import them.

- ``rank_one_residual(Δv, ŵ, ĝ)`` — on a rank-1 ``Δv = ŵ ⊗ ĝ`` (no residual) it
  returns ≈ 0; on a Δv with an orthogonal component it returns the orthogonal
  magnitude (the residual fraction ‖Δv(C') − ŵĝ‖ / ‖Δv(C')‖).
- ``metric_key_ablation(...)`` — importable with the documented signature for
  Phase 3 to call.

CPU-only; synthetic tensors; no store, no network, no GPU.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class _LazyModule:
    """Proxy that imports the target on first attribute access (TDD).

    The net-new shared module does NOT exist this round, so the first
    ``shared.<fn>`` access inside each test raises ImportError → the test FAILS
    (not skips). A module-level ``importorskip`` was rejected because it skips
    COLLECTION, so the proposed-test count could not be verified by approve-tests.
    """

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


shared = _LazyModule("explore_persona_space.analysis.leakage_predictor")

RTOL = 1e-6


# ---------------------------------------------------------------------------
# Both helpers exist + are importable from the SHARED module (Must-Fix 3).
# ---------------------------------------------------------------------------
def test_rank_one_residual_is_exported_callable():
    assert hasattr(shared, "rank_one_residual")
    assert callable(shared.rank_one_residual)


def test_metric_key_ablation_is_exported_callable():
    assert hasattr(shared, "metric_key_ablation")
    assert callable(shared.metric_key_ablation)


def test_metric_key_ablation_documented_signature():
    """Phase 3 imports this — its parameter names are a frozen contract."""
    params = set(inspect.signature(shared.metric_key_ablation).parameters)
    # The documented inputs: the per-context Δv stack, the candidate c_C keys,
    # the candidate whitening metrics (Σc⁻¹ variants), and the layer.
    required = {"dv_stack", "c_keys", "metrics"}
    assert required <= params, (
        f"metric_key_ablation missing documented params {required - params}; "
        "Phase 3 imports this signature"
    )


# ---------------------------------------------------------------------------
# rank_one_residual numerical behavior.
# ---------------------------------------------------------------------------
def test_rank_one_residual_zero_on_pure_rank_one():
    rng = np.random.default_rng(80)
    n_ctx, d = 25, 32
    w = rng.standard_normal(d)  # ŵ = Δv(C) at source
    ghat = rng.standard_normal(n_ctx)  # per-context realized gate
    # Pure rank-1 Δv(C') = ŵ * ĝ(C') → residual ≈ 0.
    dv = ghat[:, None] * w[None, :]  # (n_ctx, d)
    resid = shared.rank_one_residual(dv, w, ghat)
    resid = np.asarray(resid)
    assert np.all(np.abs(resid) < 1e-9), (
        f"rank-1 Δv must give ~0 residual, got max {np.max(np.abs(resid))}"
    )


def test_rank_one_residual_recovers_orthogonal_magnitude():
    rng = np.random.default_rng(81)
    n_ctx, d = 30, 40
    w = rng.standard_normal(d)
    w_unit = w / np.linalg.norm(w)
    ghat = rng.standard_normal(n_ctx)
    # Add an orthogonal component to each Δv(C').
    ortho_raw = rng.standard_normal((n_ctx, d))
    # Project out the ŵ direction so it is purely orthogonal.
    ortho = ortho_raw - (ortho_raw @ w_unit)[:, None] * w_unit[None, :]
    dv = ghat[:, None] * w[None, :] + ortho
    resid = np.asarray(shared.rank_one_residual(dv, w, ghat))
    # The residual fraction per context = ‖ortho‖ / ‖Δv‖.
    expected = np.linalg.norm(ortho, axis=1) / np.linalg.norm(dv, axis=1)
    assert np.allclose(resid, expected, rtol=1e-5, atol=1e-5), (
        "residual must equal the orthogonal magnitude fraction"
    )
    assert np.all(resid > 1e-3), "with an orthogonal component the residual is nonzero"


def test_rank_one_residual_matches_paper_definition():
    """‖Δv(C') − ŵ·ĝ^real(C')‖ / ‖Δv(C')‖ (plan §4j A3.8)."""
    rng = np.random.default_rng(82)
    n_ctx, d = 12, 20
    w = rng.standard_normal(d)
    dv = rng.standard_normal((n_ctx, d))
    # ĝ^real(C') = ŵᵀΔv(C') / ŵᵀŵ (the realized gate, plan §4b).
    ghat = (dv @ w) / (w @ w)
    resid = np.asarray(shared.rank_one_residual(dv, w, ghat))
    manual = np.linalg.norm(dv - ghat[:, None] * w[None, :], axis=1) / np.linalg.norm(dv, axis=1)
    assert np.allclose(resid, manual, rtol=RTOL, atol=RTOL)
