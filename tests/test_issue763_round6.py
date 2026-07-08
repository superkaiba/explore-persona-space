"""Issue #763 round-6 (fit-phase vectorization + PV-geometry guard) regression tests.

Round 6 vectorizes the fit-phase LOCO nulls (the 80x-compute-deviation reversal:
serial null ~1000 h/behavior -> minutes) and adds a fail-loud PV/v0 geometry
guard. Two permanent invariants, each failing pre-fix / passing post-fix:

1. ``vectorized-exactness-gate`` — the batched fitters in
   ``analysis.issue_763_vectorized`` MUST reproduce the serial statsmodels-GLM /
   #658-PRESS-ridge oracles before ANY behavior is fit. ``assert_matches_reference``
   is the hard fail-loud gate run at ``main()`` start; this test pins that it
   passes (batched == serial within tol, dim-selection argmax-identical) so a
   future batched-solve / seeding / standardization drift is caught in CI, not on
   the pod. (A serial vs vectorized ``_shuffle_null`` numeric-equivalence check on
   the real deception slice was run at implementation time; this CI test pins the
   synthetic gate the production run itself enforces.)

2. ``pv-baseline-staged-is-05b-smoke-not-7b`` — ``fit_behavior`` MUST fail loud
   when the PV baseline ``r_B`` geometry does not match v0's (n_layers, H). The
   PV arm reads ``direction = rb[ell]`` then ``x @ direction``, so a ``r_B``
   captured on a DIFFERENT model (a Qwen2.5-0.5B mock PV is [24, 896] vs the
   production 7B [*, 28, 3584]) would otherwise crash deep in
   ``_layer_sweep_select`` with a cryptic broadcast/IndexError, or silently
   mis-project. The invariant: a mismatched ``r_B`` raises a ``ValueError`` naming
   BOTH shapes up front; a matched ``r_B`` passes the guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


# ── (1) vectorized-exactness-gate ─────────────────────────────────────────────


def test_vectorized_fitters_match_serial_oracles():
    """The batched fitters reproduce the serial GLM/ridge/dim-select oracles.

    Calls the production fail-loud gate ``assert_matches_reference`` (which the
    fit's ``main()`` runs before any behavior) and asserts each measured delta is
    within its tolerance and the nested-CV dim selection is argmax-identical.
    A regression that breaks batched==serial equivalence FAILs here.
    """
    from explore_persona_space.analysis.issue_763_vectorized import assert_matches_reference

    res = assert_matches_reference()
    assert res["glm_delta"] <= res["tol_glm"], f"GLM batched vs serial delta too large: {res}"
    assert res["ridge_delta"] <= res["tol_ridge"], f"ridge batched vs serial delta too large: {res}"
    assert res["glm_obs_delta"] <= res["tol_glm"], f"observed GLM LOCO delta too large: {res}"
    assert res["ridge_obs_delta"] <= res["tol_ridge"], f"observed ridge LOCO delta too large: {res}"
    assert res["dim_select_identical"] is True, "nested-CV dim selection must be argmax-identical"


# ── (2) pv-baseline-staged-is-05b-smoke-not-7b ────────────────────────────────


def _tiny_e0_for(behavior: str, ctx_ids: list[str]) -> dict:
    """Minimal E0 dict with >=4 kept contexts so fit_behavior does not early-return."""
    per_ctx = {}
    for i, c in enumerate(ctx_ids):
        per_ctx[c] = {
            "graded_mean": 40.0 + i,
            "rate": 0.4 + 0.01 * i,
            "n_graded": 8,
            "n_judged": 8,
            "per_probe": [{"graded": 40.0 + i, "e0": 1 if i % 2 else 0}],
        }
    return {"e0": {behavior: per_ctx}}


def test_fit_behavior_raises_on_pv_geometry_mismatch():
    """A r_B whose (n_layers, H) mismatches v0 raises ValueError naming both shapes.

    This is the exact staged-input defect (a Qwen2.5-0.5B [24, 896] mock r_B against
    a production 7B v0). Pre-fix the mismatch crashed deep in ``_layer_sweep_select``
    with a cryptic broadcast/IndexError; post-fix ``fit_behavior`` raises up front.
    """
    import issue763_fit_predictors as f
    import torch

    n_ctx, n_layers, h = 8, 6, 12
    ctx_ids = [f"ctx{i}" for i in range(n_ctx)]
    v0 = np.random.default_rng(0).standard_normal((n_ctx, n_layers, h)).astype(np.float32)
    e0 = _tiny_e0_for("deception", ctx_ids)
    # r_B captured on a DIFFERENT geometry (wrong n_layers AND wrong H) — the mock case.
    rb_blob = {
        "r_b": torch.zeros(n_layers + 2, h + 4, dtype=torch.float32),
        "behavior": "deception",
    }

    with pytest.raises(ValueError) as ei:
        f.fit_behavior("deception", v0, ctx_ids, e0, rb_blob, n_perms=2, n_boot=2)
    msg = str(ei.value)
    assert "geometry mismatch" in msg
    # names BOTH the rb shape and the v0 geometry so the operator can act
    assert str(n_layers + 2) in msg and str(h + 4) in msg, msg
    assert f"n_layers={n_layers}" in msg and f"H={h}" in msg, msg


def test_fit_behavior_pv_guard_passes_on_matched_geometry():
    """A matched-geometry r_B does NOT trip the guard (the guard is not over-eager).

    Fits with a (n_layers, H)-matched r_B and a tiny slice; the run completes
    (returns a record dict) — proving the guard fires ONLY on a real mismatch.
    """
    import issue763_fit_predictors as f
    import torch

    n_ctx, n_layers, h = 8, 6, 12
    ctx_ids = [f"ctx{i}" for i in range(n_ctx)]
    rng = np.random.default_rng(1)
    v0 = rng.standard_normal((n_ctx, n_layers, h)).astype(np.float32)
    e0 = _tiny_e0_for("deception", ctx_ids)
    rb_blob = {
        "r_b": torch.from_numpy(rng.standard_normal((n_layers, h)).astype(np.float32)),
        "behavior": "deception",
    }

    rec = f.fit_behavior("deception", v0, ctx_ids, e0, rb_blob, n_perms=2, n_boot=2)
    assert isinstance(rec, dict)
    assert rec.get("behavior") == "deception"
