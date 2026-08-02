"""#1887 ridge lambda-selection hardening — defaults flip + refusal guard + tripwire.

Pins (task #1887, plan v4 §3.3):
  1. New defaults: the three fit825 kwarg-threaded entrypoints default to the
     registered ``inner-group-cv`` selector + ``collect_lambdas=True``; the
     module globals read ``GCV_DOF_CAP = 0.9`` / ``LEGACY_UNGUARDED_GCV =
     False`` in all three cores (fit825 / map_alignment / crossmodel).
  2. Refusal guard: a fit reaching a GCV scan with ``GCV_DOF_CAP is None`` and
     ``n_train < d`` and ``LEGACY_UNGUARDED_GCV = False`` raises RuntimeError
     naming the fix; the explicit legacy opt-in unlocks it.
  3. Demonstrative fixture: on a seeded n < d cell, legacy pure GCV selects
     within one grid step of the lambda-grid lower edge with held-out R^2 < 0
     while inner-group-cv selects an interior lambda with strictly higher
     held-out R^2 (the #1345/#1310 incident mechanism at test scale).
  4. Tripwire arms (a)/(b) fire / stay silent on constructed inputs, pinned at
     the motivating incident's artifact-read values (story cell: raw retrieval
     0.0223 ~ 45x chance 1/2018; forced-lambda kNN@1 0.2423-0.2765).
  5. Reduced-basis companion present iff n_train < d.
  6. Legacy pins reproduce the pre-#1887 selected lambda + prediction against
     an independent in-test GCV reference (bit-compatible to fp roundoff).
  7. Class-A caller pins: the deliberately-unguarded audit arms
     (issue825_selector_audit / issue825_trackm_settle_battery) still execute
     pure unguarded GCV byte-preserved (no refusal), via their explicit
     LEGACY_UNGUARDED_GCV opt-in with finally-restored globals.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: seeded n < d cell where pure GCV demonstrably interpolates.
# n=24, d=64 -> per-fold n_train ~ 19 < 64 (the degenerate regime). Y carries a
# smooth low-rank signal + noise sized so an interior lambda generalizes while
# the (near-)interpolating grid-floor lambda overfits held-out.
# ---------------------------------------------------------------------------
N_ROWS, D_IN, D_OUT, RANK, SIGMA, SEED = 24, 64, 64, 4, 2.0, 1887


def _fixture_xy(
    n: int = N_ROWS, d_in: int = D_IN, d_out: int = D_OUT, sigma: float = SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X_layers, Y_layers, conv_ids) with ONE layer axis (fit825 shape)."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d_in))
    W = (
        rng.standard_normal((d_in, RANK))
        @ rng.standard_normal((RANK, d_out))
        / np.sqrt(d_in * RANK)
    )
    Y = X @ W + sigma * rng.standard_normal((n, d_out))
    conv_ids = np.asarray([f"c{i}" for i in range(n)])  # singleton groups
    return X[:, None, :].astype(np.float32), Y[:, None, :].astype(np.float32), conv_ids


def _sweep(lambda_selection: str, **kw) -> dict:
    X, Y, ids = _fixture_xy()
    return fit825.heldout_r2_sweep(
        X,
        Y,
        ids,
        n_folds=5,
        seed=0,
        null_draws=0,
        collect_cosines=False,
        collect_lambdas=True,
        lambda_selection=lambda_selection,
        frozen_layers=(),
        **kw,
    )


@pytest.fixture()
def legacy_pins(monkeypatch):
    """The explicit pre-#1887 legacy configuration on fit825."""
    monkeypatch.setattr(fit825, "GCV_DOF_CAP", None)
    monkeypatch.setattr(fit825, "LEGACY_UNGUARDED_GCV", True)


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------
def test_new_defaults_module_globals():
    assert fit825.GCV_DOF_CAP == 0.9
    assert fit825.LAMBDA_SELECTION == "inner-group-cv"
    assert fit825.LEGACY_UNGUARDED_GCV is False
    assert isinstance(fit825.SELECTOR_LOG, dict)
    assert ma.LAMBDA_SELECTION == "inner-group-cv"
    assert ma.GCV_DOF_CAP == 0.9
    assert ma.LEGACY_UNGUARDED_GCV is False
    assert cm.LAMBDA_SELECTION == "inner-group-cv"
    assert cm.GCV_DOF_CAP == 0.9
    assert cm.LEGACY_UNGUARDED_GCV is False


def test_new_defaults_kwarg_threaded_entrypoints():
    for fn in (fit825.heldout_r2_sweep, fit825.random_projection_control, fit825.run_cell):
        params = inspect.signature(fn).parameters
        assert params["lambda_selection"].default == "inner-group-cv", fn.__name__
        # The module global is the canonical RECORD of the kwarg default.
        assert params["lambda_selection"].default == fit825.LAMBDA_SELECTION, fn.__name__
    for fn in (fit825.heldout_r2_sweep, fit825.run_cell):
        assert inspect.signature(fn).parameters["collect_lambdas"].default is True, fn.__name__
    for fn in (fit825.heldout_r2_sweep, fit825.run_cell):
        assert inspect.signature(fn).parameters["reduced_basis_companion"].default is True, (
            fn.__name__
        )


# ---------------------------------------------------------------------------
# 2. Refusal guard
# ---------------------------------------------------------------------------
def test_pure_gcv_refused_at_n_lt_d(monkeypatch):
    monkeypatch.setattr(fit825, "GCV_DOF_CAP", None)
    monkeypatch.setattr(fit825, "LEGACY_UNGUARDED_GCV", False)
    with pytest.raises(RuntimeError, match=r"#1887"):
        _sweep("gcv", reduced_basis_companion=False)


def test_pure_gcv_runs_under_explicit_legacy_opt_in(legacy_pins):
    sw = _sweep("gcv", reduced_basis_companion=False)
    assert np.isfinite(sw["r2_obs"][0])


def test_capped_gcv_not_refused_at_n_lt_d(monkeypatch):
    # The cap (default 0.9) is a registered mitigation: no refusal.
    monkeypatch.setattr(fit825, "GCV_DOF_CAP", 0.9)
    monkeypatch.setattr(fit825, "LEGACY_UNGUARDED_GCV", False)
    sw = _sweep("gcv", reduced_basis_companion=False)
    assert np.isfinite(sw["r2_obs"][0])


def test_pure_gcv_not_refused_at_n_gt_d(monkeypatch):
    # Well-posed regime (n_train > d): pure GCV stays legal without the opt-in.
    monkeypatch.setattr(fit825, "GCV_DOF_CAP", None)
    monkeypatch.setattr(fit825, "LEGACY_UNGUARDED_GCV", False)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 1, 8)).astype(np.float32)
    Y = rng.standard_normal((60, 1, 8)).astype(np.float32)
    ids = np.asarray([f"c{i}" for i in range(60)])
    sw = fit825.heldout_r2_sweep(
        X, Y, ids, n_folds=5, seed=0, null_draws=0, lambda_selection="gcv", frozen_layers=()
    )
    assert np.isfinite(sw["r2_obs"][0])


def test_forced_single_lambda_grid_not_refused(monkeypatch):
    # A deliberate 1-element forced grid (lambdas=[lam]) SELECTS nothing, so
    # the guard's degenerate-selection failure mode cannot occur — the #1887
    # audit's forced-lambda diagnostic arm relies on this carve-out.
    monkeypatch.setattr(fit825, "GCV_DOF_CAP", None)
    monkeypatch.setattr(fit825, "LEGACY_UNGUARDED_GCV", False)
    X, Y, _ = _fixture_xy()
    Xl, Yl = X[:, 0, :], Y[:, 0, :]
    tr = np.arange(N_ROWS) >= 5
    cache = fit825._prep_fold(Xl[tr], Xl[~tr])
    pred, lam = fit825._ridge_predict_cached(cache, Yl[tr], return_lam=True, lambdas=[1e3])
    assert lam == 1e3 and np.isfinite(pred).all()
    # A MULTI-element forced grid is a selection again: refused.
    with pytest.raises(RuntimeError, match=r"#1887"):
        fit825._ridge_predict_cached(cache, Yl[tr], lambdas=[1e2, 1e3])


def test_ma_and_cm_guards_refuse_and_unlock(monkeypatch):
    rng = np.random.default_rng(5)
    Xtr = rng.standard_normal((12, 48))
    Ytr = rng.standard_normal((12, 48))
    # ma: prep via _ridge_prep (stores d), GCV branch via _select_lambda.
    monkeypatch.setattr(ma, "LAMBDA_SELECTION", "gcv")
    monkeypatch.setattr(ma, "GCV_DOF_CAP", None)
    monkeypatch.setattr(ma, "LEGACY_UNGUARDED_GCV", False)
    prep = ma._ridge_prep(torch.as_tensor(Xtr, dtype=torch.float64))
    Yt = torch.as_tensor(Ytr, dtype=torch.float64)
    VtY = prep["V"].T @ (Yt - Yt.mean(0))
    tot = float(((Yt - Yt.mean(0)) ** 2).sum())
    with pytest.raises(RuntimeError, match=r"map_alignment.*#1887"):
        ma._select_lambda(prep, Yt, VtY, tot)
    monkeypatch.setattr(ma, "LEGACY_UNGUARDED_GCV", True)
    assert ma._select_lambda(prep, Yt, VtY, tot) > 0
    # cm: cache via _prep_fold (stores d), GCV branch in _ridge_predict_cached.
    monkeypatch.setattr(cm, "LAMBDA_SELECTION", "gcv")
    monkeypatch.setattr(cm, "GCV_DOF_CAP", None)
    monkeypatch.setattr(cm, "LEGACY_UNGUARDED_GCV", False)
    cache = cm._prep_fold(Xtr, Xtr[:4])
    with pytest.raises(RuntimeError, match=r"crossmodel.*#1887"):
        cm._ridge_predict_cached(cache, Ytr)
    monkeypatch.setattr(cm, "LEGACY_UNGUARDED_GCV", True)
    assert cm._ridge_predict_cached(cache, Ytr).shape == (4, 48)
    # cm fit_primal_beta's own scan site.
    monkeypatch.setattr(cm, "LEGACY_UNGUARDED_GCV", False)
    with pytest.raises(RuntimeError, match=r"fit_primal_beta.*#1887"):
        cm.fit_primal_beta(Xtr, Ytr)
    monkeypatch.setattr(cm, "LEGACY_UNGUARDED_GCV", True)
    beta, lam = cm.fit_primal_beta(Xtr, Ytr)
    assert beta.shape == (48, 48) and lam > 0


# ---------------------------------------------------------------------------
# 3. The demonstrative fixture: GCV picks the grid edge; inner-CV does not.
# ---------------------------------------------------------------------------
def test_gcv_picks_grid_edge_and_inner_cv_does_not(legacy_pins):
    sw_gcv = _sweep("gcv", reduced_basis_companion=False)
    lam_gcv = np.asarray(sw_gcv["gcv_lambda"], dtype=float).ravel()
    lam_gcv = lam_gcv[np.isfinite(lam_gcv)]
    r2_gcv = float(sw_gcv["r2_obs"][0])
    # Legacy pure GCV: (near-)interpolating selection at the grid edge with
    # negative held-out R^2 — the #1345/#1310 incident mechanism.
    assert lam_gcv.min() <= float(fit825.LAMBDAS[1]), lam_gcv
    assert r2_gcv < 0.0, r2_gcv

    sw_in = _sweep("inner-group-cv", reduced_basis_companion=False)
    lam_in = np.asarray(sw_in["gcv_lambda"], dtype=float).ravel()
    lam_in = lam_in[np.isfinite(lam_in)]
    r2_in = float(sw_in["r2_obs"][0])
    assert lam_in.min() >= float(fit825.LAMBDAS[2]), lam_in
    assert r2_in > r2_gcv, (r2_in, r2_gcv)


def test_tripwire_fires_on_the_fixture(legacy_pins):
    sw = _sweep("gcv", reduced_basis_companion=False)
    tw = sw["degeneracy_tripwire"]
    assert tw["estimator_degenerate_suspect"] is True
    assert "n_lt_d_lambda_at_grid_edge" in tw["reasons"]
    sw_in = _sweep("inner-group-cv", reduced_basis_companion=False)
    assert sw_in["degeneracy_tripwire"]["estimator_degenerate_suspect"] is False


# ---------------------------------------------------------------------------
# 4. Tripwire arms on constructed inputs (incident artifact-read values).
# ---------------------------------------------------------------------------
def test_tripwire_arm_a():
    out = fit825.degeneracy_tripwire(
        n_train=1730,
        d=3584,
        selected_lambdas=[0.01] * 5,  # grid floor, 5/5 folds (4682f0247a)
    )
    assert out["estimator_degenerate_suspect"] is True
    assert out["reasons"] == ["n_lt_d_lambda_at_grid_edge"]
    # One grid step above the edge still fires (<= LAMBDAS[1]); two steps do not.
    edge_plus1 = fit825.degeneracy_tripwire(
        n_train=1730, d=3584, selected_lambdas=[float(fit825.LAMBDAS[1])]
    )
    assert edge_plus1["estimator_degenerate_suspect"] is True
    interior = fit825.degeneracy_tripwire(
        n_train=1730, d=3584, selected_lambdas=[float(fit825.LAMBDAS[2])]
    )
    assert interior["estimator_degenerate_suspect"] is False
    # n_train >= d: arm (a) never fires even at the grid floor.
    nd_ok = fit825.degeneracy_tripwire(n_train=4724, d=3584, selected_lambdas=[0.01])
    assert nd_ok["estimator_degenerate_suspect"] is False


def test_tripwire_arm_b_incident_values():
    chance = 1.0 / 2018.0  # story-cell candidate pool (fact-check-corrected)
    # Committed story cell: ambient R^2 = -0.547 with its OWN raw retrieval
    # 0.0223 (~45x chance) -> fires.
    raw = fit825.degeneracy_tripwire(
        n_train=1730,
        d=3584,
        selected_lambdas=[1e3],
        r2_heldout=-0.547,
        knn_at_1=0.0223,
        knn_chance=chance,
    )
    assert raw["reasons"] == ["negative_r2_with_retrieval_dissociation"]
    # Forced-lambda reads 0.2423-0.2765 (~490-560x chance) -> fires a fortiori.
    for knn in (0.2423, 0.2765):
        forced = fit825.degeneracy_tripwire(
            n_train=1730,
            d=3584,
            selected_lambdas=[1e3],
            r2_heldout=-0.547,
            knn_at_1=knn,
            knn_chance=chance,
        )
        assert forced["estimator_degenerate_suspect"] is True
    # Retrieval at chance: no dissociation -> silent.
    at_chance = fit825.degeneracy_tripwire(
        n_train=1730,
        d=3584,
        selected_lambdas=[1e3],
        r2_heldout=-0.547,
        knn_at_1=chance,
        knn_chance=chance,
    )
    assert at_chance["estimator_degenerate_suspect"] is False
    # Positive R^2: arm (b) silent regardless of retrieval.
    pos = fit825.degeneracy_tripwire(
        n_train=1730,
        d=3584,
        selected_lambdas=[1e3],
        r2_heldout=0.26,
        knn_at_1=0.2765,
        knn_chance=chance,
    )
    assert pos["estimator_degenerate_suspect"] is False
    # No retrieval read (knn_at_1=None): arm (b) skipped, never fabricated.
    none = fit825.degeneracy_tripwire(
        n_train=1730, d=3584, selected_lambdas=[1e3], r2_heldout=-0.547
    )
    assert none["estimator_degenerate_suspect"] is False


# ---------------------------------------------------------------------------
# 5. Reduced-basis companion present iff n_train < d.
# ---------------------------------------------------------------------------
def test_reduced_basis_companion_present_iff_n_lt_d():
    sw = _sweep("inner-group-cv")  # fixture: n_train ~19 < d=64
    rb = sw["reduced_basis"]
    assert rb is not None
    assert rb["k"] == fit825.reduced_basis_k(sw["n_train_min"], sw["d_in"])
    assert 1 <= rb["k"] <= sw["n_train_min"] // 2
    assert len(rb["r2_per_layer"]) == 1 and np.isfinite(rb["r2_per_layer"][0])
    # n > d cell: no companion.
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 1, 8)).astype(np.float32)
    Y = rng.standard_normal((60, 1, 8)).astype(np.float32)
    ids = np.asarray([f"c{i}" for i in range(60)])
    sw_nd = fit825.heldout_r2_sweep(X, Y, ids, n_folds=5, seed=0, null_draws=0, frozen_layers=())
    assert sw_nd["reduced_basis"] is None
    # Opt-out is honored.
    sw_off = _sweep("inner-group-cv", reduced_basis_companion=False)
    assert sw_off["reduced_basis"] is None


def test_reduced_basis_k_rule():
    assert fit825.reduced_basis_k(1730, 3584) == 865
    assert fit825.reduced_basis_k(5000, 3584) == 1024
    assert fit825.reduced_basis_k(300, 64) == 64
    assert fit825.reduced_basis_k(3, 64) == 1


# ---------------------------------------------------------------------------
# 5b. run_cell end-to-end at smoke shape: the D2 cell-JSON contract.
# ---------------------------------------------------------------------------
def test_run_cell_payload_carries_selector_config_tripwire_and_companion(tmp_path):
    """Every cell JSON written through run_cell carries lambda_selection,
    gcv_dof_cap, the degeneracy-tripwire fields, and (at n_train < d) the
    reduced_basis companion block — the #1887 D2 acceptance, exercised
    through the REAL run_cell body on a synthetic in-memory bundle."""
    rng = np.random.default_rng(11)
    n, d = 24, 32  # per-fold n_train ~19 < d=32: the degenerate regime
    ll = fit825.EXPECTED_LAYERS
    slots = rng.standard_normal((n, 1, ll, d)).astype(np.float32)
    profiles = rng.standard_normal((n, 2, ll, d)).astype(np.float32)
    bundle = {
        "arrays": {"slots": slots, "profiles": profiles},
        "sidecar": {"conv_ids": np.asarray([f"c{i}" for i in range(n)])},
    }
    cell = {"cell_id": "syn_1887", "model_key": "syn", "format_key": "chat", "track": "s"}
    res = fit825.run_cell(
        cell,
        tmp_path,  # turnstore_dir unused (bundle injected)
        tmp_path,
        n_folds=5,
        seed=0,
        null_draws=0,
        n_boot=8,
        bundle=bundle,
    )
    payload = res["cell_payload"]
    assert payload["lambda_selection"] == "inner-group-cv"
    assert payload["gcv_dof_cap"] == 0.9
    assert payload["legacy_unguarded_gcv"] is False
    assert payload["estimator_degenerate_suspect"] in (True, False)
    assert isinstance(payload["reasons"], list)
    assert payload["n_train"] is not None and payload["d"] == d
    assert payload["n_train"] < d  # this fixture IS the degenerate regime
    assert payload["reduced_basis"] is not None
    assert payload["reduced_basis"]["k"] == fit825.reduced_basis_k(payload["n_train"], d)
    assert payload["selected_lambda_per_layer_fold"] is not None  # collect default ON
    import json as _json

    on_disk = _json.loads((tmp_path / "cells_syn_1887.json").read_text())
    for key in ("lambda_selection", "gcv_dof_cap", "estimator_degenerate_suspect", "reasons"):
        assert key in on_disk, key


# ---------------------------------------------------------------------------
# 6. Legacy pins reproduce the pre-#1887 behavior (independent GCV reference).
# ---------------------------------------------------------------------------
def _reference_gcv_fit(X_tr, X_ev, Y_tr):
    """Independent numpy re-implementation of the committed pure-GCV ridge
    (grid-min start, strict-< first minimum, no dof cap)."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_ev = np.asarray(X_ev, dtype=np.float64)
    Y_tr = np.asarray(Y_tr, dtype=np.float64)
    # ddof=1: torch.Tensor.std defaults to the unbiased estimator.
    xmu, xsd = X_tr.mean(0), X_tr.std(0, ddof=1) + 1e-9
    Xn = (X_tr - xmu) / xsd
    Xe = (X_ev - xmu) / xsd
    G = Xn @ Xn.T
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)
    ymu = Y_tr.mean(0)
    Yc = Y_tr - ymu
    VtY = V.T @ Yc
    ntr = X_tr.shape[0]
    sq = (VtY**2).sum(1)
    tot = float((Yc**2).sum())
    best_lam, best_gcv = float(fit825.LAMBDAS[0]), float("inf")
    for lam in fit825.LAMBDAS:
        filt = w / (w + lam)
        dof = float(filt.sum())
        rss = tot - float(((2 * filt - filt**2) * sq).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    filt = 1.0 / (w + best_lam)
    pred = ((Xe @ Xn.T @ V) * filt) @ VtY + ymu
    return pred, best_lam


def test_legacy_pins_reproduce_prior_behavior(legacy_pins):
    X, Y, _ = _fixture_xy()
    Xl, Yl = X[:, 0, :], Y[:, 0, :]
    tr = np.arange(24) >= 5
    te = ~tr
    cache = fit825._prep_fold(Xl[tr], Xl[te])
    pred, lam = fit825._ridge_predict_cached(cache, Yl[tr], return_lam=True)
    ref_pred, ref_lam = _reference_gcv_fit(Xl[tr], Xl[te], Yl[tr])
    assert lam == ref_lam
    np.testing.assert_allclose(pred, ref_pred, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. Class-A caller pins (deliberately-unguarded audit arms, byte-preserved).
# ---------------------------------------------------------------------------
def test_selector_audit_arms_run_unrefused():
    import issue825_selector_audit as sa

    X, Y, ids = _fixture_xy()
    for selector in sa.SELECTORS:
        out = sa._fit_one(X, Y, ids, selector)  # must NOT raise at n_train < d
        assert np.isfinite(out["r2"]), selector
    # The unguarded arm still executes PURE GCV: grid-floor selection survives.
    unguarded = sa._fit_one(X, Y, ids, "gcv_unguarded")
    assert unguarded["lambda_at_grid_floor_frac"] > 0.0
    # Globals restored after every arm (finally block).
    assert fit825.GCV_DOF_CAP == 0.9
    assert fit825.LEGACY_UNGUARDED_GCV is False


def test_trackm_settle_battery_ridge_r2_runs_unrefused():
    import issue825_trackm_settle_battery as bat

    X, Y, ids = _fixture_xy()
    prev_want = bat.WANT_LAYERS
    try:
        bat.WANT_LAYERS = [0]
        r_un = bat.ridge_r2(X, Y, ids, guarded=False)  # must NOT raise at n_train < d
        r_g = bat.ridge_r2(X, Y, ids, guarded=True)
    finally:
        bat.WANT_LAYERS = prev_want
    assert np.isfinite(r_un["0"]) and np.isfinite(r_g["0"])
    assert fit825.GCV_DOF_CAP == 0.9
    assert fit825.LEGACY_UNGUARDED_GCV is False


# ---------------------------------------------------------------------------
# 8. r2 concern fixes (i1887-1310-store-rev-unpinned + i1887-variant-store-
#    staging): pinned store revisions + variant store/allowlist routing.
#    Pure — no HF calls, no staging.
# ---------------------------------------------------------------------------
def _audit():
    import issue1887_lambda_audit as audit

    return audit


def _is_full_sha(s: str) -> bool:
    return len(s) == 40 and set(s) <= set("0123456789abcdef")


def test_store_rev_pins_are_full_shas():
    audit = _audit()
    pins = [
        audit.I1310_STORE_REV,
        audit.I1345_PARENT_STORE_REV,
        audit.I1345_PARENT_MATCHED_REV,
        *audit.I1345_VARIANT_STORE_REVS.values(),
        *audit.I1345_VARIANT_MATCHED_REVS.values(),
    ]
    assert len(pins) == 3 + 3 + 4
    for pin in pins:
        assert _is_full_sha(pin), pin


def test_1310_resume_key_carries_the_real_store_rev():
    """The resume predicate invalidates on store drift — no placeholder string
    (concern i1887-1310-store-rev-unpinned)."""
    audit = _audit()
    cell = audit.CellSpec(
        issue=1310,
        cell_id="instruct_Dana",
        variant="xpersona",
        committed_r2=0.0,
        published_claim_ref="ref",
        store_rev=audit.I1310_STORE_REV,
        load=lambda: [],
    )
    key = audit._resume_key(cell, "inner_group_cv")
    assert key["store_rev"] == audit.I1310_STORE_REV
    assert _is_full_sha(key["store_rev"])


def test_1345_store_resolution_map():
    audit = _audit()
    # Parent-format stems -> the parent turnstore pin, ONE shared flat dir.
    for fmt in ("chat", "naturalistic", "stories"):
        prefix, rev, subdir = audit._resolve_1345_store("assistant_named_story", fmt)
        assert prefix == audit.I1345_PARENT_STORE_PREFIX
        assert rev == audit.I1345_PARENT_STORE_REV
        assert subdir == "parent_turnstore"
    # Variant-prefixed stems -> that variant's pinned prefix.
    prefix, rev, subdir = audit._resolve_1345_store("story_slot_ablation", "stories_paired_slots")
    assert prefix == "issue1345_framing/story_slot_ablation/analysis_tensors/turnstore"
    assert rev == audit.I1345_VARIANT_STORE_REVS["story_slot_ablation"]
    assert subdir == "story_slot_ablation_turnstore"
    # No pinned source -> None (the un-refittable path, plan §5.4).
    assert audit._resolve_1345_store("ladder_rungs", "stories_paired") is None


def test_1345_allowlist_ref_tokens():
    audit = _audit()
    assert audit._allowlist_ref_1345("base", {"row_allowlist_applied": False}) is None
    assert (
        audit._allowlist_ref_1345("followup_cjk_excluded", {"cjk_exclusion": {"x": 1}})
        == "payload:cjk_exclusion"
    )
    assert (
        audit._allowlist_ref_1345("story_slot_ablation", {"row_allowlist_applied": True})
        == "git:slot_row_coverage.json"
    )
    ref = audit._allowlist_ref_1345("onpolicy_assistant_story", {"row_allowlist_applied": True})
    assert ref == f"matched:{audit.I1345_VARIANT_MATCHED_REVS['onpolicy_assistant_story']}"
    ref = audit._allowlist_ref_1345("base", {"row_allowlist_applied": True})
    assert ref == f"matched:{audit.I1345_PARENT_MATCHED_REV}"


def test_unrefittable_cell_row_and_gate_exclusion(tmp_path):
    """A load=None cell is SKIPPED by run_units and lands in the corrections
    table as a named un-refittable row, excluded from the replay gate
    (plan §5.4 — never a whole-audit failure)."""
    audit = _audit()
    cell = audit.CellSpec(
        issue=1345,
        cell_id="ladder_rungs__hypothetical",
        variant="ladder_rungs",
        committed_r2=0.5,
        published_claim_ref="ref",
        store_rev="unresolvable",
        load=None,
        notes="un-refittable — store not resolvable (stem 'x': no pinned HF turnstore prefix)",
    )
    audit.run_units([cell], audit.ARMS, tmp_path)  # skips every arm, writes nothing
    assert not (tmp_path / "cells").exists()
    table = audit.build_corrections_table([cell], tmp_path)
    row = table["rows"][0]
    assert row["verdict_label"] == audit.UNREFITTABLE_VERDICT
    assert row["notes"].startswith("un-refittable")
    gate = table["replay_gate"]
    assert gate["n_unrefittable"] == 1
    assert gate["n_cells_with_reference"] == 0  # excluded from the gate denominator
    assert gate["gate"] == "PASS"
    md = (tmp_path / "corrections_table.md").read_text()
    assert "un-refittable — store not resolvable" in md


# ---------------------------------------------------------------------------
# 9. r3 crash-fix (staged-layout consumer-open miss, artifact-reuse (h)(iv)):
#    the 825 adapter materializes the parent layout load_cell_xy opens.
#    Pure — extract_stem (the HF-download boundary) is faked signature-
#    conformantly; the staging body, cid->stem mapping, and consumer-open
#    probe run for real.
# ---------------------------------------------------------------------------
def _fake_extract_stem_writing(calls):
    """Signature-conformant fake of cm.extract_stem that writes a tiny valid
    4-layer S-track npz at the parent-layout path (boundary fake only)."""

    def fake_extract(stem, dl_dir, revision=None):
        calls.append((stem, Path(dl_dir)))
        p = Path(dl_dir) / f"{stem}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            slots=np.zeros((1, 2, 4, 3), dtype=np.float16),
            profiles=np.zeros((1, 4, 4, 3), dtype=np.float16),
            conv_ids=np.asarray(["c0"]),
            layers=np.asarray([14, 18, 19, 26]),
        )
        return p

    return fake_extract


def test_825_stage_parent_layout_body(tmp_path, monkeypatch):
    """The staging body maps cids to parent stems, stages via the parent's own
    helper at the CONSUMER's S_STORE_DIR, and the probe passes on a valid npz
    (fails FileNotFoundError-style pre-fix: nothing staged this layout)."""
    import issue825_trackm_settle_battery as bat

    audit = _audit()
    store = tmp_path / "data" / "issue_825" / "hf_dl" / "map_alignment"
    monkeypatch.setattr(bat, "S_STORE_DIR", store)
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(cm, "extract_stem", _fake_extract_stem_writing(calls))
    audit._stage_825_parent_layout(["S_instruct_chat"])
    assert calls == [("instruct_chat_s", store)]
    assert (store / "instruct_chat_s.npz").is_file()


def test_825_consumer_open_probe_fails_loud(tmp_path, monkeypatch):
    """A staging that does NOT materialize the consumer's exact path is caught
    at enumeration time with the mapping explanation — never a bare
    FileNotFoundError inside run_units (the P0 crash shape)."""
    import issue825_trackm_settle_battery as bat

    audit = _audit()
    monkeypatch.setattr(bat, "S_STORE_DIR", tmp_path / "absent")
    monkeypatch.setattr(
        cm, "extract_stem", lambda stem, dl_dir, revision=None: Path(dl_dir) / f"{stem}.npz"
    )
    with pytest.raises(RuntimeError, match="staged-layout consumer-open miss"):
        audit._stage_825_parent_layout(["S_instruct_chat"])


def test_825_probe_rejects_wrong_keys(tmp_path, monkeypatch):
    """A present-but-wrong-shape npz (missing turnstore keys) fails loud too."""
    import issue825_trackm_settle_battery as bat

    audit = _audit()
    store = tmp_path / "ma"
    monkeypatch.setattr(bat, "S_STORE_DIR", store)

    def fake_extract(stem, dl_dir, revision=None):
        p = Path(dl_dir) / f"{stem}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, wrong=np.zeros(1))
        return p

    monkeypatch.setattr(cm, "extract_stem", fake_extract)
    with pytest.raises(RuntimeError, match="lacks keys"):
        audit._stage_825_parent_layout(["S_instruct_chat"])


def test_825_cells_stage_at_enumeration_for_pilot_slice(tmp_path, monkeypatch):
    """cells_825_control stages the parent layout at ENUMERATION time (before
    run_units) for exactly the pilot-sliced cells (the r3 crash-fix wiring)."""
    audit = _audit()
    staged: list[list[str]] = []
    monkeypatch.setattr(audit, "_stage_825_parent_layout", lambda cids: staged.append(list(cids)))
    specs = audit.cells_825_control(tmp_path, tmp_path, pilot=1)
    assert staged == [["S_instruct_chat"]]
    assert [s.cell_id for s in specs] == ["control__S_instruct_chat"]
    specs_full = audit.cells_825_control(tmp_path, tmp_path, pilot=0)
    assert staged[1] == ["S_instruct_chat", "S_pretrained_chat"]
    assert len(specs_full) == 2
