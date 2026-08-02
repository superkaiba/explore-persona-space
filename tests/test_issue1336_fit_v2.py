"""#1336 Unit B: v2 registry / adaptive edge rule / lambda-audit grid pins.

Covers the plan v13 must-fixes: the parametrized `_lambda_audit` grid
(assumption 11 — the L332 `fc.LAMBDAS` hardcode), the parametrized cell
registry (assumption 12 — CELLS_V2 = 45 beside the intact v1 assert), the
adaptive edge rule (<=2 one-decade extensions per side, then the
`estimator-limited: lambda-edge` label), the §7 kill-bar form
bar_v2 = 0.20 * ex_v2, and the `--x-slot` X-slot selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fc  # noqa: E402
import issue1336_fit_cells as f36  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Registries (plan §12 assumption 12)
# ---------------------------------------------------------------------------
def test_v1_registry_assert_still_binds():
    assert len(cm.CELLS) == 20


def test_cells_v2_registry_shape():
    assert len(cm.CELLS_V2) == 45
    context = [c for c in cm.CELLS_V2 if c["x_slot"] == "context"]
    prefix = [c for c in cm.CELLS_V2 if c["x_slot"] == "prefix"]
    assert len(context) == 40  # 5 models x 8 corpus-format surfaces
    assert len(prefix) == 5  # 5 models x (lmsys23k, naturalistic)
    ids = [c["cell_id"] for c in cm.CELLS_V2]
    assert len(set(ids)) == 45
    for c in prefix:
        assert c["corpus"] == "lmsys23k" and c["format"] == "naturalistic"
        assert c["cell_id"].endswith("_xprefix")
        # Prefix-arm cells read the SAME turnstore stem as their context twin.
        assert c["cell_id"] == f"{cm.cell_id(c['model'], c['format'], c['corpus'])}_xprefix"
    for c in context:
        assert c["corpus"] in cm.V2_CORPORA
        assert c["format"] in cm.V2_CORPORA[c["corpus"]]["formats"]


def test_cells_v2_subset_seam():
    sub = cm.cells_v2_for(("base",), ("gsm8k_train_full",))
    assert [c["cell_id"] for c in sub] == ["base_chat_gsm8k_train_full"]
    smoke = cm.cells_v2_for(cm.SMOKE_MODELS, cm.SMOKE_CORPORA_V2)
    # 3 models x (chat + naturalistic) context + 3 prefix-arm cells.
    assert len(smoke) == 9
    assert sum(c["x_slot"] == "prefix" for c in smoke) == 3


def test_v2_corpora_reexport_is_same_object():
    import issue1336_stage_corpora as sc

    assert sc.V2_CORPORA is cm.V2_CORPORA


def test_merged_cell_lookup_covers_v1_and_v2():
    assert "rlvr_chat_lmsys5k" in f36.CELL_BY_ID  # v1
    assert "rlvr_chat_lmsys23k" in f36.CELL_BY_ID  # v2 context
    assert "base_naturalistic_lmsys23k_xprefix" in f36.CELL_BY_ID  # v2 prefix
    # The 5 shared gsm8k_test1319 ids resolve to the v2 dict (same cell +
    # x_slot field) — behavior-identical for v1 callers.
    assert f36.CELL_BY_ID["rlvr_chat_gsm8k_test1319"]["x_slot"] == "context"


# ---------------------------------------------------------------------------
# v2 estimator constants + bars (plan §7 / §11)
# ---------------------------------------------------------------------------
def test_lambdas_23_grid():
    g = np.asarray(cm.LAMBDAS_23)
    assert len(g) == 23
    assert g[0] == 10.0**-3 and g[-1] == 10.0**8
    assert np.all(np.diff(np.log10(g)) > 0.49) and np.all(np.diff(np.log10(g)) < 0.51)
    fc._validate_lambda_grid(g)


def test_v2_bars_kill_bar_form():
    s = 0.7
    bars = cm.v2_bars(s)
    ex = s / 0.6731
    assert bars["ex_v2"] == ex
    assert bars["bar_v2"] == 0.20 * ex  # the §7 form, EXACTLY
    assert bars["elicit_band_v2"] == 0.0201 * ex
    assert bars["practical_scale_v2"] == 0.0503 * ex
    assert bars["health_gate_v2"] == 0.05 * ex


# ---------------------------------------------------------------------------
# _lambda_audit grid parametrization (plan §12 assumption 11)
# ---------------------------------------------------------------------------
def _fake_sweep(lam_matrix):
    return {"gcv_lambda": np.asarray(lam_matrix, dtype=np.float64)}


def test_lambda_audit_default_grid_is_module_committed():
    g = [float(v) for v in fc.LAMBDAS]
    audit = f36._lambda_audit(_fake_sweep([[g[0], g[3]], [g[-1], np.nan]]), (0,))
    assert audit["grid"] == g
    assert audit["n_selected"] == 3
    assert audit["n_at_low_edge"] == 1 and audit["n_at_high_edge"] == 1


def test_lambda_audit_custom_grid_and_step_fractions():
    g = [1e-3, 1e-2, 1e-1, 1.0]
    audit = f36._lambda_audit(_fake_sweep([[1e-3, 1e-2, 1e-1, 1.0]]), (0,), grid=g)
    assert audit["grid"] == g
    assert audit["n_at_low_edge"] == 1 and audit["n_at_high_edge"] == 1
    assert audit["frac_at_low_edge"] == 0.25 and audit["frac_at_high_edge"] == 0.25
    # Within-one-step INCLUDES the exact edge (<= grid[1] / >= grid[-2]).
    assert audit["n_within_one_step_low"] == 2 and audit["n_within_one_step_high"] == 2
    assert audit["frac_within_one_step_low"] == 0.5
    assert audit["frac_within_one_step_high"] == 0.5


# ---------------------------------------------------------------------------
# Adaptive edge rule (plan §4 Phase FIT)
# ---------------------------------------------------------------------------
def test_edge_extend_grid_one_decade_half_step():
    base = np.logspace(-3, 8, 23)
    low = f36._edge_extend_grid(base, low=True, high=False)
    assert len(low) == 25
    np.testing.assert_allclose(low[0], 1e-4, rtol=1e-12)
    np.testing.assert_allclose(low[2:], base, rtol=0)
    both = f36._edge_extend_grid(base, low=True, high=True)
    assert len(both) == 27
    np.testing.assert_allclose(both[-1], 1e9, rtol=1e-12)
    steps = np.diff(np.log10(both))
    assert np.all(steps > 0.49) and np.all(steps < 0.51)


def _stub_sweep_fn(select):
    """Signature-conformant heldout_r2_sweep stand-in: selects `select(grid)`
    at every (layer, fold) of a (1 layer x 2 folds) lambda matrix."""

    def fn(X, Y, conv_ids, *, lambdas=None, **kw):
        v = select(np.asarray(lambdas, dtype=np.float64))
        return {"gcv_lambda": np.asarray([[v, v]]), "r2_obs": np.asarray([0.0])}

    return fn


def test_edge_rule_extends_twice_then_labels():
    x = np.zeros((4, 1, 2), dtype=np.float32)
    _sweep, edge, grid = f36._run_sweep_edge(
        x,
        x,
        np.asarray(["a", "b", "c", "d"]),
        base_grid=np.logspace(-3, 8, 23),
        sweep_kwargs={},
        sweep_fn=_stub_sweep_fn(lambda g: float(g[0])),  # always at the LOW edge
    )
    assert edge["extensions_low"] == 2 and edge["extensions_high"] == 0
    assert edge["estimator_limited"] == "lambda-edge"
    np.testing.assert_allclose(grid[0], 1e-5, rtol=1e-12)  # 2 one-decade extensions
    assert len(edge["history"]) == 3  # base + 2 re-runs


def test_edge_rule_clean_selection_single_sweep():
    x = np.zeros((4, 1, 2), dtype=np.float32)
    _sweep, edge, grid = f36._run_sweep_edge(
        x,
        x,
        np.asarray(["a", "b", "c", "d"]),
        base_grid=np.logspace(-3, 8, 23),
        sweep_kwargs={},
        sweep_fn=_stub_sweep_fn(lambda g: float(g[len(g) // 2])),  # mid-grid
    )
    assert edge["extensions_low"] == 0 and edge["extensions_high"] == 0
    assert edge["estimator_limited"] is None
    assert len(edge["history"]) == 1
    assert len(grid) == 23


def test_edge_rule_none_grid_is_single_v1_sweep():
    """base_grid=None = the byte-identical v1 path (real production body)."""
    rng = np.random.default_rng(0)
    n, d = 30, 8
    X = rng.normal(size=(n, 1, d)).astype(np.float32)
    Y = rng.normal(size=(n, 1, d)).astype(np.float32)
    conv = np.asarray([f"c{i}" for i in range(n)])
    kw = dict(n_folds=3, seed=0, null_draws=0, collect_cosines=False, frozen_layers=())
    sweep, edge, grid = f36._run_sweep_edge(X, Y, conv, base_grid=None, sweep_kwargs=kw)
    assert edge is None and grid is None
    ref = fc.heldout_r2_sweep(X, Y, conv, **kw)
    np.testing.assert_allclose(sweep["r2_obs"], ref["r2_obs"], rtol=0, atol=0)


def test_edge_rule_real_sweep_noise_hits_high_edge():
    """Real production body: pure-noise Y drives selection to the HIGH edge,
    the loop extends twice (cap) and labels the cell estimator-limited."""
    rng = np.random.default_rng(1)
    n, d = 60, 16  # fold-train ~48 > d=16 -> the primal path runs here too
    X = rng.normal(size=(n, 1, d)).astype(np.float32)
    Y = rng.normal(size=(n, 1, d)).astype(np.float32)
    conv = np.asarray([f"c{i}" for i in range(n)])
    kw = dict(n_folds=3, seed=0, null_draws=0, collect_cosines=False, frozen_layers=())
    _sweep, edge, grid = f36._run_sweep_edge(
        X,
        Y,
        conv,
        base_grid=np.logspace(-3, 8, 23),
        sweep_kwargs=kw,
    )
    assert edge["extensions_high"] == 2
    assert edge["estimator_limited"] == "lambda-edge"
    np.testing.assert_allclose(grid[-1], 1e10, rtol=1e-12)


# ---------------------------------------------------------------------------
# --x-slot selection (plan §4 divergence 7)
# ---------------------------------------------------------------------------
def _tiny_bundle(n=6, layers=2, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    slots = rng.normal(size=(n, 2, layers, dim)).astype(np.float32)
    profiles = rng.normal(size=(n, 2, layers, dim)).astype(np.float32)
    return {
        "arrays": {"slots": slots, "profiles": profiles},
        "sidecar": {"conv_ids": [f"s{i}" for i in range(n)]},
    }


def test_cell_xy_x_slot_selects_prefix_vs_context():
    b = _tiny_bundle()
    ctx = f36._cell_xy_1336(b, 2)  # default: context (byte-preserving)
    np.testing.assert_array_equal(ctx["X"], b["arrays"]["slots"][:, 1])
    pre = f36._cell_xy_1336(b, 2, x_slot="prefix")
    np.testing.assert_array_equal(pre["X"], b["arrays"]["slots"][:, 0])
    # Y is the a1 profile on BOTH arms.
    np.testing.assert_array_equal(ctx["Y"], b["arrays"]["profiles"][:, 1])
    np.testing.assert_array_equal(pre["Y"], b["arrays"]["profiles"][:, 1])
