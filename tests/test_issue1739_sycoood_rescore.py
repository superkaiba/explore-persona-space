"""Group-grain (cluster) uncertainty pins for scripts/issue1739_sycoood_rescore.py.

The syco-OOD rescore reads rung-level Spearman against the GROUP grain
(sycomim: 285 rows / 15 independent artifacts), so `_compute_detection_metrics`
must emit `n_groups`, a cluster-bootstrap `ci_rho_group`, `rho_groupmean`, and a
`groupmean` permutation-null variant for grouped rungs. Synthetic arrays only —
no store, no network, CPU-instant.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rescore_mod():
    spec = importlib.util.spec_from_file_location(
        "issue1739_sycoood_rescore", REPO_ROOT / "scripts" / "issue1739_sycoood_rescore.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic(n_grouped: int = 40, n_flat: int = 30, n_groups: int = 4, seed: int = 7):
    """Two rungs: 'mim' (n_grouped rows in n_groups clusters), 'ays' (flat)."""
    rng = np.random.default_rng(seed)
    n = n_grouped + n_flat
    dv = rng.uniform(0, 100, size=n)
    # arm A tracks dv with noise; arm B is noise
    layers = 3
    sc_a = np.stack([dv + rng.normal(0, 20, size=n) for _ in range(layers)])
    sc_b = np.stack([rng.normal(0, 1, size=n) for _ in range(layers)])
    scores_ev = {"armA": sc_a, "armB": sc_b}
    rungs = ["mim"] * n_grouped + ["ays"] * n_flat
    groups = [f"g{i % n_groups}" for i in range(n_grouped)] + [f"solo{i}" for i in range(n_flat)]
    frozen = {"armA": 1, "armB": 0}
    return scores_ev, dv, rungs, groups, frozen


def test_group_grain_fields_present(rescore_mod):
    scores_ev, dv, rungs, groups, frozen = _synthetic()
    rows, nulls = rescore_mod._compute_detection_metrics(
        scores_ev,
        dv,
        rungs,
        frozen,
        (0, 1, 2),
        groups_ev=groups,
        n_boot=25,
        n_perm=10,
        all16=["armA", "armB"],
    )
    by = {(r["arm"], r["rung"]): r for r in rows}
    mim = by[("armA", "mim")]
    assert mim["n_groups"] == 4
    assert mim["n_rung"] == 40
    lo, hi = mim["ci_rho_group"]
    assert lo <= hi
    assert mim["rho_groupmean"] is not None and -1.0 <= mim["rho_groupmean"] <= 1.0
    # grouped-rung group CI must be at least as wide as it is finite; the
    # signal arm's point rho should be positive (tracks dv by construction)
    assert mim["rho"] > 0.3
    # flat rung: every context its own group -> n_groups == n_rung
    ays = by[("armA", "ays")]
    assert ays["n_groups"] == ays["n_rung"] == 30


def test_groupmean_null_only_on_grouped_rungs(rescore_mod):
    scores_ev, dv, rungs, groups, frozen = _synthetic()
    _, nulls = rescore_mod._compute_detection_metrics(
        scores_ev,
        dv,
        rungs,
        frozen,
        (0, 1, 2),
        groups_ev=groups,
        n_boot=10,
        n_perm=10,
        all16=["armA", "armB"],
    )
    assert "groupmean" in nulls["mim"] and nulls["mim"]["n_groups"] == 4
    assert "groupmean" not in nulls["ays"]


def test_group_boot_wider_than_context_boot_under_cluster_structure(rescore_mod):
    """With strong within-group correlation, the cluster CI must be wider."""
    rng = np.random.default_rng(3)
    n_groups, per = 6, 12
    g_effect = rng.normal(0, 30, size=n_groups)
    dv = np.concatenate([np.full(per, 50.0 + e) + rng.normal(0, 1, per) for e in g_effect])
    sc = dv + rng.normal(0, 5, size=dv.size)
    groups = np.array([f"g{i}" for i in range(n_groups) for _ in range(per)])
    gdraws = rescore_mod._group_boot_rhos(sc, dv, groups, n_boot=200, seed=0)
    from explore_persona_space.experiments.issue_1739.arms import (
        bootstrap_rhos,
        make_bootstrap_idx,
    )

    idx = make_bootstrap_idx(dv.size, n_boot=200, seed=0)
    cdraws = bootstrap_rhos(sc[None], dv, idx)[0]
    g_w = np.nanquantile(gdraws, 0.975) - np.nanquantile(gdraws, 0.025)
    c_w = np.nanquantile(cdraws, 0.975) - np.nanquantile(cdraws, 0.025)
    assert g_w >= c_w * 0.9  # cluster CI never materially narrower


def test_backwards_compatible_without_groups(rescore_mod):
    scores_ev, dv, rungs, _, frozen = _synthetic()
    rows, nulls = rescore_mod._compute_detection_metrics(
        scores_ev,
        dv,
        rungs,
        frozen,
        (0, 1, 2),
        n_boot=10,
        n_perm=10,
        all16=["armA", "armB"],
    )
    assert rows and all("n_groups" not in r for r in rows)
    assert all("groupmean" not in v for v in nulls.values())
