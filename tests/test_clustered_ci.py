# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, ≤) in scientific docstrings + asserts.
"""Family-clustered bootstrap CI for issue-666 (plan §6 C4, §11).

The 50 battery contexts cluster by FAMILY (7 families) and share the 48-probe
pool, so n=50 has far fewer effective d.o.f. than a naive bootstrap assumes. The
headline CIs MUST resample at the CLUSTER level (resample FAMILIES, then the
contexts within each drawn family), NOT resample the 50 contexts independently
(plan §6: "resample at cluster level, NEVER naive n=50").

These tests pin: (1) the clustered bootstrap resamples FAMILIES (the same family
appears together or not at all within a resample); (2) the clustered CI is WIDER
than the naive n=50 CI (the effective-n deflation), on a synthetic dataset with
strong within-family correlation.

CPU-only; synthetic 7-family dataset; no store, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _LazyModule:
    """Proxy that imports a per-issue script on first attribute access (TDD).

    The net-new script does NOT exist this round, so the first ``ci.<fn>``
    access inside each test raises ImportError → the test FAILS (not skips).
    A module-level ``importorskip`` was rejected because it skips COLLECTION,
    so the proposed-test count could not be verified by approve-tests.
    """

    def __init__(self, dotted: str):
        object.__setattr__(self, "_dotted", dotted)

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)

    def __setattr__(self, name, value):
        # Forward attribute SETS to the real module so monkeypatch.setattr(proxy,
        # ...) patches the module function the implementation actually calls (and
        # monkeypatch's teardown restore forwards back the same way). Without this,
        # a set landed on the proxy instance and the real module stayed unpatched.
        import importlib

        setattr(importlib.import_module(self._dotted), name, value)


ci = _LazyModule("issue666_predictor")

N_FAMILIES = 7
PER_FAMILY = 8  # 7 × 8 = 56 synthetic contexts (the real battery is 50/7-family)


def _make_family_dataset(seed, within_family_corr):
    """7 families × PER_FAMILY contexts with a tunable within-family correlation.

    Returns (x, y, families): predictor x, target y, and the family label array.
    A high within_family_corr means contexts in a family move together → the
    effective n is closer to 7 than to 56.
    """
    rng = np.random.default_rng(seed)
    xs, ys, fams = [], [], []
    for f in range(N_FAMILIES):
        family_effect_x = rng.standard_normal()
        family_effect_y = rng.standard_normal()
        for _ in range(PER_FAMILY):
            ex = (
                within_family_corr * family_effect_x
                + (1 - within_family_corr) * rng.standard_normal()
            )
            ey = (
                within_family_corr * family_effect_y
                + (1 - within_family_corr) * rng.standard_normal()
            )
            xs.append(ex + 0.5 * ey)
            ys.append(ey)
            fams.append(f)
    return np.array(xs), np.array(ys), np.array(fams)


# ---------------------------------------------------------------------------
# Clustered bootstrap resamples FAMILIES, not contexts.
# ---------------------------------------------------------------------------
def test_clustered_bootstrap_resamples_families(monkeypatch):
    """Every bootstrap replicate is built from whole-family draws (cluster resampling)."""
    x, y, fams = _make_family_dataset(seed=0, within_family_corr=0.8)

    # Inspect what the estimator resamples by capturing the family draw.
    captured = {"draws": []}

    def _spy_families(family_ids, rng, n_draw):
        captured["draws"].append(tuple(sorted(family_ids)))
        return rng.choice(family_ids, size=n_draw, replace=True)

    # The estimator must expose a family-draw hook the test can observe.
    monkeypatch.setattr(ci, "draw_families", _spy_families, raising=True)

    lo, hi = ci.clustered_bootstrap_ci(
        x, y, clusters=fams, n_boot=200, seed=0, statistic="spearman"
    )
    # The family-draw hook was exercised (clusters, not contexts, are the unit).
    assert len(captured["draws"]) > 0, "clustered bootstrap must draw at the family level"
    # The drawn unit set is exactly the 7 family ids.
    assert all(set(d) == set(range(N_FAMILIES)) for d in captured["draws"])
    assert lo <= hi
    assert -1.0 <= lo <= 1.0 and -1.0 <= hi <= 1.0


def test_clustered_ci_wider_than_naive_n50():
    """Clustered CI > naive CI when within-family correlation is strong.

    With high within-family correlation the effective n collapses toward the
    family count (7), so the cluster-aware CI is materially wider than the naive
    independent-context CI (which over-counts d.o.f.).
    """
    x, y, fams = _make_family_dataset(seed=1, within_family_corr=0.85)

    lo_c, hi_c = ci.clustered_bootstrap_ci(
        x, y, clusters=fams, n_boot=2000, seed=11, statistic="spearman"
    )
    lo_n, hi_n = ci.naive_bootstrap_ci(x, y, n_boot=2000, seed=11, statistic="spearman")

    width_clustered = hi_c - lo_c
    width_naive = hi_n - lo_n
    assert width_clustered > width_naive, (
        f"clustered CI width ({width_clustered:.3f}) must exceed naive CI width "
        f"({width_naive:.3f}) under strong within-family correlation"
    )


def test_clustered_ci_brackets_point_estimate():
    from scipy.stats import spearmanr

    x, y, fams = _make_family_dataset(seed=2, within_family_corr=0.5)
    point = spearmanr(x, y).statistic
    lo, hi = ci.clustered_bootstrap_ci(
        x, y, clusters=fams, n_boot=1000, seed=22, statistic="spearman"
    )
    assert lo <= point <= hi, f"CI [{lo:.3f}, {hi:.3f}] must bracket the point estimate {point:.3f}"
