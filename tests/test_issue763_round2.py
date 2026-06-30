# ruff: noqa: RUF002, RUF003
"""Round-2 regression tests for the issue #763 code-review BLOCKERs.

Three permanent invariants, each failing pre-fix / passing post-fix:

1. ``fact-pool-distinct-probes`` — ``freeze_pool`` raises on ANY exact-duplicate
   probe (duplicate probes inflate √(r_yy) by construction), and the
   fact_expression pool builder produces DISTINCT framings (no ``flat[i % len]``
   cycle backfill).
2. ``ridge-pca-comparator`` — the ridge LOCO arm fits on the SAME nested-CV
   PCA-reduced features the GLM consumes (input dim ∈ the {2,4,6,8,10,15,20}
   grid), so the registered ρ_ridge − ρ_GLM optimism delta is apples-to-apples.

These are behavior-focused: they trip the actual guards / shared reduction path
the round-2 fixes added, not implementation details.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))


# ── BLOCKER fact-pool-distinct-probes ─────────────────────────────────────────


def test_freeze_pool_rejects_duplicate_probes():
    """A pool with ANY exact duplicate fails loud (the banned silent backfill)."""
    import issue763_build_probe_pools as bp

    dup_pool = ["probe A", "probe B", "probe A", "probe C"]  # 'probe A' twice
    with pytest.raises(RuntimeError, match="duplicate probe"):
        bp.freeze_pool("fact_expression", dup_pool, smoke=True)


def test_freeze_pool_accepts_distinct_pool(tmp_path, monkeypatch):
    """A fully-distinct pool freezes cleanly (the post-fix happy path)."""
    import issue763_build_probe_pools as bp

    # Redirect the freeze write to a tmp dir so the test never touches the repo.
    monkeypatch.setattr(bp, "probe_pool_path", lambda b: tmp_path / f"{b}.json")
    distinct = [f"distinct probe {i}" for i in range(12)]
    pool = bp.freeze_pool("fact_expression", distinct, smoke=True)
    assert pool["n_probes"] == 12
    assert len(set(pool["probes"])) == len(pool["probes"])


def test_fact_expression_real_pool_is_distinct(monkeypatch):
    """``_build_real_pool('fact_expression', 60)`` yields DISTINCT framings.

    Stubs ``build_fact_battery`` to return a 4-proposition payload (the post-fix
    shape) so the test needs no Sonnet call, and asserts the flattened pool has
    NO duplicates — the ``flat[i % len]`` cycle would have produced repeats.
    """
    import issue763_build_probe_pools as bp

    from explore_persona_space.experiments.behavior_testbed_545 import corpora

    payload = {
        "direct": [f"prop1 direct {i}" for i in range(3)],
        "ood_framings": [f"prop1 ood {i}" for i in range(11)],
        "entailed": [f"prop1 entailed {i}" for i in range(2)],
        "extra_propositions": [
            {
                "direct": [f"prop{p} direct {i}" for i in range(3)],
                "ood_framings": [f"prop{p} ood {i}" for i in range(11)],
                "entailed": [f"prop{p} entailed {i}" for i in range(2)],
            }
            for p in (2, 3, 4)
        ],
        "reversal": [f"reversal {i}" for i in range(3)],
    }

    fake_path = "/tmp/fake_fact_battery.json"
    monkeypatch.setattr(corpora, "build_fact_battery", lambda: fake_path)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: corpora.json.dumps(payload))

    probes = bp._build_real_pool("fact_expression", 60)
    # The 4-proposition payload supplies 16 + 3×16 + 3 = 67 strings, all distinct.
    assert len(probes) == 60  # truncated to the target
    assert len(set(probes)) == len(probes), "fact_expression pool contains duplicates"


def test_fact_expression_pool_reports_shortfall_not_backfill(monkeypatch, caplog):
    """A short distinct pool truncates to its size + WARNs — never cycle-padded."""
    import logging

    import issue763_build_probe_pools as bp

    from explore_persona_space.experiments.behavior_testbed_545 import corpora

    payload = {
        "direct": [f"only {i}" for i in range(5)],  # 5 distinct framings, target 60
        "ood_framings": [],
        "entailed": [],
        "extra_propositions": [],
        "reversal": [],
    }
    monkeypatch.setattr(corpora, "build_fact_battery", lambda: "/tmp/x.json")
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: corpora.json.dumps(payload))
    with caplog.at_level(logging.WARNING):
        probes = bp._build_real_pool("fact_expression", 60)
    assert len(probes) == 5  # the distinct count, NOT 60 (no cycle backfill)
    assert len(set(probes)) == 5
    assert any("under-fill" in r.message or "yield_shortfall" in r.message for r in caplog.records)


# ── BLOCKER ridge-pca-comparator ──────────────────────────────────────────────


def test_ridge_fits_on_pca_reduced_features():
    """The ridge LOCO arm consumes PCA-reduced features (input dim ∈ the grid).

    Monkeypatch ``nested_cv_pca_reduce`` to record the reduced-feature dim it
    returns each fold; assert (a) the ridge arm actually called it (so it is NOT
    fitting raw 3584-d x), and (b) every recorded dim is in the plan §11 grid.
    This is the capacity-match invariant the ρ_ridge − ρ_GLM optimism delta rests
    on (BLOCKER ridge-pca-comparator).
    """
    import issue763_fit_predictors as fit

    from explore_persona_space.analysis import issue_763_pca

    rng = np.random.default_rng(0)
    n, h = 24, 64  # n>>grid, h>>any chosen d so a raw fit would be 64-d not ≤20-d
    x = rng.standard_normal((n, h))
    y = rng.uniform(0.1, 0.9, size=n)
    n_judged = np.full(n, 30)

    grid = issue_763_pca.PCA_DIM_GRID
    seen_dims: list[int] = []
    real = issue_763_pca.nested_cv_pca_reduce

    def _spy(x_train, x_test, **kw):
        x_tr_red, x_te_red, d = real(x_train, x_test, **kw)
        seen_dims.append(x_tr_red.shape[1])  # the actual reduced feature width
        return x_tr_red, x_te_red, d

    # patch BOTH the module symbol and the fit-script's imported binding
    import explore_persona_space.analysis.issue_763_pca as pca_mod

    pca_mod.nested_cv_pca_reduce = _spy
    fit.nested_cv_pca_reduce = _spy
    try:
        pred = fit._ridge_predict_loco_pca(x, y, n_judged, fit.RIDGE_LAMBDAS)
    finally:
        pca_mod.nested_cv_pca_reduce = real
        fit.nested_cv_pca_reduce = real

    assert pred.shape == (n,)
    assert seen_dims, "ridge never called nested_cv_pca_reduce — it is fitting RAW features"
    assert len(seen_dims) == n  # one reduction per LOCO fold
    # every fold's reduced width must be a grid dim (capped at n_train//5 = 4 here)
    assert all(d in grid for d in seen_dims), seen_dims
    assert max(seen_dims) <= h, "reduced width exceeded the input width (impossible)"
    # the cap: with n_train=23, d_max = 23//5 = 4, so only grid dims ≤4 are eligible
    assert max(seen_dims) <= 4, f"p≪n cap not applied: {seen_dims}"


def test_glm_and_ridge_share_the_same_reduction_helper():
    """Both arms route through the SAME shared nested_cv_pca_reduce (matched space)."""
    import issue763_fit_predictors as fit

    from explore_persona_space.analysis import issue_763_glm

    # Both modules import the SAME function object — the capacity-match contract.
    assert fit.nested_cv_pca_reduce is issue_763_glm.nested_cv_pca_reduce
