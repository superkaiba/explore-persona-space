# ruff: noqa: RUF002
# (multiplication-sign characters intentional in docstrings)
"""Tests for issue #649 LEVEL/CHANGE decomposition analysis-script invariants.

These pin the load-bearing analysis behavior on small SYNTHETIC data (4 sources ×
8 bystanders), with NO model / NO HF dependency:
  - bystander-grouped CV uses min(5, n_unique_bystanders) folds and groups BY
    BYSTANDER (every source stays in every training fold -> M0 identifiable);
  - the six-regression ladder returns the #532 ΔCV-R² uplifts;
  - the marginal-Spearman table has 6 rows with bootstrap CIs;
  - cell exclusions (diagonal + trained negatives) fire;
  - the #391 forced-choice S/N gate + collinearity gate compute.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "issue649_level_change_decomp", SCRIPTS / "issue649_level_change_decomp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _synthetic_table(mod, n_sources=4, n_bystanders=8, seed=0):
    """Build a synthetic Phase-2 table dict: n_sources × n_bystanders off-diagonal
    cells with LEVEL driven by prior and CHANGE driven by cosine."""
    rng = np.random.default_rng(seed)
    rows = []
    src_names = [f"src{i}" for i in range(n_sources)]
    by_names = [f"by{j}" for j in range(n_bystanders)]
    for s in src_names:
        for b in by_names:
            prior = float(rng.uniform(0.0, 0.3))
            cos = float(rng.uniform(0.7, 1.0))
            kl = float(rng.uniform(0.0, 5.0))
            level = 0.8 * prior + 0.02 * rng.standard_normal() + 0.05
            change = 0.3 * (1.0 - cos) + 0.02 * rng.standard_normal()
            rows.append(
                {
                    "source": s,
                    "bystander": b,
                    "level": level,
                    "change": change,
                    "base_prior": prior,
                    "cos_L2_eos": cos,
                    "cos_L7_lp": cos,
                    "cos_L20_robust": cos,
                    "kl_L2": kl,
                    "kl_L7": kl,
                    "t_seed_42": level,
                    "t_seed_137": level,
                    "n_seeds": 2,
                }
            )

    def col(name):
        return np.array([r[name] for r in rows], dtype=np.float64)

    source_ids = [r["source"] for r in rows]
    bystander_ids = [r["bystander"] for r in rows]
    src_to_int = {s: i for i, s in enumerate(sorted(set(source_ids)))}
    by_to_int = {b: i for i, b in enumerate(sorted(set(bystander_ids)))}
    return {
        "arm": "arm_canned",
        "n_cells": len(rows),
        "rows": rows,
        "level": col("level"),
        "change": col("change"),
        "base_prior": col("base_prior"),
        "cos_L2_eos": col("cos_L2_eos"),
        "cos_L7_lp": col("cos_L7_lp"),
        "cos_L20_robust": col("cos_L20_robust"),
        "kl_L2": col("kl_L2"),
        "kl_L7": col("kl_L7"),
        "t_seed_42": col("t_seed_42"),
        "t_seed_137": col("t_seed_137"),
        "source_ids": source_ids,
        "bystander_ids": bystander_ids,
        "source_group": np.array([src_to_int[s] for s in source_ids]),
        "bystander_group": np.array([by_to_int[b] for b in bystander_ids]),
        "source_onehot": mod._one_hot(source_ids),
    }


# ── CV grouping invariants (the binding round-1 identifiability fix) ──


def test_cv_uses_min5_and_groups_by_bystander(mod, monkeypatch):
    """8 bystanders -> 5-fold grouped CV (min(5, 8)=5); the GroupKFold split must
    receive the bystander id as `groups`, and every fold must keep some of EVERY
    source in training (so M0's per-source one-hots are fit)."""
    n_sources, n_bystanders = 4, 8
    tbl = _synthetic_table(mod, n_sources=n_sources, n_bystanders=n_bystanders)

    captured = {}
    import sklearn.model_selection as msel

    real_split = msel.GroupKFold.split

    def spy_split(self, X, y=None, groups=None):
        captured["n_splits"] = self.n_splits
        captured["n_unique_groups"] = len(np.unique(groups))
        captured["folds"] = []
        for tr, te in real_split(self, X, y, groups=groups):
            captured["folds"].append((tr, te))
            yield tr, te

    monkeypatch.setattr(msel.GroupKFold, "split", spy_split)

    out = mod.six_regression_ladder(tbl, "level", tbl["bystander_group"])
    assert out["M0_source_indicators"] is not None
    assert not np.isnan(out["M0_source_indicators"]), "bystander-grouped M0 must be identifiable"
    # min(5, n_unique_bystanders) = 5
    assert captured["n_splits"] == 5
    assert captured["n_unique_groups"] == n_bystanders
    # every training fold contains at least one cell of every source (identifiability)
    onehot = tbl["source_onehot"]
    for tr, _te in captured["folds"]:
        cols_present = onehot[tr].sum(axis=0)
        assert (cols_present > 0).all(), (
            "a source vanished from a training fold (M0 not identifiable)"
        )


def test_cv_folds_equal_min5_n_bystanders_small(mod):
    """With 4 unique bystanders the headline CV degenerates to 4-fold (min(5,4));
    confirms the fold count tracks n_unique_bystanders below 5. (3 bystanders ->
    only 3 cells/source falls under _cv_r2_grouped's len(y)<5 NaN floor, which is
    why the production smoke's tiny ladder is expected to read NaN — that is the
    plumbing-only smoke, not a statistical claim.)"""
    tbl = _synthetic_table(mod, n_sources=2, n_bystanders=4)
    import sklearn.model_selection as msel

    captured = {}
    real_split = msel.GroupKFold.split

    def spy(self, X, y=None, groups=None):
        captured["n_splits"] = self.n_splits
        captured["n_unique_groups"] = len(np.unique(groups))
        yield from real_split(self, X, y, groups=groups)

    orig = msel.GroupKFold.split
    msel.GroupKFold.split = spy
    try:
        mod.six_regression_ladder(tbl, "change", tbl["bystander_group"])
    finally:
        msel.GroupKFold.split = orig
    assert captured["n_unique_groups"] == 4
    assert captured["n_splits"] == 4  # min(5, 4)


def test_source_grouped_cv_uses_source_ids(mod):
    """The robustness column groups by SOURCE (4 sources -> 4-fold)."""
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    import sklearn.model_selection as msel

    captured = {}
    real_split = msel.GroupKFold.split

    def spy(self, X, y=None, groups=None):
        captured["n_unique_groups"] = len(np.unique(groups))
        captured["n_splits"] = self.n_splits
        yield from real_split(self, X, y, groups=groups)

    orig = msel.GroupKFold.split
    msel.GroupKFold.split = spy
    try:
        mod.six_regression_ladder(tbl, "level", tbl["source_group"])
    finally:
        msel.GroupKFold.split = orig
    assert captured["n_unique_groups"] == 4
    assert captured["n_splits"] == 4  # min(5, 4)


# ── ladder + Spearman shape ──


def test_ladder_returns_six_models_and_deltas(mod):
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    out = mod.six_regression_ladder(tbl, "level", tbl["bystander_group"])
    for k in (
        "M0_source_indicators",
        "M1_plus_prior",
        "M2_plus_prior_cosine",
        "M3_cosine_only",
        "M4_plus_prior_kl",
        "M5_kl_only",
        "delta_prior_beyond_M0",
        "delta_cosine_beyond_M1",
        "delta_cosine_beyond_M0",
    ):
        assert k in out
    # On synthetic data where LEVEL is prior-driven, prior should add CV-R² over M0.
    assert out["delta_prior_beyond_M0"] > 0.0


def test_marginal_spearman_headline_rows_and_cis(mod):
    """The HEADLINE set is the canonical 3 predictors × {LEVEL, CHANGE} = 6 rows
    (plan §6.5); secondary/robustness cells (cosine_L7, kl_L7, cosine_L20_robust)
    are ALSO emitted, flagged headline=False."""
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    rows = mod.marginal_spearman_table(tbl, n_boot=50)
    headline = [r for r in rows if r["headline"]]
    assert len(headline) == 6, "headline = 3 predictors x 2 DVs"
    hpreds = {(r["predictor"], r["dv"]) for r in headline}
    assert ("prior", "LEVEL") in hpreds
    assert ("cosine_L2", "CHANGE") in hpreds
    assert ("kl_L2", "CHANGE") in hpreds
    # secondary cells present + flagged non-headline
    allpreds = {(r["predictor"], r["dv"]) for r in rows}
    assert ("cosine_L7", "CHANGE") in allpreds
    assert ("cosine_L20_robust", "LEVEL") in allpreds
    for r in rows:
        assert "ci95_low" in r and "ci95_high" in r and "ci_covers_zero" in r


def test_marginal_spearman_tolerates_missing_l20(mod):
    """A table missing cos_L20_robust must not crash; the L20 rows read NaN."""
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    del tbl["cos_L20_robust"]
    rows = mod.marginal_spearman_table(tbl, n_boot=50)
    l20 = [r for r in rows if r["predictor"] == "cosine_L20_robust"]
    assert len(l20) == 2
    assert all(np.isnan(r["spearman_rho"]) for r in l20)


def test_intercept_only_ladder_identifiable(mod):
    """The source-grouped intercept-only-M0 variant returns a non-NaN ΔCV-R² for
    prior over the intercept (apples-to-apples generalization read)."""
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    out = mod.intercept_only_ladder(tbl, "level", tbl["source_group"])
    for k in ("M0_intercept_only", "delta_prior_beyond_intercept", "delta_cosine_beyond_intercept"):
        assert k in out
    assert not np.isnan(out["M0_intercept_only"])


def test_noncircular_partials_on_trained_rate(mod):
    """The non-circular partials on t (LEVEL) controlling the other predictor are
    computed with bootstrap CIs (plan §8 risk row 3)."""
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    out = mod.noncircular_partials(tbl, n_boot=50)
    for key in ("prior_vs_t_given_cosine_L2", "cosine_L2_vs_t_given_prior"):
        assert key in out
        blk = out[key]
        assert "partial_spearman" in blk and "ci95_low" in blk and "ci95_high" in blk
    # On synthetic LEVEL=0.8*prior, prior should still predict t after partialling cosine.
    assert out["prior_vs_t_given_cosine_L2"]["partial_spearman"] > 0.0


# ── gates ──


def test_forced_choice_gate_computes_sn(mod):
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    g = mod.forced_choice_gate(tbl)
    assert "signal_to_noise" in g and "cluster_honest_wilson_half_width" in g
    assert g["cluster_honest_wilson_half_width"] > 0.0
    assert g["fired"] in (True, False)


def test_collinearity_gate_reports_pearson(mod):
    tbl = _synthetic_table(mod, n_sources=4, n_bystanders=8)
    g = mod.collinearity_gate(tbl)
    assert "pearson_abs_cos_prior" in g
    assert g["fired"] in (True, False)
    assert -1.0 <= g["pearson_abs_cos_prior"] <= 1.0


def test_wilson_half_width_known_value(mod):
    # p=0.5, n=60 -> roughly the plan's cluster-honest figure (~0.11 region).
    hw = mod._wilson_half_width(0.5, 60)
    assert 0.08 < hw < 0.15


# ── cell-exclusion logic (diagonal + trained negatives) ──


def test_build_table_excludes_diagonal_and_trained_negatives(mod):
    """build_table must drop source==bystander and per-source trained negatives."""
    # Fake panel_set: 3 personas; by1 is a trained negative for src 'villain'.
    panel_set = {
        "personas": {
            "villain": {"prompt": "p", "neg_member_for": []},
            "by1": {"prompt": "p", "neg_member_for": ["villain"]},
            "by2": {"prompt": "p", "neg_member_for": []},
        }
    }
    names = ["villain", "by1", "by2"]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    geom = {
        "idx": idx,
        "cos_L2_eos": np.full((n, n), 0.9),
        "cos_L7_lp": np.full((n, n), 0.9),
        "kl_L2": np.full((n, n), 1.0),
        "kl_L7": np.full((n, n), 1.0),
    }
    # monkeypatch the rate readers to constants (no disk/HF)
    orig_base = mod._read_base_rate
    orig_trained = mod._read_trained_rate
    mod._read_base_rate = lambda by: 0.1
    mod._read_trained_rate = lambda inputs_dir, arm, src, seed, by: 0.4
    try:
        tbl = mod.build_table(
            "arm_canned", ("villain",), names, (42, 137), panel_set, geom, Path("/tmp")
        )
    finally:
        mod._read_base_rate = orig_base
        mod._read_trained_rate = orig_trained
    # villain==villain diagonal excluded; by1 trained-negative excluded; only by2 survives.
    assert tbl["n_cells"] == 1
    assert tbl["rows"][0]["bystander"] == "by2"
    assert tbl["excluded"]["diagonal"] == 1
    assert tbl["excluded"]["trained_negative"] == 1
    # CHANGE = t - b = 0.4 - 0.1 = 0.3
    assert abs(tbl["rows"][0]["change"] - 0.3) < 1e-9
    assert abs(tbl["rows"][0]["level"] - 0.4) < 1e-9


# ── vendored Gaussian-KL parity (non-singular at >= 2k probes) ──


def test_gaussian_kl_nonsingular_with_enough_probes(mod):
    rng = np.random.default_rng(1)
    k = 16
    Xa = rng.standard_normal((40, 64))  # 40 probes >= 2k=32
    Xb = rng.standard_normal((40, 64)) + 0.5
    v = mod._gaussian_sym_kl_in_subspace_local(Xa, Xb, k)
    assert not np.isnan(v) and v >= 0.0


def test_gaussian_kl_symmetric(mod):
    rng = np.random.default_rng(2)
    k = 8
    Xa = rng.standard_normal((40, 32))
    Xb = rng.standard_normal((40, 32)) + 0.3
    v_ab = mod._gaussian_sym_kl_in_subspace_local(Xa, Xb, k)
    v_ba = mod._gaussian_sym_kl_in_subspace_local(Xb, Xa, k)
    assert abs(v_ab - v_ba) < 1e-6
