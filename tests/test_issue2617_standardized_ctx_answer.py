from __future__ import annotations

import numpy as np

from scripts.issue2617_standardized_ctx_answer import (
    PairRow,
    apply_pca_whitener,
    fit_pca_whitener,
    hedges_g,
    make_figure,
    make_folds,
)


def _pair(i: int, label: str, family: str) -> PairRow:
    gap = {"flip": 0.8, "nonflip": 0.0, "mid": 0.3}[label]
    return PairRow(
        pair_id=f"pair-{i}",
        pair_class="xstest",
        pair_source="xstest",
        artifact_family_id=family,
        a_idx=2 * i,
        b_idx=2 * i + 1,
        refusal_rate_a=gap,
        refusal_rate_b=0.0,
        behavior_gap=gap,
        outcome_group=label,
        hi_idx=2 * i,
        lo_idx=2 * i + 1,
        orientation="a_minus_b",
    )


def test_fold_assignment_is_complete_and_family_disjoint() -> None:
    labels = ["flip", "nonflip", "mid"]
    pairs = [_pair(i, labels[i % 3], f"family-{i // 2}") for i in range(60)]
    fold_of, meta = make_folds(pairs)
    assert set(fold_of.tolist()) == set(range(5))
    assert len(meta) == 5
    groups = np.array([pair.artifact_family_id for pair in pairs])
    for fold in range(5):
        assert not set(groups[fold_of == fold]) & set(groups[fold_of != fold])


def test_pca_whitening_has_unit_training_covariance() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(90, 80)) @ np.diag(np.linspace(0.5, 3.0, 80))
    mean, components, scales, diagnostics = fit_pca_whitener(x, max_rank=64)
    z = apply_pca_whitener(x, mean, components, scales, rank=32)
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(32), atol=1e-10)
    assert diagnostics["n_train_rows"] == 90


def test_hedges_g_matches_manual_small_sample_correction() -> None:
    flip = np.array([2.0, 3.0, 4.0, 5.0])
    nonflip = np.array([0.0, 1.0, 1.0, 2.0])
    result = hedges_g(flip, nonflip)
    pooled = np.sqrt((3 * flip.var(ddof=1) + 3 * nonflip.var(ddof=1)) / 6)
    d = (flip.mean() - nonflip.mean()) / pooled
    correction = 1 - 3 / (4 * 8 - 9)
    assert result["cohens_d"] == d
    assert result["hedges_g"] == correction * d


def test_figure_clamps_inverted_ci_offsets(tmp_path) -> None:
    def space(point: float) -> dict:
        return {
            "hedges_g": point,
            "hedges_g_ci95": [point + 0.2, point - 0.2],
            "nonflip": {"mean": 0.1, "mean_ci95": [0.3, -0.1]},
            "flip": {"mean": 1.5, "mean_ci95": [1.7, 1.3]},
        }

    summary = {
        "ranks": {
            "32": {
                "spaces": {"v_C": space(2.0), "v_A": space(1.8)},
                "headline_contrast": {"point": -0.2, "ci95": [-0.5, 0.1]},
            }
        }
    }
    out = tmp_path / "inverted-ci.png"
    make_figure(summary, out)
    assert out.stat().st_size > 1_000
