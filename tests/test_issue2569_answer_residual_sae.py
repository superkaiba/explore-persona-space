from __future__ import annotations

import numpy as np

from scripts import issue2569_answer_residual_sae as residual_sae


def test_stable_order_breaks_ties_by_feature_id() -> None:
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    assert residual_sae.stable_order(scores, descending=True).tolist() == [0, 1, 2, 3]
    assert residual_sae.stable_order(scores, descending=False).tolist() == [2, 3, 0, 1]


def test_feature_scores_match_small_analytic_case(monkeypatch) -> None:
    monkeypatch.setattr(residual_sae, "N_FEATURES", 3)
    values = np.array([[2.0, 0.0], [0.0, 1.0]])
    decoder = residual_sae.unit_rows(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

    scores = residual_sae.feature_scores(values, decoder, block=2)

    np.testing.assert_allclose(scores["pair_direction_rms"], np.full(3, np.sqrt(0.5)), atol=1e-12)
    np.testing.assert_allclose(
        scores["raw_energy_rms"],
        np.array([np.sqrt(2.0), np.sqrt(0.5), np.sqrt(1.25)]),
        atol=1e-12,
    )
    mean_direction = np.array([1.0, 0.5]) / np.linalg.norm([1.0, 0.5])
    np.testing.assert_allclose(
        scores["mean_direction_abs_cos"], np.abs(mean_direction @ decoder.T), atol=1e-12
    )


def test_analyze_family_reconstructs_observation(monkeypatch) -> None:
    monkeypatch.setattr(residual_sae, "D_MODEL", 2)
    monkeypatch.setattr(residual_sae, "N_FEATURES", 3)
    monkeypatch.setattr(residual_sae, "N_BOOT", 20)
    context = np.array([[1.0, 0.0], [0.0, 1.0]])
    observed = 2.0 * context
    operator = np.eye(2)
    decoder = residual_sae.unit_rows(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

    metrics, rankings, arrays = residual_sae.analyze_family(
        "synthetic",
        context,
        observed,
        ["p0", "p1"],
        operator,
        decoder,
        {},
        seed_offset=0,
    )

    assert metrics["max_abs_reconstruction_error"] == 0.0
    assert metrics["cosine_observed_predicted"]["mean"] == 1.0
    assert metrics["predicted_over_observed_norm"]["mean"] == 0.5
    assert metrics["residual_over_observed_norm"]["mean"] == 0.5
    assert set(rankings) == {"observed", "predicted", "residual"}
    assert len(arrays) == 9
