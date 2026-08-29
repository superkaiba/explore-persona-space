from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue2569_fewshot_transfer as FT  # noqa: E402


def test_affine_span_recovers_linear_map() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(32, 5)).astype(np.float32)
    operator = rng.normal(size=(5, 4)).astype(np.float32)
    bias = rng.normal(size=4).astype(np.float32)
    y = x @ operator + bias
    pred = FT.affine_span_predict(
        x[16:], x[:16], y[:16], ridge_fraction=1e-8, device="cpu"
    )
    assert np.allclose(pred, y[16:], atol=2e-4, rtol=2e-4)


def test_prediction_metrics_identity() -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(size=(20, 7))
    metrics = FT.prediction_metrics(y, y, y, y[:10].mean(0))
    assert metrics["observed_target"]["pooled_r2"] == 1.0
    assert metrics["observed_target"]["train_mean_normalized_r2"] == 1.0
    assert metrics["full_target_mapping"]["normalized_r2"] == 1.0
    assert metrics["full_target_mapping"]["relative_l2"] == 0.0


def test_repeat_summary_uses_requested_quantiles() -> None:
    rows = []
    for value in range(10):
        block = {
            "observed_target": {
                "pooled_r2": value,
                "train_mean_normalized_r2": value,
                "centered_cosine": value,
            },
            "full_target_mapping": {
                "normalized_r2": value,
                "centered_cosine": value,
                "relative_l2": value,
            },
        }
        rows.append(block)
    summary = FT.summarize_repeats(rows)
    assert summary["n_repeats"] == 10
    assert summary["observed_target"]["pooled_r2"]["median"] == 4.5
    assert np.allclose(
        summary["observed_target"]["pooled_r2"]["q10_q90"], [0.9, 8.1]
    )


def test_paired_advantage_preserves_repeat_differences() -> None:
    def row(value: float) -> dict[str, dict[str, float]]:
        return {
            "observed_target": {
                "pooled_r2": value,
                "train_mean_normalized_r2": value,
                "centered_cosine": value,
            },
            "full_target_mapping": {
                "normalized_r2": value,
                "centered_cosine": value,
                "relative_l2": value,
            },
        }

    advantage = FT.summarize_paired_advantage(
        [row(2.0), row(4.0)], [row(1.0), row(6.0)]
    )
    metric = advantage["full_target_mapping"]["normalized_r2"]
    assert metric["values"] == [1.0, -2.0]
    assert metric["median"] == -0.5
    assert metric["fraction_positive"] == 0.5
