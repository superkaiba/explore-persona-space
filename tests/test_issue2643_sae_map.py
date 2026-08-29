from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.issue2643_sae_map import (
    apply_dense_ridge,
    binary_auroc,
    binary_average_precision,
    decode_bf16_uint16,
    feature_scale_apply,
    feature_scale_fit,
    pooled_r2,
    row_scores,
)


def _bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def test_bf16_codec_roundtrip() -> None:
    x = torch.tensor([[0.0, 1.25, -2.5], [3.0, -0.125, 7.75]])
    got = decode_bf16_uint16(_bf16_bits(x))
    torch.testing.assert_close(got, x.to(torch.bfloat16).float(), rtol=0, atol=0)
    with pytest.raises(TypeError):
        decode_bf16_uint16(np.zeros((2, 2), dtype=np.float32))


def test_dense_ridge_equation_and_shape_guard() -> None:
    x = torch.tensor([[3.0, 8.0]])
    ridge = {
        "xmu": torch.tensor([1.0, 2.0]),
        "xsd": torch.tensor([2.0, 3.0]),
        "ymu": torch.tensor([0.5, -1.0]),
        "W": torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
    }
    torch.testing.assert_close(apply_dense_ridge(x, ridge), torch.tensor([[2.5, 7.0]]))
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_dense_ridge(torch.zeros(1, 3), ridge)


def test_feature_scale_is_slope_only_and_improves_scale_mismatch() -> None:
    pred = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    target = 2.0 * pred
    scale = feature_scale_fit((pred * target).sum(0), pred.square().sum(0), ridge_to_identity=0.0)
    got = feature_scale_apply(pred, scale)
    assert torch.count_nonzero(got[pred == 0]) == 0
    assert torch.square(got - target).sum() < torch.square(pred - target).sum()


def test_row_scores_flag_injected_realized_answer_anomaly() -> None:
    n, d, f = 8, 4, 6
    x = torch.ones(n, d)
    y = torch.ones(n, d)
    y[-1] = 10.0
    z = torch.ones(n, f)
    z[-1] = 10.0
    pred_z = torch.ones(n, f)
    scores = row_scores(x, x, y, torch.ones_like(y), torch.ones_like(y), z, pred_z)
    assert scores["post_dense_surprise_raw"][-1] > scores["post_dense_surprise_raw"][:-1].max()
    assert scores["post_code_relative_l2"][-1] > scores["post_code_relative_l2"][:-1].max()
    assert torch.all(scores["forecast_context_recon_nse"] == 0)


def test_metrics_perfect_and_reversed() -> None:
    labels = [0, 0, 1, 1]
    assert binary_auroc(labels, [0.0, 0.1, 0.9, 1.0]) == pytest.approx(1.0)
    assert binary_average_precision(labels, [0.0, 0.1, 0.9, 1.0]) == pytest.approx(1.0)
    assert binary_auroc(labels, [1.0, 0.9, 0.1, 0.0]) == pytest.approx(0.0)
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert pooled_r2(x, x) == pytest.approx(1.0)
