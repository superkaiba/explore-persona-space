"""Numerical contracts for raw-coordinate operator comparisons."""

import numpy as np
import torch

from scripts.issue2054_k5_map_geometry import (
    cosine_gram,
    moments_by_fold,
    restore_maps,
    retrieval,
)


def test_batched_restoration_matches_direct_ridge_with_unequal_scales():
    rng = np.random.default_rng(19)
    panels = []
    for n, offset in [(55, 30), (60, -12)]:
        x = rng.normal(size=(n, 4)) * [0.2, 2, 5, 10] + offset
        y = x @ rng.normal(size=(4, 4)) + rng.normal(size=(n, 4))
        panels.append({"x": x, "y": y, "membership": np.arange(n) % 5})
    bank = [moments_by_fold(p) for p in panels]
    penalties = [1.2, 3.4, 5.6]
    maps, biases, _, _, _ = restore_maps(bank, 2, penalties)
    for i, sources in enumerate([[panels[0]], [panels[1]], panels]):
        x = np.concatenate([p["x"][p["membership"] != 2] for p in sources])
        y = np.concatenate([p["y"][p["membership"] != 2] for p in sources])
        mu, sd = x.mean(0), x.std(0) + 1e-9
        normalized = (x - mu) / sd
        w = np.linalg.solve(
            normalized.T @ normalized + penalties[i] * np.eye(4), normalized.T @ (y - y.mean(0))
        )
        expected = w / sd[:, None]
        np.testing.assert_allclose(maps[i], expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(biases[i], y.mean(0) - mu @ expected, rtol=1e-9, atol=1e-9)


def test_operator_cosine_is_scale_invariant_but_distance_is_not():
    flat = torch.tensor([[1.0, 0], [2.0, 0], [0.0, 1]], dtype=torch.float64)
    cosine, distance = cosine_gram(flat)
    assert cosine[0, 1] == 1
    assert cosine[0, 2] == 0
    assert distance[0, 1] > 0
    assert torch.equal(cosine.diag(), torch.ones(3, dtype=torch.float64))


def test_retrieval_scores_source_maps_on_one_common_pool():
    y = torch.eye(4, dtype=torch.float64)
    prediction = torch.stack([y, y.roll(1, 0)])
    result = retrieval(prediction, y)
    assert result["euclidean_top1"] == [1, 0]
    assert result["cosine_top1"] == [1, 0]
    assert result["pool_size"] == 4
    assert result["chance_top1"] == 0.25


def test_retrieval_matches_parent_with_duplicate_targets():
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    y = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]], dtype=torch.float64)
    pred = torch.stack([y, y.roll(1, 0)])
    result = retrieval(pred, y)
    for i in range(2):
        for metric in ["euclidean", "cosine"]:
            expected = knn_retrieval(pred[i].numpy(), y.numpy(), metric=metric)
            assert result[f"{metric}_top1"][i] == expected["acc_at_k"][1]
