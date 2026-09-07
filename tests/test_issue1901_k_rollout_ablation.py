"""Statistical and cached-geometry equivalence tests for the K ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1901_k_rollout_ablation as K


def test_bootstrap_r2_matches_literal_resampled_datasets():
    """A bootstrap must re-center the target, including with repeated rows."""
    rng = np.random.default_rng(8)
    draws = rng.normal(size=(3, 7, 4)) + np.arange(7)[None, :, None]
    pred = rng.normal(size=(3, 7, 4))
    samples = np.array([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 2, 3, 3, 6]])
    counts = np.stack([np.bincount(row, minlength=7) for row in samples])
    masks = K.subset_masks(3)
    point, boot, _ = K.bootstrap_r2(draws, pred, masks, counts)
    for si, mask in enumerate(masks):
        target = draws[mask.astype(bool)].mean(0)
        for ai, p in enumerate(pred):
            np.testing.assert_allclose(point[si, ai], K.F79._recon_point(p, target)[0], atol=1e-12)
            for bi, rows in enumerate(samples):
                expected = K.F79._recon_point(p[rows], target[rows])[0]
                np.testing.assert_allclose(boot[si, ai, bi], expected, atol=1e-12)


def test_cached_geometry_matches_canonical_on_every_subset(tmp_path):
    """Exercise real scorer with an affine whitening transform and a duplicate."""
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(24, 6))
    draws = latent[None] + rng.normal(size=(5, 24, 6)) * 0.5
    draws[0, 2] = draws[0, 1]
    pred = latent[None] + rng.normal(size=(3, 24, 6)) * 0.3
    matrix = rng.normal(size=(6, 6))
    offset = rng.normal(size=6)

    def whiten(x):
        return (x - offset) @ matrix

    result = K.score_subsets(draws, pred, whiten, tmp_path)
    view = K.FINAL.make_eval_view(draws[0], 24, "keep_one")
    assert result["duplicate_audit"]["realized_n_pool"] == 23
    assert [v["n_subsets"] for v in result["all_subsets"]["per_k"].values()] == [5, 10, 10, 5, 1]
    assert [v["n_subsets"] for v in result["fresh_only"]["per_k"].values()] == [4, 6, 4, 1]
    for cell in result["subset_scores"]:
        target = draws[cell["draw_indices"]].mean(0)
        for ai, p in enumerate(pred):
            # Reuse the canonical full-geometry helper, independently of caching.
            full = K.FINAL._precompute_metric_arrays(p, target, whiten(p), whiten(target))
            ix = np.ix_(view.pred_rows, view.pool_rows)
            sim = 1 - full["whiten_cosine"][ix]
            distances = {
                "whiten_csls": -K.FINAL.MB.csls_scores(sim, 10),
                **{name: value[ix] for name, value in full.items()},
            }
            for mi, metric in enumerate(K.METRICS):
                ranks = K.FINAL._strict_ranks(distances[metric], view.true_idx)
                assert cell["top1"][ai][mi] == np.mean(ranks <= 1)
                assert cell["top5"][ai][mi] == np.mean(ranks <= 5)
    archive = np.load(tmp_path / "per_row_and_bootstrap.npz")
    assert archive["sse"].shape == (31, 3, 24)
    assert archive["retrieval_boot"].shape == (31, 3, 4, K.N_BOOT)
