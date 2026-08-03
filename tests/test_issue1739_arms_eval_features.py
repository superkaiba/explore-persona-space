"""Eval-side text/surface features for the #1739 transfer leg (arms 15/16).

Pins the ``run_transfer_cell`` ``text_emb_ev`` / ``text_features_ev``
threading: WITH eval features the text arms score the eval block (no skip);
WITHOUT them the pre-change behavior is untouched — the text arms record
their standard SKIP and every other arm's scores are BYTE-IDENTICAL to a
features-bearing call (the additive no-change pin). Everything is tiny +
CPU; no network, no staged data, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import arms, fits

TEXT_ARMS = ["arm15_text_only", "arm16_surface_feat"]
OTHER_ARMS = ["arm1_ctx_e1", "arm4_ridge_ctx"]


def _fixture(n_tr=24, n_ev=12, n_layers=2, d=6, e=5, f=3, seed=3):
    rng = np.random.default_rng(seed)
    rb = rng.normal(size=(n_layers, d))
    z_tr = rng.normal(size=(n_layers, n_tr, d))
    z_ev = rng.normal(size=(n_layers, n_ev, d))
    dv_tr = np.clip(50 + 20 * np.einsum("lnd,ld->n", z_tr, rb) / n_layers, 0, 100)
    dv_ev = np.clip(50 + 20 * np.einsum("lnd,ld->n", z_ev, rb) / n_layers, 0, 100)
    groups = [f"g{i % 6}" for i in range(n_tr)]
    text_emb = rng.normal(size=(n_tr, e))
    text_features = rng.normal(size=(n_tr, f))
    text_emb_ev = rng.normal(size=(n_ev, e))
    text_features_ev = rng.normal(size=(n_ev, f))
    data = arms.CellData(
        z_ctx=z_tr,
        dv=dv_tr,
        rb=rb,
        text_emb=text_emb,
        text_features=text_features,
        layers=(0, 1),
    )
    cell = fits.realize_budget_cell(groups, budget_l=n_tr, draw=0, seed=0)
    return data, cell, z_ev, dv_ev, text_emb_ev, text_features_ev, rng


def test_text_arms_score_eval_rows_with_eval_features():
    data, cell, z_ev, dv_ev, emb_ev, feat_ev, rng = _fixture()
    scores, skipped = arms.run_transfer_cell(
        data,
        cell,
        z_ev,
        dv_ev,
        text_emb_ev=emb_ev,
        text_features_ev=feat_ev,
        arms=TEXT_ARMS + OTHER_ARMS,
        ridge_folds=(0,),  # the production transfer leg's discarded-fold skip
    )
    assert not skipped, skipped
    for slug in TEXT_ARMS:
        assert scores[slug].shape == (1, z_ev.shape[1])  # non-layered: one row
        assert np.isfinite(scores[slug]).all(), slug
    # Frozen-predictor semantics survive: perturbing EVAL dv must not move the
    # text-arm predictions (fit on the train block only, never on eval DV).
    s2, _ = arms.run_transfer_cell(
        data,
        cell,
        z_ev,
        dv_ev + rng.normal(scale=10, size=dv_ev.shape),
        text_emb_ev=emb_ev,
        text_features_ev=feat_ev,
        arms=TEXT_ARMS,
        ridge_folds=(0,),
    )
    for slug in TEXT_ARMS:
        np.testing.assert_array_equal(scores[slug], s2[slug])


def test_no_eval_features_keeps_pre_change_behavior_and_other_arms_identical():
    data, cell, z_ev, dv_ev, emb_ev, feat_ev, _ = _fixture()
    want = TEXT_ARMS + OTHER_ARMS
    with_feats, sk_with = arms.run_transfer_cell(
        data,
        cell,
        z_ev,
        dv_ev,
        text_emb_ev=emb_ev,
        text_features_ev=feat_ev,
        arms=want,
        ridge_folds=(0,),
    )
    without, sk_without = arms.run_transfer_cell(
        data, cell, z_ev, dv_ev, arms=want, ridge_folds=(0,)
    )
    # Pre-change behavior: the text arms SKIP with their standard reasons.
    assert sk_without["arm15_text_only"] == "no text embeddings"
    assert sk_without["arm16_surface_feat"] == "no surface features"
    assert not any(a in without for a in TEXT_ARMS)
    assert not sk_with
    # Additive no-change pin: every OTHER arm's eval scores are byte-identical
    # whether or not the text features are threaded.
    for slug in OTHER_ARMS:
        np.testing.assert_array_equal(with_feats[slug], without[slug])
    # A data table with NO train-side features behaves identically (the
    # pre-change CellData shape used by both eval-rung callers).
    bare = arms.CellData(z_ctx=data.z_ctx, dv=data.dv, rb=data.rb, layers=data.layers)
    bare_scores, bare_sk = arms.run_transfer_cell(
        bare, cell, z_ev, dv_ev, arms=want, ridge_folds=(0,)
    )
    assert bare_sk["arm15_text_only"] == "no text embeddings"
    for slug in OTHER_ARMS:
        np.testing.assert_array_equal(bare_scores[slug], without[slug])


def test_eval_features_without_train_side_table_fail_loud():
    data, cell, z_ev, dv_ev, emb_ev, feat_ev, _ = _fixture()
    bare = arms.CellData(z_ctx=data.z_ctx, dv=data.dv, rb=data.rb, layers=data.layers)
    with pytest.raises(ValueError, match=r"text_emb_ev supplied but data\.text_emb is None"):
        arms.run_transfer_cell(
            bare, cell, z_ev, dv_ev, text_emb_ev=emb_ev, arms=TEXT_ARMS, ridge_folds=(0,)
        )
    with pytest.raises(ValueError, match="text_features_ev supplied"):
        arms.run_transfer_cell(
            bare, cell, z_ev, dv_ev, text_features_ev=feat_ev, arms=TEXT_ARMS, ridge_folds=(0,)
        )


def test_eval_feature_shape_mismatches_fail_loud():
    data, cell, z_ev, dv_ev, emb_ev, feat_ev, _ = _fixture()
    with pytest.raises(AssertionError):  # wrong eval row count
        arms.run_transfer_cell(
            data, cell, z_ev, dv_ev, text_emb_ev=emb_ev[:-1], arms=TEXT_ARMS, ridge_folds=(0,)
        )
    with pytest.raises(AssertionError):  # wrong feature dim vs the train table
        arms.run_transfer_cell(
            data,
            cell,
            z_ev,
            dv_ev,
            text_features_ev=feat_ev[:, :-1],
            arms=TEXT_ARMS,
            ridge_folds=(0,),
        )
