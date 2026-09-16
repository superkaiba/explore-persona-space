"""Scientific parity and artifact-boundary tests for the exact shuffled-map fit."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

from scripts import issue1739_transfer_map as transfer


def fixture_arrays():
    """A non-isotropic, intercept-bearing problem that exposes pairing mistakes."""
    rng = np.random.default_rng(51)
    X = rng.normal(size=(91, 7)).astype(np.float32) * np.linspace(1, 8, 7)
    Y = (X @ rng.normal(size=(7, 5)) + rng.normal(size=(91, 5)) * 3 + 7).astype(np.float32)
    train, val, test = np.arange(57), np.arange(57, 74), np.arange(74, 91)
    return X, Y, train, val, test


def test_shared_gram_matches_canonical_independently_permuted_refits(tmp_path):
    """The production vectorization reproduces six separate canonical ridge fits."""
    X, Y, train, val, test = fixture_arrays()
    ty, vy = transfer.permutations(train, val)
    lambdas = np.array([0.001, 0.1, 1, 10, 100])
    state = transfer.accumulate(X, Y, train, ty, block=13)
    payloads, selections = transfer.select_payloads(X, Y, val, vy, state, lambdas)
    for arm in range(6):
        permuted_y = Y.copy()
        permuted_y[train] = Y[ty[arm]]
        permuted_y[val] = Y[vy[arm]]
        _, meta, ref = transfer.parent.fit_ridge_with_weights(
            X, permuted_y, train, val, test, lambdas, torch.device("cpu"), 13
        )
        assert selections[arm]["selected_lambda"] == meta["selected_lambda"]
        for name in ("W", "xmu", "xsd", "ymu"):
            torch.testing.assert_close(payloads[arm][name], ref[name], rtol=1e-6, atol=1e-6)
        a = transfer.parent.apply_map(payloads[arm], X[test], torch.device("cpu"))
        b = transfer.parent.apply_map(ref, X[test], torch.device("cpu"))
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)


def test_crossproduct_checkpoint_is_exact_and_refuses_changed_recipe(tmp_path):
    """A checkpoint resume preserves all arrays and rejects stale regime keys."""
    X, Y, train, val, _ = fixture_arrays()
    ty, _ = transfer.permutations(train, val, seeds=(0, 1))
    ckpt = tmp_path / "cross.pt"
    got = transfer.accumulate(X, Y, train, ty, block=14, checkpoint=ckpt, regime={"seed": 9})
    resumed = transfer.accumulate(X, Y, train, ty, block=14, checkpoint=ckpt, regime={"seed": 9})
    for name in ("gram", "cross", "xmu", "xsd", "ymu"):
        torch.testing.assert_close(got[name], resumed[name], rtol=0, atol=0)
    with pytest.raises(AssertionError):
        transfer.accumulate(X, Y, train, ty, block=14, checkpoint=ckpt, regime={"seed": 10})


def test_independent_permutations_preserve_marginals_and_split_membership():
    """Validation permutation is independent and never draws a training answer."""
    _, _, train, val, _ = fixture_arrays()
    ty, vy = transfer.permutations(train, val)
    ty2, vy2 = transfer.permutations(train, val)
    for a, b, c, d in zip(ty, vy, ty2, vy2, strict=True):
        np.testing.assert_array_equal(np.sort(a), train)
        np.testing.assert_array_equal(np.sort(b), val)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        assert not set(a) & set(b)
    assert len({transfer.index_hash(a) for a in ty}) == 6


def test_chunk_loading_checks_actual_keys_hash_alignment_and_order(tmp_path, monkeypatch):
    """The production chunk consumer refuses wrong prompt alignment before writes."""
    monkeypatch.setattr(transfer, "WIDTH", 7)
    path = tmp_path / "chunk.pt"
    prompts = ["A\nB", "Other query"]
    b = {
        "layers": [14, 19, 26],
        "cx_last": torch.ones(2, 3, 7),
        "v_x": torch.zeros(2, 3, 7),
        "ci": [0, 2],
        "prompts": prompts,
    }
    torch.save(b, path)
    hashes = np.array(
        [transfer.text_hash(prompts[0]), "unused", transfer.text_hash(prompts[1])], dtype="S64"
    )
    x, y, ci = transfer.chunk_arrays(path, hashes, -1)
    assert x.shape == y.shape == (2, 7)
    np.testing.assert_array_equal(ci, [0, 2])
    with pytest.raises(AssertionError):
        transfer.chunk_arrays(path, hashes[::-1], -1)
    del b["prompts"]
    torch.save(b, path)
    with pytest.raises(AssertionError):
        transfer.chunk_arrays(path, hashes, -1)


def test_fetch_uses_pinned_canonical_stager_and_verifies_bytes(tmp_path, monkeypatch):
    """Exercise fetch body with a signature-constrained external download boundary."""
    path = tmp_path / "chunk.pt"
    path.write_bytes(b"real boundary bytes")
    rec = {
        "path": "example/chunk.pt",
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    stage = create_autospec(transfer.hub.stage_hub_file, return_value=path)
    monkeypatch.setattr(transfer.hub, "stage_hub_file", stage)
    got, disposable = transfer.fetch_chunk(rec, tmp_path)
    assert got == path and disposable
    stage.assert_called_once_with(
        transfer.REPO,
        rec["path"],
        path,
        repo_type="dataset",
        revision=transfer.REVISION,
        size_bytes=rec["size"],
    )
    with pytest.raises(AssertionError):
        transfer.verify_chunk(path, {**rec, "sha256": "bad"})


def test_hash_normalization_and_retrieval_semantics():
    """Normalization follows #779; retrieval uses exactly the realized test pool."""
    assert transfer.text_hash(" A\nB  ", True) == transfer.text_hash("a b", True)
    assert transfer.text_hash("A B") != transfer.text_hash("a b")
    x = np.eye(7)
    metrics = transfer.map_metrics(x, x)
    assert metrics["r2"] == 1 and metrics["cosine_retrieval_top1"] == 1
    assert metrics["retrieval_pool_size"] == 7 and metrics["retrieval_chance"] == 1 / 7


def test_selection_checkpoint_digests_are_enforced(tmp_path):
    """A selected map is resumable only while its persisted bytes remain intact."""
    X, Y, train, val, _ = fixture_arrays()
    state = transfer.accumulate(X, Y, train, [train], block=30)
    transfer.select_payloads(X, Y, val, [val], state, [0.001, 100], out=tmp_path)
    meta = json.loads((tmp_path / "true_refit_selection.json").read_text())
    assert meta["payload_sha256"] == transfer.sha_file(tmp_path / "true_refit.pt")
    (tmp_path / "true_refit.pt").write_bytes(b"corruption")
    with pytest.raises(AssertionError):
        transfer.select_payloads(X, Y, val, [val], state, [0.001, 100], out=tmp_path)
