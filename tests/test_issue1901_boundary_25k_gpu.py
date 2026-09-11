from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import issue1901_boundary_25k_gpu as G  # noqa: E402
import issue1901_singleturn_retrieval_final as F  # noqa: E402


def test_strict_retrieval_matches_original_helper_without_ties():
    rng = np.random.default_rng(73)
    true = rng.normal(size=(40, 12))
    pred = true + rng.normal(size=true.shape)
    view = F.make_eval_view(true.astype(np.float32), len(true), "keep_one")
    old = F.score_cell(pred, true, view, lambda x: x, seed=43)
    new = G.strict_retrieval(pred, true, view, lambda x: x, seed=43)
    assert new == {k: v["strict"] for k, v in old.items()}


def test_zero_whitened_constant_uses_strict_tie_ranks():
    rng = np.random.default_rng(91)
    true = rng.normal(size=(40, 12))
    pred = np.zeros_like(true)
    view = F.make_eval_view(true.astype(np.float32), len(true), "keep_one")
    result = G.strict_retrieval(pred, true, view, lambda x: x, seed=43)
    for name in ("raw_cosine", "whiten_cosine", "whiten_csls"):
        assert result[name]["acc_at_k"]["1"] == 0
        assert result[name]["median_rank"] == 20.5
    assert result["raw_euclidean"]["acc_at_k"]["1"] == 1 / 40


def test_article_partition_is_deterministic_and_complete():
    items = [
        {"item_id": str(i), "input_ids": [0] * n}
        for i, n in enumerate([4096, 3000, 1000, 999, 500, 100])
    ]
    first = G.partition_articles(items, 2)
    assert first == G.partition_articles(list(reversed(items)), 2)
    flat = [a for part in first for a in part]
    assert len(flat) == len(set(flat)) == len(items)
    assert set(flat) == {it["item_id"] for it in items}


def test_store_coverage_checks_tensor_rows_not_sidecar_claims(tmp_path):
    obj = {
        "row_ids": ["one"],
        "group_ids": ["article"],
        "char_ids": ["sep"],
        "arrays": {k: torch.ones((1, 1, 3584), dtype=torch.bfloat16) for k in ("x_sep", "y")},
    }
    path = tmp_path / "pairs_shard000.pt"
    torch.save(obj, path)
    side = {"layers": [19], "row_ids": ["one"], "n_rows": 1}
    G.B.write_json(path.with_suffix(".json"), side)
    assert G.validate_store(tmp_path, ["one"]) == 1
    with pytest.raises(AssertionError):
        G.validate_store(tmp_path, ["one", "two"])
    side["row_ids"] = ["one", "two"]
    side["n_rows"] = 2
    G.B.write_json(path.with_suffix(".json"), side)
    with pytest.raises(AssertionError):
        G.validate_store(tmp_path, ["one", "two"])


def test_capture_regime_changes_with_hardware_and_manifest(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="A100", total_memory=80)
    )
    meta = {
        "selected_row_ids_sha256": "rows",
        "manifest_content_sha256": "content-v1",
        "source_provenance": {"model_revision": "model"},
    }
    first = G.capture_regime(meta, 8)
    meta["manifest_content_sha256"] = "content-v2"
    assert G.capture_regime(meta, 8) != first
    meta["manifest_content_sha256"] = "content-v1"
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="H100", total_memory=80)
    )
    assert G.capture_regime(meta, 8) != first
