"""Scientific gates for matched K=1/K=5 transfer and train-only calibration."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def driver(monkeypatch):
    """Import the actual worktree driver without replacing numerical bodies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("issue825_turn_k5_fit")


@pytest.fixture
def captures(driver):
    """Small complete banks with independent answer noise and identical draw contexts."""
    rng = np.random.default_rng(8255)
    n, d = 24, 7
    captures = {}
    for model in driver.MODELS:
        rows = []
        for i in range(n):
            for turn in driver.TURNS:
                x = rng.normal(size=d)
                for draw in range(5):
                    rows.append((f"conv{i:03}", turn, draw, x, 0.6 * x + rng.normal(size=d)))
        captures[model] = {
            "conv_id": np.array([r[0] for r in rows]),
            "turn": np.array([r[1] for r in rows]),
            "draw_id": np.array([r[2] for r in rows]),
            "context": np.stack([r[3] for r in rows]),
            "answer": np.stack([r[4] for r in rows]),
        }
    return captures


def panels_from(driver, captures):
    """Execute capture validation and panel matching on the small fixture."""
    return driver.matched_panels(
        {m: driver.validate_capture(a, expected_dim=7) for m, a in captures.items()}
    )


def test_equal_weight_draw_average_and_context_invariance(driver, captures):
    """K=1 is draw zero, K=5 averages all draws, and context drift is rejected."""
    panels, _ = panels_from(driver, captures)
    a = captures["instruct"]
    rows = (a["conv_id"] == "conv000") & (a["turn"] == 1)
    target = panels["instruct"]["turns"][1]["y"]
    np.testing.assert_array_equal(target[0, 0], a["answer"][rows][0])
    np.testing.assert_array_equal(target[1, 0], a["answer"][rows].mean(0))
    a["context"][1, 0] += 1e-8
    tolerant = driver.validate_capture(a, expected_dim=7)
    np.testing.assert_array_equal(tolerant["banks"]["conv000"][1][0], a["context"][0])
    a["context"][1] *= -1
    with pytest.raises(ValueError, match="context differs"):
        driver.validate_capture(a, expected_dim=7)


def test_incomplete_conversation_removed_from_every_model_and_k(driver, captures):
    """One missing draw removes the whole paired conversation without imputation."""
    a = captures["pretrained"]
    captures["pretrained"] = {name: values[1:] for name, values in a.items()}
    panels, coverage = panels_from(driver, captures)
    assert coverage["n_conversations"] == 23
    assert "conv000" in coverage["exclusions"]["pretrained"]["incomplete"]
    assert coverage["exclusions"]["instruct"]["complete_but_unmatched"] == ["conv000"]
    np.testing.assert_array_equal(panels["instruct"]["ids"], panels["pretrained"]["ids"])
    np.testing.assert_array_equal(panels["instruct"]["folds"], panels["pretrained"]["folds"])
    assert panels["instruct"]["turns"][1]["y"].shape == (2, 23, 7)


@pytest.mark.parametrize("failure", ["duplicate", "nonfinite", "object_ids"])
def test_invalid_capture_rejected(driver, captures, failure):
    """Duplicate keys, nonfinite activations and pickle-requiring IDs fail loudly."""
    a = captures["instruct"]
    if failure == "duplicate":
        a["draw_id"][1] = 0
    elif failure == "nonfinite":
        a["answer"][0, 0] = np.nan
    else:
        a["conv_id"] = a["conv_id"].astype(object)
    with pytest.raises(ValueError):
        driver.validate_capture(a, expected_dim=7)


def test_capture_manifest_requires_completion_and_exact_hash(driver, captures, tmp_path):
    """The real loader consumes safe NPZ shards only after hash and status gates."""
    root = tmp_path / "capture"
    root.mkdir()
    chunk = root / "chunk00000"
    chunk.mkdir()
    shard = chunk / "vectors.npz"
    receipt = driver.write_npz(shard, **captures["instruct"])
    a = captures["instruct"]
    (chunk / "rows.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "conv_id": str(c),
                    "turn": int(t),
                    "draw_id": int(d),
                    "context_prefix_sha256": hashlib.sha256(f"{c}/{t}".encode()).hexdigest(),
                }
            )
            for c, t, d in zip(a["conv_id"], a["turn"], a["draw_id"], strict=True)
        )
        + "\n"
    )
    generation = {"model": "instruct", "n": 5, "turns": [1, 12]}
    generation_fingerprint = hashlib.sha256(
        json.dumps(generation, sort_keys=True).encode()
    ).hexdigest()
    config = {
        "fixture": "capture",
        "layer": 19,
        "context_cos_min": 0.995,
        "generation_fingerprint": generation_fingerprint,
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    driver.write_json(root / "config.json", config)
    driver.write_json(
        chunk / "manifest.json",
        {
            "kind": "capture",
            "fingerprint": fingerprint,
            "n_rows": 240,
            "files": {
                "vectors.npz": {"bytes": shard.stat().st_size, "sha256": receipt["sha256"]},
                "rows.jsonl": {
                    "bytes": (chunk / "rows.jsonl").stat().st_size,
                    "sha256": driver.sha256(chunk / "rows.jsonl"),
                },
            },
        },
    )
    manifest = {
        "status": "complete",
        "fingerprint": fingerprint,
        "generation_config": generation,
        "generation_fingerprint": generation_fingerprint,
        "config_sha256": driver.sha256(root / "config.json"),
        "n_captured_draws": 240,
        "n_expected_draws": 240,
        "n_excluded_draws": 0,
        "exclusions": [],
        "hook_gates": [{"status": "pass", "layer": 19, "same_forward_equal": True, "max_abs": 0}],
        "n_complete_conversations": 24,
        "complete_conversation_ids": [f"conv{i:03}" for i in range(24)],
        "chunks": [{"path": "chunk00000", "sha256": driver.sha256(chunk / "manifest.json")}],
    }
    path = root / "summary.json"
    driver.write_json(path, manifest)
    bank, provenance = driver.load_capture(root, expected_dim=7)
    assert len(bank["complete"]) == 24
    assert provenance["sha256"] == driver.sha256(path)
    with pytest.raises(ValueError, match="different model"):
        driver.load_capture(root, expected_dim=7, expected_model="pretrained")
    manifest["status"] = "running"
    driver.write_json(path, manifest)
    with pytest.raises(ValueError, match="not complete"):
        driver.load_capture(root, expected_dim=7)
    manifest["status"] = "complete"
    manifest["chunks"][0]["sha256"] = "0" * 64
    driver.write_json(path, manifest)
    with pytest.raises(ValueError, match="hash mismatch"):
        driver.load_capture(root, expected_dim=7)


def test_capture_chunk_preserves_unicode_separators_inside_json_records(driver, captures, tmp_path):
    """JSON strings containing U+2028/U+2029 stay within their physical JSONL record."""
    chunk = tmp_path / "chunk00000"
    chunk.mkdir()
    arrays = {name: values[:1] for name, values in captures["instruct"].items()}
    driver.write_npz(chunk / "vectors.npz", **arrays)
    text = "before" + chr(0x2028) + "middle" + chr(0x2029) + "after"
    row = {
        "conv_id": str(arrays["conv_id"][0]),
        "turn": int(arrays["turn"][0]),
        "draw_id": int(arrays["draw_id"][0]),
        "text": text,
    }
    (chunk / "rows.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = chunk / "manifest.json"
    driver.write_json(
        manifest_path,
        {
            "kind": "capture",
            "fingerprint": "unicode-fixture",
            "n_rows": 1,
            "files": {
                name: {
                    "bytes": (chunk / name).stat().st_size,
                    "sha256": driver.sha256(chunk / name),
                }
                for name in ("vectors.npz", "rows.jsonl")
            },
        },
    )
    saved, metadata = driver.load_capture_chunk(
        chunk, driver.sha256(manifest_path), "unicode-fixture"
    )
    assert len(metadata) == 1 and metadata[0]["text"] == text
    np.testing.assert_array_equal(saved["answer"], arrays["answer"])


def test_shared_inner_selector_matches_explicit_validation_oracle(driver, captures):
    """Compare the dispatched batched selector to explicit inner validation predictions."""
    panels, _ = panels_from(driver, captures)
    p = panels["instruct"]
    train = p["folds"] != 0
    x, y, ids = p["turns"][1]["x"][train], p["turns"][1]["y"][:, train], p["ids"][train]
    fitted = driver.fit_paired_ridge(x, y, ids, 0)
    rss = np.zeros((2, 13))
    for inner in range(4):
        fi = fitted[f"inner_{inner}_train_indices"]
        va = fitted[f"inner_{inner}_validation_indices"]
        assert not set(ids[fi]) & set(ids[va])
        xmu, xsd = x[fi].mean(0), x[fi].std(0, ddof=1) + 1e-9
        xf, xv = (x[fi] - xmu) / xsd, (x[va] - xmu) / xsd
        for k in range(2):
            ym = y[k, fi].mean(0)
            for li, lam in enumerate(driver.fit825.LAMBDAS):
                beta = np.linalg.solve(xf.T @ xf + lam * np.eye(x.shape[1]), xf.T @ (y[k, fi] - ym))
                rss[k, li] += np.square(xv @ beta + ym - y[k, va]).sum()
    np.testing.assert_allclose(fitted["inner_cv_rss"], rss, atol=1e-8, rtol=1e-10)
    np.testing.assert_array_equal(fitted["lambda"], driver.fit825.LAMBDAS[rss.argmin(1)])
    prediction = driver.predict_paired(fitted, x)
    xn = (x - fitted["xmu"]) / fitted["xsd"]
    for k in range(2):
        beta = np.linalg.solve(
            xn.T @ xn + fitted["lambda"][k] * np.eye(x.shape[1]), xn.T @ (y[k] - y[k].mean(0))
        )
        np.testing.assert_allclose(prediction[k], xn @ beta + y[k].mean(0), atol=1e-8)


def test_missing_inner_cache_never_falls_back_to_gcv(driver, monkeypatch):
    """An unavailable inner-group selector fails rather than choosing another recipe."""
    from unittest.mock import create_autospec

    original = driver.fit825._prep_inner_lambda
    monkeypatch.setattr(
        driver.fit825, "_prep_inner_lambda", create_autospec(original, return_value=None)
    )
    with pytest.raises(RuntimeError, match="no GCV fallback"):
        driver.fit_paired_ridge(
            np.ones((12, 3)), np.ones((2, 12, 3)), np.array([str(i) for i in range(12)]), 0
        )


def test_heldout_labels_cannot_change_fit_or_calibration(driver, captures):
    """Changing test answers changes scores only, including cross-K evaluation targets."""
    panels, _ = panels_from(driver, captures)
    p = panels["instruct"]
    train = p["folds"] != 0
    maps = {
        turn: driver.fit_paired_ridge(
            p["turns"][turn]["x"][train], p["turns"][turn]["y"][:, train], p["ids"][train], 0
        )
        for turn in driver.TURNS
    }
    before = driver.heldout_predictions(p, 0, maps[1], maps[12])
    p["turns"][12]["y"][:, ~train] += 100
    after = driver.heldout_predictions(p, 0, maps[1], maps[12])
    for name in ("predictions", "bias", "gain", "prediction_mean", "target_mean"):
        np.testing.assert_array_equal(before[name], after[name])
    assert not np.array_equal(before["target"], after["target"])
    # Independently reconstruct train-K-specific calibration; neither eval K
    # target can enter the calibrated predictions at scoring time.
    all_raw = driver.predict_paired(maps[1], p["turns"][12]["x"])
    for k in range(2):
        expected = (p["turns"][12]["y"][k, train] - all_raw[k, train]).mean(0)
        np.testing.assert_allclose(before["bias"][k], expected, atol=1e-12)
    maps[1]["train_ids"] = maps[1]["train_ids"][::-1]
    with pytest.raises(ValueError, match="training conversations differ"):
        driver.heldout_predictions(p, 0, maps[1], maps[12])


def test_cross_scores_use_same_fold_pool_and_match_independent_oracle(driver):
    """All four K combinations score the exact saved rows and target candidate bank."""
    rng = np.random.default_rng(5)
    pred = rng.normal(size=(2, 5, 9, 7))
    targets = rng.normal(size=(2, 9, 7))
    rows = driver.score_rows(pred, targets)
    assert np.all(rows["pool_size"] == 9)
    for k in range(2):
        for e in range(2):
            for m in range(5):
                np.testing.assert_allclose(
                    rows["row_sse"][k, e, m], np.square(pred[k, m] - targets[e]).sum(1)
                )
                distance = np.square(pred[k, m, :, None] - targets[e][None]).sum(-1)
                np.testing.assert_array_equal(
                    rows["euclidean_top1_hit"][k, e, m], distance.argmin(1) == np.arange(9)
                )


def test_bootstrap_null_ratio_and_paired_k_differences(driver):
    """No ratios for nonpositive own-map R²; identical K banks have zero paired deltas."""
    n = 24
    rows = {
        "row_sse": np.full((2, 2, 5, n), 2.0),
        "row_sst": np.ones((2, n)),
        "cosine_top1_hit": np.ones((2, 2, 5, n)),
        "euclidean_top1_hit": np.ones((2, 2, 5, n)),
        "pool_size": np.full(n, 4),
    }
    result = driver.bootstrap_summary(rows, replicates=100)
    assert all(c["r2_retention"] is None for c in result["cells"])
    assert all(c["r2_retention_ci95"] is None for c in result["cells"])
    for comparison in result["paired_k_comparisons"]:
        assert comparison["r2"] == 0
        assert comparison["r2_ci95"] == [0, 0]
    assert len(result["paired_calibration_comparisons"]) == 12
    for comparison in result["paired_calibration_comparisons"]:
        assert comparison["r2"] == 0
        assert comparison["r2_ci95"] == [0, 0]


def test_full_production_analysis_persists_reconstructable_maps_and_resumes(
    driver, captures, tmp_path
):
    """Exercise all real fit/score/reduce/checkpoint bodies and detect corrupt resumes."""
    torch.set_num_threads(2)
    panels, coverage = panels_from(driver, captures)
    out, store = tmp_path / "out", tmp_path / "store"
    result = driver.run_analysis(
        panels, coverage, {"fixture": "microbank"}, out, store, bootstraps=100
    )
    assert result["status"] == "complete"
    assert len(result["artifacts"]) == 38  # 24 maps + 12 fold predictions + 2 OOF banks.
    assert len(list((out / "folds").glob("*.json"))) == 12
    for model in driver.MODELS:
        assert len(result["models"][model]["cells"]) == 20
        assert result["models"][model]["n_conversations"] == 24
    map_path = store / "maps" / "instruct_turn1_fold0.npz"
    with np.load(map_path, allow_pickle=False) as saved:
        fitted = dict(saved)
    with np.load(store / "predictions" / "instruct_fold0.npz", allow_pickle=False) as saved:
        raw = driver.predict_paired(
            fitted, panels["instruct"]["turns"][12]["x"][saved["test_indices"]]
        )
        np.testing.assert_allclose(raw, saved["predictions"][:, 0], atol=1e-10)
    rerun = driver.run_analysis(
        panels, coverage, {"fixture": "microbank"}, out, store, bootstraps=100
    )
    assert rerun["models"] == result["models"]
    receipt_path = out / "folds" / "instruct_fold0.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][0]["sha256"] = "0" * 64
    driver.write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="corrupt checkpoint"):
        driver.run_analysis(panels, coverage, {"fixture": "microbank"}, out, store, bootstraps=100)
