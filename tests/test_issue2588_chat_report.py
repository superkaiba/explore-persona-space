"""Export and refusal tests using explicitly test-only metric/curve fixtures."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_issue2588_chat_plot import source as source

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_report as report


@pytest.fixture
def complete_source(source):
    for arm in ("a", "b"):
        path = source / f"rank_{arm}.json"
        rec = json.loads(path.read_text())
        rec.update(layer_star=12, selected_lambda=100.0, participation_ratio_x_at_star=100.0)
        rec["provenance"]["hf_revision"] = "a" * 40
        rec["parent_selected_layer_metrics"] = {
            "n": {"tr": 10000, "val": 400, "te": 1000},
            "test_r2": rec["full_test_r2"],
            "floors_test_r2": {"identity_bias": -0.2},
            "knn_test": {
                "_meta": {"n_pool": 1000},
                **{
                    method: {
                        metric: {"acc_at_k": {"1": value}} for metric in ("cosine", "euclidean")
                    }
                    for method, value in (("ridge", 0.50), ("identity_bias", 0.1))
                },
            },
        }
        rec["ceiling_two_draw_at_star"] = {"n": 990, "ceiling": 0.85}
        rec["ceiling_retrieval_at_star"] = {
            "n_pool": 990,
            "seed_pair": [43, 44],
            "chance": 1 / 990,
            "ceiling_acc1_cos": 0.9,
        }
        path.write_text(json.dumps(rec))
    return source


def test_actual_csv_and_report_export(complete_source, tmp_path):
    out = tmp_path / "test_only_report"
    metadata = report.export(complete_source, out, "https://example.org/test-only.png")
    rows = list(csv.DictReader((out / "comparison.csv").open()))
    assert len(rows) == 2
    assert rows[0]["retrieval_chance"] == "0.001"
    assert rows[1]["repeat_answer_n"] == "990"
    text = (out / "comparison.md").read_text()
    assert "unchanged operational rank" in text
    assert "https://example.org/test-only.png" in text
    assert "not an isolated causal effect" in text
    assert "13 exact prompt strings" in text
    assert metadata["rank_change_thinking_minus_no_thinking"] == 0
    for name, expected in metadata["outputs_sha256"].items():
        assert report.plot.digest(out / name) == expected


def test_higher_rank_is_valid_result(complete_source, tmp_path):
    path = complete_source / "rank_b.json"
    rec = json.loads(path.read_text())
    values = (0.5 * (1 - np.exp(-np.arange(4097) / 200))).tolist()
    threshold = 1 - 1.1 * (1 - values[-1])
    rank = int(np.flatnonzero(np.asarray(values) >= threshold - 1e-12)[0])
    rec.update(
        rank=rank,
        rank_curve={"test_r2": values, "validation_r2": values},
        full_test_r2=values[-1],
        full_validation_r2=values[-1],
        selected_rank_test_r2=values[rank],
        selected_rank_validation_r2=values[rank],
        validation_r2_threshold=threshold,
    )
    rec["parent_selected_layer_metrics"]["test_r2"] = values[-1]
    path.write_text(json.dumps(rec))
    out = tmp_path / "test_only_higher_rank"
    metadata = report.export(complete_source, out, "https://example.org/test-only.png")
    assert metadata["rank_change_thinking_minus_no_thinking"] > 0
    assert "higher operational rank" in (out / "comparison.md").read_text()


@pytest.mark.parametrize(
    "change",
    [
        "revision",
        "counts",
        "pool",
        "baseline",
        "nan",
        "accuracy",
        "missing_repeat",
        "repeat_pool",
        "seed",
        "parity",
    ],
)
def test_incomplete_or_inconsistent_report_rejected(complete_source, tmp_path, change):
    path = complete_source / "rank_b.json"
    rec = json.loads(path.read_text())
    metrics = rec["parent_selected_layer_metrics"]
    if change == "revision":
        rec["provenance"]["hf_revision"] = "b" * 40
    elif change == "counts":
        metrics["n"]["te"] -= 1
    elif change == "pool":
        metrics["knn_test"]["_meta"]["n_pool"] -= 1
    elif change == "baseline":
        metrics["floors_test_r2"]["identity_bias"] = None
    elif change == "nan":
        rec["participation_ratio_x_at_star"] = float("nan")
    elif change == "accuracy":
        metrics["knn_test"]["ridge"]["cosine"]["acc_at_k"]["1"] = 1.1
    elif change == "missing_repeat":
        rec["ceiling_two_draw_at_star"] = None
    elif change == "repeat_pool":
        rec["ceiling_retrieval_at_star"]["n_pool"] -= 1
    elif change == "seed":
        rec["ceiling_retrieval_at_star"]["seed_pair"] = [42, 43]
    elif change == "parity":
        metrics["test_r2"] += 0.01
    path.write_text(json.dumps(rec))
    out = tmp_path / "test_only_rejected_report"
    with pytest.raises(ValueError):
        report.export(complete_source, out, "https://example.org/test-only.png")
    assert not out.exists()


@pytest.mark.parametrize(
    "url",
    [
        "local.png",
        "file:///tmp/a.png",
        "http://example.org/a.png",
        "https://user:password@example.org/a.png",
    ],
)
def test_report_requires_browser_url(complete_source, tmp_path, url):
    with pytest.raises(ValueError, match="figure URL"):
        report.export(complete_source, tmp_path / "rejected", url)
