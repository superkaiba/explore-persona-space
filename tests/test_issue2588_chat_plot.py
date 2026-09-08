"""Test-only curves exercise final-input rejection and the actual figure exporter."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_plot as P


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "test_only_inputs"
    directory.mkdir()
    for arm, position in zip(("a", "b"), ("prompt_last", "cot_boundary"), strict=True):
        values = (0.5 * (1 - np.exp(-np.arange(4097) / 70))).tolist()
        threshold = 1 - 1.10 * (1 - values[-1])
        rank = int(np.flatnonzero(np.asarray(values) >= threshold - 1e-12)[0])
        rec = {
            "cell": f"q3_8b_{arm}",
            "input_position": position,
            "dimension": 4096,
            "rank": rank,
            "rank_curve": {"test_r2": values, "validation_r2": values},
            "full_test_r2": values[-1],
            "full_validation_r2": values[-1],
            "validation_r2_threshold": threshold,
            "selected_rank_test_r2": values[rank],
            "selected_rank_validation_r2": values[rank],
            "realized_rows": {"train_10k": 10000, "val_400": 400, "test_1000": 1000},
            "provenance": {
                "run_identity": {
                    "smoke": False,
                    "surface": "generic",
                    "model_id": "Qwen/Qwen3-8B",
                    "run_id": "unit-test-only",
                    "model_revision": P.MODEL_REVISION,
                    "manifest_revision": P.MANIFEST_REVISION,
                    "source_sha": "fixture",
                },
                "durable_verification": {"content_verified": True},
            },
        }
        (directory / f"rank_{arm}.json").write_text(json.dumps(rec))
    return directory


def test_actual_export_with_test_only_inputs(source, tmp_path):
    assert (
        Path(P.style.__file__).resolve()
        == P.ROOT / "src/explore_persona_space/analysis/c2a_plot_style.py"
    )
    meta = P.render(source, tmp_path / "test_only_figure")
    assert len(meta["maps"]) == 2
    assert meta["render"]["style_version"] == "c2a-v2"
    assert meta["render"]["include_width_frac"] == 0.75
    assert all(Path(row["path"]).is_file() for row in meta["outputs"].values())
    assert (tmp_path / "test_only_figure.meta.json").is_file()
    assert "one fitted metamodel per condition" in meta["caption"]


@pytest.mark.parametrize(
    "change",
    [
        "pilot",
        "rank_value",
        "recipe",
        "nonminimal",
        "bool",
        "nan",
        "model_pin",
        "manifest_pin",
        "threshold",
    ],
)
def test_inconsistent_or_pilot_inputs_are_rejected(source, change):
    path = source / "rank_b.json"
    rec = json.loads(path.read_text())
    if change == "pilot":
        rec["provenance"]["run_identity"]["smoke"] = True
    elif change == "rank_value":
        rec["selected_rank_test_r2"] += 0.2
    elif change == "recipe":
        rec["provenance"]["run_identity"]["run_id"] = "different"
    elif change == "nonminimal":
        rec["rank"] += 1
        rec["selected_rank_test_r2"] = rec["rank_curve"]["test_r2"][rec["rank"]]
        rec["selected_rank_validation_r2"] = rec["rank_curve"]["validation_r2"][rec["rank"]]
    elif change == "bool":
        rec["rank"] = True
    elif change == "nan":
        rec["full_test_r2"] = float("nan")
    elif change == "model_pin":
        rec["provenance"]["run_identity"]["model_revision"] = "incorrect"
    elif change == "manifest_pin":
        rec["provenance"]["run_identity"]["manifest_revision"] = "incorrect"
    elif change == "threshold":
        rec["validation_r2_threshold"] += 0.01
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        P.load_maps(source)
