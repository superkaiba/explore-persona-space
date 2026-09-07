"""Check the actual two-panel reasoning render against its saved source scores."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


def test_combined_bars_match_saved_scores_and_intervals(tmp_path):
    """Exercise the real renderer/exporter, checking every bar, CI and source hash."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/section45_reasoning_story.py"
    spec = importlib.util.spec_from_file_location("section45_reasoning_story", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path
    module.set_c2a_style()
    fig = module.main_plot()
    assert len(fig.axes) == 2

    data_root = root / "eval_results/issue_2546"
    conditions = [
        json.loads((data_root / f"allfit/{cell}__a3.json").read_text())["subsets"]["all"]
        for cell in ["p7_Aoff", "p7_A", "p7_D"]
    ]
    snapshot = json.loads(
        (data_root / "paper_reasoning_20260906/qwen3_necessity_table.json").read_text()
    )
    pooled = snapshot["pooled_equal_corpus_weight"]
    expected_a = [
        (row[key], row[ci_key])
        for key, ci_key in [("r2_corpus", "r2_corpus_ci"), ("acc1", "acc1_ci")]
        for row in conditions
    ]
    expected_b = [
        (pooled[readout][group]["r2_corpus_mean"], pooled[readout][group]["r2_corpus_mean_ci"])
        for readout in ["context", "end_of_thought"]
        for group in ["necessary", "both_correct"]
    ]
    for ax, expected in zip(fig.axes, [expected_a, expected_b], strict=True):
        assert ax.get_ylim()[0] == 0
        assert len(ax.patches) == len(ax.collections) == len(expected)
        for bar, collection, (score, bounds) in zip(
            ax.patches, ax.collections, expected, strict=True
        ):
            assert bar.get_y() == 0
            assert bar.get_height() == score
            assert bar.get_zorder() < collection.get_zorder()
            center = bar.get_x() + bar.get_width() / 2
            np.testing.assert_allclose(collection.get_segments()[0][:, 0], center, atol=1e-14)
            np.testing.assert_array_equal(collection.get_segments()[0][:, 1], bounds)
        assert all(line.get_marker() == "none" for line in ax.lines)

    assert [bar.get_hatch() for bar in fig.axes[0].patches] == [None] * 3 + ["///"] * 3
    meta = json.loads((tmp_path / "c1_cot_story.meta.json").read_text())
    assert meta["values"]["maps_refit"] is False
    assert meta["values"]["panels"]["B"]["readouts"] == pooled
    assert meta["values"]["panels"]["A"]["n_evaluated"] == 33810
    assert meta["values"]["panels"]["A"]["evaluation_subset"] == "all"
    assert any("ALL QUESTIONS" in text.get_text() for text in fig.axes[0].texts)
    assert meta["render"]["include_width_frac"] == 1.0
    assert meta["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert len(meta["sources_sha256"]) == 4
    for relative_path, digest in meta["sources_sha256"].items():
        assert digest == hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
    for suffix in [".pdf", ".png", "_grayscale.png", ".meta.json"]:
        assert (tmp_path / f"c1_cot_story{suffix}").stat().st_size > 0
    assert not (tmp_path / "c1_cot_necessity_comparison.pdf").exists()
