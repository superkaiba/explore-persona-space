"""Explicitly synthetic report fixtures for plot-only v2 figure contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from scripts import issue952_china_repair_figures as figures


@pytest.fixture
def synthetic_report():
    """Use the real aggregate schema, visibly marked as synthetic if rendered."""
    report = {
        "contract": figures.REPORT_CONTRACT,
        "role": "full",
        "technical_complete": True,
        "synthetic_fixture": True,
        "coverage": {
            "realized_sources": 85,
            "planned_sources": 85,
            "topics": 12,
            "languages": ["en", "zh"],
            "geometry_selected_on_behavior": False,
        },
        "regime": {"n_resample": 10000},
        "panels": [],
    }
    for mass in figures.MASSES:
        for layer in figures.LAYERS:
            panel = {
                "layer": layer,
                "squared_singular_mass": mass,
                "role": "primary" if mass == 0.99 else f"sensitivity_{mass}",
                "n_geometry_subjects": 85,
                "rank_retained": 4,
                "rank_low": 8,
                "H2": {},
                "observed_answer_write_decomposition_secondary": {},
            }
            for language, _ in figures.LANGUAGES:
                context, answer = {}, {}
                for index, (key, _) in enumerate(figures.CONTRASTS):
                    mean = 0.22 + 0.15 * index
                    summary = {
                        "mean": mean,
                        "median": mean,
                        "ci95": [mean - 0.10, mean + 0.06],
                        "n_total": 85,
                        "n_defined": 85,
                        "n_undefined": 0,
                        "bootstrap_n_defined": 10000,
                    }
                    context[key] = {
                        "low_gain_share": copy.deepcopy(summary),
                        "n_exact_zero_vectors": 0,
                    }
                    answer[key] = {
                        "low_write_share": copy.deepcopy(summary),
                        "n_exact_zero_vectors": 0,
                    }
                panel["H2"][language] = {"contrasts": context}
                panel["observed_answer_write_decomposition_secondary"][language] = {
                    "basis": "RIGHT singular vectors (answer write directions)",
                    "contrasts": answer,
                }
            report["panels"].append(panel)
    return report


def test_exact_high_write_complement_and_interval_reversal(synthetic_report):
    result = figures.extract_displayed_data(synthetic_report)
    assert len(result["panels"]) == 6
    for panel in result["panels"]:
        for contrast in panel["contrasts"]:
            low, high = contrast["answer_low_write"], contrast["answer_high_write"]
            assert high["mean"] == pytest.approx(1 - low["mean"])
            assert high["ci95"] == pytest.approx([1 - low["ci95"][1], 1 - low["ci95"][0]])
            assert high["n_defined"] == low["n_defined"] == 85


def test_undefined_share_keeps_denominator_and_draws_no_zero(synthetic_report):
    record = synthetic_report["panels"][0]["H2"]["en"]["contrasts"]["subject"]
    record["low_gain_share"].update(
        mean=None, ci95=[None, None], n_defined=0, n_undefined=85, bootstrap_n_defined=0
    )
    record["n_exact_zero_vectors"] = 85
    data = figures.extract_displayed_data(synthetic_report)
    assert data["panels"][0]["contrasts"][0]["context_low_gain"]["mean"] is None
    fig, _, _ = figures.make_figure(data, "context")
    try:
        assert len(fig.axes[0].lines) == 3
        assert "Subject (0/85)" in [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "mutation", ["layer", "language", "geometry", "basis", "denominator", "half_interval"]
)
def test_invalid_or_incomplete_report_is_rejected(synthetic_report, mutation):
    if mutation == "layer":
        synthetic_report["panels"].pop(0)
    elif mutation == "language":
        synthetic_report["coverage"]["languages"] = ["en"]
    elif mutation == "geometry":
        synthetic_report["coverage"]["geometry_selected_on_behavior"] = True
    elif mutation == "basis":
        synthetic_report["panels"][0]["observed_answer_write_decomposition_secondary"]["en"][
            "basis"
        ] = "left vectors"
    else:
        share = synthetic_report["panels"][0]["H2"]["en"]["contrasts"]["subject"]["low_gain_share"]
        if mutation == "denominator":
            share["n_defined"] = 84
        else:
            share["ci95"][0] = None
    with pytest.raises(ValueError):
        figures.extract_displayed_data(synthetic_report)


def test_saved_ci_can_exclude_mean_without_errorbar_clipping(synthetic_report):
    share = synthetic_report["panels"][0]["H2"]["en"]["contrasts"]["subject"]["low_gain_share"]
    share.update(mean=0.2, ci95=[0.3, 0.4])
    data = figures.extract_displayed_data(synthetic_report)
    fig, _, _ = figures.make_figure(data, "context")
    try:
        assert fig.axes[0].lines[0].get_xdata()[0] == 0.2
        segment = fig.axes[0].collections[0].get_segments()[0]
        assert list(segment[:, 0]) == [0.3, 0.4]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["context", "answer"])
def test_axis_labels_clear_interval_footer(synthetic_report, kind):
    fig, _, _ = figures.make_figure(figures.extract_displayed_data(synthetic_report), kind)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer = next(text for text in fig.texts if text.get_text().startswith("Error bars:"))
        footer_top = footer.get_window_extent(renderer).y1
        for ax in fig.axes[-2:]:
            assert ax.xaxis.label.get_window_extent(renderer).y0 > footer_top + 5
    finally:
        plt.close(fig)


def test_report_hash_and_full_completion_are_required(synthetic_report, tmp_path):
    path = tmp_path / "report.json"
    figures.write_json(path, synthetic_report)
    with pytest.raises(FileNotFoundError):
        figures.load_report(path)
    with pytest.raises(ValueError, match="hash differs"):
        figures.load_report(path, "0" * 64)
    figures.write_json(
        tmp_path / "done.json",
        {
            "technical_complete": True,
            "role": "full",
            "report_sha256": figures.sha256(path),
        },
    )
    loaded, source = figures.load_report(path)
    assert loaded == synthetic_report and "sentinel" in source
    path.write_text('{"bad": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        figures.read_json(path)


def test_render_exports_all_formats_metadata_and_detects_changed_bytes(synthetic_report, tmp_path):
    source = tmp_path / "report.json"
    figures.write_json(source, synthetic_report)
    report_hash = figures.sha256(source)
    out_dir = tmp_path / "figures"
    result = figures.render_report(source, out_dir, expected_sha256=report_hash)
    assert set(result) == {"context", "answer"}
    for kind, record in result.items():
        assert record["source"]["sha256"] == report_hash
        assert record["render"]["style_version"] == "c2a-v2"
        assert record["render"]["include_width_frac"] == 1
        assert record["title"].startswith("Synthetic fixture:")
        assert record["displayed_data"]["synthetic_fixture"] is True
        assert "Layer 14" in " ".join(record["render"]["text"])
        assert "Layer 26" in " ".join(record["render"]["text"])
        assert "mass0p99" in record["outputs"]["pdf"]
        assert Path(record["outputs"]["pdf"]).read_bytes().startswith(b"%PDF-")
        for name, path in record["outputs"].items():
            assert figures.sha256(Path(path)) == record["output_sha256"][name]
        with Image.open(record["outputs"]["grayscale"]) as image:
            assert image.mode == "L"
        metadata = out_dir / f"china_v2_{kind}_shares_mass0p99.meta.json"
        assert json.loads(metadata.read_text())["regime"]["report_sha256"] == report_hash
    assert figures.render_report(source, out_dir, expected_sha256=report_hash) == result
    Path(result["context"]["outputs"]["png"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="output bytes changed"):
        figures.render_report(source, out_dir, expected_sha256=report_hash)


def test_sensitivity_selection_preserves_all_facets(synthetic_report):
    for mass in (0.90, 0.999):
        result = figures.extract_displayed_data(synthetic_report, mass)
        assert result["mass"] == mass
        assert len(result["panels"]) == 6
        assert result["role"] == f"sensitivity_{mass}"
