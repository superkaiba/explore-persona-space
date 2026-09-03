"""Tests for the context-to-answer paper's shared visual system."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from explore_persona_space.analysis.c2a_plot_style import (
    PAPER,
    PREDICTOR_STYLES,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)


def test_c2a_style_contract() -> None:
    font = set_c2a_style()
    assert STYLE_VERSION == "c2a-v2"
    assert font in {"Inter", "Noto Sans", "DejaVu Sans"}
    assert matplotlib.rcParams["figure.facecolor"] == PAPER
    assert PREDICTOR_STYLES["ridge"].color == "#176B87"
    assert PREDICTOR_STYLES["ridge"].marker == "o"
    assert PREDICTOR_STYLES["mlp_w8192"].color == "#C4553D"
    assert PREDICTOR_STYLES["mlp_w8192"].marker == "D"


def test_score_axis_and_export_bundle(tmp_path) -> None:
    set_c2a_style()
    fig, ax = plt.subplots()
    style_score_axis(ax)
    assert ax.get_ylim() == (0.5, 1.01)
    assert list(ax.get_yticks()) == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()

    ax.plot([0, 1], [0.6, 0.9])
    outputs = save_c2a_figure(
        fig,
        tmp_path / "example",
        title="Example",
        subject="Test figure",
        creator="tests/test_c2a_plot_style.py",
    )
    plt.close(fig)

    assert set(outputs) == {"pdf", "png", "grayscale", "record"}
    for key in ("pdf", "png", "grayscale"):
        assert outputs[key].exists() and outputs[key].stat().st_size > 0
    assert outputs["record"]["style_version"] == STYLE_VERSION
    with Image.open(outputs["png"]) as color:
        assert color.mode in {"RGB", "RGBA"}
        assert color.getpixel((0, 0))[:3] == (255, 255, 255)
    with Image.open(outputs["grayscale"]) as grayscale:
        assert grayscale.mode == "L"
