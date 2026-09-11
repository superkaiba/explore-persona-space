"""Render the speaker panels using the explore-persona-space environment.

Run with ``uv run python /path/to/c4_shared_speakers.render.py`` from that
repository. The adjacent data JSON preserves each panel's targets and sources.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    METRIC_LABELS,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


def main():
    """Keep the shared-map numerator and denominator on matching targets."""
    stem = Path(__file__).with_name("c4_shared_speakers")
    source = stem.with_suffix(".data.json")
    data = json.loads(source.read_text())
    own = data["panel_a"]
    shared = data["panel_b"]
    assert own["k_rollouts"] == 3 and shared["k_rollouts"] == 1
    assert shared["pool_settings"] == 6
    assert len(own["data"]) == len(shared["speakers"]) == 6

    def normalize(text):
        return " ".join(text.split())

    for a, b in zip(own["data"], shared["speakers"], strict=True):
        assert normalize(a["label"]) == normalize(b["label"])
        assert np.isclose(b["frac_shared_asis"], b["shared_asis"] / b["post_own"])

    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.37)
    ax_a, ax_b = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.77, wspace=0.20)
    # Give the adjacent long assistant labels enough room at manuscript scale.
    x = np.array([0.0, 1.3, 2.5, 3.5, 4.5, 5.5])
    width = 0.38
    for role, key, offset in [("base_model", "base", -0.5), ("post_trained", "post_trained", 0.5)]:
        style = ROLES[role]
        ax_a.bar(
            x + offset * width,
            [r[key]["r2"] for r in own["data"]],
            width,
            color=style.color,
            label=style.label,
        )
    ax_a.set_ylabel(better_label(METRIC_LABELS["r2"]))
    ax_a.set_ylim(0.0, 0.85)
    panel_header(
        ax_a,
        "A",
        "Separate map",
        title="Separate maps",
        kicker_y=1.28,
        title_y=1.04,
    )

    recovery = [r["frac_shared_asis"] for r in shared["speakers"]]
    ax_b.bar(x, recovery, width * 1.6, color=ROLES["linear"].color, label="Shared map")
    ax_b.axhline(1.0, color=INK, linestyle="--", linewidth=1.6, label="Separate map")
    ax_b.set_ylabel(better_label("Fraction of own $R^2$"))
    ax_b.set_ylim(0.0, 1.5)
    panel_header(
        ax_b,
        "B",
        "Shared map",
        title="Shared vs. separate maps",
        kicker_y=1.28,
        title_y=1.04,
    )

    ticks = [r["label"].replace(" ", "\n") for r in own["data"]]
    for ax in (ax_a, ax_b):
        ax.set_xticks(x, ticks)
        # Preserve the current figure's documented three-line category size.
        ax.tick_params(axis="x", labelsize=14)
        style_axis(ax)
        ax.legend(
            loc="upper right" if ax is ax_a else "upper left",
            labelspacing=0.3,
            handlelength=1.4,
            borderaxespad=0.2,
        )
    saved = save_c2a_figure(
        fig,
        stem,
        title="Linear predictability across framings and speakers",
        subject=(
            "Separate maps: three-answer targets; six-setting shared map: single-answer targets"
        ),
        creator=Path(__file__).name,
        include_width=fraction,
    )
    meta = {
        "figure": "c4_shared_speakers",
        "panel_a": own,
        "panel_b": shared,
        "record": saved["record"],
        "source_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_deviations": "x-axis category labels use 14 pt, as in the prior manuscript figure",
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print("Rendered", saved["pdf"])


if __name__ == "__main__":
    main()
