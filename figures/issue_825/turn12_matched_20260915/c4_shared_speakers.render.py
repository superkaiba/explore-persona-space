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
from matplotlib.lines import Line2D

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
    transfer = data["panel_c"]["results"]
    assert transfer["n_conversations"] == 4975
    assert transfer["answer_draws"] == 1
    assert own["k_rollouts"] == shared["k_rollouts"] == 5
    assert shared["pool_settings"] == 6
    assert len(own["data"]) == len(shared["speakers"]) == 6

    def normalize(text):
        """Compare category names independently of line wrapping."""
        return " ".join(text.split())

    for a, b in zip(own["data"], shared["speakers"], strict=True):
        assert normalize(a["label"]) == normalize(b["label"])
        assert np.isclose(b["frac_shared_asis"], b["shared_asis"] / b["post_own"])

    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.34)
    ax_a = fig.add_axes([0.14, 0.28, 0.205, 0.58])
    ax_b = fig.add_axes([0.415, 0.28, 0.20, 0.58])
    ax_c = fig.add_axes([0.705, 0.28, 0.28, 0.58])
    y = np.arange(6)
    height = 0.33
    for role, key, offset in [("base_model", "base", -0.5), ("post_trained", "post_trained", 0.5)]:
        style = ROLES[role]
        ax_a.barh(
            y + offset * height,
            [r[key]["r2"] for r in own["data"]],
            height,
            color=style.color,
            label=style.label,
        )
    labels = ["Assistant (chat)", "Assistant (plain)", "HELIOS", "Wren", "Dana", "Vex"]
    ax_a.set(
        yticks=y,
        yticklabels=labels,
        ylim=(5.6, -0.6),
        xlim=(0, 0.8),
        xticks=[0, 0.4, 0.8],
        xlabel=better_label(METRIC_LABELS["r2"]),
    )
    # Retain the prior manuscript category-label size, now on the shared y axis.
    ax_a.tick_params(axis="y", labelsize=14)
    panel_header(ax_a, "A", "Separate maps", kicker_y=1.07)
    recovery = [r["frac_shared_asis"] for r in shared["speakers"]]
    ax_b.barh(y, recovery, height * 1.7, color=ROLES["linear"].color)
    ax_b.axvline(1.0, color=INK, linestyle="--", linewidth=1.2)
    ax_b.set(
        yticks=y,
        yticklabels=[],
        ylim=(5.6, -0.6),
        xlim=(0, 1.12),
        xticks=[0, 0.5, 1],
        xlabel=better_label("Fraction of own $R^2$"),
    )
    panel_header(ax_b, "B", "Shared map", kicker_y=1.07)
    for ax in (ax_a, ax_b):
        style_axis(ax, grid_axis="x")
    turns = np.arange(1, 13)
    sources = [
        ("1", "Turn 1", "s", ":"),
        ("3", "Turn 3", "D", "-"),
        ("12", "Turn 12", "v", "--"),
    ]
    model_roles = [("instruct", "post_trained"), ("pretrained", "base_model")]
    for model, role in model_roles:
        color = ROLES[role].color
        for source_turn, _, marker, linestyle in sources:
            rows = sorted(
                (
                    r
                    for r in transfer["models"][model]["cells"]
                    if r["source"] == source_turn and r["method"] == "raw"
                ),
                key=lambda r: r["target_turn"],
            )
            assert [r["target_turn"] for r in rows] == turns.tolist()
            values = np.array([r["r2"] for r in rows])
            ci = np.array([r["r2_ci95"] for r in rows])
            ax_c.plot(
                turns,
                values,
                color=color,
                marker=marker,
                markersize=4.5,
                markevery=[0, 2, 5, 8, 11],
                markerfacecolor=color if source_turn == "3" else "white",
                linestyle=linestyle,
                linewidth=1.8,
            )
            ax_c.fill_between(turns, ci[:, 0], ci[:, 1], color=color, alpha=0.10, linewidth=0)
    ax_c.set(
        xlim=(0.7, 12.3),
        ylim=(0.0, 0.65),
        xticks=[1, 6, 12],
        yticks=[0.2, 0.4, 0.6],
        xlabel="Evaluation turn",
        ylabel=better_label(METRIC_LABELS["r2"]),
    )
    panel_header(ax_c, "C", "Turn transfer", kicker_y=1.07)
    style_axis(ax_c)
    model_handles = [
        Line2D([], [], color=ROLES[role].color, linewidth=2, label=ROLES[role].label)
        for _, role in model_roles
    ]
    source_handles = [
        Line2D([], [], color=INK, marker=marker, linestyle=linestyle, label=label,
               markerfacecolor=INK if source == "3" else "white", markersize=6)
        for source, label, marker, linestyle in sources
    ]
    fig.legend(
        handles=model_handles,
        loc="lower left",
        bbox_to_anchor=(0.14, 0.005),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.legend(
        handles=source_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.005),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.7,
    )
    saved = save_c2a_figure(
        fig,
        stem,
        title="Linear predictability across framings, speakers, and conversation turns",
        subject=(
            "Panels A-B use matching five-answer targets; panel C uses one generated answer "
            "per turn with matched training counts and conversation IDs; main panel shows source "
            "turns 1, 3 and 12, with all five source conditions retained in the appendix"
        ),
        creator=Path(__file__).name,
        include_width=fraction,
    )
    meta = {
        "figure": "c4_shared_speakers",
        "panel_a": own,
        "panel_b": shared,
        "panel_c": data["panel_c"],
        "record": saved["record"],
        "main_panel_c_sources": ["1", "3", "12"],
        "appendix_full_conditions": "c4_turn_transfer_full.pdf",
        "source_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_deviations": (
            "shared y-axis categories retain the prior 14 pt label size; "
            "panel C uses marker shapes and line styles for training turn, color for checkpoint"
        ),
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print("Rendered", saved["pdf"])


if __name__ == "__main__":
    main()
