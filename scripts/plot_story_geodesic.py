#!/usr/bin/env python3
"""Plot story-imprinting metric correlations and the tiny-bank sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from explore_persona_space.analysis.c2a_plot_style import (
    MUTED,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval_results/story_geodesic_20261002/summary.json"
OUT = ROOT / "figures/story_geodesic_20261002"
DIAGNOSTIC = "k5_d1_ridge0.0001"
SENSITIVITY = ["k3_d1_ridge0.0001", DIAGNOSTIC, "k7_d1_ridge0.0001", "k7_d2_ridge0.0001"]


def save(fig, width: float, stem: str, panels: list[dict]) -> None:
    """Write scientific figure formats and the exact plotted input values."""
    result = save_c2a_figure(
        fig,
        OUT / stem,
        title="Story imprinting: exploratory graph distance comparison",
        subject="Eight persona centroids; five published DeepSeek behavioral aggregates per panel.",
        creator="scripts/plot_story_geodesic.py",
        include_width=width,
    )
    (OUT / f"{stem}.meta.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "diagnostic": DIAGNOSTIC,
                "panels": panels,
                "render": result["record"],
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)


def main() -> None:
    """Render all fixed layers and labeled final-layer geodesic scatters."""
    data = json.loads(SOURCE.read_text())["models"]
    set_c2a_style()
    fig, width = c2a_figure("full", aspect=0.87)
    axes = fig.subplots(2, 2, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.85, hspace=0.4, wspace=0.2)
    panels = []
    styles = [
        ("raw_cosine", "Raw cosine", ROLES["linear"], "-"),
        ("whitened_cosine", "Whitened cosine", ROLES["control"], "--"),
        ("geodesic", "Geodesic diagnostic (k=5, d=1)", ROLES["nonlinear"], "-"),
    ]
    for row, (model, records) in enumerate(data.items()):
        layers = [int(k) for k in records]
        for col, persona in enumerate(("hhh", "fred")):
            ax = axes[row, col]
            vals = {}
            for key, label, role, line in styles:
                vals[key] = [
                    (r["graphs"][DIAGNOSTIC]["metrics"] if key == "geodesic" else r["baselines"])[
                        key
                    ][persona]["direct"]["spearman_rho"]
                    for r in records.values()
                ]
                ax.plot(
                    layers,
                    vals[key],
                    label=label,
                    color=role.color,
                    marker=role.marker,
                    ls=line,
                    lw=2.3,
                    ms=7,
                )
            grid = np.array(
                [
                    [
                        r["graphs"][setting]["metrics"]["geodesic"][persona]["direct"][
                            "spearman_rho"
                        ]
                        for setting in SENSITIVITY
                    ]
                    for r in records.values()
                ]
            )
            ax.fill_between(
                layers,
                grid.min(axis=1),
                grid.max(axis=1),
                color=ROLES["nonlinear"].color,
                alpha=0.13,
                label="Geodesic settings range",
            )
            ax.axhline(0, lw=1, color=MUTED, alpha=0.6)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticks(layers)
            label = "DeepSeek-V3.1-Base" if model == "deepseek" else "Qwen3.8-27B vectors"
            ax.set_title(
                f"{label} · {persona.upper() if persona == 'hhh' else 'Fred'}", loc="left", pad=12
            )
            if col == 0:
                ax.set_ylabel(r"Spearman $\rho$ with tracer uptake")
            if row == 1:
                ax.set_xlabel("Transformer block (zero-based)")
            style_axis(ax)
            panels.append(
                {
                    "model": model,
                    "persona": persona,
                    "layers": layers,
                    "series": vals,
                    "settings": SENSITIVITY,
                    "sensitivity_rho": grid.tolist(),
                }
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.99), ncol=2, fontsize=12
    )
    fig.text(
        0.08,
        0.075,
        "Five character outcomes per panel; both models use published DeepSeek behavior.",
        color=MUTED,
        fontsize=12,
    )
    fig.text(
        0.08,
        0.04,
        "Shading shows graph-setting sensitivity, not a confidence interval. Eight personas cannot support the original recipe.",
        color=MUTED,
        fontsize=11,
    )
    save(fig, width, "layer_comparison", panels)

    fig, width = c2a_figure("full", aspect=0.83)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.86, hspace=0.55, wspace=0.22)
    panels = []
    labels = ["Dismissive", "Sarcastic", "Saboteur", "Peer", "Help-seeker"]
    for row, (model, records) in enumerate(data.items()):
        layer = list(records)[-1]
        for col, persona in enumerate(("hhh", "fred")):
            ax = axes[row, col]
            rec = records[layer]["graphs"][DIAGNOSTIC]["metrics"]["geodesic"][persona]
            x, y = np.asarray(rec["scores"]), np.asarray(rec["rates"]) * 100
            span = np.ptp(x)
            ax.set_xlim(x.min() - span * 0.2, x.max() + span * 0.2)
            ax.set_ylim(0, 14 if persona == "hhh" else 72)
            ax.set_xticks(np.linspace(x.min(), x.max(), 4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_yticks([0, 4, 8, 12] if persona == "hhh" else [0, 20, 40, 60])
            ax.errorbar(
                x,
                y,
                yerr=np.asarray(rec["digitization_bounds"]) * 100,
                fmt="o",
                color=ROLES["nonlinear"].color,
                capsize=3,
            )
            for a, b, label in zip(x, y, labels, strict=True):
                align_right = a > (x.max() + x.min()) / 2
                dy = {"Sarcastic": 9, "Help-seeker": -10, "Peer": 8}.get(label, 5)
                if persona == "fred" and label == "Help-seeker":
                    dy = 0
                ax.annotate(
                    label,
                    (a, b),
                    xytext=(-6 if align_right else 6, dy),
                    textcoords="offset points",
                    ha="right" if align_right else "left",
                    va="center",
                    fontsize=11,
                )
            title = "DeepSeek" if model == "deepseek" else "Qwen"
            ax.set_title(
                f"{title} block {layer} · {persona.upper() if persona == 'hhh' else 'Fred'}",
                loc="left",
                pad=12,
            )
            ax.text(
                0.04,
                0.96,
                f"Spearman ρ = {rec['direct']['spearman_rho']:+.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=12,
            )
            ax.set_xlabel("Negative geodesic distance (closer →)")
            if col == 0:
                ax.set_ylabel("Published tracer uptake (%)")
            style_axis(ax)
            panels.append({"model": model, "layer": int(layer), "persona": persona, **rec})
    fig.text(
        0.08, 0.94, "Story imprinting · final-layer geodesic diagnostic", fontsize=21, weight="bold"
    )
    fig.text(
        0.08,
        0.067,
        "Fixed k=5, d=1; eight-node graph. Whiskers are digitization bounds, not sampling uncertainty.",
        color=MUTED,
        fontsize=12,
    )
    fig.text(
        0.08,
        0.03,
        "Both rows use published DeepSeek outcomes; Qwen tests cross-model geometry.",
        color=MUTED,
        fontsize=12,
    )
    save(fig, width, "final_layer_scatter", panels)


if __name__ == "__main__":
    main()
