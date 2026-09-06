"""Render the synthetic-data essay's banked results; no fitting or model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/blog/synthetic-data/plot_data.json"
OUT = ROOT / "figures/blog/synthetic-data"
TRAITS = ("evil", "sycophancy", "hallucination")
LABELS = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
METHODS = {
    "pv": ("Persona Vector on context", ROLES["base_model"].color, "s"),
    "map": ("Linear map + Persona Vector", ROLES["linear"].color, "o"),
    "oracle": ("Persona Vector on actual answer", MUTED, "x"),
}


def interval(ax, x, y, lo, hi, color, marker):
    """Draw endpoints directly: a percentile interval need not contain its estimate."""
    if not all(np.isfinite([x, y, lo, hi])) or lo > hi:
        raise ValueError(f"invalid interval: {(x, y, lo, hi)}")
    ax.hlines(y, lo, hi, color=color, linewidth=2)
    ax.vlines([lo, hi], y - 0.025, y + 0.025, color=color, linewidth=1.5)
    ax.plot(x, y, marker=marker, color=color, markersize=8, linestyle="none")


def legend(fig, methods, y):
    handles = [
        Line2D(
            [],
            [],
            color=METHODS[m][1],
            marker=METHODS[m][2],
            linestyle="none",
            label=METHODS[m][0],
            markersize=8,
        )
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=1 if len(methods) == 3 else 2,
        frameon=False,
        fontsize=16,
    )


def export(fig, name, title, rows, note, data):
    rendered = save_c2a_figure(
        fig, OUT / name, title=title, subject=note, creator=Path(__file__).name
    )
    meta = {
        "title": title,
        "rows": rows,
        "notes": note,
        "sources": data["sources"],
        "render": rendered["record"],
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_module_sha256": hashlib.sha256(
            Path(
                __import__("explore_persona_space.analysis.c2a_plot_style", fromlist=["x"]).__file__
            ).read_bytes()
        ).hexdigest(),
        "output_sha256": {
            key: hashlib.sha256(rendered[key].read_bytes()).hexdigest()
            for key in ("pdf", "png", "grayscale")
        },
    }
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print(f"Rendered {name}: {len(rows)} data rows", flush=True)


def first_comparison(data):
    rows = data["initial_comparison"]
    assert len(rows) == 6
    fig, _ = c2a_figure("full", aspect=0.47)
    ax = fig.add_axes([0.24, 0.28, 0.70, 0.50])
    for i, trait in enumerate(TRAITS):
        for method, offset in [("pv", 0.13), ("map", -0.13)]:
            (row,) = [r for r in rows if r["trait"] == trait and r["method"] == method]
            y = 2 - i + offset
            _, color, marker = METHODS[method]
            interval(ax, row["point"], y, row["lo"], row["hi"], color, marker)
            ax.annotate(
                f"{row['point']:.2f}",
                (row["point"], y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=16,
                color=color,
            )
    ax.set_yticks([2, 1, 0], [LABELS[t] for t in TRAITS])
    ax.set_ylim(-0.5, 2.55)
    ax.set_xlim(-0.45, 0.85)
    ax.set_xticks(np.arange(-0.4, 0.81, 0.2))
    ax.set_xlabel("Mean within-prompt Pearson r  ↑")
    style_axis(ax, grid_axis="x")
    ax.axvline(0, color=MUTED, linewidth=1)
    fig.text(
        0.04,
        0.95,
        "My first comparison on synthetic prompts",
        fontsize=24,
        color=INK,
        weight="bold",
    )
    legend(fig, ["pv", "map"], 0.89)
    fig.text(
        0.04,
        0.055,
        "Historical run · Qwen2.5-7B-Instruct · 95% condition-bootstrap intervals\n"
        "4 / 8 / 8 informative system prompts; the replication check did not pass.",
        color=MUTED,
        fontsize=15,
    )
    export(
        fig,
        "01_initial_comparison",
        "My first comparison on synthetic prompts",
        rows,
        data["initial_notes"],
        data,
    )


def paper_correlations(data):
    rows = data["paper_correlations"]
    assert len(rows) == 6
    fig, _ = c2a_figure("full", aspect=0.46)
    axes = [fig.add_axes([0.19, 0.23, 0.33, 0.50]), fig.add_axes([0.64, 0.23, 0.33, 0.50])]
    for ax, setting, title in zip(
        axes, ["system", "many_shot"], ["System prompts", "Many-shot prompts"], strict=True
    ):
        for i, trait in enumerate(TRAITS):
            (row,) = [r for r in rows if r["trait"] == trait and r["setting"] == setting]
            y = 2 - i
            ax.plot([row["within"], row["overall"]], [y, y], color=MUTED, linewidth=2)
            for key, color, marker, offset in [
                ("overall", ROLES["base_model"].color, "s", 11),
                ("within", ROLES["linear"].color, "o", -22),
            ]:
                ax.plot(row[key], y, marker=marker, color=color, markersize=9, linestyle="none")
                ax.annotate(
                    f"{row[key]:.3f}",
                    (row[key], y),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=15,
                )
        ax.set_yticks([2, 1, 0], [LABELS[t] for t in TRAITS] if setting == "system" else [])
        ax.set_ylim(-0.55, 2.65)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(title, loc="left", pad=18)
        ax.set_xlabel("Pearson r  ↑")
        style_axis(ax, grid_axis="x")
    fig.text(
        0.04,
        0.95,
        "The paper separates two kinds of predictability",
        fontsize=24,
        color=INK,
        weight="bold",
    )
    handles = [
        Line2D(
            [],
            [],
            color=ROLES["base_model"].color,
            marker="s",
            linestyle="none",
            label="All prompts pooled",
            markersize=8,
        ),
        Line2D(
            [],
            [],
            color=ROLES["linear"].color,
            marker="o",
            linestyle="none",
            label="Within each prompt condition",
            markersize=8,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
        fontsize=16,
    )
    fig.text(
        0.04,
        0.055,
        "Source: Persona Vectors, Appendix C.2, Table 2 (v3).\n"
        "Published point estimates; no confidence intervals reported in the table.",
        fontsize=15,
        color=MUTED,
    )
    export(
        fig,
        "02_pooled_vs_within",
        "The paper separates two kinds of predictability",
        rows,
        data["paper_notes"],
        data,
    )


def transfer(data):
    rows = data["followup_comparison"]
    assert len(rows) == 33
    fig, _ = c2a_figure("full", aspect=1.0)
    layouts = [(0.63, 0.18), (0.405, 0.14), (0.16, 0.18)]
    settings = {
        "evil": [
            ("pvsynth", "Synthetic trait prompts"),
            ("hhrt", "Human red-team attempts"),
            ("toxicchat", "ToxicChat (flagged)"),
            ("wildchat_rung", "WildChat conversations"),
        ],
        "sycophancy": [
            ("pvsynth", "Synthetic trait prompts"),
            ("aita", "Held-out Reddit advice"),
            ("wildchat_rung", "WildChat conversations"),
        ],
        "hallucination": [
            ("pvsynth", "Synthetic trait prompts"),
            ("nqopen", "NQ-Open*"),
            ("simpleqa", "SimpleQA*"),
            ("wildchat_rung", "WildChat conversations"),
        ],
    }
    for trait, (bottom, height) in zip(TRAITS, layouts, strict=True):
        ax = fig.add_axes([0.40, bottom, 0.54, height])
        labels = []
        for i, (setting, label) in enumerate(settings[trait]):
            selected = [r for r in rows if r["trait"] == trait and r["setting"] == setting]
            assert len(selected) == 3 and len({r["n_eval"] for r in selected}) == 1
            labels.append(f"{label}  (n={selected[0]['n_eval']:,})")
            for method, offset in [("pv", -0.22), ("map", 0), ("oracle", 0.22)]:
                (row,) = [r for r in selected if r["method"] == method]
                _, color, marker = METHODS[method]
                interval(ax, row["rho"], i + offset, *row["ci"], color, marker)
        ax.set_yticks(range(len(labels)), labels, fontsize=16)
        ax.set_ylim(len(labels) - 0.5, -0.7)
        ax.set_xlim(-0.3, 1)
        ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1])
        ax.set_title(LABELS[trait], loc="left", pad=12)
        style_axis(ax, grid_axis="x")
        ax.axvline(0, color=MUTED, linewidth=1)
        if trait == "hallucination":
            ax.set_xlabel("Spearman ρ with response score  ↑")  # noqa: RUF001
    fig.text(
        0.04, 0.96, "A later test across evaluation datasets", fontsize=24, color=INK, weight="bold"
    )
    legend(fig, ["pv", "map", "oracle"], 0.935)
    fig.text(
        0.04,
        0.025,
        "Later map recipe · 95% context-bootstrap intervals · "
        "actual-answer readout requires generation\n"
        "*NQ-Open / SimpleQA measure fabrication rate; other rows use graded trait scores.",
        color=MUTED,
        fontsize=14,
    )
    export(
        fig,
        "03_followup_datasets",
        "A later test across evaluation datasets",
        rows,
        data["followup_notes"],
        data,
    )


def main():
    data = json.loads(SOURCE.read_text())
    set_c2a_style()
    first_comparison(data)
    paper_correlations(data)
    transfer(data)


if __name__ == "__main__":
    main()
