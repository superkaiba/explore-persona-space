"""Render the synthetic-data essay's banked results; no fitting or model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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
DISTRIBUTIONS = ROOT / "docs/blog/synthetic-data/distribution_data.json"
OUT = ROOT / "figures/blog/synthetic-data"
TRAITS = ("evil", "sycophancy", "hallucination")
LABELS = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
METHODS = {
    "pv": ("Persona Vector on context", ROLES["base_model"].color, None),
    "map": ("My method", ROLES["linear"].color, "//"),
}


def bar_interval(ax, x, y, lo, hi, color, hatch):
    """Draw a bar from zero and the original percentile interval endpoints."""
    if not all(np.isfinite([x, y, lo, hi])) or lo > hi:
        raise ValueError(f"invalid interval: {(x, y, lo, hi)}")
    ax.barh(y, x, height=0.26, color=color, hatch=hatch, edgecolor="white", linewidth=0.5)
    ax.hlines(y, lo, hi, color=INK, linewidth=1.5)
    ax.vlines([lo, hi], y - 0.035, y + 0.035, color=INK, linewidth=1.5)


def legend(fig, methods, y):
    handles = [
        Patch(
            facecolor=METHODS[m][1],
            hatch=METHODS[m][2],
            edgecolor="white",
            label=METHODS[m][0],
        )
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=2,
        frameon=False,
        fontsize=16,
    )


def export(fig, name, title, rows, note, data, source=SOURCE):
    rendered = save_c2a_figure(
        fig, OUT / name, title=title, subject=note, creator=Path(__file__).name
    )
    meta = {
        "title": title,
        "rows": rows,
        "notes": note,
        "sources": data["sources"],
        "render": rendered["record"],
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
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
    for row in rows:
        (shared,) = [
            r
            for r in data["followup_comparison"]
            if r["setting"] == "pvsynth"
            and r["trait"] == row["trait"]
            and r["method"] == row["method"]
        ]
        assert (row["point"], row["lo"], row["hi"]) == (shared["rho"], *shared["ci"])
    fig, _ = c2a_figure("full", aspect=0.47)
    ax = fig.add_axes([0.24, 0.28, 0.70, 0.50])
    for i, trait in enumerate(TRAITS):
        for method, offset in [("pv", 0.16), ("map", -0.16)]:
            (row,) = [r for r in rows if r["trait"] == trait and r["method"] == method]
            y = 2 - i + offset
            _, color, hatch = METHODS[method]
            bar_interval(ax, row["point"], y, row["lo"], row["hi"], color, hatch)
            ax.annotate(
                f"{row['point']:.2f}",
                (row["hi"], y),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=16,
                color=color,
            )
    ax.set_yticks([2, 1, 0], [LABELS[t] for t in TRAITS])
    ax.set_ylim(-0.5, 2.55)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Spearman ρ with response score  ↑")  # noqa: RUF001
    style_axis(ax, grid_axis="x")
    ax.axvline(0, color=MUTED, linewidth=1)
    fig.text(
        0.04,
        0.95,
        "Predicting traits on synthetic prompts",
        fontsize=24,
        color=INK,
        weight="bold",
    )
    legend(fig, ["pv", "map"], 0.89)
    fig.text(
        0.04,
        0.055,
        "Qwen2.5-7B-Instruct · 200 contexts per trait\n"
        "Correlation across contexts · 95% context-bootstrap intervals",
        color=MUTED,
        fontsize=15,
    )
    export(
        fig,
        "01_initial_comparison",
        "Predicting traits on synthetic prompts",
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
            for key, color, hatch, offset in [
                ("overall", ROLES["base_model"].color, None, 0.17),
                ("within", ROLES["linear"].color, "//", -0.17),
            ]:
                y = 2 - i + offset
                ax.barh(
                    y,
                    row[key],
                    height=0.29,
                    color=color,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.annotate(
                    f"{row[key]:.3f}",
                    (row[key], y),
                    xytext=(7, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=15,
                )
        ax.set_yticks([2, 1, 0], [LABELS[t] for t in TRAITS] if setting == "system" else [])
        ax.set_ylim(-0.55, 2.65)
        ax.set_xlim(0, 1.08)
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
        Patch(
            facecolor=ROLES["base_model"].color,
            label="All prompts pooled",
        ),
        Patch(
            facecolor=ROLES["linear"].color,
            hatch="//",
            edgecolor="white",
            label="Within each prompt condition",
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
        "Published Pearson correlations; no confidence intervals reported in the table.",
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
    assert len(rows) == 22
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
            assert len(selected) == 2 and len({r["n_eval"] for r in selected}) == 1
            labels.append(f"{label}  (n={selected[0]['n_eval']:,})")
            for method, offset in [("pv", -0.16), ("map", 0.16)]:
                (row,) = [r for r in selected if r["method"] == method]
                _, color, hatch = METHODS[method]
                bar_interval(ax, row["rho"], i + offset, *row["ci"], color, hatch)
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
        0.04,
        0.96,
        "Predicting traits across evaluation datasets",
        fontsize=24,
        color=INK,
        weight="bold",
    )
    legend(fig, ["pv", "map"], 0.91)
    fig.text(
        0.04,
        0.025,
        "Same experiment as the synthetic comparison · 95% context-bootstrap intervals\n"
        "*NQ-Open / SimpleQA measure fabrication rate; other rows use graded trait scores.",
        color=MUTED,
        fontsize=14,
    )
    export(
        fig,
        "03_followup_datasets",
        "Predicting traits across evaluation datasets",
        rows,
        data["followup_notes"],
        data,
    )


def expression_distributions():
    """Show empirical score distributions and the synthetic instruction split."""
    data = json.loads(DISTRIBUTIONS.read_text())
    edges = np.asarray(data["bin_edges"])
    assert np.array_equal(edges, np.arange(0, 101, 10))
    rows = data["histograms"]
    for row in rows:
        assert sum(row["counts"]) == row["n_eval"] > 0
        assert len(row["counts"]) == len(edges) - 1
        assert min(row["counts"]) >= 0
        scores = np.asarray(row["scores"])
        assert len(scores) == row["n_eval"] and np.isfinite(scores).all()
        np.testing.assert_array_equal(np.histogram(scores, bins=edges)[0], row["counts"])

    fig, _ = c2a_figure("full", aspect=0.48)
    split_rows = [r for r in rows if r["group"] in {"pos", "neg"}]
    assert len(split_rows) == 6
    for i, trait in enumerate(TRAITS):
        ax = fig.add_axes([0.075 + i * 0.305, 0.27, 0.255, 0.42])
        for sign, offset, color, hatch in [
            ("neg", 0.5, MUTED, "//"),
            ("pos", 5.0, ROLES["base_model"].color, None),
        ]:
            (row,) = [r for r in split_rows if r["trait"] == trait and r["group"] == sign]
            ax.bar(
                edges[:-1] + offset,
                100 * np.asarray(row["counts"]) / row["n_eval"],
                width=4.5,
                align="edge",
                color=color,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(LABELS[trait], loc="left", fontsize=21, pad=12)
        ax.set(xlim=(0, 100), ylim=(0, 105), xticks=[0, 50, 100], yticks=[0, 50, 100])
        ax.set_xlabel("Behavior score", fontsize=17)
        if i == 0:
            ax.set_ylabel("Contexts (%)", fontsize=17)
        style_axis(ax, grid_axis="y")
    title = "What the synthetic instructions elicit"
    fig.text(0.04, 0.95, title, fontsize=24, weight="bold", color=INK)
    fig.legend(
        handles=[
            Patch(facecolor=MUTED, edgecolor="white", hatch="//", label="Suppressing instructions"),
            Patch(facecolor=ROLES["base_model"].color, label="Promoting instructions"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=2,
        frameon=False,
        fontsize=16,
    )
    fig.text(
        0.04,
        0.07,
        "100 contexts per instruction polarity, per trait · Share within each polarity\n"
        "Per-context mean behavior score · 10-point bins · Qwen2.5-7B-Instruct",
        color=MUTED,
        fontsize=14,
    )
    export(
        fig, "04_synthetic_expression", title, split_rows, data["notes"], data, source=DISTRIBUTIONS
    )

    titles = {
        "pvsynth": "Synthetic prompts",
        "hhrt": "Human red-team\nattempts",
        "toxicchat": "ToxicChat (flagged)",
        "wildchat_rung": "WildChat",
        "aita": "Held-out Reddit\nadvice",
        "nqopen": "NQ-Open*",
        "simpleqa": "SimpleQA*",
    }
    order = {
        "evil": ["pvsynth", "hhrt", "toxicchat", "wildchat_rung"],
        "sycophancy": ["pvsynth", "aita", "wildchat_rung"],
        "hallucination": ["pvsynth", "nqopen", "simpleqa", "wildchat_rung"],
    }
    pooled = [r for r in rows if r["group"] == "all"]
    assert len(pooled) == sum(map(len, order.values())) == 11
    fig, _ = c2a_figure("full", aspect=0.94)
    grid = fig.add_gridspec(
        3, 4, left=0.075, right=0.975, bottom=0.14, top=0.87, wspace=0.28, hspace=0.72
    )
    for i, trait in enumerate(TRAITS):
        for j, setting in enumerate(order[trait]):
            ax = fig.add_subplot(grid[i, j])
            (row,) = [r for r in pooled if r["trait"] == trait and r["setting"] == setting]
            ax.bar(
                edges[:-1],
                100 * np.asarray(row["counts"]) / row["n_eval"],
                width=10,
                align="edge",
                color=MUTED,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.set_title(
                f"{titles[setting]}\nn = {row['n_eval']:,}", loc="left", fontsize=16, pad=10
            )
            ax.set(xlim=(0, 100), ylim=(0, 105), xticks=[0, 50, 100], yticks=[0, 50, 100])
            ax.tick_params(labelsize=15)
            if j == 0:
                ax.set_ylabel(f"{LABELS[trait]}\nContexts (%)", fontsize=17)
            style_axis(ax, grid_axis="y")
    title = "Behavior expression across evaluation datasets"
    fig.text(0.04, 0.955, title, fontsize=24, weight="bold", color=INK)
    fig.text(0.52, 0.095, "Behavior score (0–100)", ha="center", fontsize=20, color=INK)
    fig.text(
        0.04,
        0.025,
        "Same 11 evaluation cells as the correlation comparison · 10-point bins\n"
        "*NQ-Open / SimpleQA: fabrication rate (%) · Other panels: mean graded trait score\n"
        "WildChat uses the held-out split · Missing scores excluded",
        color=MUTED,
        fontsize=14,
    )
    export(
        fig,
        "05_expression_across_datasets",
        title,
        pooled,
        data["notes"],
        data,
        source=DISTRIBUTIONS,
    )


def main():
    data = json.loads(SOURCE.read_text())
    set_c2a_style()
    first_comparison(data)
    paper_correlations(data)
    transfer(data)
    expression_distributions()


if __name__ == "__main__":
    main()
