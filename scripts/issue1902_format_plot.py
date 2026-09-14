"""Render the requested OLMo on-policy panels from verified fit summaries."""

from pathlib import Path

import issue1902_format_common as C
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


def plot(root: Path, summary: dict):
    """Neutral descriptive labels; paired intervals are read directly from the summary."""
    set_c2a_style()
    cells = summary["cells"]
    labels = dict(B="Base", S="SFT", D="DPO", R="RLVR")
    colors = dict(base=ROLES["base_model"].color, own=ROLES["linear"].color)
    files = []

    def point(ax, x, value, ci, *, color, marker="o", filled=True):
        lo, hi = ci
        assert lo <= hi and np.isfinite([value, lo, hi]).all()
        # Draw interval endpoints directly; percentile intervals need not bracket the estimate.
        ax.vlines(x, lo, hi, color=color, linewidth=1.5)
        ax.plot(
            x,
            value,
            marker=marker,
            markersize=8,
            color=color,
            markerfacecolor=color if filled else "white",
            linestyle="none",
        )

    def save(fig, frac, letter, title):
        stem = root / "figures" / f"figure3{letter}_onpolicy"
        rendered = save_c2a_figure(
            fig,
            stem,
            title=title,
            subject=f"OLMo-2-7B layer 18; n={summary['n']}; 95% paired row-bootstrap intervals",
            creator="issue1902_format_plot.py",
            include_width=frac,
        )
        meta = stem.with_suffix(".meta.json")
        C.write_json(
            meta,
            dict(
                render=rendered["record"],
                n=summary["n"],
                target_definition=summary["target_definition"],
                source_sha256=C.sha(root / "fits/olmo/summary.json"),
            ),
        )
        files.extend([rendered["pdf"], rendered["png"], rendered["grayscale"], meta])
        plt.close(fig)

    fig, frac = c2a_figure("full", aspect=0.36)
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.76)
    ticklabels = []
    for i, stage in enumerate("BSDR"):
        for j, form in enumerate(("chat", "plain")):
            r = cells[f"fit_{stage}{stage}_{form}_fit"]
            point(
                ax,
                2 * i + j,
                r["r2"],
                r["r2_ci95"],
                color=colors["base" if stage == "B" else "own"],
                marker="o" if form == "chat" else "s",
            )
            ticklabels.append(f"{labels[stage]}\n{'Chat' if form == 'chat' else 'Plain'}")
    ax.set_xticks(range(8), ticklabels)
    ax.set_ylabel("Held-out $R^2$")
    style_axis(ax)
    panel_header(ax, "A", "Own answers", "Predictability across checkpoints and formats")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    save(fig, frac, "A", "Own-answer predictability")

    fig, frac = c2a_figure("full", aspect=0.38)
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.71)
    ticklabels = []
    for i, target in enumerate("SDR"):
        for j, form in enumerate(("chat", "plain")):
            x = 2 * i + j
            for delta, source, role in ((-0.14, "B", "base"), (0.14, target, "own")):
                r = cells[f"fit_{source}{target}_{form}_fit"]
                point(
                    ax,
                    x + delta,
                    r["r2"],
                    r["r2_ci95"],
                    color=colors[role],
                    marker="s" if source == "B" else "o",
                )
            ticklabels.append(f"{labels[target]}\n{'Chat' if form == 'chat' else 'Plain'}")
    ax.set_xticks(range(6), ticklabels)
    ax.set_ylabel("Held-out $R^2$")
    style_axis(ax)
    panel_header(
        ax, "B", "Target-generated and target-encoded answers", "Base versus target context vectors"
    )
    ax.legend(
        handles=[
            Line2D(
                [], [], color=colors["base"], marker="s", linestyle="none", label="Base context"
            ),
            Line2D(
                [], [], color=colors["own"], marker="o", linestyle="none", label="Target context"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
    )
    save(fig, frac, "B", "Prediction of fixed target representations")

    fig, frac = c2a_figure("full", aspect=0.38)
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.71)
    ticklabels = []
    for i, (source, target) in enumerate((("B", "S"), ("S", "D"), ("D", "R"))):
        for j, form in enumerate(("chat", "plain")):
            x = 2 * i + j
            for delta, mode, color, marker in (
                (-0.14, "direct", MUTED, "s"),
                (0.14, "scale_bias", colors["own"], "o"),
            ):
                r = next(
                    r
                    for r in summary["panel_c"]
                    if (r["source"], r["target"], r["format"], r["mode"])
                    == (source, target, form, mode)
                )
                point(
                    ax, x + delta, r["retention"], r["retention_ci95"], color=color, marker=marker
                )
            ticklabels.append(
                f"{labels[source]}→{labels[target]}\n{'Chat' if form == 'chat' else 'Plain'}"
            )
    ax.set_xticks(range(6), ticklabels)
    ax.set_ylabel("$R^2$ retention")
    style_axis(ax)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    panel_header(ax, "C", "Target on-policy pairs", "Transfer of the preceding checkpoint's map")
    ax.legend(
        handles=[
            Line2D([], [], color=MUTED, marker="s", linestyle="none", label="Direct transfer"),
            Line2D(
                [], [], color=colors["own"], marker="o", linestyle="none", label="Rescaling + bias"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
    )
    save(fig, frac, "C", "Frozen-map transfer on target on-policy pairs")
    return files
