"""Single-panel figure: map quality per decode arm x fitter class (#1073).

Combines the linear (ridge) and nonlinear (RBF kernel ridge, width-512 MLP)
held-out test R2 from eval_results/issue_1073/mlp_krr_decode_regime.json into
one plot: color = decode arm, linestyle/marker = fitter class.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

RESULTS = Path("eval_results/issue_1073/mlp_krr_decode_regime.json")

ARM_LABELS = {
    "avg10": "10-rollout average",
    "greedy": "Greedy (deterministic)",
    "stoch1_old": "Single stochastic",
}
FITTERS = {
    "ridge_test_r2": ("Ridge (linear)", "-", "o"),
    "krr_test_r2": ("Kernel ridge (nonlinear)", "--", "s"),
    "mlp_w512_test_r2": ("MLP width-512 (nonlinear)", ":", "^"),
}


def main() -> None:
    table = json.loads(RESULTS.read_text())["table"]
    layers = [14, 17, 19, 26, 27]
    xs = range(len(layers))

    set_paper_style("blog")
    fig, ax = plt.subplots()

    colors = dict(zip(ARM_LABELS, paper_palette(len(ARM_LABELS))))
    for arm, arm_label in ARM_LABELS.items():
        for key, (_, ls, marker) in FITTERS.items():
            ys = [table[arm][str(layer)][key] for layer in layers]
            ax.plot(
                xs,
                ys,
                ls,
                marker=marker,
                color=colors[arm],
                markersize=5,
                markeredgewidth=0.0,
                linewidth=1.6,
            )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("read-out layer")
    ax.set_ylabel("held-out test R²")
    add_direction_arrow(ax, "y", "up")

    arm_handles = [
        Line2D([0], [0], color=colors[arm], linewidth=2.2, label=label)
        for arm, label in ARM_LABELS.items()
    ]
    fitter_handles = [
        Line2D([0], [0], color="0.25", linestyle=ls, marker=marker, markersize=5, label=label)
        for label, ls, marker in FITTERS.values()
    ]
    ax.set_ylim(0.555, 0.82)
    first = ax.legend(handles=arm_handles, loc="upper left", title=None)
    ax.add_artist(first)
    ax.legend(handles=fitter_handles, loc="upper right", title=None)

    set_title_subtitle(
        ax,
        "Map quality by decoding regime and fitter class",
        "held-out R² of the context→answer map; validation-selected split, n = 5,000 contexts",
    )
    savefig_paper(fig, "issue_1073/map_quality_linear_nonlinear", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
