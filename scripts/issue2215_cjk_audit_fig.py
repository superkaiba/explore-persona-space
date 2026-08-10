"""Language-intrusion audit figure for issue #2215.

Paired-dot (dumbbell) view of the registered pooled paired 2AFC accuracy
(cosine, L19, tail-inclusive targets) per arm: all 10 banked draws vs the
CJK-intrusion-excluded recount (`eval_results/issue_2215/
cjk_intrusion_recount.json`, produced by `scripts/issue2215_cjk_recount.py`).
Reuses the arm labels/colors from `issue2215_figures.py` so the color-to-
meaning assignment matches every other figure of this issue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue2215_figures import ARM_COLORS, ARM_LABELS, ARM_ORDER, NULL_COLOR  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    res_dir = root / "eval_results" / "issue_2215"
    rec = json.loads((res_dir / "cjk_intrusion_recount.json").read_text())
    dv3 = json.loads((res_dir / "dv3_map_discrimination.json").read_text())

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = np.arange(len(ARM_ORDER))[::-1]
    band = dv3["per_config"]["779ce|L19|tail"]["pooled"]["cosine"]["null_band"]
    ax.axvspan(band[0], band[1], color=NULL_COLOR, alpha=0.5, zorder=0)
    for y, arm in zip(ys, ARM_ORDER):
        v_all = rec["validation_all_draws"][arm]["recount"]
        v_cln = rec["clean_only"][arm]
        ci = dv3["per_config"][f"{arm}|L19|tail"]["pooled"]["cosine"]["acc_ci95_clustered"]
        xerr = np.array([[v_all - ci[0]], [ci[1] - v_all]])
        ax.plot([v_all, v_cln], [y, y], color=ARM_COLORS[arm], lw=1.2, zorder=2)
        ax.errorbar(
            [v_all],
            [y],
            xerr=xerr,
            fmt="o",
            color=ARM_COLORS[arm],
            markersize=7,
            capsize=2,
            markeredgewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            [v_cln],
            [y],
            facecolors="none",
            edgecolors=ARM_COLORS[arm],
            s=70,
            linewidths=1.6,
            zorder=4,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.set_xlabel("pooled paired two-alternative accuracy (cosine)")
    add_direction_arrow(ax, axis="x", direction="up")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="#555555",
            lw=0,
            markersize=7,
            label="all 10 draws per context",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="#555555",
            lw=0,
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=1.6,
            label="CJK-intruded draws excluded",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=NULL_COLOR, alpha=0.5, label="shuffled-pair null 95% band"
        ),
    ]
    ax.legend(handles=handles, loc="lower right")
    set_title_subtitle(
        ax,
        "Pooled discrimination accuracy, with vs without language-drift rollouts",
    )
    savefig_paper(fig, "issue_2215/audit_cjk_recount", dir=str(root / "figures"))
    plt.close(fig)
    print("written:", root / "figures/issue_2215/audit_cjk_recount.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
