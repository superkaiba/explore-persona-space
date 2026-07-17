"""#1335 round-2 supporting figure: fold-granularity + CJK-exclusion refits.

Two panels from eval_results/issue_1335/refits_r2_companions.json:
  left  — the fiction-framed Q&A rung refit under scenario-grouped vs
          row-level folds, per model, with the persona-described rung and
          fiction-endpoint committed values as reference points;
  right — Q&A rungs with CJK-completion rows excluded vs all rows, per model.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

B = Path("eval_results/issue_1335")
R = json.loads((B / "refits_r2_companions.json").read_text())
LS = json.loads((B / "ladder_summary.json").read_text())


def cell_l19(cell_id: str) -> float:
    d = json.loads((B / f"cells_{cell_id}.json").read_text())
    return float(d["r2_per_layer_obs"][19])


def main() -> None:
    set_paper_style("blog")
    c = paper_palette_blog(4)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- left: fold granularity on the fiction-framed rung (full n, layer 19)
    ax = axes[0]
    models = ["base", "instruct"]
    xs = {"base": 0.0, "instruct": 1.0}
    for j, mk in enumerate(models):
        fg = R["fold_granularity_r4"][mk]
        scen = fg["committed_full_n"]
        row = fg["row_fold_full_n"]["r2"]
        r3 = cell_l19(f"r3_persona__{mk}__ctx")
        x = xs[mk]
        ax.scatter([x - 0.12], [scen], s=70, color=c[j], zorder=3)
        ax.text(x - 0.12, scen + 0.012, "scenario\nfolds", ha="center", fontsize=8)
        gb = fg["row_fold_full_n"].get("group_bootstrap_l19", {})
        lo, hi = gb.get("ci_lo"), gb.get("ci_hi")
        yerr = None
        if lo is not None and hi is not None:
            yerr = [[row - lo], [hi - row]]
        ax.errorbar(
            [x + 0.12],
            [row],
            yerr=yerr,
            fmt="o",
            ms=8,
            mfc="none",
            mec=c[j],
            markeredgewidth=1.4,
            ecolor=c[j],
            zorder=3,
        )
        ax.plot([x - 0.12, x + 0.12], [scen, row], color=c[j], lw=1.0, alpha=0.6)
        ax.text(x + 0.12, row - 0.030, "row-level\nfolds", ha="center", fontsize=8)
        ax.hlines(r3, x - 0.28, x + 0.28, color=c[j], lw=1.4, linestyle="--")
        ax.text(x + 0.30, r3, "persona-described rung", va="center", fontsize=8, color=c[j])
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["base model", "instruct model"])
    ax.set_ylabel("held-out R² (layer 19, full n)")
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0.22, 0.56)
    ax.set_title("Fold granularity on the fiction-framed rung", loc="left")

    # ---- right: CJK-exclusion refits on the Q&A rungs (full n, layer 19)
    ax = axes[1]
    rungs = [("r0_qa_full", "Q&A, full answers"), ("r1_qa_oneline", "Q&A, one-line")]
    width = 0.18
    for j, mk in enumerate(models):
        for i, (slug, _label) in enumerate(rungs):
            e = R["cjk_filtered_qa"][f"{slug}__{mk}"]
            x = i + (j - 0.5) * 2.2 * width
            ax.scatter([x - width / 2], [e["committed_full_n"]], s=60, color=c[j], zorder=3)
            ax.scatter(
                [x + width / 2],
                [e["cjk_filtered_full_n"]["r2"]],
                s=60,
                facecolors="none",
                edgecolors=c[j],
                linewidths=1.4,
                zorder=3,
            )
            ax.plot(
                [x - width / 2, x + width / 2],
                [e["committed_full_n"], e["cjk_filtered_full_n"]["r2"]],
                color=c[j],
                lw=1.0,
                alpha=0.6,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label for _slug, label in rungs])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.37, 0.53)
    ax.set_ylabel("held-out R² (layer 19, full n)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c[0], label="base"),
        plt.Line2D([], [], marker="o", ls="", color=c[1], label="instruct"),
        plt.Line2D([], [], marker="o", ls="", color="0.3", label="filled = committed read"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec="0.3",
            markeredgewidth=1.4,
            label="open = robustness refit",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Multilingual-row exclusion on the Q&A rungs", loc="left")

    savefig_paper(fig, "issue_1335/robustness_refits", dir="figures/")
    plt.close(fig)
    print("saved figures/issue_1335/robustness_refits.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
