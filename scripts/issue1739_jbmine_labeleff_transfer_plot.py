"""Figures for the label-efficiency sweep and the cross-family transfer arms.

Two panels each, simple by convention (axes + ticks + legend + panel titles only,
no caption block rendered on the canvas):

  label-efficiency: PR-AUC vs number of labelled in-domain contexts, one panel per
  layer, curves for A (context probe) / D through the in-domain map / D through the
  merged map, error bars = SD over draws, with the full-label A reference, the
  answer-space oracle E, and chance as reference lines.

  transfer: grouped bars per direction (train family -> test family) for
  A_transfer / D_transfer (in-domain map) / D_transfer (merged map) /
  E_transfer (oracle) against the within-family A reference and chance, at the
  pre-specified layer; plus the best-layer-per-arm view.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
OUT_LE = Path("docs/scratch/jailbreak_mining_pilot_label_efficiency.png")
OUT_TR = Path("docs/scratch/jailbreak_mining_pilot_transfer.png")
PRESPEC_LAYER = "19"
# one colour per arm, held constant across BOTH figures
ARM_COLOR = {"A": "C0", "D_indomain": "C2", "D_merged": "C4", "E": "0.35"}
LE_CURVES = [
    ("A", "A: probe on v_C", "o"),
    ("D_indomain", "D: v_A-probe thru M (in-dom)", "s"),
    ("D_merged", "D: v_A-probe thru M (merged)", "^"),
]
TR_BARS = [
    ("A_transfer", "A: probe on v_C", "A"),
    ("D_transfer_indomain", "D: thru M (in-dom)", "D_indomain"),
    ("D_transfer_merged", "D: thru M (merged)", "D_merged"),
    ("E_transfer_oracle", "E: probe on v_A (oracle)", "E"),
]


def plot_label_efficiency() -> None:
    """PR-AUC vs labelled-context budget, one panel per swept layer."""
    r = json.loads((DEST / "label_efficiency_results.json").read_text())
    budgets = r["budgets"]
    base = r["eval"]["base_rate"]
    layers = sorted(r["layers"], key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(6.0 * len(layers), 4.6), squeeze=False)
    for ax, L in zip(axes[0], layers):
        lay = r["layers"][L]
        for arm, label, marker in LE_CURVES:
            means = [lay["curves"][str(n)][arm]["pr_auc_mean"] for n in budgets]
            sds = [lay["curves"][str(n)][arm]["pr_auc_sd"] for n in budgets]
            ax.errorbar(
                budgets,
                means,
                yerr=sds,
                marker=marker,
                color=ARM_COLOR[arm],
                label=label,
                lw=1.6,
                ms=6,
                capsize=3,
            )
        ref = lay["full_label_ref"]
        ax.axhline(
            ref["A"],
            ls="-.",
            color=ARM_COLOR["A"],
            lw=1,
            label=f"A, all {ref['n_train']} labels ({ref['A']:.2f})",
        )
        ax.axhline(
            ref["E_oracle"],
            ls="--",
            color=ARM_COLOR["E"],
            lw=1,
            label=f"oracle E ({ref['E_oracle']:.2f})",
        )
        ax.axhline(base, ls=":", color="k", lw=1, label=f"chance ({base:.2f})")
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(n) for n in budgets])
        ax.set_xlabel("labelled in-domain contexts N")
        ax.set_ylabel("PR-AUC (average precision)")
        ax.set_ylim(0, 1.03)
        ax.set_title(f"Label efficiency, layer {L}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_LE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_LE, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT_LE}")


def plot_transfer() -> None:
    """Grouped transfer bars: pre-specified layer (left) and best-layer view (right)."""
    r = json.loads((DEST / "transfer_results.json").read_text())
    dirs = list(r["directions"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.2
    for ax, mode in ((ax1, "prespec"), (ax2, "best")):
        for j, (arm, label, ckey) in enumerate(TR_BARS):
            vals = []
            for d in dirs:
                layers = r["directions"][d]["layers"]
                if mode == "prespec":
                    vals.append(layers[PRESPEC_LAYER][arm]["pr_auc"])
                else:
                    vals.append(max(layers[L][arm]["pr_auc"] for L in layers))
            xs = [i + (j - 1.5) * width for i in range(len(dirs))]
            ax.bar(
                xs,
                vals,
                width,
                color=ARM_COLOR[ckey],
                edgecolor="k",
                linewidth=0.5,
                label=label if mode == "prespec" else None,
            )
        # within-family reference + chance, per direction
        for i, d in enumerate(dirs):
            layers = r["directions"][d]["layers"]
            wf = (
                layers[PRESPEC_LAYER]["A_within"]["pr_auc"]
                if mode == "prespec"
                else max(layers[L]["A_within"]["pr_auc"] for L in layers)
            )
            ax.plot(
                [i - 2 * width, i + 2 * width],
                [wf, wf],
                ls="-.",
                color="C1",
                lw=1.8,
                label="A within-family (ref)" if (mode == "prespec" and i == 0) else None,
            )
            base = r["directions"][d]["test_base_rate"]
            ax.plot(
                [i - 2 * width, i + 2 * width],
                [base, base],
                ls=":",
                color="k",
                lw=1.2,
                label=f"chance ({base:.2f})" if (mode == "prespec" and i == 0) else None,
            )
        ax.set_xticks(range(len(dirs)))
        ax.set_xticklabels([d.replace("->", "\n→ ") for d in dirs], fontsize=8)
        ax.set_ylabel("PR-AUC (average precision)")
        ax.set_ylim(0, 1.03)
        ax.set_title(
            f"Cross-family transfer, layer {PRESPEC_LAYER} (pre-specified)"
            if mode == "prespec"
            else "Cross-family transfer, best layer per arm",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    OUT_TR.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TR, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT_TR}")


def plot_label_efficiency_iclr() -> None:
    """--style iclr: Overleaf-paper variant, pre-specified layer only.

    PR-AUC (average precision) vs labelled in-domain contexts at layer 19.
    Colour scheme follows the paper's map-arm convention: featured blue =
    through-the-map reads (open-marker dashed = the plain context-vector
    probe, the context-side comparison), oracle purple = the answer-space
    oracle reference. Error bars: SD over 5 draws per budget.
    """
    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    r = json.loads((DEST / "label_efficiency_results.json").read_text())
    budgets = r["budgets"]
    base = r["eval"]["base_rate"]
    lay = r["layers"][PRESPEC_LAYER]
    set_paper_style("iclr")
    blue = paper_color("instruct")
    purple = paper_color("oracle_answer")
    black = paper_color("reference")
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.60))
    series = [
        ("A", "probe on the context vector", dict(ls="--", marker="o", mfc="white", mec=blue)),
        ("D_indomain", "probe through the in-domain map", dict(ls="-", marker="s", mfc=blue)),
        (
            "D_merged",
            "probe through the merged map",
            dict(ls="-", marker="^", mfc=blue, alpha=0.55),
        ),
    ]
    for arm, label, kw in series:
        means = [lay["curves"][str(n)][arm]["pr_auc_mean"] for n in budgets]
        sds = [lay["curves"][str(n)][arm]["pr_auc_sd"] for n in budgets]
        ax.errorbar(
            budgets,
            means,
            yerr=sds,
            color=blue,
            label=label,
            lw=1.0,
            ms=3.5,
            capsize=1.6,
            markeredgewidth=0.8,
            **kw,
        )
    ref = lay["full_label_ref"]
    ax.axhline(
        ref["A"],
        ls="-.",
        color=blue,
        lw=0.7,
        label=f"context probe, all {ref['n_train']:,} labels",
    )
    ax.axhline(ref["E_oracle"], ls="--", color=purple, lw=0.8, label="real-answer oracle probe")
    ax.axhline(base, ls=":", color=black, lw=0.7, label="chance")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(n) for n in budgets])
    ax.minorticks_off()
    ax.set_xlabel("labelled in-domain contexts")
    ax.set_ylabel("PR-AUC (average precision)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="center right", fontsize=6.5)
    fig.tight_layout()
    out_dir = Path("/home/thomasjiralerspong/explore-persona-space/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "c5_jailbreak_mining", dir=out_dir)
    plt.close(fig)
    print(f"wrote {out_dir / 'c5_jailbreak_mining'}.png/.pdf (iclr)")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--style",
        choices=("blog", "iclr"),
        default="blog",
        help=(
            "iclr: render ONLY the paper label-efficiency variant into figures/paper/ "
            "and exit; the scratch-report figures are untouched"
        ),
    )
    args = ap.parse_args()
    if args.style == "iclr":
        plot_label_efficiency_iclr()
        return 0
    plot_label_efficiency()
    plot_transfer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
