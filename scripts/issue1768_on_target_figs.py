"""Round-3 (on-target-prefix-corpus) figures for issue #1768.

Reads eval_results/issue_1768/on_target/ and renders three figures into
figures/issue_1768/ with `on_target_` prefixes:

1. on_target_delta_d_dumbbell — hero: per-arm D under bare (n-matched refit)
   vs own trained prefix, per layer, swapped-prefix control overlaid.
2. on_target_percell_ecdf — low-level: per-context map-change norm ECDFs for
   one exemplar arm per trained-context class, per condition, floors marked.
3. on_target_prefix_side_reads — base-side prefix effect (M0_own vs M0_bare)
   and trained-prefix representation movement.

Behavior colors follow the round-1 figures (cas #0173B2, imp #DE8F05,
syc #029E73, mk #CC79A7). Conditions ride marker fill / linestyle so no
palette pair is reused for a different factor. Saves via savefig_paper
(blog style; PNG + PDF + meta.json sidecar).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

COLORS = {"cas": "#0173B2", "imp": "#DE8F05", "syc": "#029E73", "mk": "#CC79A7"}
GRAY_BARE = "#949494"

# Top-to-bottom row order for the dumbbell (conv/icl flips lead the story).
ARM_ORDER = [
    ("syc-conv-con-lr1e5-s42", "sycophancy — conversation (con, s42)"),
    ("syc-conv-con-lr1e5-s137", "sycophancy — conversation (con, s137)"),
    ("syc-conv-po-lr1e5-s42", "sycophancy — conversation (po, s42)"),
    ("syc-icl-con-lr1e5-s42", "sycophancy — ICL demos (con, s42)"),
    ("syc-icl-po-lr3e5-s42", "sycophancy — ICL demos (po, s42)"),
    ("syc-pers-con-lr1e5-s42", "sycophancy — persona (con, s42)"),
    ("syc-pers-con-lr1e5-s137", "sycophancy — persona (con, s137)"),
    ("syc-pers-po-lr1e5-s42", "sycophancy — persona (po, s42)"),
    ("syc-pers-ft-con-s42", "sycophancy — persona (full FT, s42)"),
    ("imp-pers-con-lr3e5-s42", "impoliteness — persona (con, s42)"),
    ("cas-pers-con-lr1e5-s42", "casual writing — persona (con, s42)"),
    ("mk-pers-con-lr5e6-s42", "marker token — persona (con, s42)"),
]

ECDF_CELLS = [
    ("syc-pers-con-lr1e5-s42", 19, "sycophancy — persona prefix (layer 19)"),
    ("syc-conv-con-lr1e5-s42", 19, "sycophancy — conversation prefix (layer 19)"),
    ("syc-icl-con-lr1e5-s42", 19, "sycophancy — ICL-demo prefix (layer 19)"),
    ("mk-pers-con-lr5e6-s42", 25, "marker token — persona prefix (layer 25)"),
]

PREFIX_NAMES = {
    "pers": "persona\n(system prompt)",
    "conv": "conversation\nhistory",
    "icl_syc": "ICL demos",
}


def beh(arm_id: str) -> str:
    return arm_id.split("-")[0]


def err_pair(v: float, lo: float, hi: float) -> tuple[float, float]:
    return max(0.0, v - lo), max(0.0, hi - v)


def fig_dumbbell(root: Path, figdir: str) -> None:
    contrast = json.load(open(root / "map_change_on_target.json"))["contrast"]
    fits_dir = root / "fits"

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.8), sharey=True)
    layers = [14, 19, 25]
    ys = np.arange(len(ARM_ORDER))[::-1]  # top row first

    for ax, layer in zip(axes, layers):
        ax.axvline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
        for y, (arm, _label) in zip(ys, ARM_ORDER):
            cell = contrast[f"{arm}_L{layer}"]
            c = COLORS[beh(arm)]
            d_own, d_bare = cell["D_own"], cell["D_bare_n"]
            ax.plot([d_bare, d_own], [y, y], color="0.75", lw=1.4, zorder=2)
            lo, hi = cell["D_bare_n_ci95"]
            e = np.array(err_pair(d_bare, lo, hi)).reshape(2, 1)
            ax.errorbar(
                [d_bare],
                [y],
                xerr=e,
                fmt="o",
                mfc="white",
                mec=c,
                ecolor=c,
                ms=6.5,
                mew=1.6,
                elinewidth=1.2,
                zorder=3,
            )
            lo, hi = cell["D_own_ci95"]
            e = np.array(err_pair(d_own, lo, hi)).reshape(2, 1)
            ax.errorbar(
                [d_own],
                [y],
                xerr=e,
                fmt="o",
                color=c,
                ecolor=c,
                ms=6.5,
                elinewidth=1.2,
                zorder=4,
            )
            if cell.get("D_control") is not None:
                fj = json.load(open(fits_dir / f"{arm}_L{layer}_control.json"))
                mc = fj["map_change"]
                lo, hi = mc["D_ci95"]
                e = np.array(err_pair(mc["D"], lo, hi)).reshape(2, 1)
                ax.errorbar(
                    [cell["D_control"]],
                    [y],
                    xerr=e,
                    fmt="x",
                    color=c,
                    ecolor=c,
                    ms=8.0,
                    mew=1.9,
                    elinewidth=1.2,
                    zorder=5,
                )
        ax.set_title(f"Layer {layer}")
        ax.grid(axis="x", color="0.9", lw=0.6)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _arm, label in ARM_ORDER], fontsize=9)
    for y, (arm, _label) in zip(ys, ARM_ORDER):
        axes[0].get_yticklabels()[list(ys).index(y)].set_color(COLORS[beh(arm)])

    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="white",
            mec="0.3",
            label="bare prompts (n-matched refit)",
        ),
        Line2D([], [], ls="", marker="o", color="0.3", label="own trained prefix"),
        Line2D([], [], ls="", marker="x", color="0.3", mew=1.9, label="swapped prefix (control)"),
        Line2D([], [], ls="--", color="0.45", label="refit-noise floor (D = 0)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.supxlabel(
        "Map change D (median map difference minus refit-noise floor)", y=0.075, fontsize=11
    )
    fig.suptitle(
        "Map change at the trained-in prefix vs bare prompts (12 arms, paired rows; 95% CIs)",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    out = savefig_paper(fig, "on_target_delta_d_dumbbell", dir=figdir)
    print({k: str(v) for k, v in out.items()})


def fig_percell_ecdf(root: Path, figdir: str) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))

    for ax, (arm, layer, title) in zip(axes.flat, ECDF_CELLS):
        c = COLORS[beh(arm)]
        conds = [
            (
                "own",
                root / "percell" / f"{arm}_L{layer}_own.json",
                root / "fits" / f"{arm}_L{layer}_own.json",
                c,
                "-",
                "own trained prefix",
            ),
            (
                "bare_n",
                root / "percell" / f"{arm}_L{layer}_bare_n.json",
                root / "fits_bare_n" / f"{arm}_L{layer}.json",
                GRAY_BARE,
                "-",
                "bare prompts",
            ),
            (
                "control",
                root / "percell" / f"{arm}_L{layer}_control.json",
                root / "fits" / f"{arm}_L{layer}_control.json",
                "0.15",
                "--",
                "swapped prefix",
            ),
        ]
        for _cond, pc_path, fit_path, color, ls, label in conds:
            if not pc_path.exists():
                continue
            deltas = np.array([r["delta"] for r in json.load(open(pc_path))["rows"]])
            xs = np.sort(deltas)
            ecdf = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ecdf, color=color, ls=ls, lw=1.9, label=label, zorder=3)
            floor = json.load(open(fit_path))["map_change"]["floor_p95"]
            ax.axvline(floor, color=color, ls=":", lw=1.4, zorder=2)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Per-context map-change norm (log scale)")
        ax.set_ylabel("Fraction of test contexts")
        ax.legend(fontsize=8.5, frameon=True, loc="upper left")

    fig.suptitle(
        "Per-context map change by condition (1,000 shared test rows; dotted verticals = refit floors)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = savefig_paper(fig, "on_target_percell_ecdf", dir=figdir)
    print({k: str(v) for k, v in out.items()})


def fig_prefix_side(root: Path, figdir: str) -> None:
    m0 = json.load(open(root / "m0_prefix_effect.json"))["cells"]
    pa = json.load(open(root / "prefix_arm.json"))["prefix_delta_reads"]

    set_paper_style("blog")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # Panel A: base-map movement per prefix x layer (content-decode base unit).
    prefixes = ["pers", "conv", "icl_syc"]
    layers = [14, 19, 25]
    shades = ["0.75", "0.5", "0.25"]
    w = 0.26
    for j, (layer, shade) in enumerate(zip(layers, shades)):
        vals = [m0[f"base_content@{p}_L{layer}"]["delta_med_on_own_grid"] for p in prefixes]
        ax_a.bar(
            np.arange(len(prefixes)) + (j - 1) * w,
            vals,
            width=w,
            color=shade,
            label=f"layer {layer}",
        )
    ax_a.set_xticks(np.arange(len(prefixes)))
    ax_a.set_xticklabels([PREFIX_NAMES[p] for p in prefixes], fontsize=10)
    ax_a.set_ylabel("Base-map movement (median per-context prediction difference)")
    ax_a.set_title("The prefix alone moves the base model's map", fontsize=11.5)
    ax_a.legend(fontsize=9, frameon=True)

    # Panel B: trained-prefix representation movement (own condition, layer 19).
    rows = [r for r in pa if r["condition"] == "own" and r["layer"] == 19]
    rows.sort(key=lambda r: r["prefix_delta_norm"] / r["prefix_norm_base"], reverse=True)
    xs = np.arange(len(rows))
    vals = [r["prefix_delta_norm"] / r["prefix_norm_base"] for r in rows]
    cols = [COLORS[beh(r["arm_id"])] for r in rows]
    labels = [r["arm_id"] for r in rows]
    ax_b.bar(xs, vals, color=cols)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax_b.set_ylabel("Trained-prefix movement (fraction of base prefix norm)")
    ax_b.set_title(
        "Training barely moves the trained-prefix representation (layer 19)", fontsize=11.5
    )

    fig.tight_layout()
    out = savefig_paper(fig, "on_target_prefix_side_reads", dir=figdir)
    print({k: str(v) for k, v in out.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="figures/issue_1768")
    ap.add_argument("--root", default="eval_results/issue_1768/on_target")
    args = ap.parse_args()
    root = Path(args.root)
    fig_dumbbell(root, args.figdir)
    fig_percell_ecdf(root, args.figdir)
    fig_prefix_side(root, args.figdir)


if __name__ == "__main__":
    main()
