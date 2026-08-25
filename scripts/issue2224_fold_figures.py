"""Fold-round figures for issue #2224 (follow-up rounds 1 + 2).

Reads the committed follow-up JSONs under eval_results/issue_2224/followup_r{1,2}/
and renders four clean-result figures into figures/issue_2224/:

1. i2224_fu1_refit_rescue   — frozen vs per-corpus-refit map: score-level
   calibration r vs exact ΔP + top-500 Jaccard vs exact, per corpus × trait.
2. i2224_fu1_transport      — trait-probe AUC: same-corpus held-out vs
   cross-corpus transport, per train-corpus × trait.
3. i2224_fu1_rejudge        — sub-floor re-judge robustness: completeness
   before vs after, and the 24 affected contrast deltas before vs after.
4. i2224_fu2_seed137        — seed-42 vs seed-137 contrast deltas for the
   18 deciding cells, with response-level bootstrap CIs on both axes.

Run from the issue-2224 worktree root:
    uv run python scripts/issue2224_fold_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE matplotlib/numpy: shared-VM thread caps (#847)

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

R1 = Path("eval_results/issue_2224/followup_r1")
R2 = Path("eval_results/issue_2224/followup_r2")
OUT = "figures/"

TRAIT_LAYER = {"evil": "19", "sycophancy": "19", "hallucination": "15"}
CORPUS_LABEL = {"lmsys": "LMSYS", "ultrachat": "UltraChat"}
TRAIT_ORDER = ["evil", "hallucination", "sycophancy"]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_refit_rescue() -> None:
    """Frozen vs refit map: calibration r + top-500 Jaccard, per corpus x trait."""
    cells, r_frozen, r_refit, j_frozen, j_refit = [], [], [], [], []
    for corpus in ["lmsys", "ultrachat"]:
        data = _load(R1 / f"refit_{corpus}.json")
        layers = data["arms"]["context"]["layers"]
        for trait in TRAIT_ORDER:
            t = layers[TRAIT_LAYER[trait]]["traits"].get(trait)
            if t is None:
                continue
            cells.append(f"{CORPUS_LABEL[corpus]}\n{trait}")
            r_frozen.append(t["frozen_map_score_r_vs_exact"])
            r_refit.append(t["score_level_calibration_vs_exact"]["pearson_r"])
            j_frozen.append(t["jaccard_top_frozen_vs_exact"])
            j_refit.append(t["jaccard_top_refit_vs_exact"])

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(len(cells))
    w = 0.38
    c_frozen = paper_palette_role("baseline")
    c_refit = paper_palette_role("primary")

    ax = axes[0]
    ax.bar(x - w / 2, r_frozen, w, color=c_frozen, label="frozen map (parent run)")
    ax.bar(x + w / 2, r_refit, w, color=c_refit, label="per-corpus refit map")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=9.5)
    ax.set_ylabel("correlation with exact ΔP (per-sample r)")
    ax.set_title("Mapped-score calibration against exact ΔP", loc="left")
    ax.legend()

    ax = axes[1]
    ax.bar(x - w / 2, j_frozen, w, color=c_frozen, label="frozen map (parent run)")
    ax.bar(x + w / 2, j_refit, w, color=c_refit, label="per-corpus refit map")
    ax.axhline(0.005, color="0.4", linewidth=1.0, linestyle="--", label="chance (0.005)")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=9.5)
    ax.set_ylabel("top-500 overlap with exact ΔP (Jaccard)")
    ax.set_title("Top-500 selection overlap with exact ΔP", loc="left")
    ax.legend()

    fig.tight_layout()
    savefig_paper(fig, "issue_2224/i2224_fu1_refit_rescue", dir=OUT)
    plt.close(fig)


def fig_refit_rescue_iclr() -> None:
    """--style iclr: Overleaf-paper variant of the refit-rescue figure.

    Same cells as ``fig_refit_rescue`` at final ICLR size into figures/paper/.
    One colour (the featured-arm blue), FILL = map training pool: open = the
    frozen generic-pool map from the parent run, solid = the map refit on the
    target corpus's own 50,000 unjudged samples. No on-canvas titles beyond
    panel heads; provenance lives in the LaTeX caption.
    """
    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_color,
    )

    cells, r_frozen, r_refit, j_frozen, j_refit = [], [], [], [], []
    for corpus in ["lmsys", "ultrachat"]:
        data = _load(R1 / f"refit_{corpus}.json")
        layers = data["arms"]["context"]["layers"]
        for trait in TRAIT_ORDER:
            t = layers[TRAIT_LAYER[trait]]["traits"].get(trait)
            if t is None:
                continue
            cells.append(f"{trait} ({CORPUS_LABEL[corpus]})")
            r_frozen.append(t["frozen_map_score_r_vs_exact"])
            r_refit.append(t["score_level_calibration_vs_exact"]["pearson_r"])
            j_frozen.append(t["jaccard_top_frozen_vs_exact"])
            j_refit.append(t["jaccard_top_refit_vs_exact"])

    set_paper_style("iclr")
    blue = paper_color("instruct")
    fig, axes = plt.subplots(1, 2, figsize=figsize_iclr_panels(2, height_in=2.5))
    x = np.arange(len(cells))
    w = 0.38

    for ax, frozen, refit, ylab in (
        (axes[0], r_frozen, r_refit, "$r$ vs exact $\\Delta P$"),
        (axes[1], j_frozen, j_refit, "top-500 Jaccard"),
    ):
        ax.bar(
            x - w / 2,
            frozen,
            w,
            facecolor="white",
            edgecolor=blue,
            linewidth=0.7,
            label="frozen generic-pool map",
        )
        ax.bar(x + w / 2, refit, w, color=blue, label="corpus-refit map")
        ax.axhline(0, color=paper_color("reference"), linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(cells, rotation=35, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.set_ylabel(ylab)
    axes[1].axhline(
        0.005, color=paper_color("reference"), linewidth=0.7, linestyle="--", label="chance"
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    out_dir = Path("figures/paper")
    if not Path("eval_results").exists():
        out_dir = Path("/home/thomasjiralerspong/explore-persona-space/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "c5_screening_refit", dir=out_dir)
    plt.close(fig)
    print(f"wrote {out_dir / 'c5_screening_refit'}.png/.pdf (iclr)")


def fig_transport() -> None:
    """Trait-probe AUC: same-corpus held-out vs cross-corpus transport."""
    data = _load(R1 / "transport.json")
    labels, same, same_ci, cross, cross_ci = [], [], [], [], []
    for trait in TRAIT_ORDER:
        for corpus, other in [("lmsys", "ultrachat"), ("ultrachat", "lmsys")]:
            cell = data["cells"][f"{corpus}/{trait}"]
            s = cell["same_corpus_heldout_auc"]["trait_bearing_ge1"]
            t = cell[f"transport_to_{other}"]["auc"]["trait_bearing_ge1"]
            labels.append(f"{trait}\n{CORPUS_LABEL[corpus]}→{CORPUS_LABEL[other]}")
            same.append(s["auc"])
            same_ci.append(s["ci95"])
            cross.append(t["auc"])
            cross_ci.append(t["ci95"])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(labels))
    w = 0.38

    def _err(vals, cis):
        lo = [v - ci[0] for v, ci in zip(vals, cis)]
        hi = [ci[1] - v for v, ci in zip(vals, cis)]
        return [lo, hi]

    ax.bar(
        x - w / 2,
        same,
        w,
        yerr=_err(same, same_ci),
        capsize=2.5,
        color=paper_palette_role("baseline"),
        label="same-corpus held-out",
    )
    ax.bar(
        x + w / 2,
        cross,
        w,
        yerr=_err(cross, cross_ci),
        capsize=2.5,
        color=paper_palette_role("primary"),
        label="transported to the other corpus",
    )
    ax.axhline(0.5, color="0.4", linewidth=1.0, linestyle="--", label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC vs trait-filter judge label")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Trait-probe ranking accuracy: train corpus vs the other corpus", loc="left")
    ax.legend()

    fig.tight_layout()
    savefig_paper(fig, "issue_2224/i2224_fu1_transport", dir=OUT)
    plt.close(fig)


def fig_rejudge() -> None:
    """Sub-floor re-judge: completeness and contrast deltas, before vs after."""
    data = _load(R1 / "rejudge_updated_cells.json")
    comp_before, comp_after, names = [], [], []
    for name, cell in data["cells"].items():
        names.append(name)
        comp_before.append(cell["before"]["completeness"])
        comp_after.append(cell["after"]["completeness"])
    d_old = (
        [m["value"]["old_mean"] for m in data["contrast_moves"].values()]
        if isinstance(data["contrast_moves"], dict)
        else [m["old_mean"] for m in data["contrast_moves"]]
    )
    d_new = (
        [m["value"]["new_mean"] for m in data["contrast_moves"].values()]
        if isinstance(data["contrast_moves"], dict)
        else [m["new_mean"] for m in data["contrast_moves"]]
    )

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    ax = axes[0]
    ax.scatter(
        comp_before,
        comp_after,
        s=42,
        color=paper_palette_role("primary"),
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    lims = (0.60, 1.0)
    ax.plot(lims, lims, color="0.4", linewidth=0.9, zorder=1)
    ax.axhline(0.95, color="0.4", linewidth=0.9, linestyle="--")
    ax.axvline(0.95, color="0.4", linewidth=0.9, linestyle="--")
    screen_label = {
        "exact_dp": "exact ΔP",
        "prompttoken_dp": "prompt-token",
        "mapped_dp_context": "mapped",
        "probe_diff_context": "probe",
    }

    def _cell_label(name: str) -> str:
        _, trait, screen, tail = name.split("__")
        tail_label = "filtered top" if tail == "top_filtered" else tail
        return f"{trait} {screen_label[screen]} {tail_label}"

    worst = int(np.argmin(comp_before))
    ax.text(
        comp_before[worst] + 0.008,
        comp_after[worst],
        _cell_label(names[worst]),
        fontsize=8,
        va="center",
    )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("per-item completeness before re-judge")
    ax.set_ylabel("completeness after re-judge")
    ax.set_title(
        "Completeness, 17 sub-floor LMSYS cells\n(diagonal = unchanged; dashed = 0.95 floor)",
        loc="left",
    )

    ax = axes[1]
    ax.scatter(
        d_old,
        d_new,
        s=42,
        color=paper_palette_role("primary"),
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    lo = min(min(d_old), min(d_new)) - 1.0
    hi = max(max(d_old), max(d_new)) + 1.0
    ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=0.9, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("contrast Δ before re-judge (graded points)")
    ax.set_ylabel("contrast Δ after re-judge")
    ax.set_title("The 24 affected paired contrasts\n(diagonal = unchanged)", loc="left")

    fig.tight_layout()
    savefig_paper(fig, "issue_2224/i2224_fu1_rejudge", dir=OUT)
    plt.close(fig)


def fig_seed137() -> None:
    """Seed-42 vs seed-137 contrast deltas, 18 deciding-cell contrasts."""
    data = _load(R2 / "seed137_comparison.json")
    contrast_labels = {
        "exact_dp__top_vs_random": "exact top − random",
        "prompttoken_dp__top_vs_random": "prompt-token top − random",
        "prompttoken_dp__top_vs_exact_top": "prompt-token top − exact top",
    }
    colors = dict(zip(contrast_labels.values(), paper_palette_blog(3)))
    markers = {"lmsys": "o", "ultrachat": "s"}

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    seen = set()
    for c in (
        data["contrasts"].values() if isinstance(data["contrasts"], dict) else data["contrasts"]
    ):
        cc = c["value"] if "value" in c else c
        s42 = cc["seed42"]["response_level"]
        s137 = cc["seed137"]["response_level"]
        kind = contrast_labels[cc["contrast"]]
        label = None
        key = (cc["contrast"], cc["corpus"])
        series = f"{kind}" if cc["contrast"] not in seen else None
        seen.add(cc["contrast"])
        ax.errorbar(
            s42["mean"],
            s137["mean"],
            xerr=[[s42["mean"] - s42["ci_lo"]], [s42["ci_hi"] - s42["mean"]]],
            yerr=[[s137["mean"] - s137["ci_lo"]], [s137["ci_hi"] - s137["mean"]]],
            fmt=markers[cc["corpus"]],
            color=colors[kind],
            markersize=6,
            markeredgewidth=0.6,
            markeredgecolor="white",
            elinewidth=0.9,
            capsize=2.0,
            label=series,
            zorder=3,
        )
    lims = (-9, 20)
    ax.plot(lims, lims, color="0.4", linewidth=0.9, zorder=1)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.6", linewidth=0.8)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("contrast Δ at seed 42 (graded points, 95% bootstrap CI)")
    ax.set_ylabel("contrast Δ at seed 137")
    ax.set_title(
        "Deciding-cell contrasts, seed 42 vs seed 137\n(circles = LMSYS, squares = UltraChat; diagonal = exact replication)",
        loc="left",
    )
    ax.legend(loc="upper left")

    fig.tight_layout()
    savefig_paper(fig, "issue_2224/i2224_fu2_seed137", dir=OUT)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--style",
        choices=("blog", "iclr"),
        default="blog",
        help=(
            "iclr: render ONLY the paper refit-rescue variant into figures/paper/ "
            "and exit; the committed blog-register figures are untouched"
        ),
    )
    args = ap.parse_args()
    if args.style == "iclr":
        fig_refit_rescue_iclr()
    else:
        fig_refit_rescue()
        fig_transport()
        fig_rejudge()
        fig_seed137()
        print("done")
