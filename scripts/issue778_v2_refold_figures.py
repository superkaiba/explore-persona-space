"""Analyzer re-fold figures for issue #778 round `faithful-extraction-honest-nulls-rerun`.

Three figures over the FINAL v2 honest-null ladder JSONs (no new computation):

1. ``fwer_minp_families`` — per honest null family, the observed min empirical p
   over the 12 monitoring headline cells vs the null min-p distribution
   (registered min-p FWER construction), from ``fwer_headline_v2.json``.
2. ``fixed_layer_cell_verdicts`` — per regime-cell at the pre-registered paper
   steering layer: observed matched |r| vs the primary-family and neutral-corpus
   null 97.5th-percentile caps.
3. ``hallucination_layer_profile`` — hallucination monitoring |r| per layer with
   the paper steering layer marked and the neutral-corpus per-layer null cap.

Usage: uv run python scripts/issue778_v2_refold_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

RES = Path("eval_results/issue_778/faithful-extraction-honest-nulls-rerun")
OUT = Path("figures/issue_778/faithful-extraction-honest-nulls-rerun")

PRIMARY = {"evil": "within_class", "sycophancy": "neg_arm_only", "hallucination": "neg_arm_only"}
TRAITS = ["evil", "sycophancy", "hallucination"]
SETTINGS = ["finetune", "monitoring_corrected", "monitoring_manyshot"]
SETTING_LABEL = {
    "finetune": "finetuning shift",
    "monitoring_corrected": "8-prompt monitoring",
    "monitoring_manyshot": "many-shot monitoring",
}
STAT_LABEL = {"overall": "pooled", "within": "within-condition"}
FAMILY_LABEL = {
    "isotropic": "isotropic floor",
    "within_class": "within-class covariance",
    "neg_arm_only": "negative-arm covariance",
    "neutral_cov": "neutral-corpus covariance",
    "rb_projected_out": "direction projected out",
    "target_shuffle": "score-shuffle permutation",
    "primary_mixed": "per-trait primary family",
}


def load_cells() -> dict[tuple[str, str, str], dict]:
    cells: dict[tuple[str, str, str], dict] = {}
    for trait in TRAITS:
        for setting in SETTINGS:
            d = json.loads((RES / f"{trait}_{setting}_honestnulls_v2.json").read_text())
            for stat, block in d["stage_fixed"].items():
                cells[(trait, setting, stat)] = block["per_choice"]["paper_steering"]
    return cells


def fig_fwer() -> None:
    fwer = json.loads((RES / "fwer_headline_v2.json").read_text())
    fams = [f for f in FAMILY_LABEL if f in fwer["families"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colors = paper_palette_blog(2)
    ys = np.arange(len(fams))[::-1]
    for y, fam in zip(ys, fams, strict=True):
        v = fwer["families"][fam]
        q = v["null_min_p_quantiles"]
        ax.plot(
            [q["p2_5"], q["p97_5"]],
            [y, y],
            lw=5,
            color=colors[1],
            alpha=0.45,
            solid_capstyle="round",
            label="null min-p (2.5th-97.5th pct, 10,000 joint draws)" if y == ys[0] else None,
        )
        ax.plot([q["p50"]], [y], marker="|", ms=14, color=colors[1], mew=2)
        ax.plot(
            [v["observed_min_p"]],
            [y],
            marker="o",
            ms=9,
            color=colors[0],
            zorder=5,
            label="observed min p over the 12 monitoring cells" if y == ys[0] else None,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in fams])
    ax.set_xscale("log")
    ax.set_xlabel("min one-sided empirical p over the 12 monitoring cells (log scale)")
    ax.set_title("Observed min-p sits far below every null family's min-p distribution", pad=12)
    ax.legend(loc="lower right", frameon=False)
    savefig_paper(fig, "fwer_minp_families", dir=OUT)
    plt.close(fig)


def fig_cell_verdicts() -> None:
    cells = load_cells()
    order: list[tuple[str, str, str]] = []
    for trait in TRAITS:
        for setting in SETTINGS:
            stats = ["overall"] if setting == "finetune" else ["overall", "within"]
            for stat in stats:
                order.append((trait, setting, stat))
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    tcolors = dict(zip(TRAITS, paper_palette_blog(3), strict=True))
    ys = np.arange(len(order))[::-1]
    labels = []
    for y, key in zip(ys, order, strict=True):
        trait, setting, stat = key
        pc = cells[key]
        prim = pc["nulls"][PRIMARY[trait]]
        neut = pc["nulls"]["neutral_cov"]
        suffix = " [effect-size only]" if setting == "finetune" else ""
        labels.append(f"{trait} · {SETTING_LABEL[setting]} · {STAT_LABEL[stat]}{suffix}")
        ax.plot(
            [prim["p97_5"]],
            [y + 0.12],
            marker="|",
            ms=11,
            mew=2.4,
            color="0.35",
            label="primary honest-null 97.5th pct" if y == ys[0] else None,
        )
        ax.plot(
            [neut["p97_5"]],
            [y - 0.12],
            marker="|",
            ms=11,
            mew=2.4,
            color="0.65",
            label="neutral-corpus null 97.5th pct" if y == ys[0] else None,
        )
        passed = (prim["bh_within_family"] <= 0.05) and (neut["bh_within_family"] <= 0.05)
        ax.plot(
            [pc["observed_abs_r"]],
            [y],
            marker="o" if passed else "s",
            ms=9,
            color=tcolors[trait],
            markerfacecolor=tcolors[trait] if passed else "white",
            markeredgecolor=tcolors[trait],
            markeredgewidth=1.8,
            zorder=5,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("|Pearson r| at the pre-registered paper steering layer")
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Per-cell verdicts at the fixed layer: filled = beats both honest nulls (BH ≤ 0.05)", pad=12
    )
    ax.legend(loc="lower right", frameon=False)
    savefig_paper(fig, "fixed_layer_cell_verdicts", dir=OUT)
    plt.close(fig)


def fig_hall_layers() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = paper_palette_blog(4)
    i = 0
    for setting in ["monitoring_corrected", "monitoring_manyshot"]:
        d = json.loads((RES / f"hallucination_{setting}_honestnulls_v2.json").read_text())
        for stat in ["overall", "within"]:
            block = d["stage_maxlayer"][stat]
            r = np.abs(np.array(block["observed_r_per_layer"], dtype=float))
            ax.plot(
                np.arange(28),
                r,
                lw=1.8,
                color=colors[i],
                label=f"{SETTING_LABEL[setting]}, {STAT_LABEL[stat]}",
            )
            i += 1
    d = json.loads((RES / "hallucination_monitoring_corrected_honestnulls_v2.json").read_text())
    cap = np.array(d["stage_maxlayer"]["overall"]["per_layer_bands"]["neutral_cov"]["p97_5"])
    ax.plot(
        np.arange(28),
        cap,
        lw=1.4,
        ls="--",
        color="0.4",
        label="neutral-corpus null 97.5th pct (8-prompt pooled)",
    )
    ax.axvline(15, color="0.2", lw=1.2, ls=":", label="paper steering layer (index 15)")
    ax.set_xlabel("r_B layer index (0-indexed block outputs)")
    ax.set_ylabel("|Pearson r| (hallucination monitoring)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Hallucination monitoring predictivity peaks at layers 21-23,\n"
        "not at the paper steering layer",
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    savefig_paper(fig, "hallucination_layer_profile", dir=OUT)
    plt.close(fig)


if __name__ == "__main__":
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_fwer()
    fig_cell_verdicts()
    fig_hall_layers()
    print("done:", sorted(p.name for p in OUT.glob("*.png")))
