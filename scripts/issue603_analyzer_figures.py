#!/usr/bin/env python
"""#603 analyzer-pass figures (clean-result embeds).

Regenerates the four figures that failed the analyzer's visual verification of
the round-1 ``issue603_figures.py`` output, and splits the 3x2 hero into the
two figures the clean-result body actually embeds:

- ``hero_fact_cmf_vs_norm``      — fact family (primary axis): CMF & norm vs prior,
  per-seed connecting lines, seed legend, ordering-test annotation.
- ``extensions_cmf_vs_norm``     — refusal/EM 2x2 grid with exact-permutation
  Spearman rho + p annotated per panel.
- ``reliability_vs_prior_v2``    — guard A visual, un-clipped labels.
- ``expression_strata_v2``       — guard B, FACT cells only; refusal/EM are
  explicitly annotated as unlabeled (judge-schema bug), never zero bars.
- ``behavioral_linkage_v2``      — validity panel, family-colored, un-clipped.

Reads eval_results/issue_603/{decomposition_results,expression_strata}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval_results" / "issue_603"
FIG_DIR = REPO / "figures" / "issue_603"

FAMILY_COLORS = {
    "fact": paper_palette_role("primary"),
    "refusal": paper_palette_role("baseline"),
    "em": paper_palette_role("accent"),
}
FAMILY_LABELS = {"fact": "Fact teachers", "refusal": "Refusal sources", "em": "EM sources"}
SEED_MARKERS = {42: "o", 137: "s", 256: "^"}
PRIMARY_READ = "primary_l14_mean_resp"


def _cells(results: dict, family: str) -> list[dict]:
    return [d for d in results["per_cell"].values() if d["family"] == family]


def fig_hero_fact(results: dict, *, dis: bool = False) -> None:
    """Fact-only 1x2 hero. ``dis=True`` swaps the left panel to the
    noise-corrected (disattenuated) common-mode fraction, keeping the layout
    panel-identical to the raw figure for direct visual comparison."""
    set_paper_style("blog")
    cells = _cells(results, "fact")
    teachers = [
        "marine_biologist",
        "courthouse_architecture_historian",
        "wooden_furniture_carpenter",
    ]
    short = {
        "marine_biologist": "marine\nbiologist",
        "courthouse_architecture_historian": "courthouse\nhistorian",
        "wooden_furniture_carpenter": "furniture\ncarpenter",
    }
    cmf_label = (
        "Noise-corrected common-mode fraction" if dis else "Common-mode fraction of the write"
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, dv, ylab in (
        (axes[0], "cmf", cmf_label),
        (axes[1], "norm", "Write norm (residual-stream units)"),
    ):
        for seed in (42, 137, 256):
            xs, ys = [], []
            for t in teachers:
                d = next(c for c in cells if c["source"] == t and c["seed"] == seed)
                xs.append(d["prior"])
                if dv == "cmf" and dis:
                    ys.append(d["cmf_disattenuated"])
                else:
                    ys.append(d["reads"][PRIMARY_READ][dv])
            ax.plot(
                xs,
                ys,
                marker=SEED_MARKERS[seed],
                markeredgewidth=0.0,
                lw=1.0,
                alpha=0.85,
                color=FAMILY_COLORS["fact"],
                label=f"seed {seed}",
            )
        ax.set_xlabel("Teacher prior (log P per token, base model)")
        ax.set_ylabel(ylab)
    # teacher name ticks on top of the left panel
    pri = sorted({c["prior"] for c in cells})
    for ax in axes:
        for p, t in zip(pri, teachers):
            ax.annotate(
                short[t],
                xy=(p, 1.0),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
            )
    axes[0].annotate(
        (
            "noise-corrected predicted ordering in 1/3 seeds\n(same seed as the raw read)"
            if dis
            else "predicted ordering in 1/3 seeds\n(exact one-sided p = 0.074 needs 2/3)"
        ),
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize=8,
        color="0.25",
    )
    axes[1].annotate(
        "predicted ordering in 0/3 seeds",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize=8,
        color="0.25",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        (
            "Noise-corrected fact axis: disattenuation moves every value by less than 0.04"
            if dis
            else "Primary fact axis: neither the write's direction mix nor its size tracks"
            " teacher prior"
        ),
        x=0.02,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = "hero_fact_cmf_vs_norm_disattenuated" if dis else "hero_fact_cmf_vs_norm"
    savefig_paper(fig, stem, dir=FIG_DIR)
    plt.close(fig)


def fig_extensions(results: dict) -> None:
    set_paper_style("blog")
    stats = results["stats"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for row, family in enumerate(("refusal", "em")):
        cells = _cells(results, family)
        st = stats[family]
        for col, dv, ylab in ((0, "cmf", "Common-mode fraction"), (1, "norm", "Write norm")):
            ax = axes[row][col]
            xs = [c["prior"] for c in cells]
            ys = [c["reads"][PRIMARY_READ][dv] for c in cells]
            ax.scatter(xs, ys, color=FAMILY_COLORS[family], s=42)
            for c, x, y in zip(cells, xs, ys):
                ax.annotate(
                    c["source"].replace("_", " "),
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.5,
                    color="0.35",
                )
            key = "spearman_cmf" if dv == "cmf" else "spearman_norm"
            rho = st[key]["rho"]
            p = st[key]["p_one_sided_negative"]
            ax.annotate(
                f"Spearman rho = {rho:+.2f}\nexact one-sided p = {p:.3f} (n = 6)",
                xy=(0.03, 0.04),
                xycoords="axes fraction",
                fontsize=8,
                color="0.25",
            )
            ax.set_ylabel(f"{FAMILY_LABELS[family]}\n{ylab}")
            ax.set_xlabel("Source prior (log P per token, base model)")
            ymin, ymax = min(ys), max(ys)
            pad = 0.18 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + 2.2 * pad)
    fig.suptitle(
        "Refusal / EM extensions: direction mix leans negative on prior, norm does not",
        x=0.02,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "extensions_cmf_vs_norm", dir=FIG_DIR)
    plt.close(fig)


def fig_reliability_v2(results: dict) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for family in ("fact", "refusal", "em"):
        cells = _cells(results, family)
        ax.scatter(
            [c["prior"] for c in cells],
            [c["reliability"]["r_a_source_dir"]["r_random_mean"] for c in cells],
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
            s=40,
        )
    ax.axhline(0.3, ls="--", color="0.5", lw=1.0)
    ax.annotate(
        "pre-registered reliability floor (0.3)",
        xy=(0.02, 0.33),
        xycoords=("axes fraction", "data"),
        fontsize=8,
        color="0.4",
    )
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Source prior (log P per token, base model)")
    ax.set_ylabel("Split-half reliability of the\nsource write direction")
    ax.set_title(
        "Guard A: every write direction is estimated at ceiling reliability",
        loc="left",
        fontweight="semibold",
    )
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "reliability_vs_prior_v2", dir=FIG_DIR)
    plt.close(fig)


def fig_strata_v2(strata: dict) -> None:
    set_paper_style("blog")
    rows = [(cid, d) for cid, d in strata["per_cell"].items() if d["family"] == "fact"]
    rows.sort(key=lambda kv: (kv[1]["prior"], kv[1]["seed"]))
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    xs = range(len(rows))
    ax.bar(
        xs,
        [d["cmf_expressed"] for _, d in rows],
        color=FAMILY_COLORS["fact"],
        label="Fact-asserting questions (n per cell in tick label)",
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [
            f"{d['source'].replace('_', ' ')}\nseed {d['seed']} ({d['n_expressed']}/20 assert)"
            for _, d in rows
        ],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Common-mode fraction\n(assertion-present questions)")
    ax.set_title(
        "Guard B, fact family: expression is near-constant (179/180 questions assert the fact)",
        loc="left",
        fontweight="semibold",
    )
    ax.set_ylim(0, 1.24)
    ax.annotate(
        "Refusal/EM cells not shown: all 5,760 binary-judge labels returned null\n"
        "(verdict-schema bug) — stratified read N/A for those families, not zero.",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
        color="0.25",
    )
    fig.tight_layout()
    savefig_paper(fig, "expression_strata_v2", dir=FIG_DIR)
    plt.close(fig)


def fig_linkage_v2(results: dict) -> None:
    set_paper_style("blog")
    rows = [
        (cid, d)
        for cid, d in results["per_cell"].items()
        if d.get("behavioral_linkage") is not None
    ]
    fam_order = {"fact": 0, "refusal": 1, "em": 2}
    rows.sort(key=lambda kv: (fam_order[kv[1]["family"]], kv[1]["prior"], kv[1]["seed"]))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    xs = list(range(len(rows)))
    seen: set[str] = set()
    for x, (_, d) in zip(xs, rows):
        fam = d["family"]
        ax.bar(
            x,
            d["behavioral_linkage"]["spearman_rho_proj_vs_leak"],
            color=FAMILY_COLORS[fam],
            label=FAMILY_LABELS[fam] if fam not in seen else None,
        )
        seen.add(fam)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{d['source'].replace('_', ' ')} s{d['seed']}" for _, d in rows],
        rotation=40,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Spearman rho\n(projection vs leak, 23 bystanders)")
    ax.set_title(
        "Validity panel: shifts track leakage on the fact axis, anti-track it in refusal/EM",
        loc="left",
        fontweight="semibold",
    )
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "behavioral_linkage_v2", dir=FIG_DIR)
    plt.close(fig)


def main() -> int:
    results = json.loads((EVAL_DIR / "decomposition_results.json").read_text())
    strata = json.loads((EVAL_DIR / "expression_strata.json").read_text())
    fig_hero_fact(results)
    fig_hero_fact(results, dis=True)
    fig_extensions(results)
    fig_reliability_v2(results)
    fig_strata_v2(strata)
    fig_linkage_v2(results)
    print(f"wrote 6 figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
