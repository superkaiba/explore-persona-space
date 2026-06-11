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
- ``expression_strata_v3``       — guard B after the binary-judge re-judge
  (5,760/5,760 refusal/EM rows labeled): per-family raw vs clean-text
  direction-mix association + per-cell expressed-bystander-question counts.
  (``expression_strata_v2``, the pre-re-judge fact-only version with the
  null-labels annotation, is preserved in git history.)
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


def fig_strata_v3(strata: dict) -> None:
    """Guard B after the re-judge: 1x2 panel.

    Left: per-family rank correlation of prior with the direction mix, raw vs
    re-computed against the clean-text shared direction (EM annotated as a
    no-op rebuild). Right: per-cell counts of behavior-expressing bystander
    questions (the rows the clean-text rebuild excludes), all 21 cells.
    """
    set_paper_style("blog")
    cross = strata["cross_family"]
    caveats = strata["meta"]["family_caveats"]
    fams = ["fact", "refusal", "em"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2), width_ratios=[1.0, 1.6])

    # Left: raw vs clean-text rho per family.
    width = 0.36
    for i, fam in enumerate(fams):
        raw = cross[fam]["rho_prior_cmf_raw"]
        clean = cross[fam]["clean_u_reestimate"]["rho_prior_cmf_clean_u"]
        degenerate = bool(caveats[fam].get("clean_u_rebuild_degenerate", False))
        ax1.bar(i - width / 2, raw, width, color="0.55", label="Raw" if i == 0 else None)
        ax1.bar(
            i + width / 2,
            clean,
            width,
            color=FAMILY_COLORS[fam],
            hatch="//" if degenerate else None,
            label="Clean-text rebuild" if i == 0 else None,
        )
    ax1.axhline(0, color="0.3", lw=0.8)
    ax1.set_xticks(range(len(fams)))
    ax1.set_xticklabels(
        ["Fact teachers\n(n = 9)", "Refusal sources\n(n = 6)", "EM sources\n(n = 6)"],
        fontsize=8,
    )
    ax1.set_ylabel("Rank correlation:\nprior vs direction mix")
    ax1.set_ylim(-1.0, 0.62)
    ax1.annotate(
        "EM rebuild excludes only\n0–12 of 460 rows per cell —\na no-op by construction",
        xy=(2.0, cross["em"]["clean_u_reestimate"]["rho_prior_cmf_clean_u"] - 0.06),
        xytext=(1.05, -0.95),
        fontsize=7.5,
        color="0.25",
        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.7},
    )
    ax1.legend(loc="upper left", fontsize=8)

    # Right: per-cell expressed-bystander-question counts.
    rows = []
    for fam in fams:
        fam_rows = [
            (cid, strata["per_cell"][cid], caveats[fam]["per_cell"][cid]["n_excluded_questions"])
            for cid in caveats[fam]["per_cell"]
        ]
        fam_rows.sort(key=lambda kv: (kv[1]["prior"], kv[1]["seed"]))
        rows.extend(fam_rows)
    xs = list(range(len(rows)))
    seen: set[str] = set()
    for x, (_, d, n_excl) in zip(xs, rows):
        fam = d["family"]
        ax2.bar(
            x,
            n_excl,
            color=FAMILY_COLORS[fam],
            label=FAMILY_LABELS[fam] if fam not in seen else None,
        )
        seen.add(fam)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [f"{d['source'].replace('_', ' ')} s{d['seed']}" for _, d, _ in rows],
        rotation=40,
        ha="right",
        fontsize=6.5,
    )
    ax2.set_ylabel("Bystander questions expressing\nthe behavior (of 460)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Guard B re-judged: cleaning expressed text flips fact, spares refusal, cannot move EM",
        x=0.02,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "expression_strata_v3", dir=FIG_DIR)
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
    fig_strata_v3(strata)
    fig_linkage_v2(results)
    print(f"wrote 6 figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
