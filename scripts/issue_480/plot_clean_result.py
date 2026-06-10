"""Issue 480 clean-result figures.

Regenerates the 4 hero / supporting figures the analyzer needs for the
clean-result body, in `paper_plots.set_paper_style("blog")` register:

  1. h1_hero_marker_vs_sycophancy.png — scatter, 138 cells, color by source
  2. h1_saturation_diagnostic.png    — why the headline collapses: software_engineer runaway cells
  3. h1_source_fe_residualized.png   — same axes, source-FE residualized
  4. h2_per_source_cosine_gradient.png — 2x3 panel grid, marker_delta vs cosine_l20
  5. h2_paired_rho_vs_411.png        — paired bar chart, marker rho vs sycophancy rho per source
  6. marker_delta_distribution.png   — per-source marker_delta histogram
     (shows software_engineer outlier)

All saved to figures/issue_480/ via savefig_paper (PNG + PDF + .meta.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)

# Source -> plain-English label
SOURCE_LABELS = {
    "assistant": "assistant",
    "comedian": "comedian",
    "kindergarten_teacher": "kindergarten teacher",
    "qwen_default": "Qwen default",
    "software_engineer": "software engineer",
    "villain": "villain",
}
SOURCE_ORDER = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]


def load_matrix(eval_dir: Path) -> list[dict]:
    return json.load((eval_dir / "marker_delta_matrix.json").open())["rows"]


def load_h1_h2(eval_dir: Path) -> dict:
    return json.load((eval_dir / "h1_h2_analysis.json").open())


def figure_1_hero(matrix: list[dict], h1: dict, out_root: Path) -> Path:
    """Headline scatter: marker_delta vs sycophancy_delta, 138 cells, color by source."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    # Fixed 6-color blog palette, one per source
    blog_cols = ["#2E5C8A", "#D97F2E", "#5BA85D", "#B23B3B", "#8B5BAA", "#7A5230"]
    for i, src in enumerate(SOURCE_ORDER):
        rows = [r for r in matrix if r["source"] == src]
        x = [r["sycophancy_delta"] for r in rows]
        y = [r["marker_delta"] for r in rows]
        ax.scatter(
            x,
            y,
            label=SOURCE_LABELS[src],
            s=42,
            alpha=0.75,
            color=blog_cols[i],
            edgecolors="white",
            linewidths=0.6,
        )
    ax.axhline(0, color="#888888", linewidth=0.5, alpha=0.6)
    ax.axvline(0, color="#888888", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Sycophancy leakage (rate trained - base, from prior sycophancy run)")
    ax.set_ylabel("Marker leakage (log p(marker) trained - base, nats)")
    rho = h1["h1"]["cell_spearman_source_fe"]["rho"]
    ci_lo = h1["h1"]["cell_spearman_source_fe"]["ci_lo"]
    ci_hi = h1["h1"]["cell_spearman_source_fe"]["ci_hi"]
    ax.set_title(
        "Marker leakage doesn't track sycophancy leakage on matched (source, bystander) cells",
        fontsize=11,
        loc="left",
        pad=22,
        weight="semibold",
    )
    ax.text(
        0.0,
        1.02,
        f"Source-FE Spearman rho = {rho:+.2f} (95% CI {ci_lo:+.2f}, {ci_hi:+.2f}), "
        "n = 138 cells, seed 42",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2, columnspacing=0.7)
    fig.tight_layout()
    out = savefig_paper(fig, "issue_480/hero_marker_vs_sycophancy", dir=out_root)
    plt.close(fig)
    return out


def figure_2_saturation(matrix: list[dict], out_root: Path) -> Path:
    """Diagnostic: software_engineer's runaway cells collapse marker-delta to 0."""
    set_paper_style("blog")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2))

    # Left: emission rate vs marker_delta, by source
    blog_cols = ["#2E5C8A", "#D97F2E", "#5BA85D", "#B23B3B", "#8B5BAA", "#7A5230"]
    for i, src in enumerate(SOURCE_ORDER):
        rows = [r for r in matrix if r["source"] == src]
        x = [r["emission_rate"] for r in rows]
        y = [r["marker_delta"] for r in rows]
        ax1.scatter(
            x,
            y,
            label=SOURCE_LABELS[src],
            s=42,
            alpha=0.75,
            color=blog_cols[i],
            edgecolors="white",
            linewidths=0.6,
        )
    ax1.set_xlabel("Marker emission rate (fraction of probes where model writes a marker)")
    ax1.set_ylabel("Marker leakage (log p(marker) trained - base, nats)")
    ax1.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, columnspacing=0.7)
    ax1.set_title(
        "Runaway-emission cells collapse marker leakage to 0",
        fontsize=10,
        loc="left",
        pad=8,
        weight="semibold",
    )

    # Right: per-source bar: # of runaway cells (emission >= 0.5)
    n_runaway = {
        src: sum(1 for r in matrix if r["source"] == src and r["emission_rate"] >= 0.5)
        for src in SOURCE_ORDER
    }
    labels = [SOURCE_LABELS[s] for s in SOURCE_ORDER]
    values = [n_runaway[s] for s in SOURCE_ORDER]
    bar_colors = [blog_cols[i] for i in range(len(SOURCE_ORDER))]
    ax2.barh(
        range(len(SOURCE_ORDER)),
        values,
        color=bar_colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    ax2.set_yticks(range(len(SOURCE_ORDER)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Number of bystander cells with emission rate >=0.5 (out of 23)")
    ax2.set_xlim(0, 14)
    for i, v in enumerate(values):
        ax2.text(v + 0.2, i, f"{v}/23", va="center", fontsize=9, color="#333333")
    ax2.set_title(
        "The runaway is almost all on software engineer",
        fontsize=10,
        loc="left",
        pad=8,
        weight="semibold",
    )
    fig.suptitle(
        "The marker-leakage DV breaks under saturated emission",
        fontsize=12,
        x=0.02,
        ha="left",
        weight="bold",
        y=0.99,
    )
    fig.text(
        0.02,
        0.945,
        "When the model writes the marker to the 2048-token cap, both log p(trained) "
        "and log p(base) approach 0, so the difference collapses",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = savefig_paper(fig, "issue_480/saturation_diagnostic", dir=out_root)
    plt.close(fig)
    return out


def figure_3_source_fe(matrix: list[dict], h1: dict, out_root: Path) -> Path:
    """Source-FE residualized scatter."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    blog_cols = ["#2E5C8A", "#D97F2E", "#5BA85D", "#B23B3B", "#8B5BAA", "#7A5230"]
    # Compute source-mean residuals
    by_src_marker = {
        s: [r["marker_delta"] for r in matrix if r["source"] == s] for s in SOURCE_ORDER
    }
    by_src_syco = {
        s: [r["sycophancy_delta"] for r in matrix if r["source"] == s] for s in SOURCE_ORDER
    }
    src_marker_mean = {s: np.mean(v) for s, v in by_src_marker.items()}
    src_syco_mean = {s: np.mean(v) for s, v in by_src_syco.items()}

    for i, src in enumerate(SOURCE_ORDER):
        rows = [r for r in matrix if r["source"] == src]
        x = [r["sycophancy_delta"] - src_syco_mean[src] for r in rows]
        y = [r["marker_delta"] - src_marker_mean[src] for r in rows]
        ax.scatter(
            x,
            y,
            label=SOURCE_LABELS[src],
            s=42,
            alpha=0.75,
            color=blog_cols[i],
            edgecolors="white",
            linewidths=0.6,
        )
    ax.axhline(0, color="#888888", linewidth=0.5, alpha=0.6)
    ax.axvline(0, color="#888888", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Sycophancy leakage, residualized on source")
    ax.set_ylabel("Marker leakage (nats), residualized on source")
    rho = h1["h1"]["cell_spearman_source_fe"]["rho"]
    rl = h1["h1"]["cell_spearman_source_fe_base_rate_resp_len_partial"]["rho"]
    ax.set_title(
        "After removing each source's mean, the within-source cloud is nearly flat",
        fontsize=11,
        loc="left",
        pad=22,
        weight="semibold",
    )
    ax.text(
        0.0,
        1.02,
        f"Source-FE rho = {rho:+.2f}; partialling base rate + response length "
        f"lifts it to {rl:+.2f}",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, columnspacing=0.7)
    fig.tight_layout()
    out = savefig_paper(fig, "issue_480/source_fe_residualized", dir=out_root)
    plt.close(fig)
    return out


def figure_4_h2_panel(matrix: list[dict], h1_h2: dict, out_root: Path) -> Path:
    """Per-source cosine -> marker_delta scatter, 2x3 grid."""
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    h2 = h1_h2["h2_within_source"]

    blog_cols = ["#2E5C8A", "#D97F2E", "#5BA85D", "#B23B3B", "#8B5BAA", "#7A5230"]

    for i, src in enumerate(SOURCE_ORDER):
        ax = axes[i // 3, i % 3]
        rows = [r for r in matrix if r["source"] == src]
        x = [r["cosine_l20_baseline"] for r in rows]
        y = [r["marker_delta"] for r in rows]
        ax.scatter(x, y, color=blog_cols[i], s=42, alpha=0.85, edgecolors="white", linewidths=0.6)
        rho = h2[src]["rho"]
        p = h2[src]["perm_p"]
        ax.set_title(
            f"{SOURCE_LABELS[src]}\nrho = {rho:+.2f}, perm p = {p:.3f}, n = 23",
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("layer-20 cosine, source to bystander")
        if i % 3 == 0:
            ax.set_ylabel("marker leakage (nats)")
    fig.suptitle(
        "Per-source cosine to bystander vs marker leakage on the same panel",
        fontsize=12,
        x=0.02,
        ha="left",
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.02,
        0.955,
        "Comedian: cleanly significant gradient (rho = +0.71). Villain: nominal "
        "(rho = +0.48, perm p = 0.024 but CI crosses zero). Other four: weak or null.",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = savefig_paper(fig, "issue_480/per_source_cosine_gradient", dir=out_root)
    plt.close(fig)
    return out


def figure_5_paired_rho(h1_h2: dict, out_root: Path) -> Path:
    """Paired bar: per-source marker rho vs sycophancy rho."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    sources = h1_h2["h2_paired_delta_rho"]["sources"]
    rho_m = h1_h2["h2_paired_delta_rho"]["rho_marker_per_source"]
    rho_s = h1_h2["h2_paired_delta_rho"]["rho_syco_per_source"]

    x = np.arange(len(sources))
    w = 0.36
    ax.bar(
        x - w / 2,
        rho_m,
        w,
        label="Marker (this experiment)",
        color="#2E5C8A",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + w / 2,
        rho_s,
        w,
        label="Sycophancy (prior run, frozen)",
        color="#D97F2E",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS[s] for s in sources], rotation=25, ha="right")
    ax.set_ylabel("Within-source Spearman rho\n(cosine -> behavior leakage)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    mean_dr = h1_h2["h2_paired_delta_rho"]["mean_delta_rho"]
    ci_lo = h1_h2["h2_paired_delta_rho"]["paired_bootstrap_ci_lo"]
    ci_hi = h1_h2["h2_paired_delta_rho"]["paired_bootstrap_ci_hi"]
    ax.set_title(
        "4 of 6 sources agree in direction; only comedian is cleanly supported on the marker side",
        fontsize=10.5,
        loc="left",
        pad=24,
        weight="semibold",
    )
    ax.text(
        0.0,
        1.03,
        f"Paired mean(marker rho - syco rho) = {mean_dr:+.2f} "
        f"(95% CI {ci_lo:+.2f}, {ci_hi:+.2f}), n = 6 sources",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(pad=1.2)
    out = savefig_paper(fig, "issue_480/paired_rho_vs_411", dir=out_root)
    plt.close(fig)
    return out


def figure_6_marker_dist(matrix: list[dict], out_root: Path) -> Path:
    """Marker_delta distribution per source — shows the software_engineer cluster at 0."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    blog_cols = ["#2E5C8A", "#D97F2E", "#5BA85D", "#B23B3B", "#8B5BAA", "#7A5230"]
    bins = np.linspace(-1, 26, 28)
    for i, src in enumerate(SOURCE_ORDER):
        vals = [r["marker_delta"] for r in matrix if r["source"] == src]
        ax.hist(
            vals,
            bins=bins,
            label=SOURCE_LABELS[src],
            color=blog_cols[i],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("Marker leakage (log p(marker) trained - base, nats)")
    ax.set_ylabel("count of bystander cells")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, columnspacing=0.7)
    ax.set_title(
        "The software-engineer pile at 0 nats is the runaway pathology, not absence of leakage",
        fontsize=11,
        loc="left",
        pad=22,
        weight="semibold",
    )
    ax.text(
        0.0,
        1.02,
        "Five other sources span 9-25 nats; software engineer has 14 of 23 bystander cells "
        "pinned at the floor (emission rate >=0.5)",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    out = savefig_paper(fig, "issue_480/marker_delta_distribution_v2", dir=out_root)
    plt.close(fig)
    return out


def main() -> None:
    eval_dir = Path("eval_results/issue_480")
    out_root = Path("figures")

    matrix = load_matrix(eval_dir)
    h1_h2 = load_h1_h2(eval_dir)

    paths = []
    paths.append(figure_1_hero(matrix, h1_h2, out_root))
    paths.append(figure_2_saturation(matrix, out_root))
    paths.append(figure_3_source_fe(matrix, h1_h2, out_root))
    paths.append(figure_4_h2_panel(matrix, h1_h2, out_root))
    paths.append(figure_5_paired_rho(h1_h2, out_root))
    paths.append(figure_6_marker_dist(matrix, out_root))

    for p in paths:
        print(f"WROTE {p}")


if __name__ == "__main__":
    main()
