"""Round-2 figures for #1415 (disjoint-baseline recount, interp-critique r1).

Regenerates the three figures whose data changed under the disjoint-baseline
recount: hero_geometric_vs_behavioral (left panel now selection-matched at
steer-L20 max-over-read, shared vs disjoint, CJK annotation on the
all-position bars), h1_per_pair_scatter (per-pair shared -> disjoint), and
null_band_vs_observed (shared + disjoint histograms vs the bands).

Inputs: eval_results/issue_1415/{disjoint_baseline_recount,
disjoint_recount_figdata,null_bands,geometric_projections}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_results" / "issue_1415"
FIGDIR = ROOT / "figures" / "issue_1415"
ARMS = ["prefix", "context"]

rc = json.load(open(EVAL / "disjoint_baseline_recount.json"))
fd = json.load(open(EVAL / "disjoint_recount_figdata.json"))
nb = json.load(open(EVAL / "null_bands.json"))
gp = json.load(open(EVAL / "geometric_projections.json"))
POOLED_NULL = nb["bands"]["random_delta"]["pooled_across_pairs"]["prefix"]["p97.5"]
SHUF = {a: nb["bands"]["shuffled_pair"]["pooled_across_pairs"][a]["p97.5"] for a in ARMS}

set_paper_style("blog")
C = paper_palette_blog(6)


def mean_se(vals):
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


# ---------------- hero ----------------
def hero():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    ax = axes[0]
    cats = [
        ("Last-token Δ,\nprefix", fd["l20_delta"]["prefix"], False),
        ("Last-token Δ,\ncontext", fd["l20_delta"]["context"], False),
        ("All-position Δ,\nprefix", fd["allpos"]["prefix"], True),
        ("All-position Δ,\ncontext", fd["allpos"]["context"], True),
        ("Persona vector\nr_B evil", fd["rb"]["evil"], False),
        ("Persona vector\nr_B halluc.", fd["rb"]["hallucination"], False),
        ("Persona vector\nr_B syco.", fd["rb"]["sycophancy"], False),
    ]
    x = np.arange(len(cats))
    w = 0.38
    for i, (label, rows, cjk) in enumerate(cats):
        ms, ss = mean_se([r["shared"] for r in rows.values()])
        md, sd = mean_se([r["disj"] for r in rows.values()])
        ax.bar(
            i - w / 2,
            ms,
            w,
            yerr=ss,
            color=C[0],
            alpha=0.45,
            label="shared baseline (round-1 statistic)" if i == 0 else None,
        )
        ax.bar(
            i + w / 2,
            md,
            w,
            yerr=sd,
            color=C[0],
            label="disjoint baseline halves (corrected)" if i == 0 else None,
        )
        ax.text(i - w / 2, ms + 0.02, f"{ms:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, md + 0.02, f"{md:.2f}", ha="center", fontsize=9)
        if cjk:
            ax.text(i, max(ms, md) + 0.09, "96–98%\nCJK text", ha="center", fontsize=9, color=C[3])
    ax.axhline(
        POOLED_NULL,
        ls=":",
        color=C[3],
        lw=1.5,
        label=f"random-direction null p97.5 ({POOLED_NULL:.3f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats], fontsize=9)
    ax.set_ylabel("Answer-shift alignment with target\n(cosine, steer L20, max over 7 read layers)")
    ax.set_title(
        "Geometric: answer state moves toward the target\n(all bars share one selection rule)",
        fontsize=12,
    )
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    b = fd["behavioral_a4"]
    cats2 = [
        ("Baseline\n(no steering)", b["baseline"], False),
        ("Last-token Δ,\nprefix", b["steered_prefix_a4"], False),
        ("Last-token Δ,\ncontext", b["steered_context_a4"], False),
        ("All-position Δ,\nprefix", b["allpos_prefix"], True),
        ("All-position Δ,\ncontext", b["allpos_context"], True),
        ("Persona vector\nr_B evil", b["rb_evil"], False),
        ("Context-swap\nceiling", b["ceiling"], False),
    ]
    x2 = np.arange(len(cats2))
    for i, (label, st, cjk) in enumerate(cats2):
        ax.bar(i, st["mean"], 0.62, yerr=st["se"], color=C[1] if i < len(cats2) - 1 else C[2])
        ax.text(i, st["mean"] + st["se"] + 0.6, f"{st['mean']:.1f}", ha="center", fontsize=9)
        if cjk:
            ax.text(
                i,
                st["mean"] + st["se"] + 3.2,
                "96–98%\nCJK text",
                ha="center",
                fontsize=9,
                color=C[3],
            )
    ax.set_xticks(x2)
    ax.set_xticklabels([c[0] for c in cats2], fontsize=9)
    ax.set_ylabel("Graded judge score (0–100), α=4\nmean of per-pair means ± SE, 28 pairs")
    ax.set_title("Behavioral: the judge barely sees it", fontsize=12)

    fig.suptitle(
        "Patching Δ = V_c(c′) − V_c(c) moves answer-state geometry but not behavior "
        "(28 pairs, N=10 draws; shared- vs disjoint-baseline)",
        fontsize=13,
    )
    savefig_paper(fig, "hero_geometric_vs_behavioral", dir=FIGDIR)
    plt.close(fig)


# ---------------- per-pair scatter ----------------
def per_pair():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)
    for ax, arm in zip(axes, ARMS):
        rows = rc["h1"][arm]["per_pair"]
        ptype = {p: gp["h1"][arm][p]["pair_type"] for p in rows}
        order = sorted(rows, key=lambda p: rows[p]["disj_max"])
        for i, p in enumerate(order):
            r = rows[p]
            col = C[0] if ptype[p] == "matched" else C[1]
            ax.plot([i, i], [r["disj_max"], r["shared_max"]], color=col, lw=1.0, alpha=0.6)
            ax.scatter(
                [i],
                [r["shared_max"]],
                facecolors="none",
                edgecolors=col,
                linewidths=1.4,
                s=38,
                zorder=3,
            )
            ax.scatter([i], [r["disj_max"]], color=col, s=38, zorder=3)
            ax.plot([i - 0.35, i + 0.35], [r["band_p975"]] * 2, color="gray", lw=1.2)
            if p == "m685_07_medical_doctor":
                tx = i - 7.5 if i > 10 else i + 1.5
                ax.annotate(
                    "medical_doctor:\nnoise target\n(split-half 0.049)",
                    (i, r["disj_max"]),
                    xytext=(tx, min(r["disj_max"] + 0.28, 0.62)),
                    fontsize=8.5,
                    color=C[3],
                    arrowprops=dict(arrowstyle="-", color=C[3], lw=0.8),
                )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [p.replace("m685_", "").replace("m779_", "779_").replace("cross_", "x") for p in order],
            rotation=90,
            fontsize=7,
        )
        ax.set_title(f"{arm} arm", fontsize=12)
        ax.set_xlabel("context pair (sorted by corrected alignment)")
    axes[0].set_ylabel("Answer-shift alignment with target\n(cosine, max over 7 steer layers, α=4)")
    h = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor="k",
            label="shared baseline (round-1)",
        ),
        plt.Line2D([], [], marker="o", ls="", color="k", label="disjoint baseline (corrected)"),
        plt.Line2D([], [], color=C[0], lw=2, label="matched-query pair"),
        plt.Line2D([], [], color=C[1], lw=2, label="cross-query pair"),
        plt.Line2D([], [], color="gray", lw=1.2, label="per-pair random-direction null p97.5"),
    ]
    axes[1].legend(handles=h, fontsize=8.5, loc="upper left")
    fig.suptitle(
        "Per-pair answer-shift alignment: every pair still clears its null after the "
        "disjoint-baseline correction, at reduced magnitude",
        fontsize=13,
    )
    savefig_paper(fig, "h1_per_pair_scatter", dir=FIGDIR)
    plt.close(fig)


# ---------------- null band vs observed ----------------
def null_band():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True, constrained_layout=True)
    bins = np.linspace(-0.1, 1.0, 34)
    for ax, arm in zip(axes, ARMS):
        rows = rc["h1"][arm]["per_pair"]
        shared = [r["shared_max"] for r in rows.values()]
        disj = [r["disj_max"] for r in rows.values()]
        ax.hist(disj, bins=bins, color=C[0], alpha=0.85, label="disjoint baseline (corrected)")
        ax.hist(
            shared, bins=bins, histtype="step", color=C[1], lw=2, label="shared baseline (round-1)"
        )
        ax.axvline(
            POOLED_NULL,
            ls=":",
            color=C[3],
            lw=1.6,
            label=f"random-Δ null p97.5 ({POOLED_NULL:.3f})",
        )
        ax.axvline(
            SHUF[arm],
            ls="--",
            color=C[2],
            lw=1.6,
            label=f"shuffled-pair null p97.5 ({SHUF[arm]:.3f})",
        )
        ax.axvline(1.0, color="gray", lw=1.6, label="context-swap ceiling (1.0)")
        ax.set_title(f"{arm} arm", fontsize=12)
        ax.set_xlabel("per-pair answer-shift alignment (cosine, max over 7 steer layers)")
        ax.legend(fontsize=8.5)
    axes[0].set_ylabel("number of pairs (of 28)")
    fig.suptitle(
        "Observed per-pair alignment vs its nulls, before and after removing the "
        "shared-baseline noise term",
        fontsize=13,
    )
    savefig_paper(fig, "null_band_vs_observed", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    hero()
    per_pair()
    null_band()
    print("done:", sorted(p.name for p in FIGDIR.glob("*.png")))
