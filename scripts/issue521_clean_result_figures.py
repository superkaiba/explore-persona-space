"""Clean-result figures for #521 — cross-arm rank-one geometry test.

Four figures, one per finding:
  1. hero_direction_consistency — per-persona cos-to-top-direction dots +
     sigma1/sum(sigma) bars with nulls, per (arm, seed), same-trajectory variant.
  2. marker_structure — marker arm per-persona cos-to-U1 sorted dot plot
     (source persona highlighted) + steering-vector cosine vs random floor.
  3. robustness — variant sensitivity, within/cross-arm U1 cosines,
     magnitude-vs-similarity Spearman rho.
  4. em_gate_surfaces — EM-rate per probe surface (trivia vs canonical Betley).

Data: eval_results/issue_521/svd/*.json, em_rate_gate*/summary.json,
direction_consistency.json. All values read from raw JSONs (no aggregation
beyond what is plotted).
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
SVD = ROOT / "eval_results/issue_521/svd"
SEEDS = [42, 137, 256]

set_paper_style("blog")
mpl.rcParams["figure.constrained_layout.use"] = False

C_EM = paper_palette_role("primary")
C_MARKER = paper_palette_role("baseline")
C_ACCENT = paper_palette_role("accent")
C_NEUTRAL = paper_palette_role("neutral")


def load(variant: str, arm: str, seed: int) -> dict:
    return json.loads((SVD / f"{variant}_{arm}_seed{seed}.json").read_text())


# ---------------------------------------------------------------- figure 1
def fig1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.98, wspace=0.25)

    cells = [("marker", s) for s in SEEDS] + [("em", s) for s in SEEDS]
    rng = np.random.default_rng(0)

    ax = axes[0]
    for x, (arm, seed) in enumerate(cells):
        d = load("same", arm, seed)
        cos = np.array(d["cos_to_U1"])
        personas = d["persona_order"]
        color = C_MARKER if arm == "marker" else C_EM
        jit = rng.uniform(-0.13, 0.13, size=len(cos))
        for j, p in enumerate(personas):
            if p == "medical_doctor":
                ax.scatter(
                    x + jit[j],
                    cos[j],
                    marker="D",
                    s=46,
                    facecolor="white",
                    edgecolor=C_ACCENT,
                    linewidth=1.6,
                    zorder=5,
                )
            else:
                ax.scatter(x + jit[j], cos[j], s=26, color=color, alpha=0.75, zorder=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(
        [
            "Marker\nseed 42",
            "Marker\nseed 137",
            "Marker\nseed 256",
            "EM\nseed 42",
            "EM\nseed 137",
            "EM\nseed 256",
        ]
    )
    ax.set_ylabel("cosine(per-persona shift, top direction)")
    ax.set_ylim(0, 1.05)
    ax.scatter(
        [],
        [],
        marker="D",
        s=46,
        facecolor="white",
        edgecolor=C_ACCENT,
        linewidth=1.6,
        label="medical doctor (trained source persona)",
    )
    ax.scatter([], [], s=26, color=C_EM, label="EM arm personas")
    ax.scatter([], [], s=26, color=C_MARKER, label="marker arm personas")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title(
        "(a) How aligned is each persona context's shift\nwith the cell's top direction?",
        loc="left",
        fontsize=10.5,
    )

    ax = axes[1]
    xs = np.arange(6)
    vals = []
    shuf = []
    flip = []
    for arm, seed in cells:
        d = load("same", arm, seed)
        vals.append(d["s_top1_frac"])
        shuf.append(d["row_shuffle_p95"])
        flip.append(d["sign_flip_p95"])
    colors = [C_MARKER] * 3 + [C_EM] * 3
    ax.bar(xs, vals, width=0.55, color=colors)
    ax.scatter(
        xs,
        shuf,
        marker="_",
        s=320,
        color="#1A1A1A",
        linewidth=1.8,
        label="row-shuffle null (95th pct)",
        zorder=5,
    )
    ax.scatter(
        xs,
        flip,
        marker="_",
        s=320,
        color=C_ACCENT,
        linewidth=1.8,
        label="sign-flip null (95th pct)",
        zorder=5,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            "Marker\nseed 42",
            "Marker\nseed 137",
            "Marker\nseed 256",
            "EM\nseed 42",
            "EM\nseed 137",
            "EM\nseed 256",
        ]
    )
    ax.set_ylabel("top singular value share of spectrum")
    ax.set_ylim(0, 0.72)
    ax.set_title(
        "(b) Share of the shift spectrum carried by\nthe top direction, with both nulls",
        loc="left",
        fontsize=10.5,
    )

    fig.text(
        0.08,
        0.955,
        "EM shifts every persona context the same way; the marker implant doesn't",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.08,
        0.905,
        "Per-context activation shift (trained - base, layer 14, last response token), "
        "14 personas x 20 held-out questions per cell; same-trajectory variant",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    savefig_paper(fig, "issue_521/hero_direction_consistency", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [2.4, 1.0]})
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.17, right=0.97, wspace=0.30)

    d0 = load("same", "marker", 42)
    personas = d0["persona_order"]
    mat = np.array([load("same", "marker", s)["cos_to_U1"] for s in SEEDS])
    order = np.argsort(mat.mean(axis=0))
    trained_negs = {"comedian", "police_officer", "software_engineer", "assistant"}

    ax = axes[0]
    for row, idx in enumerate(order):
        name = personas[idx]
        ys = [row] * 3
        if name == "medical_doctor":
            ax.scatter(
                mat[:, idx],
                ys,
                marker="D",
                s=44,
                facecolor="white",
                edgecolor=C_ACCENT,
                linewidth=1.6,
                zorder=5,
            )
        elif name in trained_negs:
            ax.scatter(mat[:, idx], ys, s=30, color=C_MARKER, zorder=3)
        else:
            ax.scatter(mat[:, idx], ys, s=30, color=C_NEUTRAL, alpha=0.8, zorder=3)
    labels = []
    for idx in order:
        n = personas[idx].replace("_", " ")
        if personas[idx] == "medical_doctor":
            n += "  (source)"
        elif personas[idx] in trained_negs:
            n += "  (trained negative)"
        labels.append(n)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("cosine(persona shift, top direction) — 3 dots = 3 seeds")
    ax.set_xlim(0, 1.05)
    ax.set_title("(a) Marker arm, per persona context", loc="left", fontsize=10.5)

    ax = axes[1]
    vsteer = [load("same", "marker", s)["cos_U1_vsteer"] for s in SEEDS]
    ax.bar(range(3), vsteer, width=0.55, color=C_MARKER)
    floor = 0.033
    ax.axhline(floor, color="#1A1A1A", linewidth=1.0, linestyle="--")
    ax.axhline(-floor, color="#1A1A1A", linewidth=1.0, linestyle="--")
    ax.annotate(
        "random-vector floor (95th pct of |cos|,\n10,000 random pairs in 3,584 dims)",
        (0.02, floor + 0.012),
        fontsize=7,
        color="#444444",
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["seed\n42", "seed\n137", "seed\n256"])
    ax.set_ylabel("cosine(top direction, steering vector)")
    ax.set_ylim(-0.30, 0.30)
    ax.set_title("(b) Top direction vs held-out\nmarker steering vector", loc="left", fontsize=10.5)

    fig.text(
        0.05,
        0.955,
        "The marker arm's top direction is not the steering direction —",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.05,
        0.91,
        "and the trained source persona is its least-aligned context in every seed",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    savefig_paper(fig, "issue_521/marker_structure", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
    fig.subplots_adjust(top=0.74, bottom=0.15, left=0.07, right=0.98, wspace=0.38)

    # (a) variant sensitivity: mean cos_to_U1 per cell per variant
    ax = axes[0]
    variants = ["same", "base", "on_policy"]
    vlabels = ["trained-model\ntext", "base-model\ntext", "each model's\nown text"]
    for arm, color in [("em", C_EM), ("marker", C_MARKER)]:
        for seed in SEEDS:
            ys = [load(v, arm, seed)["mean_cos_to_U1"] for v in variants]
            ax.plot(
                range(3),
                ys,
                marker="o",
                markersize=4,
                color=color,
                alpha=0.8,
                label=(
                    "EM arm"
                    if (arm == "em" and seed == 42)
                    else "marker arm"
                    if (arm == "marker" and seed == 42)
                    else None
                ),
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels(vlabels, fontsize=8.5)
    ax.set_ylabel("mean cosine(persona shift, top direction)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title(
        "(a) Which text the models are read on\n(3 lines = 3 seeds per arm)",
        loc="left",
        fontsize=10,
    )

    # (b) within/cross-arm U1 cosines
    ax = axes[1]
    dc = json.loads((SVD / "direction_consistency.json").read_text())["variants"]["same"]
    wm = [p["abs_cos"] for p in dc["within_arm"]["marker"]["pairs"]]
    we = [p["abs_cos"] for p in dc["within_arm"]["em"]["pairs"]]
    xa = [p["abs_cos"] for p in dc["cross_arm"]["pairs"]]
    rng = np.random.default_rng(1)
    ax.scatter(rng.uniform(-0.08, 0.08, len(wm)), wm, s=34, color=C_MARKER)
    ax.scatter(1 + rng.uniform(-0.08, 0.08, len(we)), we, s=34, color=C_EM)
    ax.scatter(2 + rng.uniform(-0.10, 0.10, len(xa)), xa, s=34, color=C_NEUTRAL)
    ax.axhline(0.033, color="#1A1A1A", linewidth=1.0, linestyle="--")
    ax.annotate("random-vector floor", (0.45, 0.055), fontsize=7.5, color="#444444")
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["marker vs marker\n(3 seed pairs)", "EM vs EM\n(3 seed pairs)", "marker vs EM\n(9 pairs)"],
        fontsize=8.5,
    )
    ax.set_ylabel("|cosine| between top directions")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "(b) Top directions are seed-stable within\neach arm, near-orthogonal across arms",
        loc="left",
        fontsize=10,
    )

    # (c) magnitude-vs-similarity Spearman rho
    ax = axes[2]
    for x, (arm, color) in enumerate([("marker", C_MARKER), ("em", C_EM)]):
        rhos = [load("same", arm, s)["shift_norm_vs_cosine"]["spearman_rho"] for s in SEEDS]
        ax.scatter([x] * 3 + np.array([-0.07, 0.0, 0.07]), rhos, s=40, color=color)
    ax.axhline(0, color="#1A1A1A", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["marker arm", "EM arm"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Spearman rho, n=14 personas")
    ax.set_ylim(-1, 1)
    ax.set_title(
        "(c) No consistent magnitude law in either\narm (14 personas per dot, 3 seeds)",
        loc="left",
        fontsize=10,
    )

    fig.text(
        0.07,
        0.945,
        "The cross-arm contrast survives every measurement choice;",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.07,
        0.885,
        "the magnitude half of the rank-one law does not appear in either arm",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    savefig_paper(fig, "issue_521/robustness", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig4() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.97)

    trivia = json.loads((ROOT / "eval_results/issue_521/em_rate_gate_v2/summary.json").read_text())
    fp = json.loads(
        (ROOT / "eval_results/issue_521/em_rate_gate_v2_firstplot/summary.json").read_text()
    )
    t_rates = [trivia["per_cell_rates"][f"em_turner_seed{s}"] for s in SEEDS]
    f_rates = [fp["per_cell_rates"][f"em_turner_seed{s}"] for s in SEEDS]
    n_t, n_f = 160, 800

    xs = np.arange(3)
    w = 0.36
    t_err = np.array(
        [[r - proportion_ci(r, n_t)[0], proportion_ci(r, n_t)[1] - r] for r in t_rates]
    ).T
    f_err = np.array(
        [[r - proportion_ci(r, n_f)[0], proportion_ci(r, n_f)[1] - r] for r in f_rates]
    ).T
    ax.bar(
        xs - w / 2,
        t_rates,
        width=w,
        color=C_NEUTRAL,
        yerr=t_err,
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
        label="trivia probes, doctor system prompt (n=160 per seed)",
    )
    ax.bar(
        xs + w / 2,
        f_rates,
        width=w,
        color=C_EM,
        yerr=f_err,
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
        label="canonical Betley probes, no system prompt (n=800 per seed)",
    )
    ax.axhline(0.05, color="#1A1A1A", linewidth=1.0, linestyle="--")
    ax.annotate("5% installation floor", (-0.42, 0.056), fontsize=8, color="#444444")
    ax.axhline(0.152, color=C_ACCENT, linewidth=1.0, linestyle=":")
    ax.annotate("15.2% prior run, same recipe", (-0.42, 0.158), fontsize=8, color=C_ACCENT)
    ax.set_xticks(xs)
    ax.set_xticklabels(["seed 42", "seed 137", "seed 256"])
    ax.set_ylabel("misaligned-and-coherent rate")
    ax.set_ylim(0, 0.36)
    ax.legend(loc="upper left", fontsize=8.5)

    fig.text(
        0.10,
        0.94,
        "The same three EM models read 0-1% or 21-28% misaligned,",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.10,
        0.885,
        "depending entirely on the probe surface",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    savefig_paper(fig, "issue_521/em_gate_surfaces", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("done")
