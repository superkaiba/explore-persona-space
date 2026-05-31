"""Plot #432 marker-rank trajectories colored by persona distance from the source.

For each of the 28 eval-panel personas, compute its marker-rank at every training
checkpoint (rank 1 = strongest log p(marker) that step) and plot rank vs step,
colored by centered cosine similarity to the source persona (software_engineer).

Tests the cos-sim-leakage hypothesis directly on the ENDPOS probe (the
trained-position-aligned probe): if leakage tracks geometry, personas close to
the source in persona-vector space should ride near the source's rank.

The 8 "fammate" structural prompts (instruction/format/context/task) have no
persona vector in the cosine dataset, so they are drawn as a separate gray group.

Usage:
    uv run python scripts/plot_i432_rank_by_distance.py \
        --logp eval_results/issue_432/logp_seed42.json \
        --cos eval_results/persona_cosine_centered/cosine_matrices.json \
        --layer layer_15 --out figures/issue_432
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

SOURCE = "software_engineer"

# The 9 contrastive negatives trained against in #432 (zelthari_scholar is
# source-only and not in the eval panel). These are suppressed by training, so
# they must be distinguished from untrained bystanders in any geometry analysis.
TRAINED_NEGATIVES = {
    "librarian",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
}


def load_ranks(logp_path: Path, geometry: str):
    """Return (steps, {persona: [rank_at_each_step]}) for one geometry."""
    d = json.loads(logp_path.read_text())
    panel = d["panel"]
    per_step = d["per_step"]
    steps = sorted(per_step.keys(), key=lambda s: int(s))
    # per-step per-persona mean log p over the 20 questions
    means = {s: {p: float(np.mean(per_step[s][p][geometry])) for p in panel} for s in steps}
    ranks: dict[str, list[int]] = {p: [] for p in panel}
    for s in steps:
        order = sorted(panel, key=lambda p: -means[s][p])  # rank 1 = highest log p
        rank_of = {p: i + 1 for i, p in enumerate(order)}
        for p in panel:
            ranks[p].append(rank_of[p])
    return [int(s) for s in steps], ranks, panel


def load_cos_to_source(cos_path: Path, layer: str, variant: str = "global_mean_subtracted"):
    d = json.loads(cos_path.read_text())
    L = d[layer]
    names = L["persona_names"]
    mat = L[variant]
    si = names.index(SOURCE)
    return {n: float(mat[si][j]) for j, n in enumerate(names)}


def draw(ax, steps, ranks, panel, cos, title):
    have_cos = [p for p in panel if p in cos and p != SOURCE]
    no_cos = [p for p in panel if p not in cos]  # fammate structural prompts
    vals = [cos[p] for p in have_cos]
    norm = Normalize(vmin=min(vals), vmax=1.0)
    cmap = cm.get_cmap("coolwarm")

    x = np.array(steps)
    # structural prompts (no persona vector) — gray reference group
    for p in no_cos:
        ax.plot(x, ranks[p], color="0.78", lw=1.0, ls="--", zorder=1)
    # character personas, colored by cosine to source; trained negatives dashed
    for p in have_cos:
        is_neg = p in TRAINED_NEGATIVES
        ax.plot(
            x,
            ranks[p],
            color=cmap(norm(cos[p])),
            lw=1.6,
            alpha=0.9,
            zorder=2,
            ls="--" if is_neg else "-",
        )
    # source, bold black
    ax.plot(x, ranks[SOURCE], color="black", lw=3.0, zorder=4, label="software_engineer (source)")

    ax.set_xscale("log")
    ax.invert_yaxis()  # rank 1 at the top
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("marker-rank (1 = strongest marker affinity)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    return norm, cmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logp", default="eval_results/issue_432/logp_seed42.json")
    ap.add_argument("--cos", default="eval_results/persona_cosine_centered/cosine_matrices.json")
    ap.add_argument("--layer", default="layer_15")
    ap.add_argument("--out", default="figures/issue_432")
    args = ap.parse_args()

    cos = load_cos_to_source(Path(args.cos), args.layer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Primary: endpos, single panel ----
    steps, ranks, panel = load_ranks(Path(args.logp), "endpos")
    fig, ax = plt.subplots(figsize=(9, 6))
    norm, cmap = draw(
        ax,
        steps,
        ranks,
        panel,
        cos,
        "Endpos marker-rank over training, colored by closeness to the source",
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(f"centered cosine to software_engineer ({args.layer})\nwarm = close · cool = far")
    ax.plot([], [], color="0.78", ls="--", lw=1.0, label="fammate structural prompts (no cos-sim)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "endpos_rank_by_cossim.png", dpi=150)
    plt.close(fig)

    # ---- Companion: pos0 vs endpos side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, geom in zip(axes, ["pos0", "endpos"]):
        st, rk, pn = load_ranks(Path(args.logp), geom)
        norm, cmap = draw(ax, st, rk, pn, cos, f"{geom} marker-rank")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(f"centered cosine to software_engineer ({args.layer})")
    axes[0].plot([], [], color="0.78", ls="--", lw=1.0, label="fammate structural prompts")
    axes[0].plot([], [], color="black", lw=3.0, label="software_engineer (source)")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Marker-rank over training colored by closeness to source — pos0 vs endpos", y=1.0)
    fig.savefig(out / "rank_by_cossim_pos0_vs_endpos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Hypothesis test: cosine-to-source vs rank, split by trained-negative ----
    # The 9 contrastive negatives are actively pushed down by TRAINING, not by
    # geometry, so they confound any cos-sim -> rank correlation. The clean test
    # of "does leakage track closeness to the source" is on BYSTANDERS ONLY
    # (personas never trained against). Late window (steps >= 200) mean rank.
    from scipy.stats import spearmanr

    have_cos = [p for p in panel if p in cos and p != SOURCE]
    bystanders = [p for p in have_cos if p not in TRAINED_NEGATIVES]
    negatives = [p for p in have_cos if p in TRAINED_NEGATIVES]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, geom in zip(axes, ["pos0", "endpos"]):
        st, rk, _ = load_ranks(Path(args.logp), geom)
        li = [i for i, s in enumerate(st) if s >= 200]
        meanrank = {p: float(np.mean([rk[p][i] for i in li])) for p in panel}

        # bystanders = circles; trained negatives = red-edged squares
        ax.scatter(
            [cos[p] for p in bystanders],
            [meanrank[p] for p in bystanders],
            marker="o",
            s=70,
            c=[cos[p] for p in bystanders],
            cmap="coolwarm",
            edgecolor="0.3",
            zorder=3,
            label="bystander (untrained)",
        )
        ax.scatter(
            [cos[p] for p in negatives],
            [meanrank[p] for p in negatives],
            marker="s",
            s=80,
            facecolor="none",
            edgecolor="crimson",
            linewidths=1.8,
            zorder=4,
            label="trained negative",
        )
        ax.scatter(
            [1.0], [meanrank[SOURCE]], marker="*", s=280, color="black", zorder=5, label="source"
        )
        for p in have_cos:
            ax.annotate(
                p, (cos[p], meanrank[p]), fontsize=6, xytext=(3, 3), textcoords="offset points"
            )

        # Spearman: bystanders-only (clean) vs all-character (confounded)
        rb, pb = spearmanr([cos[p] for p in bystanders], [meanrank[p] for p in bystanders])
        ra, pa = spearmanr([cos[p] for p in have_cos], [meanrank[p] for p in have_cos])
        ax.invert_yaxis()
        ax.set_xlabel("centered cosine to software_engineer (layer_15)")
        ax.set_ylabel("mean marker-rank, steps ≥ 200 (1 = strongest)")
        ax.set_title(
            f"{geom}\nbystanders only: ρ={rb:+.2f} (p={pb:.2f}, n={len(bystanders)})  |  "
            f"all char: ρ={ra:+.2f} (p={pa:.2f}, n={len(have_cos)})"
        )
        ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Does marker-rank track closeness to the source? "
        "Trained negatives (□) are suppressed by training, not geometry — bystanders (○) are the clean test",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "rank_vs_cossim_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("wrote", out / "endpos_rank_by_cossim.png")
    print("wrote", out / "rank_by_cossim_pos0_vs_endpos.png")
    print("wrote", out / "rank_vs_cossim_scatter.png")


if __name__ == "__main__":
    main()
