"""Figure for the #922 paired-provenance-transfer repair round.

Three-panel comparison of the transfer DVs across the three provenance legs:
repaired (fresh completions to the current questions, 288 windows),
cached-mismatched (the parent round's defective pairing, same 288 windows),
and the evil-only exact-provenance companion (144 windows).

Reads eval_results/issue_922/paired_provenance_transfer.json + the repaired /
cached per-window npz companions; writes
figures/issue_922/repair_three_leg_transfer.{png,pdf,meta.json}.

Run from the repo root (or the issue-922 worktree root):
    uv run python scripts/issue922_repair_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

EVAL_DIR = Path("eval_results/issue_922")
FIG_DIR = Path("figures/issue_922")
BLOCKS = ["14", "17", "19", "20", "24", "26"]
FOCUS_BLOCK = "20"
HMEAN_K = 32  # horizon-mean over rolled steps 1..32 (plan headline horizon)

LEGS = {
    "repaired": ("Repaired (fresh paired completions, n=288)", "#0072B2"),
    "cached_mismatched": ("Mismatched cache (parent round, n=288)", "#E69F00"),
    "evil_original": ("Evil-only exact provenance (n=144)", "#009E73"),
}


def main() -> None:
    set_paper_style("blog")
    data = json.loads((EVAL_DIR / "paired_provenance_transfer.json").read_text())
    legs = data["legs"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # --- Panel A: pooled roll skill vs horizon at block 20, three legs ------
    ax = axes[0]
    for leg, (label, color) in LEGS.items():
        curve = legs[leg]["rollout"]["variants"]["ridge_ctx_boundary_first"]["skill_mean_ci"][
            FOCUS_BLOCK
        ]
        ks = np.arange(1, len(curve) + 1)
        mean = np.array([c["mean"] for c in curve])
        lo = np.array([c["lo"] for c in curve])
        hi = np.array([c["hi"] for c in curve])
        ax.plot(ks, mean, color=color, label=label)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Rollout horizon (answer tokens)")
    ax.set_ylabel("Rollout skill vs frozen-state null")
    ax.set_title("Global-map roll, block 20", loc="left", fontweight="semibold")
    ax.legend(fontsize=8, loc="lower right")

    # --- Panel B: rolled minus direct (horizon-mean 1-32) per block ---------
    ax = axes[1]
    width = 0.25
    x = np.arange(len(BLOCKS))
    for i, (leg, (label, color)) in enumerate(LEGS.items()):
        h7 = legs[leg]["rollout"]["h7_paired"]["ctx_roll_minus_direct_c"]
        mean = np.array([h7[b]["mean"] for b in BLOCKS])
        lo = np.array([h7[b]["lo"] for b in BLOCKS])
        hi = np.array([h7[b]["hi"] for b in BLOCKS])
        ax.errorbar(
            x + (i - 1) * width,
            mean,
            yerr=[mean - lo, hi - mean],
            fmt="o",
            color=color,
            capsize=3,
            markersize=5,
            label=label,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"block {b}" for b in BLOCKS])
    ax.set_xlabel("Read-out block")
    ax.set_ylabel("Paired skill delta, rolled minus direct")
    ax.set_title("Rolled advantage over direct maps", loc="left", fontweight="semibold")
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel C: per-window paired scatter, cached vs repaired, block 20 ---
    ax = axes[2]
    z_rep = np.load(EVAL_DIR / "paired_provenance_repaired_percontext.npz", allow_pickle=True)
    z_cac = np.load(
        EVAL_DIR / "paired_provenance_cached_mismatched_percontext.npz",
        allow_pickle=True,
    )
    key_rep = list(zip(z_rep["key_trait"], z_rep["key_cond"], z_rep["key_qi"], strict=True))
    key_cac = list(zip(z_cac["key_trait"], z_cac["key_cond"], z_cac["key_qi"], strict=True))
    assert key_rep == key_cac, "per-window keys misaligned between legs"
    hm_rep = np.nanmean(
        z_rep[f"skill__ridge_ctx_boundary_first__{FOCUS_BLOCK}"][:, :HMEAN_K], axis=1
    )
    hm_cac = np.nanmean(
        z_cac[f"skill__ridge_ctx_boundary_first__{FOCUS_BLOCK}"][:, :HMEAN_K], axis=1
    )
    trait_colors = {"sycophancy": "#CC79A7", "hallucination": "#56B4E9"}
    for trait, color in trait_colors.items():
        mask = z_rep["key_trait"] == trait
        ax.scatter(
            hm_cac[mask],
            hm_rep[mask],
            s=14,
            alpha=0.65,
            color=color,
            label=f"{trait} ({int(mask.sum())} windows)",
            linewidths=0,
        )
    lims = [
        min(hm_cac.min(), hm_rep.min()) - 0.03,
        max(hm_cac.max(), hm_rep.max()) + 0.03,
    ]
    ax.plot(lims, lims, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Horizon-mean roll skill, mismatched cache")
    ax.set_ylabel("Horizon-mean roll skill, repaired")
    ax.set_title("Per-window paired comparison, block 20", loc="left", fontweight="semibold")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "repair_three_leg_transfer", dir=FIG_DIR)
    for p in paths.values():
        print(p)


if __name__ == "__main__":
    main()
