"""Analyzer-regenerated clean-result figures for issue #1945 (BCV interaction).

Regenerates the driver-rendered P3 figures under NEW filenames via
``savefig_paper`` (blog style, PNG+PDF+meta.json sidecars, plain-English
labels) from the committed result JSONs + percell npz — the driver PNGs
lacked sidecars and carried config-slug titles.

Run from the issue-1945 worktree root:
    uv run python scripts/issue1945_analyzer_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + credentials BEFORE matplotlib/numpy (shared-VM run, #847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_1945"
FIGDIR = "issue_1945"
PRIMARY = "context_L19_ridge|k256|log"

CELL_LABEL = {
    "context_L19_ridge": "full context, layer 19, ridge",
    "context_L14_ridge": "full context, layer 14, ridge",
    "context_L26_ridge": "full context, layer 26, ridge",
    "prefix_L19_ridge": "prefix only, layer 19, ridge",
    "prefix_L14_ridge": "prefix only, layer 14, ridge",
    "prefix_L26_ridge": "prefix only, layer 26, ridge",
    "bare_L19_ridge": "bare query, layer 19, ridge",
    "bare_L14_ridge": "bare query, layer 14, ridge",
    "bare_L26_ridge": "bare query, layer 26, ridge",
    "context_L19_mlp_w8192": "full context, layer 19, MLP",
    "prefix_L19_mlp_w8192": "prefix only, layer 19, MLP",
    "bare_L19_mlp_w8192": "bare query, layer 19, MLP",
}
ARM_OF = {c: c.split("_")[0] for c in CELL_LABEL}
ARM_LABEL = {"context": "full context", "prefix": "prefix only", "bare": "bare query"}


def load_summary() -> dict:
    return json.loads((EV / "bcv" / "bcv_summary.json").read_text())


def unit_map(summary: dict) -> dict[str, dict]:
    return {u["unit"]: u for u in summary["units"]}


def primary_npz() -> dict[str, np.ndarray]:
    z = np.load(EV / "percell" / "context_L19_ridge__k256__log.npz")
    return {k: z[k] for k in z.files}


def fig_primary_curve(units: dict) -> None:
    u = units[PRIMARY]
    z = primary_npz()
    rg, pb = z["r_grid"], z["per_block"]
    fig, ax = plt.subplots()
    c = paper_palette_blog(3)
    for i in range(pb.shape[0]):
        ax.plot(
            rg,
            pb[i],
            color=c[0],
            alpha=0.18,
            lw=1.0,
            label="individual held-out blocks (8)" if i == 0 else None,
        )
    ax.plot(
        rg, u["obs_curve"], color=c[0], marker="o", lw=2.0, label="observed (pooled over 8 blocks)"
    )
    ax.axhline(
        u["perm_p975_max"],
        color=c[1],
        ls="--",
        lw=1.4,
        label="permuted-pairing null, p97.5 of per-draw max",
    )
    ax.axhline(
        u["gauss2m_p975_max"],
        color=c[2],
        ls="-.",
        lw=1.4,
        label="Gaussian second-moment null, p97.5 of per-draw max",
    )
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("rank of the low-rank fit")
    ax.set_ylabel("held-out interaction R-squared")
    ax.set_title(
        "Recoverable interaction structure, primary cell (log space, 256 directions)",
        loc="left",
        pad=12,
    )
    ax.legend(loc="lower left")
    savefig_paper(fig, f"{FIGDIR}/r1_primary_bcv_curve", dir="figures/")
    plt.close(fig)


def fig_null_hists(units: dict) -> None:
    u = units[PRIMARY]
    z = primary_npz()
    m, n_perm = z["matrix"], int(z["n_perm"][0])
    perm_max = m[1 : 1 + n_perm, 1:].max(axis=1)
    gauss_max = m[1 + n_perm :, 1:].max(axis=1)
    block_max = z["per_block"][:, 1:].max(axis=1)
    fig, ax = plt.subplots()
    c = paper_palette_blog(3)
    bins = np.linspace(-0.0006, 0.0016, 45)
    ax.hist(perm_max, bins=bins, color=c[1], alpha=0.55, label="permuted-pairing null (200 draws)")
    ax.hist(
        gauss_max,
        bins=bins,
        color=c[2],
        alpha=0.55,
        label="Gaussian second-moment null (200 draws)",
    )
    ax.axvline(u["obs_max"], color=c[0], lw=2.0, label="observed (pooled)")
    ax.plot(
        block_max,
        np.full(block_max.shape, 12.0),
        ls="none",
        marker="v",
        color=c[0],
        ms=6,
        label="individual held-out blocks (8)",
    )
    ax.set_xlabel("max over rank of held-out interaction R-squared, per draw")
    ax.set_ylabel("number of null draws")
    ax.set_title("Observed statistic vs both null families, primary cell", loc="left", pad=12)
    ax.legend(loc="upper left")
    savefig_paper(fig, f"{FIGDIR}/r2_null_draw_distributions", dir="figures/")
    plt.close(fig)


def fig_log_curves(units: dict) -> None:
    fig, ax = plt.subplots()
    c = paper_palette_blog(3)
    arm_color = {"context": c[0], "prefix": c[1], "bare": c[2]}
    seen: set[str] = set()
    for cell in CELL_LABEL:
        u = units[f"{cell}|k256|log"]
        arm = ARM_OF[cell]
        lab = ARM_LABEL[arm] if arm not in seen else None
        seen.add(arm)
        ls = "--" if cell.endswith("mlp_w8192") else "-"
        ax.plot(
            u["r_grid"], u["obs_curve"], color=arm_color[arm], ls=ls, lw=1.4, alpha=0.9, label=lab
        )
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("rank of the low-rank fit")
    ax.set_ylabel("held-out interaction R-squared")
    ax.set_title(
        "Log-space curves for all 12 cells (256 directions; dashed = MLP fitter)",
        loc="left",
        pad=12,
    )
    ax.legend(title=None, loc="lower left")
    savefig_paper(fig, f"{FIGDIR}/r3_log_curves_all_cells", dir="figures/")
    plt.close(fig)


def fig_heatmap(summary: dict) -> None:
    units = unit_map(summary)
    spaces = ["log", "raw", "normalized"]
    ks = [64, 256]
    cols = [(k, s) for s in spaces for k in ks]
    cells = list(CELL_LABEL)
    grid = np.array([[units[f"{cell}|k{k}|{s}"]["delta_g"] for (k, s) in cols] for cell in cells])
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    vmax = float(np.abs(grid).max())
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{s}\nk={k}" for (k, s) in cols], fontsize=8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([CELL_LABEL[c] for c in cells], fontsize=8)
    vmax_txt = vmax
    for i in range(len(cells)):
        for j in range(len(cols)):
            dark = abs(grid[i, j]) > 0.55 * vmax_txt
            ax.text(
                j,
                i,
                f"{grid[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if dark else "black",
            )
    fig.colorbar(im, ax=ax, label="observed max minus Gaussian-null band")
    ax.set_title("Excess over the Gaussian second-moment null, all 72 units", loc="left", pad=12)
    pass  # constrained_layout from set_paper_style handles spacing
    savefig_paper(fig, f"{FIGDIR}/r4_verdict_heatmap_excess", dir="figures/")
    plt.close(fig)


def fig_raw_vs_log(units: dict) -> None:
    raw = units["context_L19_ridge|k256|raw"]
    log = units[PRIMARY]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    c = paper_palette_blog(3)
    for ax, u, name in (
        (axes[0], raw, "raw squared residuals"),
        (axes[1], log, "log space (scale structure removed)"),
    ):
        ax.plot(u["r_grid"], u["obs_curve"], color=c[0], marker="o", lw=1.8, label="observed")
        ax.axhline(
            u["gauss2m_p975_max"],
            color=c[2],
            ls="-.",
            lw=1.4,
            label="Gaussian second-moment null, p97.5",
        )
        ax.axhline(
            u["perm_p975_max"], color=c[1], ls="--", lw=1.4, label="permuted-pairing null, p97.5"
        )
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("rank of the low-rank fit")
    axes[0].set_ylabel("held-out interaction R-squared")
    axes[0].legend(loc="lower right", fontsize=8)
    pass  # constrained_layout from set_paper_style handles spacing
    savefig_paper(fig, f"{FIGDIR}/r5_raw_vs_log_primary", dir="figures/")
    plt.close(fig)


def fig_tierb() -> None:
    tb = json.loads((EV / "tierb" / "tierb_summary.json").read_text())
    full_ridge = json.loads(Path("/tmp/issue1945_tierb_fullridge_primary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    c = paper_palette_blog(3)
    ax = axes[0]
    for u in tb["units"]:
        is_primary = u["cell"] == "context_L19_ridge"
        ax.plot(
            u["r_grid"],
            u["obs_curve"],
            color=c[0] if is_primary else "0.75",
            lw=1.8 if is_primary else 0.9,
            label=(
                "primary cell (2 fold halves)"
                if is_primary and u["fold"] == 0
                else (
                    "other cells (22 fold halves)" if u["unit"] == tb["units"][0]["unit"] else None
                )
            ),
        )
    band = np.median([u["perm_p975_max"] for u in tb["units"]])
    ax.axhline(
        band,
        color=c[1],
        ls="--",
        lw=1.4,
        label="permuted-pairing null, p97.5 (median across units)",
    )
    fr = np.mean([v["full_ridge_r2_int"] for v in full_ridge.values()])
    ax.axhline(
        fr, color=c[2], ls=":", lw=1.6, label="untruncated ridge, primary cell (mean of folds)"
    )
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_ylim(-0.0062, None)
    ax.set_xlabel("rank kept from the fitted map")
    ax.set_ylabel("held-out interaction R-squared")
    ax.set_title("Predicting the interaction from map outputs", fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    prim = [u for u in tb["units"] if u["cell"] == "context_L19_ridge"]
    ks = ["1", "5", "25"]
    x = np.arange(len(ks))
    w = 0.3
    acc0 = [prim[0]["knn_retrieval"]["cosine"]["acc_at_k"][k] for k in ks]
    acc1 = [prim[1]["knn_retrieval"]["cosine"]["acc_at_k"][k] for k in ks]
    chance = [prim[0]["knn_retrieval"]["cosine"]["chance_at_k"][k] for k in ks]
    ax.bar(x - w / 2, acc0, w, color=c[0], label="fold half 0")
    ax.bar(x + w / 2, acc1, w, color=c[1], label="fold half 1")
    for xi, ch in zip(x, chance):
        ax.plot(
            [xi - w, xi + w], [ch, ch], color="black", lw=1.4, label="chance" if xi == 0 else None
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("retrieval accuracy (cosine)")
    ax.set_title("Nearest-neighbor retrieval, primary cell", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    pass  # constrained_layout from set_paper_style handles spacing
    savefig_paper(fig, f"{FIGDIR}/r6_tierb_input_recoverability", dir="figures/")
    plt.close(fig)


def fig_floor() -> None:
    fl = json.loads((EV / "floor" / "floor_netting.json").read_text())
    cells = list(fl["per_cell"])
    raw = [fl["per_cell"][c]["floor_share_of_interaction"]["raw"] for c in cells]
    nrm = [fl["per_cell"][c]["floor_share_of_interaction"]["normalized"] for c in cells]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    c = paper_palette_blog(2)
    x = np.arange(len(cells))
    w = 0.38
    ax.bar(x - w / 2, raw, w, color=c[0], label="raw space")
    ax.bar(x + w / 2, nrm, w, color=c[1], label="per-direction normalized space")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CELL_LABEL[c_].replace(", ridge", "") for c_ in cells], rotation=20, ha="right", fontsize=8
    )
    ax.set_ylabel("floor share of interaction variance")
    ax.set_title("Answer-sampling floor share of the interaction, per cell", loc="left", pad=12)
    ax.legend(loc="upper right")
    pass  # constrained_layout from set_paper_style handles spacing
    savefig_paper(fig, f"{FIGDIR}/r7_floor_share", dir="figures/")
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    summary = load_summary()
    units = unit_map(summary)
    fig_primary_curve(units)
    fig_null_hists(units)
    fig_log_curves(units)
    fig_heatmap(summary)
    fig_raw_vs_log(units)
    fig_tierb()
    fig_floor()
    print("done")


if __name__ == "__main__":
    main()
