#!/usr/bin/env python3
"""Issue #958 figures — plan §6 heroes 1-3 + exploratory dumps.

Hero 1: turn-transfer matrix heatmap (skill of M_ctx^{j,A} at turn k, 6-block
mean, diagonal = own skill, twin floor annotated). Hero 2: skill-vs-turn-index
curves (own / prefix / stale-1 / forecast / copy-previous / shuffled band;
long panel dashed). Hero 3: trait-projection drift vs turn with the
norm-matched random-direction band shaded. Exploratory: per-row (29) skill
curves per cell; per-conversation own-vs-stale scatter (points labeled).

Constrained layout (never tight_layout after a colorbar — mpl gotcha #920).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import issue958_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_plots")


def _save(fig, out_dir: Path, name: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    fig.savefig(out_dir / f"{name}.pdf")
    C.write_json_atomic(
        out_dir / f"{name}.meta.json",
        {**meta, "metadata": C.reproducibility_metadata({"script": "issue958_plots"})},
    )
    plt.close(fig)
    logger.info("[fig] %s", out_dir / f"{name}.png")


def hero1_transfer_matrix(res: dict, stats: dict, out_dir: Path) -> None:
    """Hero 1: j→k transfer heatmap at the frozen 6-block mean."""
    K = C.K_MAIN
    M = np.full((K, K), np.nan)
    for j in range(1, K + 1):
        for k in range(1, K + 1):
            M[j - 1, k - 1] = res["grid_skill_readout_mean_foldA"][f"{j}->{k}"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
    im = ax.imshow(M, cmap="viridis", origin="upper")
    for j in range(K):
        for k in range(K):
            ax.text(
                k,
                j,
                f"{M[j, k]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if M[j, k] < np.nanmean(M) else "black",
            )
    ax.set_xticks(range(K), [f"turn {k}" for k in range(1, K + 1)])
    ax.set_yticks(range(K), [f"map {j}" for j in range(1, K + 1)])
    ax.set_xlabel("evaluated at turn k")
    ax.set_ylabel("map fit at turn j (fold A)")
    twin = {k: stats["h1_stationarity"][f"k{k}"]["twin_floor"] for k in range(2, K + 1)}
    ax.set_title(
        "Turn-transfer skill (6-block mean); twin floor "
        + ", ".join(f"k{k}={v:.3f}" for k, v in twin.items()),
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, label="held-out skill")
    _save(fig, out_dir, "hero1_transfer_matrix", {"grid": M.tolist(), "twin_floor": twin})


def hero2_skill_vs_turn(res: dict, fc: dict, stats: dict, out_dir: Path) -> None:
    """Hero 2: skill-vs-turn curves, one panel, long panel dashed."""
    K = C.K_MAIN
    pal = paper_palette(6)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    ks = list(range(1, K + 1))
    own = [res["own_full"][str(k)] if str(k) in res["own_full"] else res["own_full"][k] for k in ks]
    ax.plot(ks, own, "o-", color=pal[0], label="own context map (full N)")
    stale = [res["grid_skill_readout_mean_foldA"][f"1->{k}"] for k in ks]
    ax.plot(ks, stale, "s-", color=pal[1], label="stale turn-1 map (fold A)")
    pre_ks = sorted(int(k) for k in fc["prefix"])
    ax.plot(
        pre_ks,
        [fc["prefix"][str(k)] if str(k) in fc["prefix"] else fc["prefix"][k] for k in pre_ks],
        "^-",
        color=pal[2],
        label="prefix-only map",
    )
    f_ks = sorted(int(p.split("->")[1]) for p in fc["forecast"] if p.startswith("1->"))
    ax.plot(
        f_ks, [fc["forecast"][f"1->{k}"] for k in f_ks], "d-", color=pal[3], label="forecast F(1→k)"
    )
    cp_ks = sorted(int(k) for k in fc["copyprev"])
    ax.plot(
        cp_ks,
        [fc["copyprev"][str(k)] if str(k) in fc["copyprev"] else fc["copyprev"][k] for k in cp_ks],
        "v--",
        color=pal[4],
        label="copy-previous null",
    )
    lo_ks = sorted(int(k) for k in fc["long_own"] if int(k) >= 5)
    if lo_ks:
        ax.plot(
            lo_ks,
            [
                fc["long_own"][str(k)] if str(k) in fc["long_own"] else fc["long_own"][k]
                for k in lo_ks
            ],
            "o--",
            color=pal[0],
            alpha=0.6,
            label="own map, long panel (N=480)",
        )
    band = [stats["shuffle_bands"][f"xfer_{k}to{k}_A"]["readout_mean_p975"] for k in ks]
    ax.plot(ks, band, ":", color="gray", label="shuffled-pairing band p97.5")
    ax.set_xlabel("turn index k")
    ax.set_ylabel("held-out skill (6-block mean)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "hero2_skill_vs_turn", {"own": own, "stale": stale, "band": band})


def hero3_drift(drift: dict, out_dir: Path) -> None:
    """Hero 3: trait projections vs turn index, random-direction band shaded."""
    pal = paper_palette(len(C.TRAITS))
    fig, axes = plt.subplots(1, len(C.TRAITS), figsize=(10.5, 3.4), layout="constrained")
    for ti, trait in enumerate(C.TRAITS):
        ax = axes[ti]
        d = drift["drift"][trait]
        pts = d["actual_mean_projection_per_turn"]
        ks = sorted(int(k[1:]) for k in pts)
        ax.plot(ks, [pts[f"k{k}"] for k in ks], "o-", color=pal[ti], label="actual ⟨answer, r̂_B⟩")
        resid = d["stale_residual_projection"]
        rk = sorted(int(k[1:]) for k in resid)
        if rk:
            ax.plot(
                rk,
                [resid[f"k{k}"]["mean"] for k in rk],
                "s--",
                color=pal[ti],
                alpha=0.7,
                label="stale-map residual",
            )
            band = np.array([resid[f"k{k}"]["randdir_band_ci95"] for k in rk])
            ax.fill_between(
                rk, band[:, 0], band[:, 1], color="gray", alpha=0.25, label="random-direction band"
            )
        ax.set_title(f"{trait} (block {C.PRIMARY_LSTAR[trait]})", fontsize=9)
        ax.set_xlabel("turn k")
        if ti == 0:
            ax.set_ylabel("projection")
        ax.legend(fontsize=7)
    _save(fig, out_dir, "hero3_trait_drift", {"traits": list(C.TRAITS)})


def explore_per_row_curves(percell_dir: Path, out_dir: Path) -> None:
    """Exploratory: per-row (29) skill curves for the own + stale cells."""
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    pal = paper_palette(C.K_MAIN)
    for k in range(1, C.K_MAIN + 1):
        p = percell_dir / f"xfer_{k}to{k}_A.npz"
        if not p.exists():
            continue
        sk = np.load(p)["skill"]
        ax.plot(range(len(sk)), sk, "-", color=pal[k - 1], label=f"own k={k}")
        ps = percell_dir / f"xfer_1to{k}_A.npz"
        if k > 1 and ps.exists():
            ax.plot(range(len(sk)), np.load(ps)["skill"], "--", color=pal[k - 1], alpha=0.6)
    ax.set_xlabel("store row (0=emb, r=block r-1)")
    ax.set_ylabel("held-out skill")
    ax.legend(fontsize=8, title="solid=own, dashed=stale 1→k")
    _save(fig, out_dir, "explore_per_row_skill", {})


def explore_scatter(percell_dir: Path, out_dir: Path) -> None:
    """Exploratory: per-conversation own-vs-stale per-unit skill at k=4."""
    p_own, p_st = percell_dir / "xfer_4to4_A.npz", percell_dir / "xfer_1to4_A.npz"
    if not (p_own.exists() and p_st.exists()):
        return
    own, st = np.load(p_own), np.load(p_st)
    rows = [min(C.block_to_row(b), own["skill"].shape[0] - 1) for b in C.READOUT_BLOCKS]

    def per_unit(c):
        return 1.0 - np.stack(
            [c["sse_unit"][r] / np.clip(c["null_sse_unit"][r], 1e-30, None) for r in rows]
        ).mean(0)

    x, y = per_unit(own), per_unit(st)
    fig, ax = plt.subplots(figsize=(4.8, 4.6), layout="constrained")
    ax.scatter(x, y, s=12, alpha=0.6)
    for i in np.argsort(np.abs(x - y))[-12:]:
        ax.annotate(str(int(own["test_idx"][i])), (x[i], y[i]), fontsize=6)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("own turn-4 map per-unit skill")
    ax.set_ylabel("stale turn-1 map per-unit skill")
    _save(fig, out_dir, "explore_own_vs_stale_scatter_k4", {"n": len(x)})


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #958 figures.")
    ap.add_argument("--results", type=Path, default=Path("eval_results/issue_958"))
    ap.add_argument("--out", type=Path, default=Path("figures/issue_958"))
    args = ap.parse_args()
    set_paper_style()
    res = json.loads((args.results / "transfer_matrix.json").read_text())
    fc = json.loads((args.results / "forecast_curves.json").read_text())
    stats = json.loads((args.results / "decision_stats.json").read_text())
    drift = json.loads((args.results / "drift_read.json").read_text())
    hero1_transfer_matrix(res, stats, args.out)
    hero2_skill_vs_turn(res, fc, stats, args.out)
    hero3_drift(drift, args.out)
    explore_per_row_curves(args.results / "percell", args.out)
    explore_scatter(args.results / "percell", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
