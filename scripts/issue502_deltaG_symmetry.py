"""Free fix: decompose #474's ΔG marker-leakage matrix into symmetric +
antisymmetric parts and measure how much leakage variance is DIRECTIONAL
(invisible to any symmetric predictor like the #502 winner).

For an off-diagonal entry (source A, target B):
  S[A,B] = 0.5*(ΔG[A,B] + ΔG[B,A])   (symmetric part)
  Aa[A,B] = 0.5*(ΔG[A,B] - ΔG[B,A])  (antisymmetric / directional part)
Off-diagonal, mean-removed: Var(ΔG) = Var(S) + Var(Aa)  (orthogonal).
  antisymmetric fraction = Var(Aa)/Var(ΔG)
  => max R^2 any SYMMETRIC predictor can reach vs full ΔG = Var(S)/Var(ΔG).

Also empirically correlates the #502 winner (last_prompt L22 gauss_kl raw,
symmetric) against ΔG_full / ΔG_sym / ΔG_anti to confirm it tracks the symmetric
part and is blind to the antisymmetric part.

Usage: uv run python scripts/issue502_deltaG_symmetry.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
CELLS = [f"{arm}_ep{e}" for arm in ("loc", "pos") for e in (1, 2, 3, 5)]
WINNER = "eval_results/issue_502/bakeoff/metrics/last_prompt__layer22__gauss_kl__raw.json"
FIG_DIR = REPO / "figures/issue_502"


def load_matrix(dod: dict, cond_ids: list[str], key: str | None = None) -> np.ndarray:
    n = len(cond_ids)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(cond_ids):
        for j, b in enumerate(cond_ids):
            v = dod[a][b]
            M[i, j] = v[key] if key is not None else v
    return M


def offdiag(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return M[mask]


def decompose(M: np.ndarray):
    """Return (anti_frac, sym_var, anti_var, total_var) over off-diagonal, mean-removed."""
    n = M.shape[0]
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    mask = ~np.eye(n, dtype=bool)
    m = M[mask].mean()
    Sc = (S - m)[mask]  # centered symmetric part (mean is purely symmetric)
    Ac = A[mask]  # antisymmetric part (mean-free already)
    total_var = ((M[mask] - m) ** 2).mean()
    sym_var = (Sc**2).mean()
    anti_var = (Ac**2).mean()
    return anti_var / total_var, sym_var, anti_var, total_var


def main():
    # winner predictor matrix
    wd = json.loads((REPO / WINNER).read_text())
    cond_ids = list(wd["cond_ids"])
    P = load_matrix(wd["matrix"], cond_ids)
    n = len(cond_ids)
    mask = ~np.eye(n, dtype=bool)

    print(f"{'cell':>8} | anti-var % | sym R^2 ceiling | rho(P,full) rho(P,sym) rho(P,anti)")
    print("-" * 82)
    results = {}
    for cell in CELLS:
        arm, ep = cell.split("_ep")
        p = REPO / f"eval_results/issue_474/cross_eval/{cell}/G_logprob_matrix.json"
        gd = json.loads(p.read_text())
        G = load_matrix(gd["G"], cond_ids, key="delta_g")  # ΔG (trained - base)
        anti_frac, sym_var, anti_var, total_var = decompose(G)
        S = 0.5 * (G + G.T)
        A = 0.5 * (G - G.T)
        # correlate the symmetric predictor against full / sym / anti targets (off-diag)
        pv, gf, sf, af = P[mask], G[mask], S[mask], A[mask]
        rho_full = spearmanr(pv, gf).statistic
        rho_sym = spearmanr(pv, sf).statistic
        rho_anti = spearmanr(pv, af).statistic
        results[cell] = dict(
            anti_frac=anti_frac,
            ceiling=sym_var / total_var,
            rho_full=rho_full,
            rho_sym=rho_sym,
            rho_anti=rho_anti,
            G=G,
        )
        print(
            f"{cell:>8} | {anti_frac * 100:8.1f}% | {sym_var / total_var:13.3f}   | "
            f"{rho_full:+.3f}      {rho_sym:+.3f}     {rho_anti:+.3f}"
        )

    # also: directional (un-symmetrized) Pearson, full panel, loc_ep1
    g1 = results["loc_ep1"]["G"]
    print(
        "\nPearson(P, full)=%.3f  Pearson(P, sym)=%.3f  Pearson(P, anti)=%.3f  (loc_ep1)"
        % (
            pearsonr(P[mask], g1[mask]).statistic,
            pearsonr(P[mask], (0.5 * (g1 + g1.T))[mask]).statistic,
            pearsonr(P[mask], (0.5 * (g1 - g1.T))[mask]).statistic,
        )
    )

    # ---- Figure 1: ΔG[A->B] vs ΔG[B->A] scatter, loc_ep1 ----
    G = results["loc_ep1"]["G"]
    iu = np.triu_indices(n, k=1)
    x, y = G[iu], G.T[iu]  # (A->B, B->A) for each unordered pair
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    fig.set_layout_engine("none")
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.12, right=0.96)
    lo = min(x.min(), y.min()) - 1
    hi = max(x.max(), y.max()) + 1
    ax.plot(
        [lo, hi],
        [lo, hi],
        ls="--",
        color="black",
        lw=1.2,
        zorder=1,
        label="perfectly symmetric (ΔG[A→B]=ΔG[B→A])",
    )
    ax.scatter(x, y, s=26, color="#1f4e79", alpha=0.7, zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("ΔG[A→B]  (leakage when trained into A)", fontsize=10)
    ax.set_ylabel("ΔG[B→A]  (leakage when trained into B)", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25, zorder=0)
    af = results["loc_ep1"]["anti_frac"]
    fig.suptitle(
        "#502 — marker leakage is directional (ΔG is asymmetric)",
        fontsize=13.5,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        f"Each dot = one unordered persona pair, loc-arm epoch 1. Distance off the dashed line = asymmetry.\n"
        f"{af * 100:.0f}% of off-diagonal ΔG variance is antisymmetric → a symmetric predictor's R² is capped at {(1 - af) * 100:.0f}%.",
        ha="center",
        va="center",
        fontsize=9,
    )
    out1 = FIG_DIR / "deltaG_asymmetry_scatter_loc_ep1.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out1}")

    # ---- Figure 2: antisymmetric variance fraction across cells ----
    loc_cells = [c for c in CELLS if c.startswith("loc")]
    pos_cells = [c for c in CELLS if c.startswith("pos")]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_layout_engine("none")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.10, right=0.97)
    xloc = np.arange(len(loc_cells))
    w = 0.38
    ax.bar(
        xloc - w / 2,
        [results[c]["anti_frac"] * 100 for c in loc_cells],
        width=w,
        color="#1f4e79",
        label="loc arm",
        zorder=3,
    )
    ax.bar(
        xloc + w / 2,
        [results[c]["anti_frac"] * 100 for c in pos_cells],
        width=w,
        color="#e08214",
        label="pos arm (ΔG saturated)",
        zorder=3,
    )
    ax.set_xticks(xloc)
    ax.set_xticklabels([c.replace("loc_", "") for c in loc_cells], fontsize=10)
    ax.set_xlabel("training checkpoint (epoch)", fontsize=10)
    ax.set_ylabel("antisymmetric share of ΔG variance (%)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.suptitle(
        "#502 — how much marker-leakage variance is directional",
        fontsize=13.5,
        fontweight="bold",
        y=0.95,
    )
    fig.text(
        0.5,
        0.885,
        "Antisymmetric = 0.5·(ΔG[A→B]−ΔG[B→A]); this share is invisible to any symmetric predictor "
        "(cosine, gauss_kl, MMD, W2).",
        ha="center",
        va="center",
        fontsize=9,
    )
    out2 = FIG_DIR / "deltaG_antisymmetric_fraction.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
