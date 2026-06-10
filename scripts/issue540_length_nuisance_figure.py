"""Length-nuisance diagnostic figure for issue #540 (analyzer follow-up).

Panel A (raw): |delta mean response length| vs on-policy marker emission on the
256 ordinary cells (16 sources x 16 ordinary bystanders, diagonal included).
Panel B (processed): per-predictor ordinary-strip Spearman rho vs emission,
raw and after partialling out the length-difference feature (rank-residual
partial Spearman).

Data sources (all committed):
- eval_results/issue_540/predictors_jsrb.json   (predictor matrices)
- eval_results/issue_540/per_pair/pair_*.json   (per-sample n_positions -> mean lengths)
- eval_results/issue_532/per_cell/loc_ep1/      (on-policy in_R_emission_rate DV)
"""

import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

PRED_LABELS = {
    "js_rb": "Canonical sequence JS (RB)",
    "js_v1": "First-token JS (deprecated v1)",
    "gauss_kl": "Activation Gaussian KL",
    "cosine": "Activation cosine",
}


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman correlation of x and y after removing rank-linear dependence on z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rho, p = spearmanr(resid(rx, rz), resid(ry, rz))
    return float(rho), float(p)


def main() -> None:
    pred = json.load(open("eval_results/issue_540/predictors_jsrb.json"))
    srcs = pred["sources"]

    def mat(name: str) -> dict[tuple[str, str], float]:
        return {
            (s, c): pred[name][i][j]
            for i, s in enumerate(srcs)
            for j, c in enumerate(pred["bystanders"])
        }

    mats = {
        "js_rb": mat("js_rb_matrix"),
        "js_v1": mat("js_v1_matrix"),
        "gauss_kl": mat("gauss_kl_matrix"),
        "cosine": mat("cosine_matrix"),
    }

    emis: dict[tuple[str, str], float] = {}
    for f in glob.glob("eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_*.json"):
        a, b = f.split("cell_loc_ep1_")[-1][:-5].split("__")
        emis[(a, b)] = json.load(open(f))["summary"]["in_R_emission_rate"]

    pairlen: dict[tuple[str, str], tuple[float, float]] = {}
    for f in glob.glob("eval_results/issue_540/per_pair/pair_*.json"):
        d = json.load(open(f))
        a, b = d["pair"]["a"], d["pair"]["b"]
        la = float(np.mean([r["n_positions"] for r in d["per_sample"] if r["side"] == "a"]))
        lb = float(np.mean([r["n_positions"] for r in d["per_sample"] if r["side"] == "b"]))
        pairlen[(a, b)] = (la, lb)

    def dlen(x: str, y: str) -> float:
        if x == y:
            return 0.0
        if (x, y) in pairlen:
            la, lb = pairlen[(x, y)]
        else:
            lb, la = pairlen[(y, x)]
        return abs(la - lb)

    cells = [(s, c) for s in srcs for c in srcs]  # ordinary strip, diagonal included
    y = np.array([emis[c] for c in cells])
    xd = np.array([dlen(*c) for c in cells])
    is_diag = np.array([a == b for a, b in cells])

    set_paper_style("blog")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel A — raw scatter: length difference vs emission
    rho_d, p_d = spearmanr(xd, y)
    rng = np.random.default_rng(42)
    jit = rng.uniform(-1.5, 1.5, size=len(xd))
    ax_a.scatter(
        xd[~is_diag] + jit[~is_diag],
        y[~is_diag],
        s=18,
        alpha=0.55,
        color=paper_palette_role("primary"),
        label="cross-context cells (n=240)",
    )
    ax_a.scatter(
        xd[is_diag],
        y[is_diag],
        s=26,
        alpha=0.9,
        color=paper_palette_role("accent"),
        marker="D",
        label="same-context (self) cells (n=16)",
    )
    ax_a.set_xlabel("|mean response length side A − side B| (tokens)")
    ax_a.set_ylabel("Marker emission rate (trained model, on-policy)")
    ax_a.legend(loc="upper right")
    ax_a.text(
        0.97,
        0.62,
        f"Spearman ρ = {rho_d:.2f}\np = {p_d:.1e}, n = 256",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
    )
    set_title_subtitle(ax_a, "A cheap length feature predicts leakage", "")

    # Panel B — raw vs length-partialled rho per predictor
    names = ["js_rb", "js_v1", "gauss_kl", "cosine"]
    raw_rhos, par_rhos, par_ps = [], [], []
    for n in names:
        x = np.array([mats[n][c] for c in cells])
        raw_rhos.append(spearmanr(x, y)[0])
        pr, pp = partial_spearman(x, y, xd)
        par_rhos.append(pr)
        par_ps.append(pp)

    xpos = np.arange(len(names))
    w = 0.38
    ax_b.bar(
        xpos - w / 2,
        raw_rhos,
        w,
        color=paper_palette_role("primary"),
        label="raw ρ vs emission",
    )
    ax_b.bar(
        xpos + w / 2,
        par_rhos,
        w,
        color=paper_palette_role("neutral"),
        label="ρ after removing length diff",
    )
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.axhline(
        rho_d,
        color=paper_palette_role("accent"),
        lw=1.2,
        ls="--",
        label=f"length diff alone (ρ = {rho_d:.2f})",
    )
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels(
        [
            "Canonical\nsequence JS (RB)",
            "First-token JS\n(deprecated v1)",
            "Activation\nGaussian KL",
            "Activation\ncosine",
        ],
        fontsize=8,
    )
    for i, (pr, pp) in enumerate(zip(par_rhos, par_ps)):
        ax_b.text(
            xpos[i] + w / 2,
            pr - 0.03 if pr < 0 else pr + 0.015,
            f"p={pp:.0e}" if pp < 0.01 else f"p={pp:.2f}",
            ha="center",
            va="top" if pr < 0 else "bottom",
            fontsize=7,
        )
    ax_b.set_ylabel("Spearman ρ vs marker emission (ordinary strip)")
    ax_b.set_ylim(-0.78, 0.78)
    ax_b.legend(loc="upper left", fontsize=7, framealpha=0.9)
    set_title_subtitle(ax_b, "What survives the length control", "")

    fig.tight_layout()
    savefig_paper(fig, "issue_540/length_nuisance_ordinary", dir="figures/")
    plt.close(fig)
    print("raw rhos:", dict(zip(names, [f"{r:.3f}" for r in raw_rhos])))
    print("partial rhos:", dict(zip(names, [f"{r:.3f}" for r in par_rhos])))
    print("partial ps:", dict(zip(names, [f"{p:.2e}" for p in par_ps])))


if __name__ == "__main__":
    main()
