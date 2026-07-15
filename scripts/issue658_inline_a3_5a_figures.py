"""Figures for issue #658 inline A3.5a within-condition coherence test.
Low-level per-unit view (required): per-condition labeled scatters, family-colored.
  fig_a35a_spread_vs_jensen_L{L}.png   : whitened spread vs behavior Jensen gap
  fig_a35a_spread_vs_residual_L{L}.png  : whitened spread vs context->profile residual
  fig_a35a_layerwise_rho.png            : within-layer Spearman(sW,J/R) across 28 layers
"""

import json
import os
import subprocess

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib/numpy imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = "eval_results/issue_658/inline_a3_5a_coherence"
FIG = "figures/issue_658"
os.makedirs(FIG, exist_ok=True)
PRIMARY_L = 14  # mid-network; A3.5 locked region (ridge cos 0.31 / mlp cos 0.78 at L14)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)
# colorblind-safe (Okabe-Ito)
FAM_COLORS = {
    "persona": "#0072B2",
    "wildchat": "#D55E00",
    "icl": "#009E73",
    "rephrase": "#CC79A7",
    "format": "#E69F00",
    "behavior": "#56B4E9",
    "default": "#000000",
}


def short(cid):
    return cid.split("_", 1)[1] if "_" in cid else cid


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def scatter(x, y, fams, cids, xlab, ylab, title, rho, fname):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    seen = set()
    for i in range(len(x)):
        f = fams[i]
        ax.scatter(
            x[i],
            y[i],
            c=FAM_COLORS.get(f, "#888"),
            s=42,
            edgecolor="white",
            linewidth=0.5,
            label=f if f not in seen else None,
            zorder=3,
        )
        seen.add(f)
        ax.annotate(
            short(cids[i]),
            (x[i], y[i]),
            fontsize=5.4,
            alpha=0.75,
            xytext=(3, 2),
            textcoords="offset points",
        )
    # OLS guide line
    b, a = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color="0.4", ls="--", lw=1, zorder=2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{title}\nSpearman rho = {rho:+.2f} (assumption predicts POSITIVE)", fontsize=10)
    ax.legend(title="family", fontsize=7, title_fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    return os.path.join(FIG, fname)


def main():
    with open(os.path.join(RES, "coherence_results.json")) as fh:
        d = json.load(fh)
    arr = np.load(os.path.join(RES, "per_condition_layer.npz"))
    cids = d["meta"]["ctx_ids"]
    fams = d["meta"]["families"]
    L = PRIMARY_L
    sW = arr["spread_W"][L]
    J = arr["J_max"][L]
    R = arr["R_max"][L]
    rhoJ = d["per_layer"][str(L)]["spearman_sW_J_all"]["rho"]
    rhoR = d["per_layer"][str(L)]["spearman_sW_R_all"]["rho"]

    paths = []
    paths.append(
        scatter(
            sW,
            J,
            fams,
            cids,
            f"within-condition whitened spread  s_W(C)   [layer {L}]",
            "behavior-relevant Jensen gap  J(C) = max_B |mean f_B(c_x) - f_B(c_hat_C)|  "
            f"[layer {L}]",
            f"A3.5a: within-condition spread vs Jensen gap (n=50 conditions, layer {L})",
            rhoJ,
            f"fig_a35a_spread_vs_jensen_L{L}.png",
        )
    )
    paths.append(
        scatter(
            sW,
            R,
            fams,
            cids,
            f"within-condition whitened spread  s_W(C)   [layer {L}]",
            f"context->profile residual  R(C) = max_B |r_B.v0(C) - f_B(c_hat_C)|  [layer {L}]",
            f"A3.5a: within-condition spread vs prediction residual (n=50, layer {L})",
            rhoR,
            f"fig_a35a_spread_vs_residual_L{L}.png",
        )
    )

    # layerwise rho curve
    n_layers = d["meta"]["n_layers"]
    rj = [d["per_layer"][str(k)]["spearman_sW_J_all"]["rho"] for k in range(n_layers)]
    rr = [d["per_layer"][str(k)]["spearman_sW_R_all"]["rho"] for k in range(n_layers)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(0, color="0.3", lw=1)
    ax.plot(range(n_layers), rj, "-o", ms=4, color="#0072B2", label="rho(s_W, Jensen gap)")
    ax.plot(range(n_layers), rr, "-s", ms=4, color="#D55E00", label="rho(s_W, residual)")
    ax.axvline(PRIMARY_L, color="0.7", ls=":", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("within-layer Spearman rho across 50 conditions")
    ax.set_title(
        "A3.5a: within-layer rho(spread, gap/residual) is negative at EVERY layer\n"
        "(assumption predicts POSITIVE; 0/28 layers positive)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    p3 = os.path.join(FIG, "fig_a35a_layerwise_rho.png")
    fig.savefig(p3)
    plt.close(fig)
    paths.append(p3)

    for p in paths:
        meta = {
            "figure": os.path.basename(p),
            "issue": 658,
            "analysis": "A3.5a within-condition coherence",
            "git_commit": commit(),
            "primary_layer": PRIMARY_L,
            "source": "eval_results/issue_658/inline_a3_5a_coherence/coherence_results.json",
        }
        with open(p.replace(".png", ".meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print("WROTE", p)


if __name__ == "__main__":
    main()
