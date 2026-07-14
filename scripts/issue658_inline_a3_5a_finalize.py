"""Finalize the issue #658 A3.5a coherence artifact: add the honest out-of-sample
verdict + interpretation to coherence_results.json, and regenerate figures.

Key methodological point (drives the corrected headline):
  * The RELIABLE test of the assumption is the OUT-OF-SAMPLE residual: for each
    condition, fit the c->profile map on the OTHER 49 conditions, predict the
    held-out condition's centroid. Linear-ridge leave-one-condition-out (LOCO)
    residual rises with within-condition spread (rho ~ +0.8), robust to dropping
    format and WITHIN the persona family -> SUPPORTS the assumption.
  * The in-sample all-data nonlinear MLP gap/residual anti-correlate with spread,
    but that is a spread-dependent in-sample-optimism artifact (a flexible map
    interpolates a high-spread condition's own centroid having trained on its
    probes; format in-sample residual 652 vs LOCO residual 77572) -> NOT a valid
    sign for the assumption.
"""

import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RES = "eval_results/issue_658/inline_a3_5a_coherence"
FIG = "figures/issue_658"
PRIMARY_L = 14
FAM_COLORS = {
    "persona": "#0072B2",
    "wildchat": "#D55E00",
    "icl": "#009E73",
    "rephrase": "#CC79A7",
    "format": "#E69F00",
    "behavior": "#56B4E9",
    "default": "#000000",
}
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


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def short(cid):
    return cid.split("_", 1)[1] if "_" in cid else cid


def main():
    d = json.load(open(os.path.join(RES, "coherence_results.json")))
    arr = np.load(os.path.join(RES, "per_condition_layer.npz"))
    cids = d["meta"]["ctx_ids"]
    fams = np.array(d["meta"]["families"])
    nL = d["meta"]["n_layers"]

    sW = arr["spread_W"]
    Rloco = arr["Rlin_loco_max"]
    Rin = arr["R_max"]
    Jin = arr["J_max"]

    rloco = [spearmanr(sW[L], Rloco[L])[0] for L in range(nL)]
    rin_R = [spearmanr(sW[L], Rin[L])[0] for L in range(nL)]
    rin_J = [spearmanr(sW[L], Jin[L])[0] for L in range(nL)]

    # robustness of the honest LOCO read (layer-mean)
    sWm, Rlom = sW.mean(0), Rloco.mean(0)
    nf = fams != "format"
    pm = fams == "persona"
    d["overall_honest_loco"] = {
        "measure": "linear-ridge leave-one-condition-out residual (out-of-sample)",
        "median_layerwise_rho_sW_Rloco": float(np.median(rloco)),
        "frac_layers_positive": float(np.mean([r > 0 for r in rloco])),
        "layer_mean_rho_all": float(spearmanr(sWm, Rlom)[0]),
        "layer_mean_rho_excl_format": float(spearmanr(sWm[nf], Rlom[nf])[0]),
        "within_persona_rho": float(spearmanr(sWm[pm], Rlom[pm])[0]),
    }
    d["interpretation"] = {
        "verdict": "SUPPORTED (by the reliable out-of-sample read)",
        "one_line": (
            "Within-condition context-vector spread POSITIVELY predicts the held-out "
            "(leave-one-condition-out) mean-vector-predictor residual (rho ~ +0.8, robust to "
            "dropping format and within the persona family) -> the assumption's prediction holds: "
            "less-coherent conditions are worse for the prefix-average summary."
        ),
        "in_sample_confound": (
            "The in-sample all-data nonlinear MLP gap/residual anti-correlate with spread "
            f"(median rho {np.median(rin_R):+.2f}), but this is a spread-dependent in-sample-optimism "
            "artifact (a flexible map interpolates a high-spread condition's own centroid having "
            "trained on its probes; e.g. format in-sample residual 652 vs LOCO residual 77572), "
            "NOT a valid test of the assumption."
        ),
        "jensen_curvature_mechanism": (
            "The specific Jensen-curvature mechanism was NOT cleanly isolated: the nonlinear Jensen "
            "gap is identically 0 for a linear map and could only be measured here with an in-sample "
            "MLP (optimism-confounded). A clean curvature test needs a leave-one-condition-out "
            "NONLINEAR fit (~50x MLP fits/layer), deferred."
        ),
        "family_prediction": (
            "Partially matches: persona is tightest in context space (whitened spread 1067) and "
            "wildchat is among the most scattered (1458) as predicted; but format is the MOST "
            "scattered in context space (1467) though its read-out behaviors are floored, so its "
            "context scatter does not translate to behavior variance."
        ),
    }
    json.dump(d, open(os.path.join(RES, "coherence_results.json"), "w"), indent=1)

    # ---- Figure: layerwise rho, honest LOCO (positive) vs in-sample (negative) ----
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axhline(0, color="0.3", lw=1)
    ax.plot(
        range(nL),
        rloco,
        "-o",
        ms=4,
        color="#009E73",
        label="OUT-OF-SAMPLE: rho(s_W, linear-ridge LOCO residual)  [reliable]",
    )
    ax.plot(
        range(nL),
        rin_R,
        "--s",
        ms=3,
        color="#D55E00",
        label="in-sample MLP: rho(s_W, residual)  [optimism-confounded]",
    )
    ax.plot(
        range(nL),
        rin_J,
        ":^",
        ms=3,
        color="#E69F00",
        label="in-sample MLP: rho(s_W, Jensen gap)  [optimism-confounded]",
    )
    ax.axvline(PRIMARY_L, color="0.7", ls=":", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("within-layer Spearman rho across 50 conditions")
    ax.set_title(
        "A3.5a: within-condition spread vs mean-vector-predictor error\n"
        "reliable out-of-sample read is POSITIVE at every layer (supports the assumption)",
        fontsize=10,
    )
    ax.legend(fontsize=7.2, loc="lower left")
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    p = os.path.join(FIG, "fig_a35a_layerwise_rho.png")
    fig.savefig(p)
    plt.close(fig)

    # ---- Figure: honest LOCO scatter (spread vs out-of-sample residual), log-y ----
    L = PRIMARY_L
    x, y = sW[L], Rloco[L]
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
    ax.set_yscale("log")
    ax.set_xlabel(f"within-condition whitened spread  s_W(C)   [layer {L}]")
    ax.set_ylabel(
        f"out-of-sample residual  R_loco(C) = max_B |r_B.v0(C) - h_-C(c_hat_C)|  [layer {L}, log]"
    )
    ax.set_title(
        f"A3.5a (reliable read): spread vs OUT-OF-SAMPLE residual (n=50, layer {L})\n"
        f"Spearman rho = {spearmanr(x, y)[0]:+.2f} (assumption predicts POSITIVE) — SUPPORTED",
        fontsize=10,
    )
    ax.legend(title="family", fontsize=7, title_fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    p2 = os.path.join(FIG, "fig_a35a_spread_vs_residual_loco_L%d.png" % L)
    fig.savefig(p2)
    plt.close(fig)

    # ---- relabel the two in-sample scatters with the optimism-confound caveat ----
    def insample_scatter(yv, ylab, kind, fname):
        xv = sW[L]
        rho = spearmanr(xv, yv)[0]
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        seen = set()
        for i in range(len(xv)):
            fm = fams[i]
            ax.scatter(
                xv[i],
                yv[i],
                c=FAM_COLORS.get(fm, "#888"),
                s=42,
                edgecolor="white",
                linewidth=0.5,
                label=fm if fm not in seen else None,
                zorder=3,
            )
            seen.add(fm)
            ax.annotate(
                short(cids[i]),
                (xv[i], yv[i]),
                fontsize=5.4,
                alpha=0.75,
                xytext=(3, 2),
                textcoords="offset points",
            )
        b, a = np.polyfit(xv, yv, 1)
        xs = np.array([xv.min(), xv.max()])
        ax.plot(xs, a + b * xs, color="0.4", ls="--", lw=1, zorder=2)
        ax.set_xlabel(f"within-condition whitened spread  s_W(C)   [layer {L}]")
        ax.set_ylabel(ylab)
        ax.set_title(
            f"A3.5a IN-SAMPLE MLP (optimism-confounded — see LOCO figure for the "
            f"reliable read):\nspread vs {kind} (n=50, layer {L}). Spearman rho = "
            f"{rho:+.2f} — negative sign is an in-sample artifact, NOT a valid test",
            fontsize=9,
        )
        ax.legend(title="family", fontsize=7, title_fontsize=7, loc="best", framealpha=0.9)
        fig.tight_layout()
        outp = os.path.join(FIG, fname)
        fig.savefig(outp)
        plt.close(fig)
        return outp

    p3 = insample_scatter(
        Jin[L],
        f"behavior Jensen gap J(C) [layer {L}, in-sample MLP]",
        "Jensen gap",
        f"fig_a35a_spread_vs_jensen_L{L}.png",
    )
    p4 = insample_scatter(
        Rin[L],
        f"context->profile residual R(C) [layer {L}, in-sample MLP]",
        "residual",
        f"fig_a35a_spread_vs_residual_L{L}.png",
    )

    for pth in [p, p2, p3, p4]:
        json.dump(
            {
                "figure": os.path.basename(pth),
                "issue": 658,
                "analysis": "A3.5a within-condition coherence",
                "git_commit": commit(),
                "primary_layer": PRIMARY_L,
                "source": os.path.join(RES, "coherence_results.json"),
            },
            open(pth.replace(".png", ".meta.json"), "w"),
            indent=1,
        )
        print("WROTE", pth)
    print("HONEST LOCO:", json.dumps(d["overall_honest_loco"], indent=1))


if __name__ == "__main__":
    main()
