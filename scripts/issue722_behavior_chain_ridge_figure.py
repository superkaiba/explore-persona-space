#!/usr/bin/env python3
"""Combined ridge + KRR-RBF readout figure for the #722 behavior chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

RIDGE_JSON = PROJECT_ROOT / "eval_results/issue_722/structural/behavior_chain_ridge_readout.json"
KRR_JSON = PROJECT_ROOT / "eval_results/issue_722/structural/behavior_chain_krr_readout.json"
OUT = PROJECT_ROOT / "figures/issue_722/behavior_chain_ridge_readout.png"

BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
LABELS = {
    "broad_em": "Broad EM",
    "harmful_compliance": "Harmful\ncompliance",
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
}
STRONG = ["sycophancy", "refusal"]


def main():
    ridge = json.loads(RIDGE_JSON.read_text())
    krr = json.loads(KRR_JSON.read_text())
    rpb, kpb = ridge["per_behavior"], krr["per_behavior"]
    cc_variant = ridge["cc_variant"]
    kdim = krr.get("pca_target_dim_krr", 16)
    set_paper_style()
    c = paper_palette(5)  # ridge-dir, ridge-med, krr-dir, krr-med, cc-dir

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # ── left: grouped best-layer rho bars (5 readouts per behavior) ─────────
    series = [
        ("ridge direct (true v0, PCA-48)", lambda b: rpb[b]["best_ridge_direct"]["rho"], c[0]),
        ("ridge mediated (c_C->v0)", lambda b: rpb[b]["best_ridge_mediated"]["rho"], c[1]),
        (f"KRR-RBF direct (true v0, PCA-{kdim})", lambda b: kpb[b]["best_krr_direct"]["rho"], c[2]),
        ("KRR-RBF mediated (c_C->v0)", lambda b: kpb[b]["best_krr_mediated"]["rho"], c[3]),
        ("ridge direct from c_C", lambda b: rpb[b]["best_cc_direct"]["rho"], c[4]),
    ]
    x = np.arange(len(BEHAVIORS))
    w = 0.16
    for j, (lab, fn, col) in enumerate(series):
        ax_bar.bar(x + (j - 2) * w, [fn(b) for b in BEHAVIORS], w, label=lab, color=col)
    ax_bar.axhline(0, color="0.4", lw=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABELS[b] for b in BEHAVIORS])
    ax_bar.set_ylabel(r"best-layer held-out Spearman $\rho$ (pred vs actual $E_0$)")
    ax_bar.set_title("Behavior decoding: ridge vs KRR-RBF readout")
    ax_bar.legend(fontsize=6.6, loc="upper left", ncol=1)
    ax_bar.set_ylim(-0.05, 1.0)
    # nonlinear gap + degradation + KRR shuffle-null in text only
    txt = []
    for b in BEHAVIORS:
        nlg = kpb[b]["nonlinear_gap_best"]
        rdeg = rpb[b]["best_layer_degradation"]
        sh = kpb[b].get("sanity_shuffle_null_krr_direct")
        short = LABELS[b].replace("\n", " ")
        shs = f" kShuf={sh:+.2f}" if sh is not None else ""
        txt.append(f"{short[:11]:11s} NLgap={nlg:+.2f} rDeg={rdeg:+.2f}{shs}")
    txt.append("(KRR LOO-null is strongly negative at n=50 — see report)")
    ax_bar.text(
        0.985,
        0.015,
        "\n".join(txt),
        transform=ax_bar.transAxes,
        fontsize=6.0,
        va="bottom",
        ha="right",
        family="monospace",
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.92},
    )

    # ── right: per-layer rho lines, strong behaviors ────────────────────────
    layers = krr["layers"]
    ls = {"sycophancy": "-", "refusal": "--"}
    for b in STRONG:
        rows = kpb[b]["per_layer"]
        ax_line.plot(
            layers,
            [r["rho_ridge_direct"] for r in rows],
            ls[b],
            color=c[0],
            lw=1.6,
            marker="o",
            ms=2.5,
            label=f"{b} — ridge direct",
        )
        ax_line.plot(
            layers,
            [r["rho_krr_direct"] for r in rows],
            ls[b],
            color=c[2],
            lw=1.6,
            marker="^",
            ms=2.5,
            label=f"{b} — KRR direct",
        )
        ax_line.plot(
            layers,
            [r["rho_krr_mediated"] for r in rows],
            ls[b],
            color=c[3],
            lw=1.6,
            marker="s",
            ms=2.5,
            label=f"{b} — KRR mediated",
        )
    ax_line.axhline(0, color="0.4", lw=0.8)
    ax_line.set_xlabel("layer")
    ax_line.set_ylabel(r"held-out Spearman $\rho$")
    ax_line.set_title("Per-layer ridge vs KRR-RBF (strong behaviors)")
    ax_line.legend(fontsize=6.5, ncol=2, loc="lower center")
    ax_line.set_xlim(-0.5, 27.5)

    fig.suptitle(
        "Does the c_C->v0 linear approximation degrade BEHAVIOR decoding — under a "
        "strong linear (ridge, PCA-48) AND a nonlinear (KRR-RBF, PCA-16) readout?\n"
        f"#722 chain  |  LOCO n=50  |  c_C = {cc_variant}  |  Betley genre  |  "
        "NLgap = best-layer rho(KRR) - rho(ridge); rDeg = ridge direct - mediated",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, OUT)
    print(f"[info] wrote {OUT}")


if __name__ == "__main__":
    main()
